"""Code-level refinement of IQ3_XXS / IQ2_XXS routed experts under the full input Gram (gate T).

Port of inkling-2-mini-quant/scripts/refine-codes.py (Addendum H amendments 9, 9b, 9c) to bailingmoe3:

    .venv/bin/python -I scripts/refine_codes.py --quant models/tmp/U3.gguf --grams models/grams/mixc-c512-n400 \
        --layers 1 2 3 --out-dir models/refined/R3 [--experts 0:128:1] [--workers 10]

Per layer L and expert e: gate and up (blk.L.ffn_{gate,up}_exps, 512 rows x 1536 inputs each) share one input Gram
(the gate Gram; same input vector), so their 1024 rows are refined together against it; down (1536 rows x 512
inputs) against its own Gram. Every block keeps its fp16 d and 4-bit sub-block scales; only grid indices and signs
change. With E = dequantized minus BF16 (float64) and the symmetric Gram G (upper triangle of the capture,
symmetrised), the objective is J = tr(E H E^T), H = G + shrink * Diag(G) (amendment 9b, shrink 0.1), separable over
rows; R = E H. Guards as in the original: only experts with at least --min-tokens Gram tokens are refined, default
= the input dim (1536 gate/up, 512 down), others keep --quant's bytes; experts with a negative fp16 d keep theirs;
amendment 9c output-gain floor (a move is taken only if the row's numerator q_r H w_r^T stays >= llama-quantize's,
tracked as a change from 0; --no-gain-guard drops it).

One group = the values sharing one grid index: 4 for IQ3_XXS, 8 for IQ2_XXS. Candidates, exhaustive:
  IQ3_XXS: 256 grids x the 8 sign patterns whose parity keeps the 8-sign word even (partner group held fixed).
  IQ2_XXS: 256 grids x the 128 even-parity sign patterns of the group's own 7-bit sign index (each 8-weight group
    has its own sign word in IQ2_XXS, so no partner constraint): 32,768 candidates.
Ranking: score(c) = c^T Gb c + c . (2 b / db), b = R_idx - Gb cur, which is dJ(db c) / db^2 up to a constant
(db > 0; rows with db = 0 cannot move), in float32 by a C kernel (scripts/rank_kernel.c), the best 4 per row kept; with the gain guard, candidates with
c . uw < thr / db are not ranked (the floor binds often here: an unguarded ranking's winner broke it in ~30% of rows). The ranked best 4 per row are then scored exactly in float64 with v = db * grid
* sign in float32 exactly as gguf-py decodes it: dJ = 2 d.R_idx + d Gb d, d = v - cur; accepted if dJ < 0 (and the
exact gain floor holds). Accepted rows update E, R += d H[idx, :] and the raw bytes. Groups in ascending input column,
all rows in parallel; sweeps until (J_before - J_after) / J_before < 1e-4 or 4 sweeps; R and J recomputed at each
sweep start.

Output: per layer, OUT_DIR/blk.L.gguf with the three refined expert tensors (types, shapes and bytes layout as in
--quant) and OUT_DIR/blk.L.json (per expert and tensor: J under H and G before/after each sweep, accepted moves, gain
floor blocks, seconds; checks). Every refined tensor is checked: our decode equals gguf-py's dequantize bit for bit
before and after, d and scale bits are unchanged, every value is db * level * (+-1), IQ3_XXS words keep even parity,
and the final J under H is <= the starting J.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import gguf
from gguf import GGMLQuantizationType as Q
from gguf import quants as gq
from gguf.quants import dequantize

QK_K = 256
KSIGNS = np.frombuffer(gq.IQ2_XXS.ksigns, dtype=np.uint8)
POPC = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
TENSORS = ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")


class Fmt:
    def __init__(self, qtype, k, bsize, cls):
        cls.init_grid()
        self.qtype, self.k, self.bsize = qtype, k, bsize
        self.gpb = QK_K // k  # groups per block
        self.grid = np.ascontiguousarray(cls.grid.reshape(-1, k), dtype=np.float32)  # [256, k], unsigned
        self.levels = np.array(cls.grid_map, dtype=np.float32)
        assert gq.GGML_QUANT_SIZES[qtype] == (QK_K, bsize)
        # candidate sets, pattern-major: cand c = pats[c // 256], grid[c % 256]; for IQ3_XXS one set per parity
        if qtype == Q.IQ2_XXS:
            pats = ((KSIGNS[:, None] >> np.arange(8, dtype=np.uint8)) & 1).astype(bool)  # [128, 8], even parity
            self.pat_sets = [pats]
        else:
            p16 = np.array([[(p >> i) & 1 for i in range(4)] for p in range(16)], dtype=bool)
            self.pat_sets = [p16[np.array([bin(p).count("1") % 2 == q for p in range(16)])] for q in (0, 1)]
        self.cands = [(np.where(p, -1.0, 1.0)[:, None, :] * self.grid[None].astype(np.float64)).reshape(-1, k)
                      for p in self.pat_sets]


FMTS = {Q.IQ3_XXS: Fmt(Q.IQ3_XXS, 4, 98, gq.IQ3_XXS), Q.IQ2_XXS: Fmt(Q.IQ2_XXS, 8, 66, gq.IQ2_XXS)}


def _u32(b4: np.ndarray) -> np.ndarray:
    s = b4.astype(np.uint32)
    return s[..., 0] | (s[..., 1] << 8) | (s[..., 2] << 16) | (s[..., 3] << 24)


def fields(f: Fmt, raw: np.ndarray):
    """raw uint8 [rows, nb * bsize] -> (db float32 [rows, ngroups], idx int64 [rows, ngroups], neg bool [rows, ngroups, k]),
    db computed exactly as gguf-py computes it; neg marks a minus sign."""
    rows = raw.shape[0]
    b = raw.reshape(rows, -1, f.bsize)
    nb = b.shape[1]
    d = b[..., :2].copy().view(np.float16).reshape(rows, nb, 1).astype(np.float32)
    g = np.arange(f.gpb)
    if f.qtype == Q.IQ3_XXS:
        idx = b[..., 2:66].astype(np.int64)
        aux = _u32(b[..., 66:98].reshape(rows, nb, 8, 4))  # [rows, nb, 8]
        a = aux[..., g // 8]
        db = d * (np.float32(0.5) + (a >> 28).astype(np.float32)) * np.float32(0.5)
        byte = KSIGNS[(a >> (7 * ((g % 8) // 2)).astype(np.uint32)) & 127]
        nib = (byte >> (4 * (g % 2)).astype(np.uint8)) & 0x0F
        neg = ((nib[..., None] >> np.arange(4, dtype=np.uint8)) & 1).astype(bool)
    else:  # IQ2_XXS: per 32 weights 4 index bytes then a uint32 of 4 x 7 sign bits and a 4-bit scale
        q = b[..., 2:66].reshape(rows, nb, 8, 8)
        ib, l = g // 4, g % 4
        idx = q[:, :, ib, l].astype(np.int64)
        aux = _u32(q[..., 4:8])  # [rows, nb, 8]
        a = aux[..., ib]
        db = d * (np.float32(0.5) + (a >> 28).astype(np.float32)) * np.float32(0.25)
        byte = KSIGNS[(a >> (7 * l).astype(np.uint32)) & 127]
        neg = ((byte[..., None] >> np.arange(8, dtype=np.uint8)) & 1).astype(bool)
    return db.reshape(rows, -1), idx.reshape(rows, -1), neg.reshape(rows, -1, f.k)


def values(f: Fmt, db: np.ndarray, idx: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """Dequantized float32 [rows, ngroups * k], same operation order as gguf-py: (db * grid) * sign."""
    sign = np.where(neg, np.float32(-1), np.float32(1))
    return (db[..., None] * f.grid[idx] * sign).reshape(db.shape[0], -1)


def decode(f: Fmt, raw: np.ndarray) -> np.ndarray:
    return values(f, *fields(f, raw))


def set_group(f: Fmt, raw: np.ndarray, rows: np.ndarray, gi: int, idx: np.ndarray, neg: np.ndarray) -> None:
    """In place: group gi of the given rows gets grid index idx [n] and signs neg [n, k] (True = minus); d, scales and
    every other group's bits are unchanged."""
    blk, g = divmod(gi, f.gpb)
    o = blk * f.bsize
    idx = np.asarray(idx, dtype=np.int64)
    if np.any((idx < 0) | (idx >= 256)):
        raise ValueError("grid index out of range")
    bits = (np.asarray(neg, dtype=np.uint8) << np.arange(f.k, dtype=np.uint8)).sum(axis=1).astype(np.uint8)
    if f.qtype == Q.IQ3_XXS:
        raw[rows, o + 2 + g] = idx.astype(np.uint8)
        w = o + 66 + 4 * (g // 8)
        l, h = (g % 8) // 2, g % 2
    else:
        ib, l = divmod(g, 4)
        raw[rows, o + 2 + 8 * ib + l] = idx.astype(np.uint8)
        w = o + 2 + 8 * ib + 4
    aux = _u32(raw[rows, w:w + 4])
    if f.qtype == Q.IQ3_XXS:
        byte = KSIGNS[(aux >> np.uint32(7 * l)) & 127]
        byte = (byte & np.uint8(~(0x0F << (4 * h)) & 0xFF)) | (bits << np.uint8(4 * h))
    else:
        byte = bits
    if np.any(POPC[byte] % 2):
        raise ValueError("sign word with an odd number of minus signs is not representable")
    aux = (aux & np.uint32(~(127 << (7 * l)) & 0xFFFFFFFF)) | ((byte.astype(np.uint32) & 127) << np.uint32(7 * l))
    for i in range(4):
        raw[rows, w + i] = ((aux >> np.uint32(8 * i)) & 0xFF).astype(np.uint8)


def fixed_bits_mask(f: Fmt, row_bytes: int) -> np.ndarray:
    """uint8 mask [row_bytes] of the bits refinement must not touch (d, scales)."""
    m = np.zeros((row_bytes // f.bsize, f.bsize), dtype=np.uint8)
    m[:, :2] = 0xFF
    if f.qtype == Q.IQ3_XXS:
        m[:, 66 + 3:98:4] = 0xF0  # top 4 bits of each little-endian uint32
    else:
        m[:, 2 + 7:66:8] = 0xF0
    return m.reshape(-1)


def validate(f: Fmt, raw: np.ndarray, ref: np.ndarray | None = None) -> None:
    """gguf-py's dequantize equals our decode bit for bit (and ref, if given); every value is finite and
    db * level * (+-1); every 8-sign word has even parity."""
    db, idx, neg = fields(f, raw)
    mine = values(f, db, idx, neg)
    theirs = dequantize(raw, f.qtype).reshape(mine.shape)
    if not np.array_equal(mine.view(np.uint32), theirs.astype(np.float32).view(np.uint32)):
        raise AssertionError("decode differs from gguf-py dequantize")
    if ref is not None and not np.array_equal(mine.view(np.uint32), ref.view(np.uint32)):
        raise AssertionError("decode differs from the tracked values")
    if not np.all(np.isfinite(mine)):
        raise AssertionError("non-finite values")
    lv = np.abs(mine).reshape(*db.shape, f.k)
    cand = np.abs(db)[..., None, None] * f.levels
    if not np.all(np.any(lv[..., None] == cand, axis=-1)):
        raise AssertionError("a value is not db * level")
    word = neg.reshape(neg.shape[0], -1, 8).sum(axis=2)
    if np.any(word % 2):
        raise AssertionError("odd sign parity")


# ---- ranking ------------------------------------------------------------------------------------------------------

class Ranker:
    """Exhaustive candidate ranking in float32 (C kernel scripts/rank_kernel.c, build/librank.dylib, one thread).
    pick(...) -> the `top` best per row as (grid idx [r, top], neg [r, top, k]); float32 can misorder near-ties,
    so the caller rescores these exactly in float64. A slot with no feasible candidate repeats the row's first."""

    def __init__(self, f: Fmt, top: int = 4):
        import ctypes
        lib = ctypes.CDLL(str(Path(__file__).resolve().parents[1] / "build/librank.dylib"))
        self.fn = lib.rank_top
        self.fn.restype = None
        self.f, self.top = f, top
        C = np.vstack(f.cands)  # [n_total, k], sets back to back
        self.n = [len(c) for c in f.cands]
        self.off = np.cumsum([0] + self.n[:-1]).astype(np.int32)
        self.C = C
        self.CT = np.ascontiguousarray(C.T, dtype=np.float32)
        self.ptr = lambda a: a.ctypes.data_as(ctypes.c_void_p)

    def pick(self, Gb, b, db, cls, uw=None, thr=None):
        """Gb [k, k] float64, b [r, k] = R_idx - cur Gb, db [r] float32 (> 0 where cls >= 0), cls [r] candidate set
        per row (IQ3_XXS parity; zeros for IQ2_XXS; -1 = skip), uw [r, k], thr [r] (gain guard, or None)."""
        f, r = self.f, len(db)
        with np.errstate(all="ignore"):  # Accelerate's BLAS raises spurious FP flags under numpy on macOS
            qc = np.ascontiguousarray(((self.C @ Gb) * self.C).sum(1), dtype=np.float32)
        if not np.all(np.isfinite(qc)):
            raise FloatingPointError("non-finite candidate norms")
        d64 = db.astype(np.float64)
        bp = np.ascontiguousarray(2 * b / d64[:, None], dtype=np.float32)
        cls32 = np.ascontiguousarray(cls, dtype=np.int32)
        out = np.empty((r, self.top), dtype=np.int32)
        if thr is not None:
            u32 = np.ascontiguousarray(uw, dtype=np.float32)
            th = np.ascontiguousarray(thr / d64, dtype=np.float32)
        n = np.array(self.n, dtype=np.int32)
        self.fn(ctypes_int(r), ctypes_int(f.k), ctypes_int(self.top), self.ptr(self.CT), ctypes_int(self.CT.shape[1]),
                self.ptr(self.off), self.ptr(n), self.ptr(qc), self.ptr(bp),
                self.ptr(u32) if thr is not None else None, self.ptr(th) if thr is not None else None,
                self.ptr(cls32), self.ptr(out))
        out = np.where(out < 0, np.maximum(out[:, :1], 0), out).astype(np.int64)  # infeasible slots: rescored, rejected
        pat = np.empty((r, self.top, f.k), dtype=bool)
        gidx = out % 256
        for q, ps in enumerate(f.pat_sets):
            m = cls32 == q
            pat[m] = ps[out[m] // 256]
        return gidx, pat


def ctypes_int(x):
    import ctypes
    return ctypes.c_int(int(x))


def refine(f: Fmt, raw: np.ndarray, w: np.ndarray, g: np.ndarray, ranker: Ranker, max_sweeps: int = 4,
           tol: float = 1e-4, g_report: np.ndarray | None = None, gain_guard: bool = True):
    """raw uint8 [rows, row_bytes] (not modified), w float32 [rows, cols], g float64 [cols, cols] symmetric (the
    objective's matrix H). Returns (new raw, new values float32, list of per-sweep dicts)."""
    raw = raw.copy()
    db, idx, neg = fields(f, raw)
    dq = values(f, db, idx, neg)
    w64 = w.astype(np.float64)
    E = dq.astype(np.float64) - w64
    with np.errstate(all="ignore"):  # Accelerate's BLAS raises spurious FP flags under numpy on macOS
        U = w64 @ g
    den_r = np.einsum("ri,ri->r", w64, U)
    num0 = np.einsum("ri,ri->r", dq.astype(np.float64), U)
    gch = np.zeros(len(num0))
    k, ngroups = f.k, db.shape[1]
    sweeps = []
    for s in range(max_sweeps):
        t0 = time.time()
        with np.errstate(all="ignore"):
            R = E @ g
        j0 = float(np.sum(E * R))
        if g_report is not None:
            with np.errstate(all="ignore"):
                jg0 = float(np.sum(E * (E @ g_report)))
        tracked, acc_n, blocked = j0, 0, 0
        for gi in range(ngroups):
            c = slice(gi * k, gi * k + k)
            Gb, Rg, cur = g[c, c], R[:, c], dq[:, c].astype(np.float64)
            uw = U[:, c]
            dbg = db[:, gi]
            live = dbg > 0
            if f.qtype == Q.IQ3_XXS:
                cls = neg[:, gi ^ 1].sum(axis=1) % 2  # this group's parity must equal its partner's
            else:
                cls = np.zeros(len(dbg), dtype=np.int64)
            cls = np.where(live, cls, -1)
            thr = np.einsum("ri,ri->r", cur, uw) - gch if gain_guard else None
            with np.errstate(all="ignore"):
                b = Rg - cur @ Gb
            dpos = np.where(live, dbg, np.float32(1))
            cis, cns = ranker.pick(Gb, b, dpos, cls, uw, thr)  # [r, m], [r, m, k]
            vs = dbg[:, None, None] * f.grid[cis] * np.where(cns, np.float32(-1), np.float32(1))  # float32, as decode
            dds = vs.astype(np.float64) - cur[:, None]
            with np.errstate(all="ignore"):
                djs = 2 * np.einsum("rmi,ri->rm", dds, Rg) + np.einsum("rmi,ij,rmj->rm", dds, Gb, dds)
            dnums = np.einsum("rmi,ri->rm", dds, uw)
            feas = gch[:, None] + dnums >= 0 if gain_guard else np.ones_like(djs, dtype=bool)
            pick_any = np.argmin(djs, axis=1)  # exact best among the ranked, ignoring the floor (for the count)
            pick = np.argmin(np.where(feas, djs, np.inf), axis=1)
            rr = np.arange(len(dbg))
            ci, cn, v, dd = cis[rr, pick], cns[rr, pick], vs[rr, pick], dds[rr, pick]
            dj, dnum = djs[rr, pick], dnums[rr, pick]
            ok = (dj < 0) & live
            if gain_guard:
                blocked += int(np.sum((djs[rr, pick_any] < 0) & live & ~feas[rr, pick_any] & ~ok))
                ok &= feas[rr, pick]
            a = np.nonzero(ok)[0]
            if len(a) == 0:
                continue
            dq[a, c] = v[a]
            E[a, c] = v[a].astype(np.float64) - w64[a, c]
            with np.errstate(all="ignore"):
                R[a] += dd[a] @ g[c, :]
            set_group(f, raw, a, gi, ci[a], cn[a])
            idx[a, gi], neg[a, gi] = ci[a], cn[a]
            gch[a] += dnum[a]
            tracked += float(dj[a].sum())
            acc_n += len(a)
        with np.errstate(all="ignore"):
            j1 = float(np.sum(E * (E @ g)))
        num_now = np.einsum("ri,ri->r", dq.astype(np.float64), U)
        pos = den_r > 0
        sweeps.append({"sweep": s + 1, "J_before": j0, "J_after": j1, "J_tracked": tracked, "accepted": acc_n,
                       "gain_blocked": blocked,
                       "min_row_ogain_change": float(np.min((num_now[pos] - num0[pos]) / den_r[pos])) if pos.any() else 0.0,
                       "seconds": round(time.time() - t0, 2)})
        if g_report is not None:
            with np.errstate(all="ignore"):
                sweeps[-1].update(JG_before=jg0, JG_after=float(np.sum(E * (E @ g_report))))
        if j0 <= 0 or (j0 - j1) / j0 < tol:
            break
    return raw, dq, sweeps


# ---- driver -------------------------------------------------------------------------------------------------------

def symmetrise(u: np.ndarray) -> np.ndarray:
    u = np.triu(u)
    return u + u.T - np.diag(np.diag(u))


def load_gram(grams: str, name: str, n_exp: int, n_in: int):
    base = f"{grams}/{name}.gram"
    if os.path.getsize(base + ".f32") != n_exp * n_in * n_in * 4:
        sys.exit(f"STOP: {base}.f32 has {os.path.getsize(base + '.f32')} bytes")
    u = np.memmap(base + ".f32", dtype=np.float32, mode="r", shape=(n_exp, n_in, n_in))
    counts = np.fromfile(base + ".counts.i64", dtype=np.int64)
    if counts.shape != (n_exp,):
        sys.exit(f"STOP: counts shape {counts.shape}")
    return u, counts


def bf16(t: gguf.ReaderTensor, e: int) -> np.ndarray:
    if t.tensor_type != Q.BF16:
        sys.exit(f"STOP: {t.name} in the reference is {t.tensor_type.name}, not BF16")
    return (np.asarray(t.data[e]).view(np.uint16).astype(np.uint32) << 16).view(np.float32)


def _ratio(a, b):
    return float(a) / float(b) if b > 0 else None


def refine_part(f, raw0, w, g, shrink, min_tokens, count, ranker, sweeps, tol, gain_guard):
    """One refinement unit (gate+up stacked, or down) of one expert. Returns (raw, record)."""
    t0 = time.time()
    validate(f, raw0)
    w64 = w.astype(np.float64)
    h = g + shrink * np.diag(np.diag(g))
    with np.errstate(all="ignore"):
        den, den_h = float(np.sum(w64 * (w64 @ g))), float(np.sum(w64 * (w64 @ h)))
    rec = {"count": int(count), "tr_WGWt": den, "tr_WHWt": den_h}
    neg_d = int(np.sum(fields(f, raw0)[0] < 0))
    if not np.trace(g) > 0 or count < min_tokens or neg_d:
        why = ("empty Gram" if not np.trace(g) > 0 else f"{int(count)} tokens < {min_tokens}" if count < min_tokens
               else f"{neg_d} groups with negative d")
        rec.update(skipped=why, sweeps=[], seconds=round(time.time() - t0, 2))
        return raw0, rec
    raw1, dq1, sw = refine(f, raw0, w, h, ranker, sweeps, tol, g_report=g, gain_guard=gain_guard)
    if gain_guard and sw and sw[-1]["min_row_ogain_change"] < -1e-9:
        raise AssertionError(f"a row's output gain fell by {sw[-1]['min_row_ogain_change']:.3g}")
    if sw[-1]["J_after"] > sw[0]["J_before"]:
        raise AssertionError(f"J rose: {sw[0]['J_before']} -> {sw[-1]['J_after']}")
    validate(f, raw1, dq1)
    mask = fixed_bits_mask(f, raw0.shape[1])
    if not np.array_equal(raw0 & mask, raw1 & mask):
        raise AssertionError("d or scale bits changed")
    for x in sw:
        x["relreg_before"], x["relreg_after"] = _ratio(x["J_before"], den_h), _ratio(x["J_after"], den_h)
        x["rel_before"], x["rel_after"] = _ratio(x["JG_before"], den), _ratio(x["JG_after"], den)
    rec.update(sweeps=sw, blocks_changed=int(np.sum(np.any((raw0 != raw1).reshape(raw0.shape[0], -1, f.bsize), axis=2))),
               seconds=round(time.time() - t0, 2))
    return raw1, rec


_W: dict = {}


def _init(args):
    _W["args"] = args
    _W["q"] = {t.name: t for t in gguf.GGUFReader(args.quant).tensors}
    _W["b"] = {t.name: t for t in gguf.GGUFReader(args.bf16).tensors}
    _W["rk"] = {}


def _expert(job):
    """Refine one expert of one layer (gate+up against the gate Gram, down against its own)."""
    L, e = job
    args, qt, bt = _W["args"], _W["q"], _W["b"]
    t0 = time.time()
    names = {p: f"blk.{L}.{p}.weight" for p in TENSORS}
    raws = {p: np.array(qt[n].data[e], dtype=np.uint8) for p, n in names.items()}
    parts = args.parts.split(",")
    rec = {"expert": e}
    if "gu" in parts:
        f = FMTS[qt[names["ffn_gate_exps"]].tensor_type]
        rk = _W["rk"].setdefault(f.qtype, Ranker(f))
        ug, cg = load_gram(args.grams, names["ffn_gate_exps"], qt[names["ffn_gate_exps"]].data.shape[0], 1536)
        raw = np.vstack([raws["ffn_gate_exps"], raws["ffn_up_exps"]])
        w = np.vstack([bf16(bt[names["ffn_gate_exps"]], e), bf16(bt[names["ffn_up_exps"]], e)])
        with np.errstate(all="ignore"):
            g = symmetrise(np.asarray(ug[e], dtype=np.float32)).astype(np.float64)
        raw1, rec["gu"] = refine_part(f, raw, w, g, args.shrink, args.min_tokens_gu, cg[e], rk,
                                      args.sweeps, args.tol, args.gain_guard)
        n = raws["ffn_gate_exps"].shape[0]
        raws["ffn_gate_exps"], raws["ffn_up_exps"] = raw1[:n], raw1[n:]
    if "down" in parts:
        f = FMTS[qt[names["ffn_down_exps"]].tensor_type]
        rk = _W["rk"].setdefault(f.qtype, Ranker(f))
        ud, cd = load_gram(args.grams, names["ffn_down_exps"], qt[names["ffn_down_exps"]].data.shape[0], 512)
        w = bf16(bt[names["ffn_down_exps"]], e)
        with np.errstate(all="ignore"):
            g = symmetrise(np.asarray(ud[e], dtype=np.float32)).astype(np.float64)
        raws["ffn_down_exps"], rec["down"] = refine_part(f, raws["ffn_down_exps"], w, g, args.shrink,
                                                         args.min_tokens_down, cd[e], rk, args.sweeps, args.tol,
                                                         args.gain_guard)
    rec["seconds"] = round(time.time() - t0, 2)
    return e, raws["ffn_gate_exps"], raws["ffn_up_exps"], raws["ffn_down_exps"], rec


def _dense(name):
    """Refine one dense 2D tensor against its single Gram (patch 0012 layout, n_mat = 1)."""
    args, qt, bt = _W["args"], _W["q"], _W["b"]
    t0 = time.time()
    t = qt[name]
    f = FMTS[t.tensor_type]
    rk = _W["rk"].setdefault(f.qtype, Ranker(f))
    raw = np.array(t.data, dtype=np.uint8)  # [rows, row_bytes]
    n_in = int(t.shape[0])
    u, cnt = load_gram(args.grams, name, 1, n_in)
    tb = bt[name]
    if tb.tensor_type != Q.BF16:
        sys.exit(f"STOP: {name} in the reference is {tb.tensor_type.name}, not BF16")
    w = (np.asarray(tb.data).view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(raw.shape[0], n_in)
    with np.errstate(all="ignore"):
        g = symmetrise(np.asarray(u[0], dtype=np.float32)).astype(np.float64)
    min_tok = n_in if args.min_tokens_dense < 0 else args.min_tokens_dense
    raw1, rec = refine_part(f, raw, w, g, args.shrink, min_tok, cnt[0], rk, args.sweeps, args.tol, args.gain_guard)
    rec.update(name=name, type=t.tensor_type.name, seconds=round(time.time() - t0, 2))
    return name, raw1, rec


def main_dense(args) -> None:
    """--dense REGEX: refine every matching 2D tensor of a supported type; writes OUT_DIR/dense.gguf + dense.json."""
    import re
    rx = re.compile(args.dense)
    qt = {t.name: t for t in gguf.GGUFReader(args.quant).tensors}
    names = [n for n, t in qt.items() if rx.search(n)]
    bad = [n for n in names if qt[n].tensor_type not in FMTS or len(qt[n].shape) != 2]
    if bad:
        sys.exit(f"STOP: not 2D IQ3_XXS/IQ2_XXS: {bad[:5]} ({len(bad)} tensors)")
    if not names:
        sys.exit(f"STOP: --dense {args.dense} matches nothing")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        pool = ProcessPoolExecutor(args.workers, initializer=_init, initargs=(args,))
        res = list(pool.map(_dense, names)); pool.shutdown()
    else:
        _init(args); res = [_dense(n) for n in names]
    log = {"script": "scripts/refine_codes.py --dense", "args": vars(args), "source": args.quant, "tensors": []}
    part = out_dir / f"dense.gguf.part{os.getpid()}"
    wr = gguf.GGUFWriter(str(part), arch="bailingmoe3")
    for name, raw1, rec in res:
        log["tensors"].append(rec)
        sw = rec["sweeps"]
        print(f"{name}: " + (f"{_f(sw[0]['relreg_before'])}->{_f(sw[-1]['relreg_after'])} x{len(sw)}" if sw
                             else rec["skipped"]) + f" ({rec['seconds']} s)", flush=True)
        wr.add_tensor(name, raw1, raw_dtype=qt[name].tensor_type)
    wr.write_header_to_file(); wr.write_kv_data_to_file(); wr.write_tensors_to_file(); wr.close()
    chk = {t.name: t for t in gguf.GGUFReader(str(part)).tensors}
    for name, raw1, _ in res:
        t = chk[name]
        if t.tensor_type != qt[name].tensor_type or list(t.shape) != list(qt[name].shape) or not np.array_equal(np.asarray(t.data), raw1):
            sys.exit(f"STOP: re-read of {part} differs for {name}")
    log["seconds"] = round(time.time() - t_all, 1)
    (out_dir / "dense.json").write_text(json.dumps(log, indent=1) + "\n", encoding="utf-8")
    os.replace(part, out_dir / "dense.gguf")
    print(f"dense: {len(res)} tensors in {log['seconds']} s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", required=True)
    ap.add_argument("--bf16", default="models/ling-3.0-tiny-BF16.gguf")
    ap.add_argument("--grams", default="models/grams/mixc-c512-n400")
    ap.add_argument("--layers", type=int, nargs="+", help="expert layers to refine")
    ap.add_argument("--dense", help="regex: refine matching dense 2D tensors instead of experts (Grams from patch 0012)")
    ap.add_argument("--min-tokens-dense", type=int, default=-1, help="default: the tensor's input dim")
    ap.add_argument("--experts", default="0:128:1", help="start:stop:step; experts outside stay unchanged")
    ap.add_argument("--select", help="JSON {layer: [experts]}: refine only these (overrides --experts per layer)")
    ap.add_argument("--parts", default="gu,down", help="gu (gate+up) and/or down")
    ap.add_argument("--sweeps", type=int, default=4)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--shrink", type=float, default=0.1)
    ap.add_argument("--min-tokens-gu", type=int, default=1536)
    ap.add_argument("--min-tokens-down", type=int, default=512)
    ap.add_argument("--no-gain-guard", dest="gain_guard", action="store_false")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=1, help="experts refined in parallel processes")
    args = ap.parse_args()
    if args.dense:
        return main_dense(args)
    if not args.layers:
        sys.exit("STOP: --layers or --dense required")
    experts = list(range(*(int(x) for x in args.experts.split(":"))))
    qr = gguf.GGUFReader(args.quant)
    qt = {t.name: t for t in qr.tensors}
    bt = {t.name: t for t in gguf.GGUFReader(args.bf16).tensors}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pool = None
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        pool = ProcessPoolExecutor(args.workers, initializer=_init, initargs=(args,))
    else:
        _init(args)
    t_all = time.time()
    for L in args.layers:
        if (out_dir / f"blk.{L}.gguf").exists():
            print(f"layer {L}: done already", flush=True)
            continue
        names = {p: f"blk.{L}.{p}.weight" for p in TENSORS}
        types = {p: qt[n].tensor_type for p, n in names.items()}
        for p, n in names.items():
            if types[p] not in FMTS:
                sys.exit(f"STOP: {n} is {types[p].name}; only IQ3_XXS and IQ2_XXS are supported")
            if f"blk.{L}.{p}.scale" in qt:
                sys.exit(f"STOP: {n} has a .scale tensor")
        if types["ffn_gate_exps"] != types["ffn_up_exps"]:
            sys.exit(f"STOP: layer {L} gate and up have different types")
        out = {p: np.array(qt[n].data, dtype=np.uint8) for p, n in names.items()}  # [n_exp, rows, row_bytes]
        n_exp = out["ffn_gate_exps"].shape[0]
        ug, cg = load_gram(args.grams, names["ffn_gate_exps"], n_exp, 1536)
        ud, cd = load_gram(args.grams, names["ffn_down_exps"], n_exp, 512)
        if not np.array_equal(cg, cd):
            print(f"note: layer {L} gate and down Gram counts differ", flush=True)
        log = {"script": "scripts/refine_codes.py", "args": vars(args), "numpy": np.__version__,
               "source": {"path": args.quant, "bytes": os.path.getsize(args.quant)},
               "types": {p: t.name for p, t in types.items()}, "experts": []}
        t_layer = time.time()
        sel = json.loads(Path(args.select).read_text())[str(L)] if args.select else experts
        jobs = [(L, e) for e in sel]
        res_iter = pool.map(_expert, jobs) if pool else map(_expert, jobs)
        for e, rg, ru, rd, rec in res_iter:
            out["ffn_gate_exps"][e], out["ffn_up_exps"][e], out["ffn_down_exps"][e] = rg, ru, rd
            log["experts"].append(rec)
            msg = []
            for p in ("gu", "down"):
                if p in rec:
                    sw = rec[p]["sweeps"]
                    msg.append(f"{p} " + (f"{_f(sw[0]['relreg_before'])}->{_f(sw[-1]['relreg_after'])} x{len(sw)}"
                                          if sw else rec[p]["skipped"]))
            print(f"[{time.time() - t_layer:6.0f} s] L{L} e{e}: {'; '.join(msg)} ({rec['seconds']} s)", flush=True)
        log["seconds"] = round(time.time() - t_layer, 1)
        part = out_dir / f"blk.{L}.gguf.part{os.getpid()}"
        wr = gguf.GGUFWriter(str(part), arch="bailingmoe3")
        wr.add_string("refine.source", args.quant)
        wr.add_string("refine.grams", args.grams)
        for p, n in names.items():
            wr.add_tensor(n, out[p], raw_dtype=types[p])
        wr.write_header_to_file()
        wr.write_kv_data_to_file()
        wr.write_tensors_to_file()
        wr.close()
        chk = {t.name: t for t in gguf.GGUFReader(str(part)).tensors}
        for p, n in names.items():
            t = chk[n]
            if (t.tensor_type != types[p] or list(t.shape) != list(qt[n].shape)
                    or not np.array_equal(np.asarray(t.data), out[p])):
                sys.exit(f"STOP: re-read of {part} differs for {n}")
        del chk
        (out_dir / f"blk.{L}.json").write_text(json.dumps(log, indent=1) + "\n", encoding="utf-8")
        os.replace(part, out_dir / f"blk.{L}.gguf")
        print(f"layer {L} written in {log['seconds']} s", flush=True)
    if pool:
        pool.shutdown()
    print(f"all layers in {time.time() - t_all:.0f} s", flush=True)


def _f(x):
    return "n/a" if x is None else f"{x:.4f}"


if __name__ == "__main__":
    main()
