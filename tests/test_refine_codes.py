"""Tests for scripts/refine_codes.py: decode vs gguf-py, byte writes, exhaustive ranking, a refinement on a random
small expert with a known Gram.  Run: .venv/bin/python -m pytest -q tests/test_refine_codes.py"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("rc", Path(__file__).resolve().parents[1] / "scripts/refine_codes.py")
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)
Q = rc.Q
FM = [rc.FMTS[Q.IQ3_XXS], rc.FMTS[Q.IQ2_XXS]]


def random_raw(f, rows, nblk, rng, dval=0.002):
    raw = rng.integers(0, 256, size=(rows, nblk * f.bsize), dtype=np.uint8)
    b = raw.reshape(rows, nblk, f.bsize)
    d = (rng.uniform(0.5, 1.5, size=(rows, nblk)) * dval).astype(np.float16)
    b[..., :2] = d.view(np.uint8).reshape(rows, nblk, 2)
    return raw


@pytest.mark.parametrize("f", FM, ids=lambda f: f.qtype.name)
def test_decode_matches_gguf_py(f):
    rng = np.random.default_rng(0)
    raw = random_raw(f, 16, 3, rng)
    rc.validate(f, raw)


@pytest.mark.parametrize("f", FM, ids=lambda f: f.qtype.name)
def test_set_group_roundtrip(f):
    rng = np.random.default_rng(1)
    raw = random_raw(f, 8, 2, rng)
    db, idx, neg = rc.fields(f, raw)
    mask = rc.fixed_bits_mask(f, raw.shape[1])
    for gi in rng.integers(0, idx.shape[1], size=20):
        rows = np.arange(8)
        ni = rng.integers(0, 256, size=8)
        if f.qtype == Q.IQ2_XXS:
            nn = f.pat_sets[0][rng.integers(0, 128, size=8)]
        else:
            par = neg[:, gi ^ 1].sum(1) % 2
            nn = np.stack([f.pat_sets[p][rng.integers(0, 8)] for p in par])
        before = raw.copy()
        rc.set_group(f, raw, rows, int(gi), ni, nn)
        db2, idx2, neg2 = rc.fields(f, raw)
        assert np.array_equal(idx2[:, gi], ni) and np.array_equal(neg2[:, gi], nn)
        other = np.ones(idx.shape[1], bool); other[gi] = False
        assert np.array_equal(idx2[:, other], idx[:, other]) and np.array_equal(neg2[:, other], neg[:, other])
        assert np.array_equal(before & mask, raw & mask) and np.array_equal(db2, db)
        idx, neg = idx2, neg2
    rc.validate(f, raw)


@pytest.mark.parametrize("f", FM, ids=lambda f: f.qtype.name)
def test_ranker_is_exhaustive(f):
    """Ranker's pick equals the float64 brute-force argmin of the exact dJ over every representable candidate."""
    rng = np.random.default_rng(2)
    k, r = f.k, 32
    X = rng.normal(size=(200, k)) @ rng.normal(size=(k, k))
    Gb = X.T @ X
    db = rng.uniform(0.5, 2.0, size=r).astype(np.float32) * np.float32(0.01)
    cur = rng.normal(size=(r, k)) * 0.2
    Rg = rng.normal(size=(r, k)) * 5
    b = Rg - cur @ Gb
    cls = rng.integers(0, len(f.cands), size=r)
    ci, cn = rc.Ranker(f).pick(Gb, b, db, cls)
    for i in range(r):
        C = f.cands[cls[i]]
        v = (db[i] * np.abs(C).astype(np.float32)).astype(np.float64) * np.sign(C)  # decode's float32 order
        dd = v - cur[i]
        dj = 2 * dd @ Rg[i] + np.einsum("ci,ij,cj->c", dd, Gb, dd)
        best = np.argmin(dj)
        got = (db[i] * f.grid[ci[i]] * np.where(cn[i], np.float32(-1), np.float32(1))).astype(np.float64)  # [top, k]
        dd2 = got - cur[i]
        dj_got = (2 * dd2 @ Rg[i] + np.einsum("ci,ij,cj->c", dd2, Gb, dd2)).min()  # exact rescoring of the top
        assert dj_got <= dj[best] + 1e-12 * abs(dj[best]), (i, dj_got, dj[best])


@pytest.mark.parametrize("f", FM, ids=lambda f: f.qtype.name)
def test_refine_random_expert(f):
    """Small expert (24 rows x 512 inputs) with a known correlated Gram: J falls, never rises, values stay
    representable, d/scales untouched, and the gain floor holds."""
    rng = np.random.default_rng(3)
    rows, n_in = 24, 512
    raw = random_raw(f, rows, n_in // 256, rng)
    w = (rc.decode(f, raw) + rng.normal(scale=0.01, size=(rows, n_in))).astype(np.float32)
    A = rng.normal(size=(n_in, n_in)) * 0.1 + np.eye(n_in)
    X = rng.normal(size=(2000, n_in)) @ A
    G = X.T @ X
    H = G + 0.1 * np.diag(np.diag(G))
    raw1, dq1, sw = rc.refine(f, raw, w, H, rc.Ranker(f), g_report=G, gain_guard=True)
    assert sw[-1]["J_after"] < sw[0]["J_before"]
    for s in sw:
        assert s["J_after"] <= s["J_before"] * (1 + 1e-12)
        assert abs(s["J_after"] - s["J_tracked"]) <= 1e-8 * s["J_before"]
    assert sw[-1]["min_row_ogain_change"] >= -1e-9
    rc.validate(f, raw1, dq1)
    m = rc.fixed_bits_mask(f, raw.shape[1])
    assert np.array_equal(raw & m, raw1 & m)
    E = dq1.astype(np.float64) - w
    assert np.isclose(np.sum((X @ E.T) ** 2) + 0.1 * np.sum(E ** 2 * np.diag(G)), sw[-1]["J_after"], rtol=1e-9)


@pytest.mark.parametrize("f", FM, ids=lambda f: f.qtype.name)
def test_ranker_guarded(f):
    """With the gain floor, the pick is the best feasible candidate (c . uw >= thr / db)."""
    rng = np.random.default_rng(4)
    k, r = f.k, 24
    X = rng.normal(size=(200, k)); Gb = X.T @ X
    db = np.full(r, 0.01, np.float32)
    b = rng.normal(size=(r, k)) * 5
    uw = rng.normal(size=(r, k))
    cls = rng.integers(0, len(f.cands), size=r)
    thr = np.array([np.median(db[i] * f.cands[cls[i]] @ uw[i]) for i in range(r)])
    ci, cn = rc.Ranker(f).pick(Gb, b, db, cls, uw, thr)
    for i in range(r):
        C = f.cands[cls[i]]
        s = np.einsum("ci,ij,cj->c", C, Gb, C) + C @ (2 * b[i] / db[i])
        feas = db[i] * (C @ uw[i]) >= thr[i] + 1e-9
        best = s[feas].min()
        got = f.grid[ci[i, 0]] * np.where(cn[i, 0], -1.0, 1.0)
        assert db[i] * got @ uw[i] >= thr[i] - 1e-5
        assert got @ Gb @ got + got @ (2 * b[i] / db[i]) <= best + 1e-4 * abs(best)


def test_dense_mode_end_to_end(tmp_path):
    """--dense on synthetic files: a 2D IQ3_XXS tensor (32 rows x 512), its BF16 reference and a known Gram in the
    patch-0012 layout ([1][d][d] upper triangle + int64 [1]). Output decodes, keeps d/scales, J falls."""
    import json, subprocess, sys
    import gguf
    rng = np.random.default_rng(5)
    f = rc.FMTS[Q.IQ3_XXS]
    rows, n_in, name = 32, 512, "blk.1.ssm_f_a.weight"
    raw = random_raw(f, rows, n_in // 256, rng)
    w = (rc.decode(f, raw) + rng.normal(scale=0.01, size=(rows, n_in))).astype(np.float32)
    X = rng.normal(size=(3000, n_in)) @ (np.eye(n_in) + 0.1 * rng.normal(size=(n_in, n_in)))
    G = X.T @ X
    for path, arr, dt in ((tmp_path / "q.gguf", raw, Q.IQ3_XXS),
                          (tmp_path / "b.gguf", (w.view(np.uint32) >> 16).astype(np.uint16).view(np.uint8), Q.BF16)):
        wr = gguf.GGUFWriter(str(path), arch="bailingmoe3")
        wr.add_tensor(name, arr, raw_dtype=dt)
        wr.write_header_to_file(); wr.write_kv_data_to_file(); wr.write_tensors_to_file(); wr.close()
    gd = tmp_path / "grams"; gd.mkdir()
    np.triu(G).astype(np.float32).tofile(gd / f"{name}.gram.f32")
    np.array([3000], dtype=np.int64).tofile(gd / f"{name}.gram.counts.i64")
    script = Path(__file__).resolve().parents[1] / "scripts/refine_codes.py"
    subprocess.run([sys.executable, "-I", str(script), "--quant", str(tmp_path / "q.gguf"), "--bf16", str(tmp_path / "b.gguf"),
                    "--grams", str(gd), "--dense", r"ssm_f_a\.weight$", "--out-dir", str(tmp_path / "out")], check=True,
                   capture_output=True)
    log = json.loads((tmp_path / "out/dense.json").read_text())
    sw = log["tensors"][0]["sweeps"]
    assert sw and sw[-1]["J_after"] < sw[0]["J_before"]
    t = {t.name: t for t in gguf.GGUFReader(str(tmp_path / "out/dense.gguf")).tensors}[name]
    raw1 = np.asarray(t.data)
    rc.validate(f, raw1)
    m = rc.fixed_bits_mask(f, raw.shape[1])
    assert np.array_equal(raw & m, raw1 & m) and not np.array_equal(raw, raw1)
    wb = ((w.view(np.uint32) >> 16) << 16).view(np.float32).astype(np.float64)  # the BF16 reference as stored
    G32 = rc.symmetrise(np.triu(G).astype(np.float32)).astype(np.float64)  # the Gram as stored
    E = rc.decode(f, raw1).astype(np.float64) - wb
    assert np.isclose(np.sum(E * (E @ G32)) + 0.1 * np.sum(E ** 2 * np.diag(G32)), sw[-1]["J_after"], rtol=1e-9)
