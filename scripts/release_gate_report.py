"""Release gate: candidate vs bloomer010 IQ3_XXS and bartowski Q2_K_L on the four held-out sets (registered 25 Nov).

    .venv/bin/python -I scripts/release_gate_report.py --gguf cand=<release.gguf> bartowski=<Q2_K_L.gguf> \
        > results/release-gate.md

Writes results/release-gate.tsv (the only source of card figure A) and prints the Markdown report. The verdict is
decided once, here, on unrounded values; the TSV stores full-precision numbers and the verdict, and figures.py
re-validates the TSV with validate() below and refuses to draw anything that does not agree.

Inputs (per-token KL in scoring order, patch 0011): runs/release/NAME-SET.f32 for SET in zh512, code512, tool512,
and runs/release/NAME-pg19-BOOK.f32 for the four PG19 books (8 x 4096 each, pooled in BOOKS order into one
32-chunk set). Every file must be scored against the same BF16 base per set (same token ids, same chunk order);
the dumps carry no ids, so this script checks equal lengths and that every log reports the same Mean PPL(base),
a fingerprint of the base file (token ids and chunk order are fixed by that file).
Top-1 from logs/rg-kld-NAME-SET.log (PG19: mean of the four book logs, equal token counts).
Bytes and SHA-256 of every GGUF from disk; a missing file is an error, not a FAIL.

Statistics (docs/figure-stats.md): chunk means from (chunks, ctx - 1 - ctx // 2) = 255 at 512, 2,047 at 4,096;
mean KL = mean of chunk means; p99 over pooled tokens; paired CI = boot() of scripts/gate_t_report.py, exactly
(seed 0, 10,000, percentiles 2.5 / 97.5). The eight intervals are pointwise 95%; the all-four-sets rule is an
intersection-union test per comparator, not simultaneous coverage. PASS vs a comparator: on all 4 sets diff < 0
and ci_hi < 0, and candidate bytes <= comparator bytes. Q8_0 is a reference point only. PG19 also gets a
book-stratified interval (books resampled, then chunks within books), reported in text only.
"""
import argparse
import hashlib
import importlib.util
import math
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS, LOGS = ROOT / "runs/release", ROOT / "logs"
SETS = {"zh": ("zh512", 512, 128), "code": ("code512", 512, 128), "tool": ("tool512", 512, 128),
        "pg19": ("pg19", 4096, 32)}
BOOKS = ["10321", "10356", "10762", "22424"]
CAND, COMPS, FLOOR = "cand", ["bloomer", "bartowski"], "Q8"
FILES = [CAND, *COMPS, FLOOR]
GGUF = {"cand": None,  # --gguf cand=PATH
        "bloomer": ROOT / "models/public/bloomer010/Ling-3.0-tiny-IQ3_XXS.gguf",
        "bartowski": ROOT / "models/public/bartowski/Ling-3.0-tiny-Q2_K_L.gguf",
        "Q8": ROOT / "models/ling-3.0-tiny-Q8_0.gguf"}
HDR = ["set", "ctx", "chunks", "file", "gguf", "bytes", "sha256", "mean_kl", "p99", "top1", "diff", "ci_lo", "ci_hi",
       "verdict"]

_spec = importlib.util.spec_from_file_location("gate_t_report", ROOT / "scripts/gate_t_report.py")
_gt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gt)
boot = _gt.boot   # the gate-T bootstrap, reused unchanged


def n_scored(ctx):
    return ctx - 1 - ctx // 2


def chunks(path, ctx, want):
    v = np.fromfile(path, dtype=np.float32).astype(np.float64)
    n = n_scored(ctx)
    if len(v) != want * n:
        raise SystemExit(f"{path}: {len(v)} values, expected {want} chunks x {n}")
    if not np.all(np.isfinite(v)):
        raise SystemExit(f"{path}: non-finite KL values")
    return v.reshape(want, n)


def log_info(tag):
    p = LOGS / f"rg-kld-{tag}.log"
    if not p.exists():
        raise SystemExit(f"missing log {p}")
    t = p.read_text()
    m = re.search(r"Same top p:\s+([0-9.]+)", t)
    base = re.search(r"Mean PPL\(base\)\s*:\s*([0-9.]+)", t)   # fingerprint of the BF16 base file
    if not m:
        raise SystemExit(f"{p}: no 'Same top p' line")
    return float(m.group(1)), base.group(1) if base else None


def load(name, s):
    """(chunk x token KL, book index per chunk or None, top-1 %, set of BF16 base names seen in logs)."""
    tag, ctx, want = SETS[s]
    if s != "pg19":
        t1, base = log_info(f"{name}-{tag}")
        return chunks(RUNS / f"{name}-{tag}.f32", ctx, want), None, t1, {base}
    per = want // len(BOOKS)
    parts, t1s, bases = [], [], set()
    for b in BOOKS:
        parts.append(chunks(RUNS / f"{name}-pg19-{b}.f32", ctx, per))
        t1, base = log_info(f"{name}-pg19-{b}")
        t1s.append(t1)
        bases.add((b, base))
    book = np.concatenate([np.full(per, i) for i in range(len(BOOKS))])
    return np.concatenate(parts), book, float(np.mean(t1s)), bases


def boot_books(r, u, book, n=10_000, seed=0):
    """Book-stratified paired bootstrap: resample books, then chunks within each drawn book."""
    rng = np.random.default_rng(seed)
    d = r - u
    by = [d[book == b] for b in np.unique(book)]
    out = np.empty(n)
    for i in range(n):
        picks = rng.integers(0, len(by), len(by))
        out[i] = np.concatenate([by[k][rng.integers(0, len(by[k]), len(by[k]))] for k in picks]).mean()
    return np.percentile(out, [2.5, 97.5])


def file_id(name):
    p = GGUF.get(name)
    if not p or not Path(p).exists():
        raise SystemExit(f"GGUF for {name} not found ({p}); pass --gguf {name}=PATH (missing bytes are an error)")
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return Path(p).name, Path(p).stat().st_size, h.hexdigest()


def decide(rows):
    """Per-comparator verdict from unrounded values: all 4 sets diff < 0 and ci_hi < 0, cand bytes <= comp bytes."""
    by = {(r["set"], r["file"]): r for r in rows}
    out = {}
    for c in COMPS:
        miss = [f"{s}: diff {by[(s, c)]['diff']:+.4f}" for s in SETS if not by[(s, c)]["diff"] < 0]
        miss += [f"{s}: CI upper {by[(s, c)]['ci_hi']:+.4f} not below 0" for s in SETS if not by[(s, c)]["ci_hi"] < 0]
        if not by[(next(iter(SETS)), CAND)]["bytes"] <= by[(next(iter(SETS)), c)]["bytes"]:
            miss.append("file larger")
        out[c] = ("PASS" if not miss else "FAIL", miss)
    return out


def validate(rows, tol=1e-9):
    """Hard checks on gate rows (dicts with typed values). Raises ValueError; returns the per-comparator verdicts."""
    keys = [(r["set"], r["file"]) for r in rows]
    want = {(s, f) for s in SETS for f in FILES}
    if len(keys) != len(set(keys)) or set(keys) != want:
        raise ValueError(f"rows must be exactly {{zh, code, tool, pg19}} x {FILES}, once each; got {sorted(keys)}")
    by = dict(zip(keys, rows))
    for f in FILES:
        ids = {(by[(s, f)]["gguf"], by[(s, f)]["bytes"], by[(s, f)]["sha256"]) for s in SETS}
        if len(ids) != 1:
            raise ValueError(f"{f}: file name, bytes or sha256 differ across sets: {ids}")
        if not isinstance(by[("zh", f)]["bytes"], int) or by[("zh", f)]["bytes"] <= 0:
            raise ValueError(f"{f}: bytes missing")
    for (s, f), r in by.items():
        _, ctx, ch = SETS[s]
        if r["ctx"] != ctx or r["chunks"] != ch:
            raise ValueError(f"{s}/{f}: ctx {r['ctx']} chunks {r['chunks']}, registered {ctx} x {ch}")
        for k in ("mean_kl", "p99", "top1"):
            if not math.isfinite(r[k]):
                raise ValueError(f"{s}/{f}: {k} not finite")
        if f in COMPS:
            for k in ("diff", "ci_lo", "ci_hi"):
                if r[k] is None or not math.isfinite(r[k]):
                    raise ValueError(f"{s}/{f}: {k} missing")
            if not r["ci_lo"] <= r["ci_hi"]:
                raise ValueError(f"{s}/{f}: ci_lo > ci_hi")
            if abs(r["diff"] - (by[(s, CAND)]["mean_kl"] - r["mean_kl"])) > tol:
                raise ValueError(f"{s}/{f}: diff does not equal cand mean - {f} mean")
        elif any(r[k] is not None for k in ("diff", "ci_lo", "ci_hi")):
            raise ValueError(f"{s}/{f}: only comparators carry a difference")
    verdict = decide(rows)
    for (s, f), r in by.items():
        if f in COMPS and r["verdict"] is not None and r["verdict"] != verdict[f][0]:
            raise ValueError(f"{s}/{f}: stored verdict {r['verdict']} disagrees with the rule ({verdict[f][0]})")
    return verdict


def read_tsv(path):
    """TSV -> typed rows for validate()."""
    lines = Path(path).read_text().splitlines()
    hdr = lines[0].split("\t")
    if hdr != HDR:
        raise ValueError(f"{path}: header {hdr} != {HDR}")
    rows = []
    for line in lines[1:]:
        r = dict(zip(hdr, line.split("\t")))
        num = lambda x: float(x) if x != "" else None  # noqa: E731
        rows.append({"set": r["set"], "ctx": int(r["ctx"]), "chunks": int(r["chunks"]), "file": r["file"],
                     "gguf": r["gguf"], "bytes": int(r["bytes"]) if r["bytes"].isdigit() else None,
                     "sha256": r["sha256"], "mean_kl": float(r["mean_kl"]), "p99": float(r["p99"]),
                     "top1": float(r["top1"]), "diff": num(r["diff"]), "ci_lo": num(r["ci_lo"]),
                     "ci_hi": num(r["ci_hi"]), "verdict": r["verdict"] or None})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", nargs="*", default=[], metavar="NAME=PATH")
    ap.add_argument("--tsv", default=str(ROOT / "results/release-gate.tsv"))
    a = ap.parse_args()
    for kv in a.gguf:
        k, v = kv.split("=", 1)
        GGUF[k] = Path(v)
    ident = {n: file_id(n) for n in FILES}
    rows, text = [], []
    for s in SETS:
        data = {n: load(n, s) for n in FILES}
        bases = {frozenset(d[3]) for d in data.values()}
        if len(bases) != 1 or any(None in b or any(isinstance(x, tuple) and x[1] is None for x in b) for b in bases):
            raise SystemExit(f"{s}: logs do not report one common BF16 base (Mean PPL(base)): {bases}")
        mr = data[CAND][0].mean(1)
        for n, (c, book, t1, _) in data.items():
            m = c.mean(1)
            diff = lo = hi = None
            if n in COMPS:
                d, ci, _, _ = boot(mr, m)
                diff, lo, hi = float(d), float(ci[0]), float(ci[1])
                if book is not None:
                    bl, bh = boot_books(mr, m, book)
                    text.append(f"- PG19 cand - {n}, book-stratified 95% interval (4 books x 8 chunks, sensitivity "
                                f"check): [{bl:+.4f}, {bh:+.4f}]; registered chunk interval [{lo:+.4f}, {hi:+.4f}]")
            g, b, h = ident[n]
            rows.append({"set": s, "ctx": SETS[s][1], "chunks": len(m), "file": n, "gguf": g, "bytes": b, "sha256": h,
                         "mean_kl": float(m.mean()), "p99": float(np.percentile(c, 99)), "top1": t1,
                         "diff": diff, "ci_lo": lo, "ci_hi": hi, "verdict": None})
    verdict = validate(rows)
    for r in rows:
        if r["file"] in COMPS:
            r["verdict"] = verdict[r["file"]][0]
    validate(rows)
    fmt = lambda x: "" if x is None else (format(x, ".17g") if isinstance(x, float) else str(x))  # noqa: E731
    Path(a.tsv).write_text("\n".join(["\t".join(HDR)] + ["\t".join(fmt(r[k]) for k in HDR) for r in rows]) + "\n")
    validate(read_tsv(a.tsv))   # the file the figure reads must pass the same checks

    for c in COMPS:
        v, miss = verdict[c]
        print(f"- Release gate vs {c}: {v}" + (f" ({'; '.join(miss)})" if miss else ""))
    print("\nRule: candidate KL lower on all 4 sets with the paired 95% CI of cand - comparator below 0, and cand "
          "bytes <= comparator bytes; per comparator. Intervals are pointwise.\n")
    print("| set | ctx | chunks | file | bytes | mean KL | p99 | top-1 % | cand - file [95% CI] |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        ci = f"{r['diff']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]" if r["diff"] is not None else ""
        print(f"| {r['set']} | {r['ctx']} | {r['chunks']} | {r['file']} | {r['bytes']:,} | {r['mean_kl']:.4f} | "
              f"{r['p99']:.3f} | {r['top1']:.2f} | {ci} |")
    print()
    for f in FILES:
        g, b, h = ident[f]
        print(f"- {f}: `{g}`, {b:,} B, sha256 `{h}`")
    for line in text:
        print(line)


if __name__ == "__main__":
    main()
