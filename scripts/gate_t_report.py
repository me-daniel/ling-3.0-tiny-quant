"""Gate T table: paired per-chunk KL of refined (R) vs unrefined (U) builds against the BF16 reference.

    .venv/bin/python -I scripts/gate_t_report.py [--levels 3 2] [--floor Q8] > results/gate-t.md

Reads runs/gate-t/NAME-SET.f32 (per-token KL in scoring order, patch 0011) and logs/gt-kld-NAME-SET.log (top-1).
Per chunk: mean KL over its scored tokens. Paired bootstrap over chunks (10,000 resamples, seed 0) of mean(R - U)
and of mean(R) / mean(U). Gate T (registered before data): PASS for a level if mean R <= 0.9 mean U on ood512 AND
on wiki512, with the 95% CI of R - U below zero on both. wiki4096 is the longer-context check, reported only.
"""
import argparse
import re
from pathlib import Path

import numpy as np

K = Path("runs/gate-t")
SETS = {"wiki512": 512, "ood512": 512, "wiki4096": 4096}


def per_chunk(name, s):
    v = np.fromfile(K / f"{name}-{s}.f32", dtype=np.float32).astype(np.float64)
    n = SETS[s] - 1 - SETS[s] // 2
    if len(v) % n:
        raise SystemExit(f"{name}-{s}: {len(v)} values not a multiple of {n}")
    return v.reshape(-1, n), v


def top1(name, s):
    m = re.search(r"Same top p:\s+([0-9.]+)", Path(f"logs/gt-kld-{name}-{s}.log").read_text())
    return float(m.group(1)) if m else float("nan")


def boot(r, u, n=10_000, seed=0):
    idx = np.random.default_rng(seed).integers(0, len(r), size=(n, len(r)))
    d = (r - u)[idx].mean(1)
    q = r[idx].mean(1) / u[idx].mean(1)
    return (r - u).mean(), np.percentile(d, [2.5, 97.5]), r.mean() / u.mean(), np.percentile(q, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", nargs="+", default=["3", "2"])
    ap.add_argument("--floor", default="Q8")
    ap.add_argument("--pair", nargs=2, metavar=("U", "R"), help="compare these two dump names instead of U<lv>/R<lv>")
    a = ap.parse_args()
    print("| level | set | chunks | KL U | KL R | R - U [95% CI] | R / U [95% CI] | p99 U / R | top-1 U / R % |")
    print("|---|---|---|---|---|---|---|---|---|")
    verdict = {}
    for lv in a.levels:
        ok = True
        for s in SETS:
            try:
                nu, nr = a.pair if a.pair else (f"U{lv}", f"R{lv}")
                (cu, tu), (cr, tr) = per_chunk(nu, s), per_chunk(nr, s)
            except FileNotFoundError:
                print(f"| {lv} | {s} | missing | | | | | | |")
                ok = False if s != "wiki4096" else ok
                continue
            mu, mr = cu.mean(1), cr.mean(1)
            d, dci, q, qci = boot(mr, mu)
            p99 = f"{np.percentile(tu, 99):.3f} / {np.percentile(tr, 99):.3f}"
            print(f"| {lv} | {s} | {len(mu)} | {mu.mean():.4f} | {mr.mean():.4f} | {d:+.4f} [{dci[0]:+.4f}, {dci[1]:+.4f}] | "
                  f"{q:.3f} [{qci[0]:.3f}, {qci[1]:.3f}] | {p99} | {top1(nu, s):.2f} / {top1(nr, s):.2f} |")
            if s != "wiki4096":
                ok &= (q <= 0.9) and (dci[1] < 0)
        verdict[lv] = ok
    for s in SETS:
        try:
            c, t = per_chunk(a.floor, s)
            print(f"\n{a.floor} floor {s}: KL {c.mean():.4f}, top-1 {top1(a.floor, s):.2f}%", end="")
        except FileNotFoundError:
            pass
    print("\n")
    for lv, ok in verdict.items():
        print(f"- {' vs '.join(a.pair[::-1]) if a.pair else f'Gate T, IQ{lv}_XXS'}: {'PASS' if ok else 'FAIL'} (rule: R/U <= 0.90 and CI of R - U below 0 on wiki512 and ood512)")


if __name__ == "__main__":
    main()
