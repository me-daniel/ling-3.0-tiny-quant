"""Gain kept by a cheap mode: (orig - mode) / (orig - full), paired bootstrap over chunks (10,000, seed 0).

    .venv/bin/python -I scripts/gain_kept.py bloomer twin twin-s1 [twin-half ...]
"""
import sys
import numpy as np
from pathlib import Path

K = Path("runs/gate-t"); SETS = {"wiki512": 512, "ood512": 512, "wiki4096": 4096}
o, full, modes = sys.argv[1], sys.argv[2], sys.argv[3:]


def cm(n, s):
    v = np.fromfile(K / f"{n}-{s}.f32", dtype=np.float32).astype(np.float64)
    return v.reshape(-1, SETS[s] - 1 - SETS[s] // 2).mean(1)


print("| mode | set | gain full | gain mode | kept [95% CI] |")
print("|---|---|---|---|---|")
for m in modes:
    for s in SETS:
        try:
            a, b, c = cm(o, s), cm(full, s), cm(m, s)
        except FileNotFoundError:
            continue
        idx = np.random.default_rng(0).integers(0, len(a), size=(10_000, len(a)))
        kept = (a[idx] - c[idx]).mean(1) / (a[idx] - b[idx]).mean(1)
        lo, hi = np.percentile(kept, [2.5, 97.5])
        print(f"| {m} | {s} | {(a - b).mean():.4f} | {(a - c).mean():.4f} | {(a - c).mean() / (a - b).mean():.2f} [{lo:.2f}, {hi:.2f}] |")
