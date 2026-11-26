"""HellaSwag per-task correctness from llama-perplexity's cumulative accuracy lines, Wilson CIs, exact McNemar.

    .venv/bin/python -I scripts/hellaswag_pairs.py m2r bloomer bartowski      # first name = candidate
Reads logs/hs-NAME.log. Task order is the same for every model (fixed mt19937(1) shuffle of the same 1,000 tasks).
"""
import math
import re
import sys
from pathlib import Path


def per_task(name):
    acc = [float(m.group(2)) for m in re.finditer(r"^(\d+)\t([0-9.]+)%\t\[", Path(f"logs/hs-{name}.log").read_text(), re.M)]
    cum = [round(a * (i + 1) / 100) for i, a in enumerate(acc)]
    c = [cum[0]] + [cum[i] - cum[i - 1] for i in range(1, len(cum))]
    assert set(c) <= {0, 1} and len(c) == 1000, (name, len(c), set(c))
    return c


def binom_two_sided(x, n):
    """Exact two-sided binomial p (p = 0.5): twice the smaller tail, capped at 1."""
    k = min(x, n - x)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def wilson(k, n, z=1.959964):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


names = sys.argv[1:]
res = {n: per_task(n) for n in names}
print("| file | acc_norm | 95% Wilson CI |")
print("|---|---|---|")
for n, c in res.items():
    lo, hi = wilson(sum(c), len(c))
    print(f"| {n} | {100 * sum(c) / len(c):.1f}% | [{100 * lo:.1f}, {100 * hi:.1f}] |")
print("\n| candidate vs | cand right, other wrong | cand wrong, other right | difference (points) | exact McNemar p |")
print("|---|---|---|---|---|")
a = res[names[0]]
for n in names[1:]:
    b = res[n]
    x = sum(1 for i, j in zip(a, b) if i and not j); y = sum(1 for i, j in zip(a, b) if j and not i)
    p = binom_two_sided(x, x + y) if x + y else 1.0
    print(f"| {n} | {x} | {y} | {100 * (x - y) / len(a):+.1f} | {p:.3f} |")
