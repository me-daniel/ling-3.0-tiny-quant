"""13-gram overlap (lowercased \\w+ words, as Inkling's calib-wt.py) of eval texts against a calibration text.

    python scripts/ngram_overlap.py CALIB EVAL... [--n 13] [--calib-chars N]
"""
import argparse
import re

ap = argparse.ArgumentParser()
ap.add_argument("calib")
ap.add_argument("evals", nargs="+")
ap.add_argument("--n", type=int, default=13)
ap.add_argument("--calib-chars", type=int, default=0, help="use only the first N characters of CALIB (0 = all)")
a = ap.parse_args()


def grams(text):
    w = re.findall(r"\w+", text.lower())
    return {tuple(w[i:i + a.n]) for i in range(len(w) - a.n + 1)}


c = open(a.calib, encoding="utf-8", errors="replace").read()
if a.calib_chars:
    c = c[:a.calib_chars]
cg = grams(c)
print(f"{a.calib}: {len(cg):,} distinct {a.n}-grams")
for e in a.evals:
    eg = grams(open(e, encoding="utf-8", errors="replace").read())
    sh = eg & cg
    print(f"{e}: {len(eg):,} distinct, shared {len(sh):,} ({100 * len(sh) / max(1, len(eg)):.3f}%)")
    for g in list(sh)[:3]:
        print("   e.g.", " ".join(g))
