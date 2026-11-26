"""Per-layer Gram token counts of the routed experts vs the refiner's min-token rule (input dim).

    .venv/bin/python -I scripts/gram_counts.py models/grams/mixc-c512-n400
"""
import sys
import numpy as np

d = sys.argv[1]
print("| layer | experts < 1536 tokens (gate/up skipped) | experts < 512 (down skipped) | zero | min | median | max |")
print("|---|---|---|---|---|---|---|")
tot = [0, 0]
for L in range(1, 24):
    c = np.fromfile(f"{d}/blk.{L}.ffn_gate_exps.weight.gram.counts.i64", dtype=np.int64)
    cd = np.fromfile(f"{d}/blk.{L}.ffn_down_exps.weight.gram.counts.i64", dtype=np.int64)
    assert np.array_equal(c, cd), L
    a, b = int((c < 1536).sum()), int((c < 512).sum())
    tot[0] += a; tot[1] += b
    print(f"| {L} | {a} | {b} | {int((c == 0).sum())} | {c.min()} | {int(np.median(c))} | {c.max()} |")
print(f"| all | {tot[0]} of 2944 | {tot[1]} of 2944 | | | | |")
