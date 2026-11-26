"""Per-layer offline summary of a refine_codes.py run: J / tr(W G W^T) (unregularised G) before and after, summed
over refined experts; experts skipped; accepted moves; floor blocks; sweeps; seconds.

    .venv/bin/python -I scripts/refine_summary.py models/refined/R3
"""
import json
import sys
from pathlib import Path

d = Path(sys.argv[1])
print(f"Offline, {d}: rel = sum J_G / sum tr(W G W^T) over refined experts (G unregularised, in-sample Grams).\n")
print("| layer | gu rel before → after | gu skipped | down rel before → after | down skipped | moves | floor blocks | layer s |")
print("|---|---|---|---|---|---|---|---|")
tot = {"gu": [0.0, 0.0, 0.0], "down": [0.0, 0.0, 0.0]}
skip = {"gu": 0, "down": 0}
for L in range(1, 24):
    f = d / f"blk.{L}.json"
    if not f.exists():
        continue
    log = json.loads(f.read_text())
    row, moves, blocks = [], 0, 0
    for p in ("gu", "down"):
        b = a = den = 0.0
        n_skip = 0
        for e in log["experts"]:
            r = e[p]
            if not r["sweeps"]:
                n_skip += 1
                continue
            b += r["sweeps"][0]["JG_before"]; a += r["sweeps"][-1]["JG_after"]; den += r["tr_WGWt"]
            moves += sum(s["accepted"] for s in r["sweeps"]); blocks += sum(s["gain_blocked"] for s in r["sweeps"])
        tot[p][0] += b; tot[p][1] += a; tot[p][2] += den; skip[p] += n_skip
        row += [f"{b / den:.4f} → {a / den:.4f}", str(n_skip)]
    print(f"| {L} | " + " | ".join(row) + f" | {moves:,} | {blocks:,} | {log['seconds']:.0f} |")
print(f"| all | {tot['gu'][0] / tot['gu'][2]:.4f} → {tot['gu'][1] / tot['gu'][2]:.4f} | {skip['gu']} | "
      f"{tot['down'][0] / tot['down'][2]:.4f} → {tot['down'][1] / tot['down'][2]:.4f} | {skip['down']} | | | |")
