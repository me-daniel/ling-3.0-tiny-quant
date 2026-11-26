"""Write {layer: [experts]}: per layer the half of the experts with the largest starting J under H (gate/up + down),
read from a refine_codes.py run's logs (sweeps[0].J_before is the start, known before any move; units skipped by
the min-token rule count 0).  python scripts/select_half.py models/refined/twin > runs/twin/select-half.json"""
import json
import sys
from pathlib import Path

d, out = Path(sys.argv[1]), {}
for f in sorted(d.glob("blk.*.json")):
    L = f.stem.split(".")[1]
    ex = json.loads(f.read_text())["experts"]
    j = {r["expert"]: sum(r[p]["sweeps"][0]["J_before"] for p in ("gu", "down") if r[p]["sweeps"]) for r in ex}
    top = sorted(j, key=lambda e: -j[e])[: len(j) // 2]
    out[L] = sorted(top)
print(json.dumps(out))
