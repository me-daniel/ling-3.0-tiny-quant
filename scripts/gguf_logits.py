"""Run build/gguf_logits on the token ids and positions stored in an hf_ref.py npz.

  .venv/bin/python scripts/gguf_logits.py --ref runs/parity/hf.npz --gguf models/ling-3.0-tiny-BF16.gguf \
      --out runs/parity/gguf.npz [--hidden] [--cpu] [--ubatch N] [--kv f16|f32]
"""
import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--ref", required=True)
ap.add_argument("--gguf", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--hidden", action="store_true", help="also dump l_out-N per layer")
ap.add_argument("--cpu", action="store_true")
ap.add_argument("--ubatch", type=int, default=0)
ap.add_argument("--kv", default="f16")
ap.add_argument("--only", default="", help="comma list of prompt names")
a = ap.parse_args()

ref = np.load(a.ref)
names = sorted({k.split("/")[0] for k in ref.files if "/" in k})
if a.only:
    names = [n for n in names if n in a.only.split(",")]
env = dict(os.environ, GL_KV=a.kv)
if a.cpu:
    env["GL_CPU"] = "1"
if a.ubatch:
    env["GL_UBATCH"] = str(a.ubatch)
out = {"meta": json.dumps({"gguf": a.gguf, "cpu": a.cpu, "ubatch": a.ubatch or "T", "kv": a.kv})}
with tempfile.TemporaryDirectory() as td:
    for n in names:
        ids, pos = ref[f"{n}/ids"].astype(np.int32), ref[f"{n}/pos"].astype(np.int32)
        td_ = Path(td)
        ids.tofile(td_ / "ids"); pos.tofile(td_ / "pos")
        cmd = ["build/gguf_logits", a.gguf, td_ / "ids", td_ / "pos", td_ / "logits"]
        if a.hidden:
            cmd.append(td_ / "hidden")
        r = subprocess.run([str(c) for c in cmd], env=env, capture_output=True, text=True)
        if r.returncode:
            print(r.stderr[-3000:]); raise SystemExit(f"gguf_logits failed on {n}")
        print(n, [l for l in r.stderr.splitlines() if l.startswith(("T=", "dumped"))])
        out[f"{n}/ids"], out[f"{n}/pos"] = ids, pos
        out[f"{n}/logits"] = np.fromfile(td_ / "logits", dtype=np.float32).reshape(len(pos), -1)
        if a.hidden:
            out[f"{n}/hidden"] = np.fromfile(td_ / "hidden", dtype=np.float32).reshape(-1, len(pos), ref[f"{n}/hidden"].shape[-1])
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
np.savez(a.out, **out)
print("wrote", a.out)
