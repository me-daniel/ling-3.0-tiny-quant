"""Check a spliced R build against its U build and the refined layer files.

    .venv/bin/python -I scripts/check_refined.py U.gguf R.gguf REFINED_DIR [REFINED_DIR ...] [--ggml-layers 12 ...]

1. Same tensor names, order, types, shapes and byte sizes in U and R; file sizes equal.
2. Every tensor outside the refined expert tensors is byte-identical; refined tensors equal the layer files' bytes.
3. For --ggml-layers: ggml's own CPU dequant (build/ggml_dequant) of R's three expert tensors equals gguf-py's
   dequant and refine_codes' decode bit for bit.
"""
import argparse
import importlib.util
import os
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import gguf
from gguf.quants import dequantize

spec = importlib.util.spec_from_file_location("rc", Path(__file__).resolve().parent / "refine_codes.py")
rc = importlib.util.module_from_spec(spec); spec.loader.exec_module(rc)

ap = argparse.ArgumentParser()
ap.add_argument("u"); ap.add_argument("r"); ap.add_argument("refined", nargs="+", help="one or more refined dirs")
ap.add_argument("--ggml-layers", type=int, nargs="*", default=[])
a = ap.parse_args()
U, R = gguf.GGUFReader(a.u), gguf.GGUFReader(a.r)
assert os.path.getsize(a.u) == os.path.getsize(a.r), "file sizes differ"
assert [t.name for t in U.tensors] == [t.name for t in R.tensors], "names/order differ"
refd = {}
for f in [f for d in a.refined for f in list(Path(d).glob("blk.*.gguf")) + list(Path(d).glob("dense.gguf"))]:
    for t in gguf.GGUFReader(str(f)).tensors:
        refd[t.name] = t
changed, same = 0, 0
for tu, tr in zip(U.tensors, R.tensors):
    assert tu.tensor_type == tr.tensor_type and list(tu.shape) == list(tr.shape) and tu.n_bytes == tr.n_bytes, tu.name
    bu, br = np.asarray(tu.data).view(np.uint8), np.asarray(tr.data).view(np.uint8)
    if tu.name in refd:
        assert np.array_equal(br, np.asarray(refd[tu.name].data).view(np.uint8)), f"{tu.name}: R differs from layer file"
        changed += int(not np.array_equal(bu, br))
    else:
        assert np.array_equal(bu, br), f"{tu.name} changed outside the refined set"
        same += 1
print(f"OK: {len(U.tensors)} tensors, same names/types/shapes/bytes; {same} untouched byte-identical; "
      f"{len(refd)} refined tensors from {a.refined} ({changed} with changed bytes); file size {os.path.getsize(a.r):,} B")
rt = {t.name: t for t in R.tensors}
for L in a.ggml_layers:
    for p in rc.TENSORS:
        n = f"blk.{L}.{p}.weight"
        t = rt[n]
        with tempfile.TemporaryDirectory() as td:
            subprocess.run(["build/ggml_dequant", a.r, n, f"{td}/x.f32"], check=True, capture_output=True)
            g = np.fromfile(f"{td}/x.f32", dtype=np.float32)
        raw = np.asarray(t.data)
        py = dequantize(raw.reshape(-1, raw.shape[-1]), t.tensor_type).astype(np.float32).reshape(-1)
        f = rc.FMTS[t.tensor_type]
        mine = rc.decode(f, raw.reshape(-1, raw.shape[-1])).reshape(-1)
        ok = np.array_equal(g.view(np.uint32), py.view(np.uint32)) and np.array_equal(g.view(np.uint32), mine.view(np.uint32))
        print(f"{n} {t.tensor_type.name}: ggml == gguf-py == refine decode bit-exact: {ok} ({g.size:,} values)")
        assert ok
