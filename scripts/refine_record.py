"""Refinement verification record for one GGUF, keyed by its SHA-256. The banner says "codes refined" only with it.

    .venv/bin/python -I scripts/refine_record.py models/ling-3.0-tiny-m2.gguf <candidate.gguf> results/refinement-<id>.json

Byte-level checks of the candidate against its start file (no model run, no dequant):
1. Same file size, tensor names, order, types, shapes and byte counts (equal layout).
2. Every tensor that is not IQ3_XXS / IQ2_XXS is byte-identical (unchanged bytes outside the codes).
3. In IQ3_XXS / IQ2_XXS tensors the fp16 block scale d and every 4-bit sub-block scale are unchanged; only grid
   indices and sign bits may differ (preserved scales).
The record lists SHA-256 of both files and the changed tensors. `refined` is true only if 1-3 hold and at least one
tensor's codes changed. scripts/figures.py --record checks the SHA-256 against the layout TSV of the scored file.
"""
import hashlib
import json
import os
import sys

import numpy as np
from gguf import GGMLQuantizationType as Q
from gguf import GGUFReader

# block bytes, offset and count of the uint32 words whose top 4 bits are sub-block scales
FMT = {Q.IQ3_XXS: (98, 66, 8, 1), Q.IQ2_XXS: (66, 2, 16, 2)}   # (block, words offset, n words, take every k-th)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def scales(blocks, fmt):
    size, off, n, k = fmt
    w = np.ascontiguousarray(blocks[:, off:off + 4 * n]).view(np.uint32)[:, k - 1::k]
    return blocks[:, :2], w >> 28


def main(start, cand, out):
    S, C = GGUFReader(start), GGUFReader(cand)
    problems, changed = [], []
    if os.path.getsize(start) != os.path.getsize(cand):
        problems.append("file sizes differ")
    if [t.name for t in S.tensors] != [t.name for t in C.tensors]:
        problems.append("tensor names or order differ")
    for ts, tc in zip(S.tensors, C.tensors):
        if ts.tensor_type != tc.tensor_type or list(ts.shape) != list(tc.shape) or ts.n_bytes != tc.n_bytes:
            problems.append(f"{ts.name}: type/shape/bytes differ")
            continue
        bs, bc = np.asarray(ts.data).view(np.uint8).reshape(-1), np.asarray(tc.data).view(np.uint8).reshape(-1)
        if np.array_equal(bs, bc):
            continue
        fmt = FMT.get(ts.tensor_type)
        if fmt is None:
            problems.append(f"{ts.name}: {ts.tensor_type.name} bytes changed")
            continue
        (ds, ss), (dc, sc) = scales(bs.reshape(-1, fmt[0]), fmt), scales(bc.reshape(-1, fmt[0]), fmt)
        if not (np.array_equal(ds, dc) and np.array_equal(ss, sc)):
            problems.append(f"{ts.name}: block scales changed")
        changed.append(ts.name)
    rec = {"sha256": sha256(cand), "file": os.path.basename(cand), "bytes": os.path.getsize(cand),
           "start_sha256": sha256(start), "start_file": os.path.basename(start),
           "equal_layout": not any("differ" in p for p in problems),
           "scales_preserved": not any("scales" in p for p in problems),
           "unchanged_outside_codes": not any("bytes changed" in p for p in problems),
           "changed_tensors": changed, "problems": problems}
    rec["refined"] = not problems and len(changed) > 0
    with open(out, "w") as f:
        json.dump(rec, f, indent=1)
    print(f"{out}: refined={rec['refined']}, {len(changed)} tensors with changed codes, {len(problems)} problems")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
