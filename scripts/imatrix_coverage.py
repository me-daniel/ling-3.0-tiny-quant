"""Q4: which tensors and experts an imatrix covers, and how evenly experts are hit.

Reads `<tensor>.counts` (one count per matrix; per expert for *_exps) from an imatrix GGUF.
Writes results/q4-imatrix.md. Compares the tensor list against the model's quantizable 2D/3D weights.
usage: .venv/bin/python -I scripts/imatrix_coverage.py <imatrix.gguf> <model.gguf>
"""

import re
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFReader


def main(imat, model):
    r = GGUFReader(imat)
    counts = {t.name[: -len(".counts")]: np.asarray(t.data, dtype=np.float64).ravel()
              for t in r.tensors if t.name.endswith(".counts")}
    weights = [t.name for t in GGUFReader(model).tensors
               if t.name.endswith(".weight") and len(t.shape) >= 2 and t.shape[1] > 1
               and not re.search(r"norm|ffn_gate_inp|ssm_conv1d", t.name)]
    missing = [w for w in weights if w not in counts]
    out = ["# Q4: imatrix coverage", "", f"imatrix `{imat}`; model `{model}`.", "",
           f"- Quantizable weight matrices: {len(weights)}; with imatrix data: {len(weights) - len(missing)}.",
           f"- Missing: {', '.join(f'`{m}`' for m in missing) or 'none'}."]
    exps = {k: v for k, v in counts.items() if k.endswith("ffn_gate_exps.weight")}
    if exps:
        allc = np.stack([exps[k] for k in sorted(exps, key=lambda s: int(s.split('.')[1]))])
        zero = (allc == 0).sum(axis=1)
        share = allc / allc.sum(axis=1, keepdims=True)
        out += ["", "## Routed experts (from `ffn_gate_exps` counts; tokens routed per expert)", "",
                f"- MoE layers: {allc.shape[0]}, experts per layer: {allc.shape[1]}.",
                f"- Experts with zero tokens: {int(zero.sum())} of {allc.size} (per layer max {int(zero.max())}).",
                f"- Tokens per expert: min {allc.min():.0f}, median {np.median(allc):.0f}, max {allc.max():.0f}.",
                f"- Least-hit expert share vs uniform (1/{allc.shape[1]}): min over layers {share.min() * allc.shape[1]:.3f}×.",
                f"- Experts under 10% of the uniform share: {int((share < 0.1 / allc.shape[1]).sum())}.", "",
                "| layer | zero | min | median | max | max/min |", "|---|---|---|---|---|---|"]
        for k, row in zip(sorted(exps, key=lambda s: int(s.split('.')[1])), allc):
            mn = row.min()
            out.append(f"| {k.split('.')[1]} | {int((row == 0).sum())} | {mn:.0f} | {np.median(row):.0f} | {row.max():.0f} | "
                       f"{(row.max() / mn) if mn else float('inf'):.1f} |")
    Path("results").mkdir(exist_ok=True)
    Path("results/q4-imatrix.md").write_text("\n".join(out) + "\n")
    print("\n".join(out[:12]))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
