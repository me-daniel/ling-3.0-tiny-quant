"""Q6: parameter counts that drive cost, for Ling-3.0-tiny (GGUF) and Ling-3.2-air (safetensors headers).

Active params per token = everything except routed experts + top_k/E of routed experts (embedding lookup excluded
from compute, counted separately). Writes results/q6-params.md. Timing scale-up is in CHANGELOG, from these ratios.
usage: .venv/bin/python -I scripts/q6_cost.py models/ling-3.0-tiny-BF16.gguf scratch/air
"""

import json
import math
import struct
import sys
from pathlib import Path

from gguf import GGUFReader


def tiny(path):
    g = {"experts": 0, "embd": 0, "other": 0}
    for t in GGUFReader(path).tensors:
        n = math.prod(int(x) for x in t.shape)
        g["experts" if "_exps." in t.name else "embd" if t.name in ("token_embd.weight", "output.weight") else "other"] += n
    return g, 8 / 128


def air(d):
    g = {"experts": 0, "embd": 0, "other": 0, "mtp": 0, "vision": 0}
    for f in sorted(Path(d).glob("*.safetensors")):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            meta = json.loads(fh.read(n))
        for name, v in meta.items():
            if name == "__metadata__" or name.endswith(("_scale_inv", "_scale")):
                continue
            k = math.prod(v["shape"])
            if any(s in name for s in ("vision", "visual", "vit", "mm_proj", "linear_proj")):
                g["vision"] += k
            elif ".layers.42." in name or "mtp" in name:
                g["mtp"] += k
            elif ".experts." in name and "shared" not in name:
                g["experts"] += k
            elif "embed_tokens" in name or "lm_head" in name:
                g["embd"] += k
            else:
                g["other"] += k
    return g, 8 / 512


def row(label, g, frac):
    text = g["experts"] + g["embd"] + g["other"]
    active = g["other"] + g["experts"] * frac + g["embd"] / 2  # lm_head is compute, the lookup is not
    return label, text, g["experts"], active, g


def main(t, a):
    rows = [row("Ling-3.0-tiny", *tiny(t)), row("Ling-3.2-air", *air(a))]
    out = ["# Q6: parameter counts for the cost model", "",
           "| model | text params (B) | routed experts (B) | experts share | active per token (B) | extra (B) |",
           "|---|---|---|---|---|---|"]
    for label, text, ex, act, g in rows:
        extra = ", ".join(f"{k} {g[k]/1e9:.2f}" for k in ("mtp", "vision") if k in g)
        out.append(f"| {label} | {text/1e9:.2f} | {ex/1e9:.2f} | {ex/text:.1%} | {act/1e9:.2f} | {extra or '-'} |")
    (_, t_text, _, t_act, _), (_, a_text, _, a_act, _) = rows
    out += ["", f"Ratios Air / tiny: total text params {a_text/t_text:.1f}×, active per token {a_act/t_act:.1f}×."]
    Path("results/q6-params.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
