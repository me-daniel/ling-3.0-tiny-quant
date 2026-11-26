"""Q3: do requested types land where asked? llama-quantize --dry-run per tensor group and per stock mix.

Writes results/q3-types.md. Parses the per-tensor dry-run lines, so no GGUF is written.
usage: .venv/bin/python -I scripts/q3_types.py models/ling-3.0-tiny-BF16.gguf
"""

import collections
import re
import subprocess
import sys
from pathlib import Path

BIN = Path("vendor/llama.cpp/build/bin/llama-quantize")
KDA = "0|1|2|4|5|6|8|9|10|12|13|14|16|17|18|20|21|22"
MLA = "3|7|11|15|19|23"

# group -> regex used both for --tensor-type and to classify parsed tensor names (anchored on .weight per the audit)
GROUPS = {
    "kda_qkv": r"attn_[qkv]\.weight",
    "kda_gates": r"ssm_(f_a|g_a|beta)\.weight",
    "kda_out": rf"^blk\.({KDA})\.attn_output\.weight",
    "mla_proj": r"attn_(q_a|q_b|kv_a_mqa|v_b)\.weight",
    "mla_k_b": r"attn_k_b\.weight",
    "mla_gate": r"attn_gate\.weight",
    "mla_out": rf"^blk\.({MLA})\.attn_output\.weight",
    "exp_gate_up": r"ffn_(gate|up)_exps\.weight",
    "exp_down": r"ffn_down_exps\.weight",
    "shexp": r"ffn_(gate|up|down)_shexp\.weight",
    "dense0": r"^blk\.0\.ffn_(gate|up|down)\.weight",
    "embd_out": r"^(token_embd|output)\.weight",
}
TYPES = ["iq2_xxs", "iq3_xxs", "q2_k", "q4_k", "iq4_xs"]
STOCK = ["Q4_K_M", "Q2_K", "IQ3_XXS", "IQ2_XXS"]

LINE = re.compile(r"\]\s+(\S+)\s+- \[([^\]]*)\], type =\s*(\S+), (?:size =\s*[\d.]+ MiB -> \s*[\d.]+ MiB \((\S+)\)|size =)")


def group_of(name):
    for g, rx in GROUPS.items():
        if re.search(rx, name):
            return g
    return "other"


def dry_run(gguf, ftype, overrides=()):
    args = [str(BIN), "--dry-run"]
    for rx, t in overrides:
        args += ["--tensor-type", f"{rx}={t}"]
    p = subprocess.run(args + [gguf, ftype], capture_output=True, text=True)
    text = p.stdout + p.stderr
    rows = [(m[1], m[3], m[4] or m[3]) for m in LINE.finditer(text)]
    fallback = re.search(r"(\d+) of (\d+) tensor\(s\) required fallback", text)
    bpw = re.search(r"quant size\s*=\s*([\d.]+) MiB \(([\d.]+) BPW\)", text)
    if not rows:
        raise SystemExit(f"no tensor lines parsed for {ftype} {overrides}:\n{text[-2000:]}")
    return rows, int(fallback[1]) if fallback else 0, bpw.groups() if bpw else ("?", "?")


def by_group(rows):
    hist = collections.defaultdict(collections.Counter)
    for name, _, new in rows:
        hist[group_of(name)][new.lower()] += 1
    return hist


def fmt(counter):
    return ", ".join(f"{t}×{n}" for t, n in counter.most_common())


def main(gguf):
    out = ["# Q3: requested vs actual tensor types (llama-quantize --dry-run)", "",
           f"Source `{gguf}`, llama.cpp `c661033`. Base mix Q8_0; one group overridden at a time.", "",
           "## Per-group overrides", "",
           "| group | requested | tensors | actual types in group | fallbacks (whole model) |", "|---|---|---|---|---|"]
    base_rows, _, _ = dry_run(gguf, "Q8_0")
    unclassified = collections.Counter(
        f"{re.sub(r'^blk\.\d+\.', 'blk.N.', n)} ({new.lower()})" for n, _, new in base_rows if group_of(n) == "other")
    for g, rx in GROUPS.items():
        for t in TYPES:
            rows, nfb, _ = dry_run(gguf, "Q8_0", [(rx, t)])
            hist = by_group(rows)[g]
            ok = "" if set(hist) == {t} else " **≠**"
            out.append(f"| {g} | {t} | {sum(hist.values())} | {fmt(hist)}{ok} | {nfb} |")
    out += ["", "## Stock mixes (no overrides)", "", "| mix | BPW | fallbacks | per-group actual types |", "|---|---|---|---|"]
    for mix in STOCK:
        rows, nfb, (mib, bpw) = dry_run(gguf, mix)
        hist = by_group(rows)
        cells = "; ".join(f"{g}: {fmt(hist[g])}" for g in list(GROUPS) + ["other"] if hist[g])
        out.append(f"| {mix} | {bpw} | {nfb} | {cells} |")
    out += ["", "## Tensors in no group (left at the mix's choice)", "", ", ".join(f"`{n}`×{c}" for n, c in sorted(unclassified.items())) or "none"]
    Path("results").mkdir(exist_ok=True)
    Path("results/q3-types.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1])
