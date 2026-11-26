"""Q7: single-group sensitivity S = (KLD_group - KLD_base) / GB saved vs the all-Q8_0 build. Writes results/q7-sensitivity.md."""
import csv
from pathlib import Path

rows = {r["name"]: r for r in csv.DictReader(open("results/ablations.tsv"), delimiter="\t")}
base = rows["q8_all"]
b_bytes, b_kld = int(base["bytes"]), float(base["mean_kld"])
out = ["# Q7: single-group IQ2_XXS sensitivity on a Q8_0 base (Ling-3.0-tiny)", "",
       "KLD vs BF16, 16 × 512 wikitext-2 test, Metal, default -ub, imatrix wiki-train 100 × 512. "
       f"Base all-Q8_0: {b_bytes/1e9:.3f} GB, mean KLD {b_kld:.4f}. S = ΔKLD / GB saved (higher = keep the bits).", "",
       "| group at IQ2_XXS | GB saved | ΔKLD | ± (SE of group run) | S per GB | top-1 % | quantize s |", "|---|---|---|---|---|---|---|"]
groups = [r for n, r in rows.items() if n.startswith("g_")]
for r in sorted(groups, key=lambda r: -(float(r["mean_kld"]) - b_kld) / max(b_bytes - int(r["bytes"]), 1)):
    saved = (b_bytes - int(r["bytes"])) / 1e9
    d = float(r["mean_kld"]) - b_kld
    out.append(f"| {r['name'][2:].replace('_iq2_xxs','')} | {saved:.3f} | {d:.4f} | {float(r['kld_se']):.4f} | {d/saved:.2f} | {r['top1']} | {r['quant_s']} |")
out += ["", "| stock mix | GB | BPW | mean KLD | PPL | top-1 % | quantize s |", "|---|---|---|---|---|---|---|"]
for n, r in rows.items():
    if n.startswith(("stock_", "q8_all")):
        out.append(f"| {n} | {int(r['bytes'])/1e9:.3f} | {r['bpw']} | {r['mean_kld']} | {r['ppl']} | {r['top1']} | {r['quant_s']} |")
Path("results/q7-sensitivity.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
