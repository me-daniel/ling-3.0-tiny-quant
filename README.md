# ling-3.0-tiny-quant

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/banner-dark.png">
  <img alt="Ling-3.0-tiny, 3.12 GB. 24 layers: 18 KDA (layer 0 with a dense FFN) and 6 gated MLA. Routed experts IQ3_XXS with refined codes, 86% of the weights." src="figures/banner-light.png">
</picture>

Code, measurements and figures behind [Ling-3.0-tiny-IQ3_XXS-refined.gguf](https://huggingface.co/Me-Daniel/Ling-3.0-tiny-3GB-GGUF), a 3 GB GGUF of [inclusionAI/Ling-3.0-tiny](https://huggingface.co/inclusionAI/Ling-3.0-tiny). On four held-out text sets it stays 21% to 46% closer to the original model than bloomer010's IQ3_XXS and 21% to 51% closer than bartowski's Q2_K_L, at the smallest size of the three, the same speed and the same HellaSwag score.

This repository is the test run for our Ling-3.2-air quantization: on the small sibling model we checked what llama.cpp supports on the `bailingmoe3` architecture, what each part of the model tolerates, and whether our refiner works here, before spending compute on Air. Every run used one Apple M5 Pro with 48 GB, llama.cpp `c661033` and Metal.

- Model: [huggingface.co/Me-Daniel/Ling-3.0-tiny-3GB-GGUF](https://huggingface.co/Me-Daniel/Ling-3.0-tiny-3GB-GGUF)
- Findings for Ling-3.2-air: [results/findings.md](results/findings.md)
- Dated log, registered rules and the status of every number: [CHANGELOG.md](CHANGELOG.md)

## Release gate

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/release-gate-dark.png">
  <img alt="Mean KL per set for the three files, and ours minus each comparator with paired 95% intervals. Ours is lower on every set: Chinese 0.33 vs 0.41 and 0.41, code 0.15 vs 0.25 and 0.22, tool calls 0.46 vs 0.58 and 0.61, long books 0.12 vs 0.23 and 0.25; every interval is below zero." src="figures/release-gate-light.png">
</picture>

| KL vs BF16 | Ours (3.12 GB) | bloomer010 IQ3_XXS (3.13 GB) | bartowski Q2_K_L (3.24 GB) |
|---|---|---|---|
| Chinese (128 × 512) | 0.328 | 0.413 | 0.413 |
| Code (128 × 512) | 0.146 | 0.250 | 0.217 |
| Tool calls (128 × 512) | 0.459 | 0.583 | 0.614 |
| Long books, PG19 (32 × 4,096) | 0.123 | 0.229 | 0.251 |

We registered the gate before any release scores existed: against each comparator, our file passes only if its KL is lower on all four sets with the paired 95% interval below zero, and it is not larger. It passes against both. Top-1 agreement is higher and p99 KL lower on every set; a book-level interval for PG19 agrees. Full table: [results/release-gate.md](results/release-gate.md).

Speed and tasks ([results/checks-v1.md](results/checks-v1.md)): generation runs at about 81–87 tokens/s for all files on Metal, and HellaSwag (1,000 tasks) gives 55.9% for ours, 55.1% for bloomer010 and 54.5% for bartowski, no difference beyond noise. A first loop check found no repetition loops; longer-generation checks will follow.

## Allocation and refinement

### Which tensor groups can go to 2 bits

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/sensitivity-dark.png">
  <img alt="Added KL vs MB saved, one group at IQ2_XXS on Q8_0: MLA gate +0.019 for 0.118 MB (161 KL per GB); MLA projections +0.208 for 15 MB (13.8); MLA output +0.124 for 15.2 MB (8.14); KDA output +0.32 for 45.6 MB (7.03); shared expert +0.218 for 43.7 MB (4.99); dense FFN +0.0691 for 17.1 MB (4.04); KDA q/k/v +0.223 for 137 MB (1.63); KDA gates +0.0827 for 91.5 MB (0.904); experts down +0.317 for 1,863 MB (0.17); experts gate/up +0.499 for 3,726 MB (0.134)." src="figures/sensitivity-light.png">
</picture>

During selection, we measured KL on wiki.test at 16 × 512 with one group at IQ2_XXS on a Q8_0 base (`results/ablations.tsv`, ratios in `results/q7-sensitivity.md`). Per GB saved, the routed experts add 0.13 (gate/up) and 0.17 (down) KL; KDA output adds 7.0, and the three MLA groups 8.1 to 161. That is why m2 keeps the experts at IQ3_XXS and spends the bytes elsewhere. The bars are the sum of the two runs' SEs, a bound under token independence, not paired chunk intervals.

### m2 against stock and public quantizations

| 16 × 512, `results/ablations.tsv` | wiki.test KL | eval-ood KL | m2 relative |
|---|---|---|---|
| m2 (3.12 GB) | 0.367 | 0.328 | |
| stock IQ3_XXS, same mixc imatrix (3.13 GB) | 0.436 | 0.419 | −16% / −22% |
| bloomer010 IQ3_XXS (3.13 GB) | 0.457 | 0.442 | −20% / −26% |

We chose m2 from four candidates measured on the first 16 wiki chunks; eval-ood was held out from that choice.

### Refinement

The refiner keeps every block scale and re-chooses each low-bit grid code to minimise an estimate of expert-output error on calibration inputs (input Gram matrices from 400 × 512 tokens of mixed text), so types and byte counts stay the same.

- Gate T, experts only on a Q8_0 base (`results/gate-t.md`, 64 × 512): IQ3_XXS ratio 0.818 (wiki) and 0.804 (ood), IQ2_XXS 0.785 and 0.781, every paired interval below zero. PASS.
- m2 with refined experts (m2r, the released file), 64 × 512: ratio 0.874 [0.853, 0.896] on wiki.test and 0.867 [0.839, 0.896] on eval-ood. Refining the KDA gates on top (m2rd) gave no gain beyond noise, so it was not adopted.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/ladder-dark.png">
  <img alt="KL from bloomer010's IQ3_XXS to our mix to our mix with refined experts, on two selection sets: wiki 0.50, 0.40, 0.35; held-out 0.34, 0.23, 0.20." src="figures/ladder-light.png">
</picture>

### Refined codes on a public file (twin test)

We refined the expert codes in bloomer010's IQ3_XXS and kept its scales and every other tensor byte for byte (`results/twin.md`, paired bootstrap over chunks).

| Set | Chunks | bloomer010 | Twin | Twin − bloomer010 [95% CI] | Ratio [95% CI] |
|---|---|---|---|---|---|
| wiki.test (512) | 64 | 0.5030 | 0.4520 | −0.0510 [−0.0619, −0.0401] | 0.899 [0.879, 0.918] |
| eval-ood (512) | 64 | 0.3372 | 0.3095 | −0.0277 [−0.0371, −0.0186] | 0.918 [0.891, 0.945] |
| wiki.test (4,096), not in the rule | 16 | 0.4037 | 0.3731 | −0.0306 [−0.0402, −0.0210] | 0.924 [0.904, 0.946] |

The registered rule required a ratio of 0.90 or lower on both 512 sets. The twin failed because its ratio on eval-ood was 0.918. In absolute terms the reductions are similar to gate T's on a Q8_0 base (−0.0540 and −0.0344).

## Findings

| # | Question | Verdict |
|---|---|---|
| Q0 | Does llama.cpp's `bailingmoe3` forward match the HF reference? | Works. KL 9e-4 to 1.8e-3 on a 2,000-token prompt, top-1 ≥ 98.7% |
| Q1 | BF16 safetensors → GGUF | Works. 57 s, 526 tensors |
| Q2 | Q8_0 | Works, with a high floor. KL 0.0075 ± 0.0004 |
| Q3 | Do requested types land where asked? | Works, except MLA `attn_k_b` (row width 128), which falls back |
| Q4 | Does an imatrix cover KDA, MLA and every expert? | With mixed calibration text, every expert gets tokens. Wiki text alone leaves 63 to 97 of 2,944 unseen |
| Q5 | Does KL grow with context? | Flat for KDA groups, rising for MLA (direction only) |
| Q7 | Does a sensitivity-guided mix beat stock IQ3_XXS? | m2 vs stock IQ3_XXS (same imatrix): −16% (wiki), −22% (ood), at a smaller file |
| Q8 | Does the converter take Ling-3.2-air? | Works in dry-run |
| T | Does the full-Gram code refiner cut KL on `bailingmoe3` experts? | Yes, on a Q8_0 base; see gate T above |

Details, evidence and the cost model: [results/findings.md](results/findings.md).

## What this repository holds

| Path | What it holds |
|---|---|
| `scripts/` | Conversion, ablation, Gram capture, refiner, splice and checks, reports, figures (`figures.py`, `style.py`) |
| `results/` | Generated reports and TSVs (`ablations.tsv`, `gate-t.md`, `twin.md`, `layout-m2.tsv`, ...) and `findings.md` |
| `patches/` | Patches on llama.cpp `c661033` (below) |
| `figures/` | Card figures as PNG and SVG, plus `alt.md` |
| `tests/` | Tests for the refiner |
| `CHANGELOG.md` | Dated log and registrations |

Not in git: model files (`models/`), calibration and eval text (`data/`), logs and per-token dumps (`logs/`, `runs/`), the llama.cpp clone (`vendor/`). The release test sets come from public sources (Chinese Wikipedia, llama.cpp source, Hermes function-calling chats, four PG19 books); the script that builds them, with hashes and book ids, is part of our Ling-3.2-air study and will be published with it.

## Patches

These patches apply to llama.cpp `c661033`. Running a GGUF requires none of them.

| Patch | What it does |
|---|---|
| `patches/0010-imatrix-gram-capture.diff` | `llama-imatrix` also writes a full input Gram per routed expert, for the refiner |
| `patches/0011-perplexity-kld-dump.diff` | `llama-perplexity --kl-divergence` dumps per-token KL (`KLD_DUMP`), so intervals come from exact chunk means |
| `patches/0012-imatrix-gram-capture-dense.diff` | 0010 plus Grams for dense matrices; apply instead of 0010 |

## Reproduce

```sh
sh scripts/convert-q8.sh                                          # BF16 GGUF and Q8_0
IMATRIX=<imatrix.gguf> EVAL=<text> sh scripts/ablate.sh <name> <base_type> 'regex=type' ...   # one ablation row
sh scripts/gate_t.sh build U3 && sh scripts/gate_t.sh base        # gate T builds and BF16 bases
sh scripts/twin_chain.sh                                          # twin, m2r and cheap modes
.venv/bin/python -I scripts/gate_t_report.py > results/gate-t.md
sh scripts/release_score.sh                                        # release-gate scoring; dumps in runs/release/
.venv/bin/python -I scripts/release_gate_report.py --gguf cand=<release.gguf> bartowski=<Q2_K_L.gguf> > results/release-gate.md
.venv/bin/python -I scripts/gguf_layout.py <release.gguf> results/layout-release.tsv
.venv/bin/python -I scripts/refine_record.py models/ling-3.0-tiny-m2.gguf <release.gguf> results/refinement-release.json
for t in light dark; do .venv/bin/python -I scripts/figures.py --theme $t \
  --layout results/layout-release.tsv --record results/refinement-release.json; done
```

The figures use Red Hat Display and Red Hat Text from Google Fonts, kept in `build/fonts/` (git-ignored). Without them `scripts/style.py` falls back to DejaVu Sans and says so.

Checks: `.venv/bin/python -I -m pytest -q tests` and `.venv/bin/python scripts/test_fla_torch.py`.

## Status

v1.0 is released. KL against BF16 measures closeness to the original model; HellaSwag is the one task check so far. Everything ran on one machine, at one engine commit, with one calibration text. The findings for Ling-3.2-air are pilot results, not results about that model. Changes and registrations: [CHANGELOG.md](CHANGELOG.md).

Code: MIT License ([LICENSE](LICENSE)). The model file follows the base model's MIT License.
