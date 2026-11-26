# Gate T: does the full-Gram code refiner give a real KL gain on bailingmoe3?

Ling-3.0-tiny rev `9a98e35`, llama.cpp `c661033` plus `patches/0010` (Gram capture) and `patches/0011` (per-token KL dump), Metal, 25 Nov 2026. Every number is measured unless marked inferred. Method and logs: `CHANGELOG.md` (gate T entries).

## Setup

- U3 / U2: `llama-quantize` from BF16, base Q8_0, every routed expert tensor (`^blk\.[0-9]+\.ffn_(gate|up|down)_exps\.weight$`, 69 tensors) at IQ3_XXS / IQ2_XXS, imatrix from the Gram capture (`data/calib-mixc.txt`, 400 × 512, `--parse-special`). 3,687,236,800 B / 2,819,015,872 B.
- R3 / R2: U with refined expert codes (`scripts/refine_codes.py`, guards 9b and 9c, 4 sweeps). Same file size, tensor names, order, types and shapes as U; 457 other tensors byte-identical (`scripts/check_refined.py`).
- Scoring: KL against BF16 with one `--kl-divergence-base` file per set, per-chunk means from per-token KL, paired bootstrap over chunks (10,000 resamples, seed 0). eval sets share 0 13-grams with the calibration text.
- Rule, registered before any refined build existed: PASS for a level if mean KL of R ≤ 0.90 × U on wiki512 and on ood512, with the paired 95% CI of R − U below zero on both. wiki4096 is a check only.

## Result

| level | set | chunks | KL U | KL R | R - U [95% CI] | R / U [95% CI] | p99 U / R | top-1 U / R % |
|---|---|---|---|---|---|---|---|---|
| 3 | wiki512 | 64 | 0.2974 | 0.2434 | -0.0540 [-0.0643, -0.0440] | 0.818 [0.790, 0.847] | 2.591 / 2.202 | 77.75 / 79.55 |
| 3 | ood512 | 64 | 0.1754 | 0.1410 | -0.0344 [-0.0437, -0.0265] | 0.804 [0.771, 0.836] | 2.192 / 1.813 | 87.41 / 88.64 |
| 3 | wiki4096 | 16 | 0.2032 | 0.1646 | -0.0386 [-0.0476, -0.0312] | 0.810 [0.776, 0.840] | 1.833 / 1.559 | 80.92 / 82.90 |
| 2 | wiki512 | 64 | 0.7094 | 0.5571 | -0.1523 [-0.1731, -0.1322] | 0.785 [0.761, 0.809] | 5.313 / 4.729 | 66.02 / 70.12 |
| 2 | ood512 | 64 | 0.4512 | 0.3522 | -0.0989 [-0.1202, -0.0786] | 0.781 [0.748, 0.816] | 4.794 / 3.931 | 79.17 / 81.95 |
| 2 | wiki4096 | 16 | 0.5141 | 0.4256 | -0.0886 [-0.1028, -0.0742] | 0.828 [0.804, 0.852] | 4.184 / 3.559 | 70.60 / 72.81 |

Q8 floor wiki512: KL 0.0092, top-1 96.11%
Q8 floor ood512: KL 0.0046, top-1 97.86%
Q8 floor wiki4096: KL 0.0063, top-1 96.71%

- Gate T, IQ3_XXS: PASS (rule: R/U <= 0.90 and CI of R - U below 0 on wiki512 and ood512)
- Gate T, IQ2_XXS: PASS (rule: R/U <= 0.90 and CI of R - U below 0 on wiki512 and ood512)

Offline (in-sample Grams, J_G / tr(W G W^T) summed over refined experts): IQ3_XXS gate/up 0.0238 → 0.0136, down 0.0273 → 0.0163 (`gate-t-offline-R3.md`); IQ2_XXS gate/up 0.0855 → 0.0474, down 0.1093 → 0.0611 (`gate-t-offline-R2.md`). Skipped by the min-token rule (input dim): 223 of 2,944 gate/up units, 85 of 2,944 down units (`gate-t-counts.md`).

## Cost on this Mac (M5 Pro, 5 P + 10 E cores, 48 GB)

| Step | Tiny (measured) | Ling-3.2-air (inferred) |
|---|---|---|
| Gram capture, 400 × 512 tokens | 4 passes × ~268 s = 18 min; ≤ 27.3 GB max RSS incl. the 15.8 GB mmapped model (Grams held in RAM: 1.21 GB per gate layer, 0.13 GB per down layer); 29 GB on disk | Gate Gram 2560² × 4 B = 26.2 MB per expert, 13.4 GB per layer, 537 GB for 40 MoE layers; down (768²) 48 GB. The patch keeps all matched Grams in RAM, so at most one or two gate layers per pass, ~40 passes. Per-expert tokens scale as 8/512 of the calibration tokens (tiny: 8/128), so ~4× the tokens (~800k) for a similar per-expert count against a higher min-token bar (2,560). |
| Refinement IQ3_XXS | 3,855 s wall, 10 workers; 33,135 CPU-s = 11.3 CPU-s per expert; ~1.1 GB per worker | Air expert: 1,536 × 2,560 gate/up + 2,560 × 768 down = 5.9 M weights vs 2.36 M (2.5×), and the R update grows with input dim (×1.7): ~28–45 CPU-s per expert × 20,480 = 160–260 CPU-h; ~19–30 h on this Mac |
| Refinement IQ2_XXS | 15,458 s wall, 10 workers; 145,420 CPU-s = 49.4 CPU-s per expert | ~125–200 CPU-s per expert, 700–1,100 CPU-h; ~80–130 h on this Mac, ~11–18 h on a 64-core box |
| Splice + check | < 1 min per build | disk-bound, minutes |
| KLD scoring, 64 × 512 + 64 × 512 + 16 × 4096 | 18–39 s per set per model; bases 5.1 + 5.1 + 10.3 GB | see `findings.md` cost model |

The refiner is embarrassingly parallel over experts and needs only CPU, the Grams and the U build, so on Air it is a CPU job on the rented box (inferred).
