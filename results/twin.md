# Twin of a public file: refined codes in bloomer010's Ling-3.0-tiny IQ3_XXS

Start: `models/public/bloomer010/Ling-3.0-tiny-IQ3_XXS.gguf` (rev `1689ed17`, 3,129,076,064 B, all 69 routed expert tensors IQ3_XXS, everything else in bloomer010's own low-bit mix). Twin: same file with the expert codes refined by `scripts/refine_codes.py` (bloomer010's own d and scales kept, guards 9b/9c, 4 sweeps, Grams `models/grams/mixc-c512-n400`); 457 other tensors byte-identical, same size, ggml dequant == gguf-py == refiner decode bit-exact (layers 1, 23). Scoring as gate T (`results/gate-t.md`). Measured, 25 Nov 2026. Rule registered before scoring (CHANGELOG): PASS if twin ≤ 0.90 × original on wiki512 and ood512 with the CI of twin − original below 0 on both.

"U" = bloomer010's file, "R" = twin.

| level | set | chunks | KL U | KL R | R - U [95% CI] | R / U [95% CI] | p99 U / R | top-1 U / R % |
|---|---|---|---|---|---|---|---|---|
| 3 | wiki512 | 64 | 0.5030 | 0.4520 | -0.0510 [-0.0619, -0.0401] | 0.899 [0.879, 0.918] | 3.720 / 3.483 | 71.16 / 72.44 |
| 3 | ood512 | 64 | 0.3372 | 0.3095 | -0.0277 [-0.0371, -0.0186] | 0.918 [0.891, 0.945] | 3.333 / 3.292 | 81.98 / 82.47 |
| 3 | wiki4096 | 16 | 0.4037 | 0.3731 | -0.0306 [-0.0402, -0.0210] | 0.924 [0.904, 0.946] | 2.946 / 2.765 | 73.65 / 74.43 |

Q8 floor wiki512: KL 0.0092, top-1 96.11%
Q8 floor ood512: KL 0.0046, top-1 97.86%
Q8 floor wiki4096: KL 0.0063, top-1 96.71%

- twin vs bloomer: FAIL (rule: R/U <= 0.90 and CI of R - U below 0 on wiki512 and ood512)

Offline (in-sample, `twin-offline.md`): gate/up 0.0244 → 0.0138, down 0.0341 → 0.0170 (gate T's U3: 0.0238 → 0.0136, 0.0273 → 0.0163). Refinement 2,147 s wall, 10 workers, 20,193 CPU-s (6.9 per expert), 2.7 GB parent RSS.

## Reading

- The absolute gain is about the same as in gate T: −0.051 vs −0.054 on wiki512, −0.028 vs −0.034 on ood512. Both CIs are below 0.
- The ratio is smaller because the starting KL is higher: bloomer010's non-expert tensors are low-bit, while gate T had Q8_0 around the experts. Wiki512 reaches 0.899 (inside the rule). Ood512 reaches 0.918 [0.891, 0.945], so the registered rule fails on ood.
- Reading (inferred): the refiner removes a roughly fixed amount of expert error whatever scales it inherits. The relative gain shrinks as the error from the rest of the file grows. It works on a foreign file's scales without any re-quantization.

Later runs (CHANGELOG 25–26 Nov): test 2 (m2 mix) PASS, ratios 0.874 wiki and 0.867 ood. Cheap modes on this twin, gain kept = (original − mode) / (original − full twin), rule ≥ 0.75 on both 512 sets:

| mode | set | gain full | gain mode | kept [95% CI] |
|---|---|---|---|---|
| twin-s1 | wiki512 | 0.0510 | 0.0445 | 0.87 [0.73, 1.02] |
| twin-s1 | ood512 | 0.0277 | 0.0137 | 0.49 [0.07, 0.78] |
| twin-s1 | wiki4096 | 0.0306 | 0.0212 | 0.69 [0.53, 0.83] |
| twin-half | wiki512 | 0.0510 | 0.0310 | 0.61 [0.46, 0.75] |
| twin-half | ood512 | 0.0277 | 0.0273 | 0.99 [0.79, 1.20] |
| twin-half | wiki4096 | 0.0306 | 0.0211 | 0.69 [0.55, 0.88] |

Neither (a) 1 sweep nor (b) half the experts keeps the gain by that rule. (c) 1/4-token Grams was postponed (registered, not run).
