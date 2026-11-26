# Ling-3.0-tiny proof of concept: what transfers to Ling-3.2-air

Pilot for the Ling-3.2-air quantization, run on 24 November 2026 on a 48 GB Apple M-series Mac. Model `inclusionAI/Ling-3.0-tiny` rev `9a98e35` (BF16, 7.89 B text parameters, 128 experts); llama.cpp `c661033` (21 Nov 2026), Metal. Every number below is measured unless marked otherwise; the dated detail, commands and caveats are in `CHANGELOG.md`, raw rows in `results/ablations.tsv`. Numbers from this repo are smoke tests for tooling and method, not results about Ling-3.2-air.

## Verdicts

| # | Question | Verdict | Evidence |
|---|---|---|---|
| Q0 | Does llama.cpp's `bailingmoe3` forward match the HF reference? | Works | Reference: HF code with pure-PyTorch copies of fla-core 0.6.0's naive KDA ops (a line-by-line check found no math difference; precision details only). Random F32 copy of the architecture: KL 2e-10, top-1 100% over 2,000 tokens (measured; the synthetic-model builder was not kept). Real BF16 GGUF: KL 7e-7 to 2.5e-5 on short prompts; on a 2,000-token prompt (548 scored positions) 9e-4 to 1.8e-3, top-1 ≥ 98.7% (Metal). Residual: MoE router near-ties, the f16 KV cache, and backend/ubatch rounding. |
| Q1 | BF16 safetensors → GGUF | Works | 57 s, 3.1 GB RSS, 526 tensors, layer layout as in the config. |
| Q2 | Q8_0 build and run | Works, with a high floor | KLD vs BF16 0.0075 ± 0.0004, top-1 96.3%. Routed experts alone at Q8_0 give 0.0048, about two thirds of it; the non-expert groups together give the rest, though none moves it alone. |
| Q3 | Do requested types land where asked? | Works, one fallback | `--tensor-type` lands exactly for every group and type tested, except MLA `attn_k_b` (row width 128), which falls back to IQ4_NL, Q4_0 or Q5_0. Stock mixes silently move KDA projections (IQ3_XXS puts 36 KDA q/k at IQ2_S). |
| Q4 | Does an imatrix cover KDA, MLA and every expert? | Works with a workaround | Covers 309 of 311 quantizable matrices (the embedding lookup never gets data; `output` needs `--process-output`). With wiki text only, 63 to 97 of 2,944 experts get no tokens, mostly in deep layers; mixed calibration text (prose, chat, code) at 400 × 512 tokens leaves 0, with a minimum of 6 tokens per expert. |
| Q5 | Does KLD grow with context? | KDA flat, MLA grows (direction only) | On identical 4 × 4,096-token windows, KL by position from IQ2_XXS KDA weights does not rise (last/first 512-token bucket 0.84–0.91), while IQ2_XXS MLA projections rise from 0.19 at positions 0–511 to 0.25–0.31 after (1.63×). Four windows give bucket SEs of 0.01–0.11, so this is a direction, not a slope; only two KDA groups and one MLA group were tested. m2 runs at 32K with finite PPL and no NaN. |
| Q6 | What does a run cost? | Measured on tiny, scaled below | See "Cost model". |
| Q7 | Does the single-group ablation run end to end? | Works | Ten groups at IQ2_XXS on a Q8_0 base, each ΔKLD ≥ 20× its SE at 16 × 512 chunks. A build guided by the map (m2) beats stock IQ3_XXS at a smaller file size: KLD −16% (wiki) and −22% (out-of-domain); against bloomer010's public IQ3_XXS −20% and −26%. A variant with a better embedding (m5) does better still (−23%, −27% vs stock) but is 0.16% larger than stock, so it does not count. |
| Q8 | Does the converter take Ling-3.2-air? | Works in dry-run | All 938 text tensors (255 GB at BF16) and the 334-tensor vision projector map; FP8 E4M3 128 × 128 dequant path and MTP (`blk.42`) recognised. Dequant numerics untested. |
| T | Does Inkling's full-Gram code refiner cut KL on `bailingmoe3` experts? | Works (gate T PASS) | Routed experts only, Q8_0 base, equal bytes: IQ3_XXS KL 0.297 → 0.243 on wiki (R/U 0.82 [0.79, 0.85]) and 0.175 → 0.141 out-of-domain (0.80 [0.77, 0.84]); IQ2_XXS 0.709 → 0.557 (0.79) and 0.451 → 0.352 (0.78); holds at 4,096 context. Cost and Air extrapolation in `results/gate-t.md`: Grams ~585 GB and 160–1,100 CPU-h on Air (inferred). |

## What the Air repo should do differently

1. Reference and quants on the same backend, `-ub` and KV type. llama.cpp's BF16 forward already differs from fp32 HF by ~1e-3 KL on long text, and that gap depends on all three.
2. Calibrate with mixed text and count zero-hit experts before every build. Air has 512 experts per layer, four times tiny's, so coverage gets harder; quantize does not stop for unseen experts and only prints "partial data".
3. Measure MLA groups at ≥ 4K context, on identical target tokens across context lengths (`scripts/q5_positions.py`). Treat a KDA ranking at 512 as provisional until the same check holds on Air.
4. Allocate by marginal cost at each bit level, not by one sensitivity ranking. Protecting every small sensitive group at Q8_0 while leaving expert gate/up at IQ2_XXS lost to stock at equal size (0.537 vs 0.461); the build that won spent bits on experts first (IQ3_XXS) and put the small groups at IQ3_XXS to Q5_K, with the embedding at Q2_K and the output at Q3_K.
5. Do not treat the embedding and output as free. From a Q8_0 base, `token_embd` at Q2_K alone costs KLD 0.106 and `output` at Q3_K 0.024, while `output` at Q6_K costs nothing measurable. Keep `token_embd` at Q3_K or above (m5 vs m2: −0.03 KLD for +18 MB, though m5 also moves the shared expert from Q5_K to Q4_K). Without an imatrix, `token_embd` at IQ2_XXS, IQ2_XS or IQ1_S aborts `llama-quantize` on `GGML_ASSERT(imatrix != NULL)`, because the imatrix never covers the embedding lookup.
6. Name tensor groups with the anchored regexes in `CHANGELOG.md` (Q3 entry): matching is unanchored and first-match-wins, so `attn_q` also hits the MLA `attn_q_a` and `attn_q_b`.
7. Check Air's `attn_k_b` row width (128 on tiny) and count its fallback type in the size. Match sizes on real file bytes: `--dry-run` "quant size" leaves out ~6 MB of GGUF metadata.
8. Always quantize from a quantized base type. With base `BF16`, `llama-quantize` ignores every `--tensor-type` override and writes an unchanged copy, with no warning.
9. Pair small effects. At 16 × 512 chunks the SE is ~0.0004, so effects below ~0.001 are not resolvable from unpaired means; use per-token differences against the same BF16 reference (`scripts/gguf_logits.py` dumps logits for fixed token ids).
10. Pin nvcc ≥ 13.2.2 on Blackwell (llama.cpp #28581: IQ1_S/IQ2_S/IQ3_S garbage output under 13.2.0–13.2.1; from a source search, not tested here).
11. Check one FP8 tensor's dequant against a hand computation on the first real Air shard.

## Cost model

Parameter counts from tensor shapes (`results/q6-params.md`): tiny 7.89 B text, 1.14 B active per token; Air 124.41 B text, 5.30 B active, plus 3.07 B MTP and 0.43 B vision. "Active" here leaves out the embedding lookup and counts the output matmul; the model cards say 1.3 B and 5.5 B. Air / tiny: 15.8× total, 4.7× active.

| Step | Tiny on the Mac (measured) | Scales with | Air on the same Mac (inferred) |
|---|---|---|---|
| Convert to BF16 GGUF | 57 s, 3.1 GB RSS | total bytes | ~15 min, plus FP8 dequant; 255 GB output |
| Quantize Q8_0 | 7–9 s | total params | ~2 min |
| Quantize IQ2_XXS (full) | 109 s | total params, type | ~30 min |
| Quantize IQ3_XXS (full) | 174–307 s | total params, type | 45–80 min |
| Imatrix | ~1,090 tok/s, 16.4 GB | active params | ~230 tok/s on Metal; does not fit in 48 GB at BF16 |
| KLD pass, 16 × 512 | 4–8 s | active params | ~30 s if it fits; it does not at BF16 |

These ratios are rough proxies. Quantize runs on the CPU and also depends on type, FP8 dequant and disk I/O, so the rented box's core count matters more than its GPU for the build loop. Forward passes also depend on expert traffic, MLA context work, logits scoring and the device split. The BF16 GGUF for Air is ~255 GB with MTP (~249 GB for the text trunk); weights alone exceed any single GPU, so plan resident weights, KV and recurrent state, scratch and logits per device, and keep reference logits files on disk (16 × 4,096 chunks at Air's vocabulary is ~10.3 GB). Time conversion, quantization, imatrix and one KLD chunk separately on the rented box before budgeting; this table only says which of them will dominate.

## Not settled here

- CUDA and multi-GPU behaviour of `bailingmoe3` (all runs were Metal).
- FP8 dequant numerics, MTP and vision quality (tiny has none of them).
- Effects of 512 experts on routing noise; tiny's Q8_0 floor of 0.0075 may be larger or smaller on Air.
- Additivity: stock IQ2_XXS (1.49) is below the sum of single-group ΔKLD (~2.1), but stock also changes types the groups did not, so additivity was not tested with a matched joint build.
- Whether the m2 gain survives on Air. It came from four measured candidates on tiny, picked by size (m2) or wiki KLD (m4, m5), with the out-of-domain set held out; Air's public quants already use sensitivity-based promotion (bartowski's validation is on Flash-VL, not Air).
- MTP importance data (`--nextn`), Air's own chat template in the calibration set, and YaRN: llama.cpp's `bailingmoe3` has no YaRN mscale, so Air's optional 1M-token mode is unverified.
- Looping reports on Air (5–15K tokens, llama.cpp): the 25 Nov same-suffix test found no drift of llama.cpp from the HF reference on tiny up to 8K (KL on identical suffixes 2–4e-3 at 1K, 4K and 8K histories; CHANGELOG addendum). Past 8K, Air's clamp layers and free decoding remain untested.
