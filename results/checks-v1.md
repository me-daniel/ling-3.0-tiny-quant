# v1.0 checks: speed, HellaSwag, loops

Released file: `m2r.gguf` (m2 with refined expert codes; 3,116,281,408 B, sha256 1282ce37…79bd7). Metal, Apple M5 Pro, llama.cpp c661033 with patches 0010–0013 (none touch inference). Registered as descriptive, with no pass rule (CHANGELOG 26 Nov). Measured 26 Nov 2026.

## In short

- Speed: the released file runs as fast as m2. They have the same tensor types and byte layout, and the bench cannot separate them (prompt processing ~2,200 t/s, generation ~81–87 t/s, with large run-to-run spread). bloomer010 and bartowski are in the same range.
- HellaSwag (1,000 tasks): 55.9% for the released file, 55.1% for bloomer010 IQ3_XXS, 54.5% for bartowski Q2_K_L. The differences (+0.8 and +1.4 points) are within noise (paired McNemar p = 0.52 and 0.26). BF16 was cancelled at task 483 (59.0% at that point; not comparable).
- Loops: postponed (cancelled before any comparison existed).

## Speed (`llama-bench -ngl 99 -p 512 -n 128 -r 5`, one run, files back to back; `logs/bench-v1.md`)

| file | bytes | pp512 t/s (mean ± sd, 5 reps) | tg128 t/s |
|---|---|---|---|
| m2 (`ling-3.0-tiny-m2.gguf`) | 3,116,281,408 | 2207 ± 425 | 80.6 ± 2.4 |
| m2r (released, `m2r.gguf`) | 3,116,281,408 | 2198 ± 475 | 86.5 ± 0.8 |
| bloomer010 IQ3_XXS | 3,129,076,064 | 2637 ± 870 | 83.8 ± 3.0 |
| bartowski Q2_K_L | 3,235,879,360 | 2546 ± 789 | 81.6 ± 1.7 |

llama-bench names models by their majority type, so its m2 and m2r rows both read "Q4_K - Medium"; the row order above is the `-m` order. The pp512 spread (20–35%) means differences of this size are noise. m2 and m2r run identical kernels on identical shapes; only the code bytes differ.

## HellaSwag (first 1,000 tasks of `hellaswag_val_full.txt`, sha256 d5725393…; `llama-perplexity --hellaswag --hellaswag-tasks 1000`)

Every model sees the same task order, from llama-perplexity's fixed-seed shuffle. Per-task correctness is recovered exactly from the cumulative accuracy printed after each task (`scripts/hellaswag_pairs.py`).

| file | acc_norm | 95% Wilson CI |
|---|---|---|
| m2r | 55.9% | [52.8, 58.9] |
| bloomer | 55.1% | [52.0, 58.2] |
| bartowski | 54.5% | [51.4, 57.6] |

| candidate vs | cand right, other wrong | cand wrong, other right | difference (points) | exact McNemar p |
|---|---|---|---|---|
| bloomer | 64 | 56 | +0.8 | 0.523 |
| bartowski | 73 | 59 | +1.4 | 0.258 |

BF16 reference: cancelled at task 483 of 1,000 to free the GPU (59.0% [54.6, 63.3] on those 483; not comparable with the 1,000-task numbers).

## Loops

Postponed (registered, not completed): cancelled during the first file's pass when v1.0 was declared done, so there is no comparison to report.

## Correction (26 Nov 2026): the loop check got further than first logged

`logs/loop-check.out` shows complete passes for m2r (20 of 20 prompts) and bloomer010 IQ3_XXS (20 of 20), and 15 of 20 for BF16 before the cancel. With the registered detector (a span of 8–128 tokens repeated 3 times back to back), no output looped: 0 of 20, 0 of 20 and 0 of 15. Greedy, 512 new tokens, so this says nothing about the 5–15K-token looping reported for Ling-3.2-air.
