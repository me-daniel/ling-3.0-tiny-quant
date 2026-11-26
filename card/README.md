---
license: mit
base_model: inclusionAI/Ling-3.0-tiny
base_model_relation: quantized
library_name: llama.cpp
pipeline_tag: text-generation
tags:
  - gguf
  - quantized
---

# Ling-3.0-tiny GGUF, 3 GB

The best measured ~3 GB GGUF of [Ling-3.0-tiny](https://huggingface.co/inclusionAI/Ling-3.0-tiny). On four held-out text sets it stays 21% to 46% closer to the original model (KL to BF16) than bloomer010's IQ3_XXS and 21% to 51% closer than bartowski's Q2_K_L, the leading files at this size. It is also the smallest of the three, runs at the same speed and loses nothing on HellaSwag.

Two things get it there: bits placed where a measured sensitivity map says the model needs them, and a refiner that re-chooses the expert weights' low-bit codes so their outputs stay closer to the original, at no cost in size or speed. We publish the measurements and the code.

![Ling-3.0-tiny, 3.12 GB: 24 layers, 18 Kimi Delta Attention and 6 gated MLA. Routed experts at IQ3_XXS with refined codes make up 86% of the weights.](figures/banner-light.png)

Which file to pick: if about 3 GB is what fits, this one. If you have room for a 4-bit file (Q4_K_M, about 4.8 GB), that stays closer to the original still.

## How it compares

| KL vs BF16 (lower is closer) | Ours | bloomer010 IQ3_XXS | bartowski Q2_K_L |
|---|---|---|---|
| Size | 3.12 GB | 3.13 GB | 3.24 GB |
| Chinese | 0.33 | 0.41 | 0.41 |
| Code | 0.15 | 0.25 | 0.22 |
| Tool calls | 0.46 | 0.58 | 0.61 |
| Long books (4K context) | 0.12 | 0.23 | 0.25 |
| Top-1 agreement with BF16 | 75–89% | 72–86% | 72–87% |

![Mean KL per set for the three files, and our file minus each comparator with paired 95% intervals. Ours is lower on every set: Chinese 0.33 vs 0.41 and 0.41, code 0.15 vs 0.25 and 0.22, tool calls 0.46 vs 0.58 and 0.61, long books 0.12 vs 0.23 and 0.25; every interval is below zero.](figures/release-gate-light.png)

None of the three files was tuned on these texts. We fixed the test and its pass rule before scoring; every difference is clear of zero. Details: [GitHub](https://github.com/me-daniel/ling-3.0-tiny-quant).

## How it was made

1. Bits where they matter: we quantized one part of the model at a time and measured the damage. The routed experts tolerate low bits best, per GB saved; attention and the shared expert do not. So the experts stay at IQ3_XXS and the saved space goes to attention.
2. Refined expert codes: our refiner re-chooses each low-bit code so the experts' outputs stay closer to the original, without changing size or tensor types. On its own it lowered KL by 12–13%.

![KL from bloomer010's IQ3_XXS to our mix to our mix with refined experts, on two selection sets: wiki 0.50, 0.40, 0.35; held-out 0.34, 0.23, 0.20.](figures/ladder-light.png)

![Added KL when one part goes to 2 bits, against the MB it saves. The routed experts save the most space for the least damage; MLA and KDA attention and the shared expert cost the most per MB.](figures/sensitivity-light.png)

## Run it

Tested with llama.cpp on a Mac (Metal). Recommended settings from the base model: temperature 1.0, top-p 0.95, top-k 20.

```sh
llama-cli    -m Ling-3.0-tiny-IQ3_XXS-refined.gguf -ngl 99 --jinja --temp 1.0 --top-p 0.95 --top-k 20
llama-server -m Ling-3.0-tiny-IQ3_XXS-refined.gguf -ngl 99 --jinja --temp 1.0 --top-p 0.95 --top-k 20 --port 8080
```

## Speed and tasks

- Speed on an M5 Pro (Metal): about 81–87 tokens/s generation for all three files and for our unrefined mix; the differences are within run-to-run noise. Refinement does not cost speed.
- HellaSwag (1,000 tasks): ours 55.9%, bloomer010 55.1%, bartowski 54.5%. No task loss from the smaller, closer file.
- Repetition: a first loop check went cleanly, with no loops in any of our test chats. Longer-generation checks will follow.

## Limits

- KL measures how close the file stays to the original; HellaSwag differences of a point or two are within its uncertainty.
- Tested on one Mac with one llama.cpp version.

## License

A quantization of Ling-3.0-tiny by inclusionAI, released under the MIT License, which also covers this file.
