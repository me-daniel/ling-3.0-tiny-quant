# Q4: imatrix coverage

imatrix `models/imatrix-wiki-train-c512-n400.gguf`; model `models/ling-3.0-tiny-BF16.gguf`.

- Quantizable weight matrices: 311; with imatrix data: 309.
- Missing: `token_embd.weight`, `output.weight`.

## Routed experts (from `ffn_gate_exps` counts; tokens routed per expert)

- MoE layers: 23, experts per layer: 128.
- Experts with zero tokens: 63 of 2944 (per layer max 10).
- Tokens per expert: min 0, median 6336, max 147791.
- Least-hit expert share vs uniform (1/128): min over layers 0.000×.
- Experts under 10% of the uniform share: 787.

| layer | zero | min | median | max | max/min |
|---|---|---|---|---|---|
| 1 | 0 | 61 | 9074 | 105860 | 1735.4 |
| 2 | 1 | 0 | 8520 | 62997 | inf |
| 3 | 0 | 18 | 10098 | 83928 | 4662.7 |
| 4 | 0 | 32 | 7874 | 113853 | 3557.9 |
| 5 | 0 | 55 | 6516 | 83004 | 1509.2 |
| 6 | 0 | 16 | 6406 | 100755 | 6297.2 |
| 7 | 0 | 6 | 6300 | 105962 | 17660.3 |
| 8 | 0 | 4 | 7363 | 97045 | 24261.2 |
| 9 | 0 | 1 | 4806 | 111396 | 111396.0 |
| 10 | 3 | 0 | 4302 | 147791 | inf |
| 11 | 1 | 0 | 3742 | 141839 | inf |
| 12 | 1 | 0 | 5364 | 137306 | inf |
| 13 | 1 | 0 | 5293 | 123200 | inf |
| 14 | 1 | 0 | 3680 | 100727 | inf |
| 15 | 5 | 0 | 7200 | 116748 | inf |
| 16 | 2 | 0 | 6542 | 96416 | inf |
| 17 | 4 | 0 | 7106 | 90127 | inf |
| 18 | 6 | 0 | 5860 | 143379 | inf |
| 19 | 7 | 0 | 7458 | 95633 | inf |
| 20 | 7 | 0 | 5502 | 116288 | inf |
| 21 | 6 | 0 | 4904 | 110435 | inf |
| 22 | 8 | 0 | 4417 | 106269 | inf |
| 23 | 10 | 0 | 4764 | 115117 | inf |
