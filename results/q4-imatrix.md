# Q4: imatrix coverage

imatrix `models/imatrix-mixc-c512-n400.gguf`; model `models/ling-3.0-tiny-BF16.gguf`.

- Quantizable weight matrices: 311; with imatrix data: 309.
- Missing: `token_embd.weight`, `output.weight`.

## Routed experts (from `ffn_gate_exps` counts; tokens routed per expert)

- MoE layers: 23, experts per layer: 128.
- Experts with zero tokens: 0 of 2944 (per layer max 0).
- Tokens per expert: min 6, median 11299, max 82051.
- Least-hit expert share vs uniform (1/128): min over layers 0.000×.
- Experts under 10% of the uniform share: 193.

| layer | zero | min | median | max | max/min |
|---|---|---|---|---|---|
| 1 | 0 | 695 | 12098 | 39959 | 57.5 |
| 2 | 0 | 456 | 11758 | 41747 | 91.6 |
| 3 | 0 | 351 | 12464 | 49852 | 142.0 |
| 4 | 0 | 984 | 10954 | 55467 | 56.4 |
| 5 | 0 | 1163 | 11337 | 45212 | 38.9 |
| 6 | 0 | 1375 | 10478 | 43955 | 32.0 |
| 7 | 0 | 1015 | 10638 | 52098 | 51.3 |
| 8 | 0 | 908 | 11568 | 47869 | 52.7 |
| 9 | 0 | 992 | 10054 | 53965 | 54.4 |
| 10 | 0 | 441 | 11010 | 58663 | 133.0 |
| 11 | 0 | 392 | 10163 | 44332 | 113.1 |
| 12 | 0 | 523 | 10649 | 82051 | 156.9 |
| 13 | 0 | 449 | 11080 | 44966 | 100.1 |
| 14 | 0 | 319 | 9743 | 49320 | 154.6 |
| 15 | 0 | 278 | 11252 | 51839 | 186.5 |
| 16 | 0 | 261 | 11188 | 57052 | 218.6 |
| 17 | 0 | 233 | 11568 | 49538 | 212.6 |
| 18 | 0 | 154 | 12514 | 45749 | 297.1 |
| 19 | 0 | 58 | 12130 | 48646 | 838.7 |
| 20 | 0 | 6 | 10631 | 45544 | 7590.7 |
| 21 | 0 | 16 | 11378 | 40114 | 2507.1 |
| 22 | 0 | 10 | 12332 | 44428 | 4442.8 |
| 23 | 0 | 23 | 12582 | 39223 | 1705.3 |
