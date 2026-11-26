# Q7: single-group IQ2_XXS sensitivity on a Q8_0 base (Ling-3.0-tiny)

KLD vs BF16, 16 × 512 wikitext-2 test, Metal, default -ub, imatrix wiki-train 100 × 512. Base all-Q8_0: 8.408 GB, mean KLD 0.0075. S = ΔKLD / GB saved (higher = keep the bits).

| group at IQ2_XXS | GB saved | ΔKLD | ± (SE of group run) | S per GB | top-1 % | quantize s |
|---|---|---|---|---|---|---|
| mla_gate | 0.000 | 0.0190 | 0.0008 | 160.54 | 92.059 | 8 |
| mla_proj | 0.015 | 0.2081 | 0.0057 | 13.84 | 79.363 | 8 |
| mla_out | 0.015 | 0.1236 | 0.0035 | 8.14 | 84.289 | 9 |
| kda_out | 0.046 | 0.3201 | 0.0078 | 7.03 | 74.951 | 9 |
| shexp | 0.044 | 0.2177 | 0.0058 | 4.99 | 79.559 | 9 |
| dense0 | 0.017 | 0.0691 | 0.0023 | 4.04 | 87.230 | 9 |
| kda_qkv | 0.137 | 0.2227 | 0.0061 | 1.63 | 78.015 | 10 |
| kda_gates | 0.091 | 0.0827 | 0.0025 | 0.90 | 85.564 | 9 |
| exp_down | 1.863 | 0.3167 | 0.0093 | 0.17 | 74.926 | 40 |
| exp_gate_up | 3.726 | 0.4993 | 0.0141 | 0.13 | 69.534 | 51 |

| stock mix | GB | BPW | mean KLD | PPL | top-1 % | quantize s |
|---|---|---|---|---|---|---|
| q8_all | 8.408 | 8.52 | 0.007487 | 15.9772 | 96.250 | 7 |
| stock_IQ2_XXS | 2.208 | 2.23 | 1.491225 | 51.5350 | 48.456 | 109 |
| stock_IQ3_XXS | 3.129 | 3.16 | 0.462682 | 20.2596 | 69.853 | 174 |
| stock_Q4_K_M | 4.824 | 4.88 | 0.121997 | 17.2505 | 83.211 | 40 |
