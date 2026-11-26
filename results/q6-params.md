# Q6: parameter counts for the cost model

| model | text params (B) | routed experts (B) | experts share | active per token (B) | extra (B) |
|---|---|---|---|---|---|
| Ling-3.0-tiny | 7.89 | 6.95 | 88.0% | 1.14 | - |
| Ling-3.2-air | 124.41 | 120.80 | 97.1% | 5.30 | mtp 3.07, vision 0.43 |

Ratios Air / tiny: total text params 15.8×, active per token 4.7×.
