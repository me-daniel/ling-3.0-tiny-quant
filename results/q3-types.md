# Q3: requested vs actual tensor types (llama-quantize --dry-run)

Source `models/ling-3.0-tiny-BF16.gguf`, llama.cpp `c661033`. Base mix Q8_0; one group overridden at a time.

## Per-group overrides

| group | requested | tensors | actual types in group | fallbacks (whole model) |
|---|---|---|---|---|
| kda_qkv | iq2_xxs | 54 | iq2_xxs×54 | 0 |
| kda_qkv | iq3_xxs | 54 | iq3_xxs×54 | 0 |
| kda_qkv | q2_k | 54 | q2_k×54 | 0 |
| kda_qkv | q4_k | 54 | q4_k×54 | 0 |
| kda_qkv | iq4_xs | 54 | iq4_xs×54 | 0 |
| kda_gates | iq2_xxs | 54 | iq2_xxs×54 | 0 |
| kda_gates | iq3_xxs | 54 | iq3_xxs×54 | 0 |
| kda_gates | q2_k | 54 | q2_k×54 | 0 |
| kda_gates | q4_k | 54 | q4_k×54 | 0 |
| kda_gates | iq4_xs | 54 | iq4_xs×54 | 0 |
| kda_out | iq2_xxs | 18 | iq2_xxs×18 | 0 |
| kda_out | iq3_xxs | 18 | iq3_xxs×18 | 0 |
| kda_out | q2_k | 18 | q2_k×18 | 0 |
| kda_out | q4_k | 18 | q4_k×18 | 0 |
| kda_out | iq4_xs | 18 | iq4_xs×18 | 0 |
| mla_proj | iq2_xxs | 24 | iq2_xxs×24 | 0 |
| mla_proj | iq3_xxs | 24 | iq3_xxs×24 | 0 |
| mla_proj | q2_k | 24 | q2_k×24 | 0 |
| mla_proj | q4_k | 24 | q4_k×24 | 0 |
| mla_proj | iq4_xs | 24 | iq4_xs×24 | 0 |
| mla_k_b | iq2_xxs | 6 | iq4_nl×6 **≠** | 6 |
| mla_k_b | iq3_xxs | 6 | iq4_nl×6 **≠** | 6 |
| mla_k_b | q2_k | 6 | q4_0×6 **≠** | 6 |
| mla_k_b | q4_k | 6 | q5_0×6 **≠** | 6 |
| mla_k_b | iq4_xs | 6 | iq4_nl×6 **≠** | 6 |
| mla_gate | iq2_xxs | 6 | iq2_xxs×6 | 0 |
| mla_gate | iq3_xxs | 6 | iq3_xxs×6 | 0 |
| mla_gate | q2_k | 6 | q2_k×6 | 0 |
| mla_gate | q4_k | 6 | q4_k×6 | 0 |
| mla_gate | iq4_xs | 6 | iq4_xs×6 | 0 |
| mla_out | iq2_xxs | 6 | iq2_xxs×6 | 0 |
| mla_out | iq3_xxs | 6 | iq3_xxs×6 | 0 |
| mla_out | q2_k | 6 | q2_k×6 | 0 |
| mla_out | q4_k | 6 | q4_k×6 | 0 |
| mla_out | iq4_xs | 6 | iq4_xs×6 | 0 |
| exp_gate_up | iq2_xxs | 46 | iq2_xxs×46 | 0 |
| exp_gate_up | iq3_xxs | 46 | iq3_xxs×46 | 0 |
| exp_gate_up | q2_k | 46 | q2_k×46 | 0 |
| exp_gate_up | q4_k | 46 | q4_k×46 | 0 |
| exp_gate_up | iq4_xs | 46 | iq4_xs×46 | 0 |
| exp_down | iq2_xxs | 23 | iq2_xxs×23 | 0 |
| exp_down | iq3_xxs | 23 | iq3_xxs×23 | 0 |
| exp_down | q2_k | 23 | q2_k×23 | 0 |
| exp_down | q4_k | 23 | q4_k×23 | 0 |
| exp_down | iq4_xs | 23 | iq4_xs×23 | 0 |
| shexp | iq2_xxs | 69 | iq2_xxs×69 | 0 |
| shexp | iq3_xxs | 69 | iq3_xxs×69 | 0 |
| shexp | q2_k | 69 | q2_k×69 | 0 |
| shexp | q4_k | 69 | q4_k×69 | 0 |
| shexp | iq4_xs | 69 | iq4_xs×69 | 0 |
| dense0 | iq2_xxs | 3 | iq2_xxs×3 | 0 |
| dense0 | iq3_xxs | 3 | iq3_xxs×3 | 0 |
| dense0 | q2_k | 3 | q2_k×3 | 0 |
| dense0 | q4_k | 3 | q4_k×3 | 0 |
| dense0 | iq4_xs | 3 | iq4_xs×3 | 0 |
| embd_out | iq2_xxs | 2 | iq2_xxs×2 | 0 |
| embd_out | iq3_xxs | 2 | iq3_xxs×2 | 0 |
| embd_out | q2_k | 2 | q2_k×2 | 0 |
| embd_out | q4_k | 2 | q4_k×2 | 0 |
| embd_out | iq4_xs | 2 | iq4_xs×2 | 0 |

## Stock mixes (no overrides)

| mix | BPW | fallbacks | per-group actual types |
|---|---|---|---|
| Q4_K_M | 4.88 | 6 | kda_qkv: q4_k×45, q6_k×9; kda_gates: q4_k×54; kda_out: q4_k×18; mla_proj: q4_k×24; mla_k_b: q5_0×6; mla_gate: q4_k×6; mla_out: q4_k×6; exp_gate_up: q4_k×46; exp_down: q4_k×12, q6_k×11; shexp: q4_k×58, q6_k×11; dense0: q4_k×2, q6_k×1; embd_out: q6_k×1, q4_k×1; other: f32×215 |
| Q2_K | 3.02 | 6 | kda_qkv: q2_k×36, q3_k×18; kda_gates: q2_k×54; kda_out: q3_k×18; mla_proj: q2_k×24; mla_k_b: q4_0×6; mla_gate: q2_k×6; mla_out: q3_k×6; exp_gate_up: q2_k×46; exp_down: q3_k×23; shexp: q2_k×46, q3_k×23; dense0: q2_k×2, q3_k×1; embd_out: q6_k×1, q2_k×1; other: f32×215 |
| IQ3_XXS | 3.31 | 6 | kda_qkv: iq2_s×36, iq3_s×18; kda_gates: iq3_xxs×54; kda_out: iq3_s×18; mla_proj: iq3_xxs×24; mla_k_b: iq4_nl×6; mla_gate: iq3_xxs×6; mla_out: iq3_s×6; exp_gate_up: iq3_xxs×46; exp_down: q3_k×21, q4_k×2; shexp: iq3_xxs×46, q3_k×21, q4_k×2; dense0: iq3_xxs×2, q4_k×1; embd_out: q5_k×1, iq3_s×1; other: f32×215 |
| IQ2_XXS | 2.23 | 6 | kda_qkv: iq2_xxs×36, q4_k×18; kda_gates: iq2_xxs×54; kda_out: iq2_xxs×18; mla_proj: iq2_xxs×24; mla_k_b: iq4_nl×6; mla_gate: iq2_xxs×6; mla_out: iq2_xxs×6; exp_gate_up: iq2_xxs×46; exp_down: iq2_xxs×22, q2_k×1; shexp: iq2_xxs×68, q2_k×1; dense0: iq2_xxs×2, q2_k×1; embd_out: q5_k×1, q2_k×1; other: f32×215 |

## Tensors in no group (left at the mix's choice)

`blk.N.attn_kv_a_norm.weight (f32)`×6, `blk.N.attn_norm.weight (f32)`×24, `blk.N.attn_q_a_norm.weight (f32)`×6, `blk.N.exp_probs_b.bias (f32)`×23, `blk.N.ffn_gate_inp.weight (f32)`×23, `blk.N.ffn_norm.weight (f32)`×24, `blk.N.ssm_a (f32)`×18, `blk.N.ssm_conv1d_k.weight (f32)`×18, `blk.N.ssm_conv1d_q.weight (f32)`×18, `blk.N.ssm_conv1d_v.weight (f32)`×18, `blk.N.ssm_dt.bias (f32)`×18, `blk.N.ssm_norm.weight (f32)`×18, `output_norm.weight (f32)`×1
