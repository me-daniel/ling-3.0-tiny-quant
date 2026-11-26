"""Isolation tests for scripts/fla_torch.py on tiny random inputs. The naive KDA refs cast to float32 internally, so KDA tolerances are fp32-level.
Run: .venv/bin/python scripts/test_fla_torch.py"""
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import fla_torch as ft

torch.manual_seed(0)
B, H, K, V = 1, 3, 16, 16


def rnd(T, dtype=torch.float64):
    q, k, v = (torch.randn(B, T, H, K, dtype=dtype) for _ in range(3))
    graw = torch.randn(B, T, H, K, dtype=dtype)
    beta = torch.rand(B, T, H, dtype=dtype)
    A_log = torch.log(torch.empty(H, dtype=torch.float32).uniform_(1, 16))
    dt_bias = torch.randn(H * K, dtype=torch.float32)
    return q, k, v, graw, beta, A_log, dt_bias


def explicit_delta(q, k, v, graw, beta, A_log, dt_bias, lb):
    """Independent loop written from the KDA paper's update rule, not from naive.py."""
    T = q.shape[1]
    out = torch.zeros(B, T, H, V, dtype=torch.float64)
    for h in range(H):
        S = torch.zeros(K, V, dtype=torch.float64)
        for t in range(T):
            qt = q[0, t, h].double(); kt = k[0, t, h].double()
            qt = qt / math.sqrt((qt @ qt).item() + 1e-6); kt = kt / math.sqrt((kt @ kt).item() + 1e-6)
            a = (graw[0, t, h].double() + dt_bias.view(H, K)[h].double()) * A_log[h].double().exp()
            decay = torch.exp(lb * torch.sigmoid(a))          # per-k forget in (e^lb, 1)
            S = decay[:, None] * S
            S = S + beta[0, t, h].double() * torch.outer(kt, v[0, t, h].double() - kt @ S)
            out[0, t, h] = (qt * K ** -0.5) @ S
    return out


def check(name, a, b, tol):
    err = (a.double() - b.double()).abs().max().item()
    print(f"{name:50s} max|diff| = {err:.3e}  {'OK' if err < tol else 'FAIL'}")
    assert err < tol, name


kw = dict(use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, lower_bound=-5.0, output_final_state=True)
for T in (5, 64, 150):
    q, k, v, graw, beta, A_log, dt_bias = rnd(T)
    o_r, S_r = ft.fused_recurrent_kda(q, k, v, graw, beta, A_log=A_log, dt_bias=dt_bias, **kw)
    o_c, S_c = ft.chunk_kda(q, k, v, graw, beta, A_log=A_log, dt_bias=dt_bias, safe_gate=True, **kw)
    check(f"T={T} recurrent vs chunk output", o_r, o_c, 2e-5)
    check(f"T={T} recurrent vs chunk final state", S_r, S_c, 2e-5)
    check(f"T={T} recurrent vs explicit loop", o_r, explicit_delta(q, k, v, graw, beta, A_log, dt_bias, -5.0), 2e-5)

# initial state carried across a split must equal one long run
q, k, v, graw, beta, A_log, dt_bias = rnd(130)
o_full, S_full = ft.chunk_kda(q, k, v, graw, beta, A_log=A_log, dt_bias=dt_bias, **kw)
o1, S1 = ft.chunk_kda(q[:, :70], k[:, :70], v[:, :70], graw[:, :70], beta[:, :70], A_log=A_log, dt_bias=dt_bias, **kw)
o2, S2 = ft.fused_recurrent_kda(q[:, 70:], k[:, 70:], v[:, 70:], graw[:, 70:], beta[:, 70:], A_log=A_log,
                                dt_bias=dt_bias, initial_state=S1, **kw)
check("split 70+60 (chunk then recurrent) vs one chunk run", torch.cat([o1, o2], 1), o_full, 2e-5)

# float32 inputs (what the model run uses): chunk vs recurrent at T=2000-ish scale is checked in fp32
q, k, v, graw, beta, A_log, dt_bias = rnd(300, torch.float32)
o_r, _ = ft.fused_recurrent_kda(q, k, v, graw, beta, A_log=A_log, dt_bias=dt_bias, **kw)
o_c, _ = ft.chunk_kda(q, k, v, graw, beta, A_log=A_log, dt_bias=dt_bias, **kw)
check("fp32 T=300 recurrent vs chunk", o_r, o_c, 1e-4)

# ShortConvolution vs explicit causal loop
D, T, W = 8, 11, 4
conv = ft.ShortConvolution(D, W, activation='silu').double()
x = torch.randn(1, T, D, dtype=torch.float64)
y, fs = conv(x, output_final_state=True)
ref = torch.zeros_like(x)
for t in range(T):
    for j in range(W):  # weight[..., W-1] multiplies the current token
        if t - (W - 1 - j) >= 0:
            ref[0, t] += conv.weight[:, 0, j] * x[0, t - (W - 1 - j)]
check("ShortConvolution vs explicit loop", y, F.silu(ref), 1e-12)
check("ShortConvolution final state = last W inputs", fs, x[0, -W:].T[None], 1e-12)

# FusedRMSNormGated (sigmoid) vs formula
n = ft.FusedRMSNormGated(K, eps=1e-6, activation='sigmoid').double()
n.weight.data = torch.randn(K, dtype=torch.float64)
x, g = torch.randn(2, 5, H, K, dtype=torch.float64), torch.randn(2, 5, H, K, dtype=torch.float64)
ref = x / torch.sqrt((x ** 2).mean(-1, keepdim=True) + 1e-6) * n.weight * torch.sigmoid(g)
check("FusedRMSNormGated(sigmoid) vs formula (fp32 math)", n(x, g), ref, 1e-5)
print("all stub tests passed")
