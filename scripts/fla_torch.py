"""Pure-PyTorch stand-ins for the flash-linear-attention (fla) pieces that
Ling-3.0-tiny's modeling_bailing_moe_v3.py imports. No Triton, runs on CPU/MPS.

Reference: fla-core 0.6.0 (PyPI wheel, read not installed):
  fla/ops/kda/naive.py             naive_recurrent_kda, naive_chunk_kda (copied below)
  fla/ops/kda/gate.py              naive_kda_gate, naive_kda_lowerbound_gate
  fla/ops/kda/chunk.py / fused_recurrent.py   argument handling (l2norm eps, gate, scale)
  fla/modules/norm/l2norm.py       x / sqrt(sum x^2 + 1e-6)
  fla/modules/norm/fused_norm_gate.py   FusedRMSNormGated: rmsnorm(x)*w*act(g), fp32 math
  fla/modules/causal_conv1d/ops.py      ShortConvolution(nn.Conv1d): depthwise causal conv + silu

Semantics checked against the 0.6.0 kernels:
  * use_gate_in_kernel=True: g_raw -> g_raw + dt_bias; if lower_bound is not None
    gate = lower_bound * sigmoid(exp(A_log) * g), else gate = -exp(A_log) * softplus(g).
    Both chunk_kda and fused_recurrent_kda take the lower-bound branch whenever
    lower_bound is set; safe_gate only changes intra-chunk numerics, not the math.
  * use_qk_l2norm_in_kernel=True: q,k l2-normalised over K with eps 1e-6 inside the sqrt.
  * beta is already post-sigmoid (use_beta_sigmoid_in_kernel defaults False).
  * scale defaults to K**-0.5.

install() puts these into sys.modules under the fla.* names the modeling file uses.
"""
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

FLA_REF_VERSION = "fla-core 0.6.0"


# ---------------- copied from fla/ops/kda/naive.py (fla-core 0.6.0) ----------------
def naive_recurrent_kda(q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False):
    dtype = v.dtype
    B, T, H, K, HV, V = *q.shape, v.shape[2], v.shape[-1]
    G = HV // H
    if scale is None:
        scale = K ** -0.5
    q, k, v, g, beta = map(lambda x: x.to(torch.float), [q, k, v, g, beta])
    q = q.repeat_interleave(G, dim=2) * scale
    k = k.repeat_interleave(G, dim=2)
    S = k.new_zeros(B, HV, K, V).to(q)
    if initial_state is not None:
        S += initial_state
    o = torch.zeros_like(v)
    for i in range(0, T):
        q_i, k_i, v_i, g_i, b_i = q[:, i], k[:, i], v[:, i], g[:, i], beta[:, i]
        S = S * g_i[..., None].exp()
        S = S + torch.einsum('b h k, b h v -> b h k v', b_i[..., None] * k_i, v_i - (k_i[..., None] * S).sum(-2))
        o[:, i] = torch.einsum('b h k, b h k v -> b h v', q_i, S)
    if not output_final_state:
        S = None
    return o.to(dtype), S


def naive_chunk_kda(q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False, chunk_size=64):
    dtype = v.dtype
    B, T, H, K, HV, V = *q.shape, v.shape[2], v.shape[-1]
    G = HV // H
    BT = chunk_size
    NT = T // BT
    if scale is None:
        scale = K ** -0.5
    assert T % BT == 0
    q, k = [rearrange(x, 'b (n c) h ... -> b h n c ...', c=BT).to(torch.float) for x in [q, k]]
    v, g, beta = [rearrange(x, 'b (n c) h ... -> b h n c ...', c=BT).to(torch.float) for x in [v, g, beta]]
    q = q.repeat_interleave(G, dim=1) * scale
    k = k.repeat_interleave(G, dim=1)
    g = g.cumsum(-2)
    mask = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=0)
    A = torch.zeros(*g.shape[:-1], BT, dtype=torch.float, device=q.device)
    for i in range(BT):
        k_i = k[..., i, :]
        g_i = g[..., i:i + 1, :]
        A[..., i] = torch.einsum('... c d, ... d -> ... c', k * (g - g_i).exp(), k_i)
    A = A * beta[..., None]
    A = -A.masked_fill(mask, 0)
    for i in range(1, BT):
        A[..., i, :i] = A[..., i, :i].clone() + (A[..., i, :, None].clone() * A[..., :, :i].clone()).sum(-2)
    A = (A + torch.eye(BT, dtype=torch.float, device=q.device)) * beta[..., None, :]
    w = A @ (g.exp() * k)
    u = A @ v
    S = k.new_zeros(B, HV, K, V).to(q)
    if initial_state is not None:
        S += initial_state
    o = torch.zeros_like(v)
    mask = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=1)
    for i in range(0, NT):
        q_i, k_i, u_i, g_i, w_i = q[:, :, i], k[:, :, i], u[:, :, i], g[:, :, i], w[:, :, i]
        Aqk = torch.zeros(B, HV, BT, BT, dtype=torch.float, device=q.device)
        for j in range(BT):
            k_j = k[:, :, i, j]
            g_j = g[:, :, i, j:j + 1, :]
            Aqk[..., j] = torch.einsum('... c d, ... d -> ... c', q_i * (g_i - g_j).exp(), k_j)
        Aqk = Aqk.masked_fill(mask, 0)
        v_i = u_i - w_i @ S
        o[:, :, i] = (q_i * g_i.exp()) @ S + Aqk @ v_i
        S = S * rearrange(g_i[:, :, -1].exp(), 'b h k -> b h k 1')
        S += rearrange((g_i[:, :, -1:] - g_i).exp() * k_i, 'b h c k -> b h k c') @ v_i
    if not output_final_state:
        S = None
    return rearrange(o, 'b h n c d -> b (n c) h d').to(dtype), S
# ------------------------------------------------------------------------------------


def _f(x):  # at least float32; keeps float64 for the tests
    return x.to(torch.promote_types(x.dtype, torch.float32))


def l2norm(x, eps=1e-6):  # fla/modules/norm/l2norm.py
    x = _f(x)
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + eps)


def kda_gate(g, A_log, dt_bias, lower_bound):  # fla/ops/kda/gate.py naive_* refs
    H = g.shape[-2]
    g = _f(g)
    if dt_bias is not None:
        g = g + dt_bias.to(g.dtype).view(H, -1)
    if lower_bound is not None:
        if A_log is not None:
            g = A_log.to(g.dtype).view(H, 1).exp() * g
        return lower_bound * torch.sigmoid(g)
    return -A_log.to(g.dtype).view(H, 1).exp() * F.softplus(g)


def _prep(q, k, g, use_qk_l2norm_in_kernel, use_gate_in_kernel, A_log, dt_bias, lower_bound):
    if use_qk_l2norm_in_kernel:
        q, k = l2norm(q), l2norm(k)
    if use_gate_in_kernel:
        g = kda_gate(g, A_log, dt_bias, lower_bound)
    return q, k, g


def _no_varlen(cu_seqlens, q):
    if cu_seqlens is not None and not (len(cu_seqlens) == 2 and int(cu_seqlens[-1]) == q.shape[1]):
        raise NotImplementedError("varlen not supported in the torch stub")


def fused_recurrent_kda(q, k, v, g, beta, A_log=None, dt_bias=None, scale=None, initial_state=None,
                        output_final_state=False, use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
                        use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False, lower_bound=None,
                        cu_seqlens=None, **kwargs):
    _no_varlen(cu_seqlens, q)
    q, k, g = _prep(q, k, g, use_qk_l2norm_in_kernel, use_gate_in_kernel, A_log, dt_bias, lower_bound)
    if use_beta_sigmoid_in_kernel:
        beta = beta.float().sigmoid() * (2 if allow_neg_eigval else 1)
    return naive_recurrent_kda(q, k, v, g, beta, scale, initial_state, output_final_state)


def chunk_kda(q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False,
              use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False, use_beta_sigmoid_in_kernel=False,
              allow_neg_eigval=False, safe_gate=False, lower_bound=None, cu_seqlens=None, **kwargs):
    _no_varlen(cu_seqlens, q)
    A_log, dt_bias = kwargs.get("A_log"), kwargs.get("dt_bias")
    chunk_size = kwargs.get("chunk_size", 64)
    if safe_gate and use_gate_in_kernel:
        assert lower_bound is not None and -5 <= lower_bound < 0
    q, k, g = _prep(q, k, g, use_qk_l2norm_in_kernel, use_gate_in_kernel, A_log, dt_bias, lower_bound)
    if use_beta_sigmoid_in_kernel:
        beta = beta.float().sigmoid() * (2 if allow_neg_eigval else 1)
    T = q.shape[1]
    pad = (-T) % chunk_size
    if pad:  # zero k/v/beta and zero log-gate: padded steps leave the state untouched
        q, k, v, g = (F.pad(x, (0, 0, 0, 0, 0, pad)) for x in (q, k, v, g))
        beta = F.pad(beta, (0, 0, 0, pad))
    o, S = naive_chunk_kda(q, k, v, g, beta, scale, initial_state, output_final_state, chunk_size)
    return o[:, :T], S


class ShortConvolution(nn.Conv1d):
    def __init__(self, hidden_size, kernel_size, bias=False, activation='silu', backend='triton',
                 device=None, dtype=None, **kwargs):
        super().__init__(hidden_size, hidden_size, kernel_size, padding=kernel_size - 1,
                         groups=hidden_size, bias=bias, device=device, dtype=dtype)
        self.hidden_size = hidden_size
        self.activation = activation

    def forward(self, x, residual=None, mask=None, cache=None, output_final_state=False,
                cu_seqlens=None, **kwargs):
        _no_varlen(cu_seqlens, x)
        if cache is not None:
            raise NotImplementedError("decode cache not supported in the torch stub")
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        T, W = x.shape[1], self.kernel_size[0]
        xt = x.transpose(1, 2)
        y = F.conv1d(xt, self.weight, self.bias, padding=W - 1, groups=self.hidden_size)[..., :T]
        if self.activation in ('silu', 'swish'):
            y = F.silu(y)
        y = y.transpose(1, 2)
        if residual is not None:
            y = y + residual
        final = F.pad(xt, (max(W - T, 0), 0))[..., -W:] if output_final_state else None
        return y, final


class FusedRMSNormGated(nn.Module):
    def __init__(self, hidden_size, elementwise_affine=True, eps=1e-5, activation="swish", device=None, dtype=None):
        super().__init__()
        self.hidden_size, self.eps, self.activation = hidden_size, eps, activation
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype)) if elementwise_affine else None
        self.register_parameter("bias", None)

    def forward(self, x, g, residual=None, prenorm=False, residual_in_fp32=False):
        dtype = x.dtype
        xf = x.float()
        if residual is not None:
            xf = xf + residual.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.weight is not None:
            y = y * self.weight.float()
        gf = g.float()
        y = y * (gf * torch.sigmoid(gf) if self.activation in ("swish", "silu") else torch.sigmoid(gf))
        y = y.to(dtype)
        return (y, xf.to(dtype)) if prenorm else y


def _unsupported(*a, **k):
    raise NotImplementedError("simple_gla / varlen helpers are not used by Ling-3.0-tiny inference")


def tensor_cache(fn):
    return fn


def install():
    mods = {}
    for name in ["fla", "fla.ops", "fla.ops.simple_gla", "fla.ops.simple_gla.fused_recurrent",
                 "fla.ops.simple_gla.chunk", "fla.modules", "fla.ops.kda", "fla.ops.utils",
                 "fla.ops.utils.index", "fla.utils"]:
        m = types.ModuleType(name)
        m.__path__ = []  # mark as package
        mods[name] = m
    mods["fla"].__version__ = "0.6.0-torchstub"
    mods["fla.ops.simple_gla.fused_recurrent"].fused_recurrent_simple_gla = _unsupported
    mods["fla.ops.simple_gla.chunk"].chunk_simple_gla = _unsupported
    mods["fla.modules"].FusedRMSNormGated = FusedRMSNormGated
    mods["fla.modules"].ShortConvolution = ShortConvolution
    mods["fla.ops.kda"].chunk_kda = chunk_kda
    mods["fla.ops.kda"].fused_recurrent_kda = fused_recurrent_kda
    mods["fla.ops.utils.index"].prepare_cu_seqlens_from_mask = _unsupported
    mods["fla.ops.utils.index"].prepare_lens_from_mask = _unsupported
    mods["fla.utils"].tensor_cache = tensor_cache
    sys.modules.update(mods)
