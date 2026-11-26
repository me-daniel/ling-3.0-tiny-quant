"""Compare HF reference logits (hf_ref.py) with GGUF logits (gguf_logits.py), per position.

  .venv/bin/python scripts/parity.py runs/parity/hf.npz runs/parity/gguf.npz

KL(HF || GGUF) in nats from fp64 log-softmax, top-1 agreement, max |logit diff|. If both files hold
hidden states, prints per-layer relative L2 error of the residual stream (layers 0..22).
"""
import json
import sys

import numpy as np


def log_softmax(x):
    x = x.astype(np.float64)
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


hf, gg = np.load(sys.argv[1]), np.load(sys.argv[2])
print("hf:", hf["meta"], "\ngguf:", gg["meta"])
names = sorted({k.split("/")[0] for k in gg.files if "/" in k})
print(f"{'prompt':8s} {'T':>5s} {'npos':>5s} {'meanKL':>9s} {'p99KL':>9s} {'maxKL':>9s} {'top1':>7s} {'maxAbs':>8s} {'meanAbs':>8s}")
rows = {}
for n in names:
    assert (hf[f"{n}/ids"] == gg[f"{n}/ids"]).all() and (hf[f"{n}/pos"] == gg[f"{n}/pos"]).all()
    a, b = hf[f"{n}/logits"], gg[f"{n}/logits"]
    la, lb = log_softmax(a), log_softmax(b)
    kl = (np.exp(la) * (la - lb)).sum(-1)
    top1 = (a.argmax(-1) == b.argmax(-1)).mean()
    d = np.abs(a - b)
    T = len(hf[f"{n}/ids"])
    rows[n] = dict(T=T, npos=len(a), mean_kl=kl.mean(), p99_kl=np.quantile(kl, .99), max_kl=kl.max(),
                   argmax_kl_pos=int(hf[f"{n}/pos"][kl.argmax()]), top1=top1, max_abs=d.max(), mean_abs=d.mean())
    r = rows[n]
    print(f"{n:8s} {T:5d} {r['npos']:5d} {r['mean_kl']:9.2e} {r['p99_kl']:9.2e} {r['max_kl']:9.2e} "
          f"{top1:7.4f} {r['max_abs']:8.4f} {r['mean_abs']:8.4f}  (maxKL at pos {r['argmax_kl_pos']})")
    if f"{n}/hidden" in hf.files and f"{n}/hidden" in gg.files:
        # HF hidden_states = [emb, out of layer 0..22, norm(out of layer 23)]; llama.cpp l_out-0..23.
        # So l_out-i == HF[i+1] for i <= 22; the last layer has no un-normed HF counterpart.
        H, G = hf[f"{n}/hidden"][1:-1], gg[f"{n}/hidden"][:-1]
        rel = np.linalg.norm(H - G, axis=-1) / np.linalg.norm(H, axis=-1)
        print("   per-layer rel L2 (mean / max over positions):",
              " ".join(f"L{i}:{rel[i].mean():.1e}/{rel[i].max():.1e}" for i in range(len(G))))
