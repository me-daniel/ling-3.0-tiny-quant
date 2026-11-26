"""Independent HF reference forward for Ling-3.0-tiny on CPU, no Triton.

fla is replaced by pure-PyTorch stubs (scripts/fla_torch.py, from fla-core 0.6.0 naive refs).
Precision: weights stay bf16 in RAM (~16 GB); every nn.Linear upcasts its weight to fp32 per call
and all activations, norms, router, attention and KDA run in fp32. Small non-Linear params
(embeddings, norms, conv, A_log, dt_bias, router) are converted to fp32 once.

  .venv/bin/python scripts/hf_ref.py --prompts data/parity_prompts.jsonl --out results/parity/hf.npz

Prompts file: JSONL, one {"name", "text" or "ids", optional "max_tokens", "stride", "save_last"} per line.
Saved per prompt <name>: ids, pos (saved positions), logits [len(pos), vocab] fp32,
hidden [n_layers+1, len(pos), hidden] fp32 (embeddings input of each layer ..., final-normed).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import fla_torch  # noqa: E402

fla_torch.install()

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def fp32_linear(self, x):
    w = self.weight.float()
    b = self.bias.float() if self.bias is not None else None
    return F.linear(x.float(), w, b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Ling-3.0-tiny")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kda-mode", choices=["auto", "fused_recurrent"], default="auto",
                    help="auto = modeling file's rule (recurrent if T<=64 else chunk); fused_recurrent forces it")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(a.model, trust_remote_code=True, dtype="auto",  # checkpoint dtype (bf16 for Ling)
                                                 attn_implementation="eager", low_cpu_mem_usage=True)
    model.eval()
    nn.Linear.forward = fp32_linear
    linear_w = {id(m.weight) for m in model.modules() if isinstance(m, nn.Linear)}
    for p in model.parameters():
        if id(p) not in linear_w:
            p.data = p.data.float()
    for m in model.modules():
        for n, b in m.named_buffers(recurse=False):
            if b.is_floating_point():
                setattr(m, n, b.float())
    if a.kda_mode != "auto":  # forward() uses self.mode for T>64
        for m in model.modules():
            if type(m).__name__ == "BailingMoeV3KimiDeltaAttention":
                m.mode = a.kda_mode
    print(f"loaded in {time.time() - t0:.0f}s; transformers model class {type(model).__name__}", flush=True)

    out = {"meta": json.dumps({"fla_ref": fla_torch.FLA_REF_VERSION, "kda_mode": a.kda_mode,
                               "precision": "bf16 weights, fp32 compute (per-call upcast)"})}
    for line in Path(a.prompts).read_text().splitlines():
        if not line.strip():
            continue
        p = json.loads(line)
        ids = p["ids"] if "ids" in p else tok(p["text"])["input_ids"]
        if p.get("max_tokens"):
            ids = ids[: p["max_tokens"]]
        T = len(ids)
        stride = p.get("stride", 1)
        if p.get("save_last"):  # same-suffix design: only the final N positions are compared
            pos = np.arange(T - p["save_last"], T)
        else:
            pos = np.unique(np.r_[np.arange(0, T, stride), np.arange(max(0, T - 64), T)])
        t0 = time.time()
        with torch.no_grad():
            r = model(torch.tensor([ids]), use_cache=False, output_hidden_states=True)
        logits = r.logits[0].float()
        hs = torch.stack([h[0].float() for h in r.hidden_states])
        n = p["name"]
        out[f"{n}/ids"] = np.array(ids, dtype=np.int32)
        out[f"{n}/pos"] = pos.astype(np.int32)
        out[f"{n}/logits"] = logits[pos].numpy()
        out[f"{n}/hidden"] = hs[:, pos].numpy()
        print(f"{n}: T={T} saved {len(pos)} positions, {time.time() - t0:.1f}s; "
              f"first ids {ids[:8]}; argmax[-1]={int(logits[-1].argmax())}", flush=True)
        del r, logits, hs
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
