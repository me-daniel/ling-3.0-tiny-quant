"""Q5, clean version: KL(BF16 || quant) by token position on the same 4,096-token windows.

Both models see identical tokens; only position varies, so growth with position is not confounded by
chunking or text. Uses scripts/gguf_logits.py (libllama, Metal, default ubatch, KV f16) for every model.
usage: .venv/bin/python -I scripts/q5_positions.py make                      # writes runs/q5pos/ids.npz
       .venv/bin/python -I scripts/q5_positions.py run <name> <gguf>          # writes runs/q5pos/<name>.npz
       .venv/bin/python -I scripts/q5_positions.py report <name>...           # writes results/q5-positions.md
"""

import subprocess
import sys
from pathlib import Path

import numpy as np

D = Path("runs/q5pos")
T, N, STEP, BUCKET = 4096, 4, 8, 512  # 4 windows of 4,096 tokens, logits every 8th position, 512-token buckets


def make():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("models/Ling-3.0-tiny")
    ids = np.array(tok(open("data/wiki.test.raw").read())["input_ids"], dtype=np.int32)
    out = {}
    for w in range(N):
        start = w * (len(ids) // N)
        out[f"w{w}/ids"] = ids[start:start + T]
        out[f"w{w}/pos"] = np.arange(0, T, STEP, dtype=np.int32)
    D.mkdir(parents=True, exist_ok=True)
    np.savez(D / "ids.npz", **out)
    print(f"{N} windows × {T} tokens from {len(ids)} tokens")


def run(name, gguf):
    subprocess.run([sys.executable, "-I", "scripts/gguf_logits.py", "--ref", str(D / "ids.npz"), "--gguf", gguf,
                    "--out", str(D / f"{name}.npz")], check=True)


def logsoftmax(x):
    x = x.astype(np.float64)
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def report(names):
    ref = np.load(D / "bf16.npz")
    pos = np.arange(0, T, STEP)
    buckets = pos // BUCKET
    lines = ["# Q5: KL(BF16 ‖ build) by position, same tokens", "",
             f"{N} windows × {T} tokens of wikitext-2 test, logits every {STEP}th position, Metal, KV f16. "
             "Mean KL per 512-token bucket; ± is the SE over windows.", "",
             "| build | " + " | ".join(f"{b*BUCKET}–{(b+1)*BUCKET-1}" for b in range(T // BUCKET)) + " | last/first |",
             "|---|" + "---|" * (T // BUCKET + 1)]
    for n in names:
        q = np.load(D / f"{n}.npz")
        per_w = []
        for w in range(N):
            lp, lq = logsoftmax(ref[f"w{w}/logits"]), logsoftmax(q[f"w{w}/logits"])
            kl = (np.exp(lp) * (lp - lq)).sum(-1)
            per_w.append([kl[buckets == b].mean() for b in range(T // BUCKET)])
        a = np.array(per_w)
        m, se = a.mean(0), a.std(0, ddof=1) / np.sqrt(N)
        lines.append(f"| {n} | " + " | ".join(f"{x:.4f} ± {s:.4f}" for x, s in zip(m, se)) + f" | {m[-1]/m[0]:.2f} |")
    Path("results/q5-positions.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    {"make": lambda: make(), "run": lambda: run(sys.argv[2], sys.argv[3]), "report": lambda: report(sys.argv[2:])}[sys.argv[1]]()
