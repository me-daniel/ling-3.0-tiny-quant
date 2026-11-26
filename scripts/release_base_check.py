"""Check each release BF16 base file's token ids against the stored Air ids (../ling-3.2-air-quant/data/*.ids.npy).

llama-perplexity takes chunk i as tokens[i*n_ctx : (i+1)*n_ctx] of the tokenized text and puts BOS at position 0 when
the vocab adds one, so positions 1.. of every chunk are compared with the stored ids at the same offsets (position 0
too when no BOS is added).   .venv/bin/python -I scripts/release_base_check.py
"""
import numpy as np
from pathlib import Path

D, K = Path("../ling-3.2-air-quant/data"), Path("runs/release")
SETS = {"zh512": "zh-test", "code512": "code-test", "tool512": "tool-test", **{f"pg19-{b}": f"long-pg19-{b}"
        for b in ("10321", "10356", "10762", "22424")}}
ok = True
for s, stem in SETS.items():
    f = K / f"base-{s}.kld"
    with open(f, "rb") as fh:
        assert fh.read(8) == b"_logits_"
        n_ctx = int(np.frombuffer(fh.read(4), np.uint32)[0]); n_vocab, n_chunk = np.frombuffer(fh.read(8), np.int32)
        tok = np.frombuffer(fh.read(4 * n_ctx * n_chunk), np.int32).reshape(n_chunk, n_ctx)
    ref = np.load(D / f"{stem}.ids.npy")
    need = n_ctx * n_chunk
    r = ref[:need].reshape(n_chunk, n_ctx) if len(ref) >= need else None
    if r is None:
        print(f"{s}: stored ids too short ({len(ref)} < {need})"); ok = False; continue
    same_rest = np.array_equal(tok[:, 1:], r[:, 1:])
    pos0 = "BOS" if not np.array_equal(tok[:, 0], r[:, 0]) and len(set(tok[:, 0])) == 1 else "text"
    print(f"{s}: n_ctx {n_ctx} chunks {n_chunk} vocab {n_vocab}; positions 1.. equal stored ids: {same_rest}; pos 0: {pos0}")
    ok &= same_rest
print("ALL OK" if ok else "MISMATCH")
