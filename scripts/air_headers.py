"""Q8: build a weightless local copy of Ling-3.2-air for `convert_hf_to_gguf.py --dry-run`.

Fetches only each shard's safetensors header (8-byte length + JSON) by HTTP range request and writes
sparse files of the true size, so the converter sees every tensor name, dtype and shape without the
124 GB of data. Data regions read as zeros; dry-run never materializes them.
usage: .venv/bin/python -I scripts/air_headers.py <out_dir> [revision]
"""

import json
import os
import shutil
import struct
import sys
import urllib.request
from pathlib import Path

REPO = "inclusionAI/Ling-3.2-air"
SMALL = ["config.json", "generation_config.json", "model.safetensors.index.json", "preprocessor_config.json",
         "special_tokens_map.json", "tokenizer_config.json", "tokenizer.json", "chat_template.jinja"]


def get(url, rng=None, timeout=60):
    req = urllib.request.Request(url, headers={"Range": f"bytes={rng}"} if rng else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read(), r.headers


def main(out, rev):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    base = f"https://huggingface.co/{REPO}/resolve/{rev}/"
    for f in SMALL:
        if not (out / f).exists():
            try:
                (out / f).write_bytes(get(base + f)[0])
            except urllib.error.HTTPError as e:
                print(f"skip {f}: {e.code}")
    tree = json.loads(get(f"https://huggingface.co/api/models/{REPO}/tree/{rev}")[0])
    sizes = {e["path"]: e["size"] for e in tree if e["path"].endswith(".safetensors")}
    summary = {}
    for name, size in sorted(sizes.items()):
        dst = out / name
        if dst.exists() and dst.stat().st_size == size:
            continue
        n = struct.unpack("<Q", get(base + name, "0-7")[0])[0]
        head = get(base + name, f"8-{8 + n - 1}")[0]
        tmp = dst.with_suffix(".part")
        with open(tmp, "wb") as fh:
            fh.write(struct.pack("<Q", n) + head)
            fh.truncate(size)  # sparse on APFS: no data blocks allocated
        os.replace(tmp, dst)
        meta = json.loads(head)
        summary[name] = len(meta) - ("__metadata__" in meta)
        print(f"{name}: {summary[name]} tensors, header {n} B, file {size / 1e9:.2f} GB", flush=True)
    print(f"done: {len(sizes)} shards, {sum(sizes.values()) / 1e9:.1f} GB logical; disk used:",
          shutil.disk_usage(out).used // 2**30, "GiB on volume")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "main")
