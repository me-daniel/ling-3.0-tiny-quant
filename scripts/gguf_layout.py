"""Dump a GGUF's tensor layout (name, type, bytes) to TSV, from the header only. Source of the card banner.

    .venv/bin/python -I scripts/gguf_layout.py models/ling-3.0-tiny-m2.gguf results/layout-m2.tsv

gguf-py memory-maps the file and reads tensor infos; no tensor data is read. The last lines are `#file <name>`, `#file_bytes <n>`
(size on disk, metadata included) and `#sha256 <hex>` (ties the banner and the refinement record to one file).
"""
import hashlib
import sys
from pathlib import Path

from gguf import GGUFReader


def main(src, out):
    r = GGUFReader(src)
    lines = ["name\ttype\tbytes"]
    lines += [f"{t.name}\t{t.tensor_type.name}\t{int(t.n_bytes)}" for t in r.tensors]
    lines.append(f"#file\t{Path(src).name}")
    lines.append(f"#file_bytes\t{Path(src).stat().st_size}")
    h = hashlib.sha256()
    with open(src, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    lines.append(f"#sha256\t{h.hexdigest()}")
    Path(out).write_text("\n".join(lines) + "\n")
    print(f"{out}: {len(r.tensors)} tensors, {Path(src).stat().st_size:,} B on disk")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
