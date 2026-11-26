"""Same-suffix drift test: score one fixed 512-token suffix after 1K, 4K and 8K tokens of history.

If HF-vs-llama.cpp KL on the identical suffix grows with history length, llama.cpp's recurrent
(KDA) state or MLA cache drifts from the reference with context; if it stays flat, the gap seen on
the 2,000-token parity prompt was content, not position.
Writes data/drift_prompts.jsonl: 3 windows (offsets spread over wikitext-2 test) x 3 history lengths.
"""
import json
from pathlib import Path

from transformers import AutoTokenizer

T, SUFFIX, HIST = 8192, 512, [1024, 4096, 8192]
tok = AutoTokenizer.from_pretrained("models/Ling-3.0-tiny")
ids = tok(Path("data/wiki.test.raw").read_text())["input_ids"]
rows = []
for w in range(3):
    end = (w + 1) * (len(ids) // 3)  # window ends spread over the file
    for h in HIST:
        rows.append({"name": f"w{w}h{h}", "ids": ids[end - h:end], "save_last": SUFFIX})
Path("data/drift_prompts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
print(f"{len(ids)} tokens; wrote {len(rows)} prompts to data/drift_prompts.jsonl")
