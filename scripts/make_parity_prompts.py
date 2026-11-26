"""Write data/parity_prompts.jsonl: 3 short prompts (<=64 tokens -> KDA fused_recurrent path in HF)
and 1 long wiki.test.raw prompt cut to 2000 tokens (-> KDA chunk path, T not a multiple of 64)."""
import json
from pathlib import Path

short = [
    "The capital of France is Paris, and the capital of Germany is Berlin. The capital of Italy is",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) +",
    "In 1905 Albert Einstein published four papers that changed physics, including one on",
]
wiki = Path("data/wiki.test.raw").read_text()[:12000]
rows = [{"name": f"short{i}", "text": t} for i, t in enumerate(short)]
rows.append({"name": "long", "text": wiki, "max_tokens": 2000, "stride": 4})
Path("data/parity_prompts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
print("wrote data/parity_prompts.jsonl")
