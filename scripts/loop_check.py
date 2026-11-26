"""Loop check (registered 26 Nov): 20 fixed chat prompts, greedy, 512 new tokens, llama-completion on Metal.

    .venv/bin/python -I scripts/loop_check.py run NAME GGUF      # writes runs/loops/NAME.json
    .venv/bin/python -I scripts/loop_check.py report NAME ...    # table

Prompts data/loop-prompts.json rendered with models/Ling-3.0-tiny/chat_template.jinja (HF apply_chat_template,
generation prompt on). Output re-tokenized with the tiny tokenizer. Loop: some span of L tokens, 8 <= L <= 128,
occurs 3 times back to back; "ends looping": such a triple reaches the last L tokens.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs/loops"


def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(ROOT / "models/Ling-3.0-tiny"))


def detect(ids):
    n = len(ids)
    found, ends = None, False
    for L in range(8, 129):
        for i in range(0, n - 3 * L + 1):
            if ids[i:i + L] == ids[i + L:i + 2 * L] == ids[i + 2 * L:i + 3 * L]:
                found = found or (L, i)
                if i + 3 * L >= n - L:
                    ends = True
                break
    return found, ends


def run(name, gguf):
    t = tok()
    prompts = json.loads((ROOT / "data/loop-prompts.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    res = []
    for k, p in enumerate(prompts):
        text = t.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        r = subprocess.run([str(ROOT / "vendor/llama.cpp/build/bin/llama-completion"), "-m", gguf, "-p", text, "-n", "512",
                            "--temp", "0", "--seed", "0", "-no-cnv", "--no-display-prompt", "-ngl", "99", "-c", "4096"],
                           capture_output=True, text=True, stdin=subprocess.DEVNULL)
        if r.returncode:
            raise SystemExit(f"prompt {k}: llama-completion failed: {r.stderr[-500:]}")
        out = r.stdout
        ids = t.encode(out, add_special_tokens=False)
        f, e = detect(ids)
        res.append({"k": k, "tokens": len(ids), "loop": f is not None, "loop_L_at": f, "ends_looping": e, "text": out})
        print(k, len(ids), "LOOP" if f else "-", f, "ends" if e else "", flush=True)
    (OUT / f"{name}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


def report(names):
    print("| file | outputs | looping | ends looping | median tokens |")
    print("|---|---|---|---|---|")
    for n in names:
        r = json.loads((OUT / f"{n}.json").read_text())
        toks = sorted(x["tokens"] for x in r)
        print(f"| {n} | {len(r)} | {sum(x['loop'] for x in r)} | {sum(x['ends_looping'] for x in r)} | {toks[len(toks) // 2]} |")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3])
    else:
        report(sys.argv[2:])
