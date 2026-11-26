#!/bin/sh
# Ling-3.0-tiny: safetensors -> BF16 GGUF -> Q8_0 GGUF -> smoke tests. Rerunnable; skips finished steps.
set -eu
cd "$(dirname "$0")/.."
BIN=vendor/llama.cpp/build/bin
HF=models/Ling-3.0-tiny
BF16=models/ling-3.0-tiny-BF16.gguf
Q8=models/ling-3.0-tiny-Q8_0.gguf
mkdir -p logs results

dump() { .venv/bin/python -I vendor/llama.cpp/gguf-py/gguf/scripts/gguf_dump.py "$@"; }

if [ ! -f "$BF16" ]; then
  /usr/bin/time -l .venv/bin/python vendor/llama.cpp/convert_hf_to_gguf.py "$HF" --outtype bf16 --outfile "$BF16.part" \
    > logs/convert-bf16.log 2>&1
  mv "$BF16.part" "$BF16"
fi
dump --no-tensors "$BF16" > logs/dump-bf16-meta.txt 2>&1
dump "$BF16" > logs/dump-bf16.txt 2>&1

# shortcut: safetensors deleted after a verified BF16 GGUF because the disk holds ~38 GB; re-download at rev 9a98e35 to reconvert.
if [ -d "$HF" ] && [ -n "${DELETE_SAFETENSORS:-}" ]; then
  rm -f "$HF"/*.safetensors
fi

[ -n "${STOP_AFTER_BF16:-}" ] && { echo "BF16 done"; exit 0; }

if [ ! -f "$Q8" ]; then
  need=$(( $(stat -f %z "$BF16") * 55 / 100 / 1024 ))  # Q8_0 is ~53% of BF16
  free=$(df -k models | awk 'NR==2 {print $4}')
  [ "$free" -gt "$need" ] || { echo "not enough disk for Q8_0: ${free} KB free, ${need} KB needed" >&2; exit 1; }
  /usr/bin/time -l "$BIN/llama-quantize" "$BF16" "$Q8.part" Q8_0 > logs/quantize-q8.log 2>&1
  mv "$Q8.part" "$Q8"
fi
dump "$Q8" > logs/dump-q8.txt 2>&1

# Smoke test 1: greedy generation from a raw prompt (no chat template).
"$BIN/llama-completion" -m "$Q8" -no-cnv -p "The capital of France is" -n 32 --temp 0 -s 1 -t 10 \
  > logs/gen-q8.txt 2> logs/gen-q8.log || true

# Smoke test 2: KLD of Q8_0 against BF16 on 16 wikitext-2 chunks of 512 tokens.
/usr/bin/time -l "$BIN/llama-perplexity" -m "$BF16" -f data/wiki.test.raw -c 512 --chunks 16 -t 10 \
  --kl-divergence-base logs/kld-base-bf16.bin > logs/ppl-bf16.log 2>&1
/usr/bin/time -l "$BIN/llama-perplexity" -m "$Q8" -f data/wiki.test.raw -c 512 --chunks 16 -t 10 \
  --kl-divergence-base logs/kld-base-bf16.bin --kl-divergence > logs/kld-q8.log 2>&1
echo done
