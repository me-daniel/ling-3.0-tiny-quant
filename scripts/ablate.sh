#!/bin/sh
# One ablation: quantize BF16 with a base type plus --tensor-type overrides, KLD vs the BF16 base, append a row, delete the build.
# usage: scripts/ablate.sh <name> <base_type> [regex=type ...]
# env: PREBUILT=<gguf> scores an existing file (no quantize, never deleted); IMATRIX=<file> (needed for IQ1-IQ3), CHUNKS (default 16), CTX (default 512), KEEP=1 keeps the GGUF.
# Skips if <name> already has a row in results/ablations.tsv (rerunnable).
set -eu
cd "$(dirname "$0")/.."
name=$1 base=$2; shift 2
BIN=vendor/llama.cpp/build/bin
BF16=models/ling-3.0-tiny-BF16.gguf
CHUNKS=${CHUNKS:-16} CTX=${CTX:-512} EVAL=${EVAL:-data/wiki.test.raw}
ev=$(basename "$EVAL" .txt); ev=${ev%.raw}
KLDBASE=logs/kld-base-bf16-$ev-c${CTX}-n${CHUNKS}.bin
TSV=results/ablations.tsv
OUT=${PREBUILT:-models/tmp-$name.gguf}
mkdir -p results logs
[ -f "$TSV" ] || printf 'name\tbase\toverrides\timatrix\teval\tctx\tchunks\tbytes\tbpw\tppl\tmean_kld\tkld_se\tp99_kld\tmax_kld\ttop1\tquant_s\tkld_s\n' > "$TSV"
if awk -F'\t' -v n="$name" '$1==n {f=1} END {exit !f}' "$TSV"; then echo "skip $name"; exit 0; fi

if [ ! -f "$KLDBASE" ]; then
  "$BIN/llama-perplexity" -m "$BF16" -f "$EVAL" -c "$CTX" --chunks "$CHUNKS" -t 10 \
    --kl-divergence-base "$KLDBASE" > "logs/kldbase-$ev-c${CTX}-n${CHUNKS}.log" 2>&1
fi

args=""
for o in "$@"; do args="$args --tensor-type $o"; done
im=""; [ -n "${IMATRIX:-}" ] && im="--imatrix $IMATRIX"
t0=$(date +%s)
# shellcheck disable=SC2086
if [ -z "${PREBUILT:-}" ]; then
  "$BIN/llama-quantize" $im $args "$BF16" "$OUT" "$base" > "logs/q-$name.log" 2>&1
else
  echo "prebuilt $PREBUILT" > "logs/q-$name.log"
fi
t1=$(date +%s)
"$BIN/llama-perplexity" -m "$OUT" -f "$EVAL" -c "$CTX" --chunks "$CHUNKS" -t 10 \
  --kl-divergence-base "$KLDBASE" --kl-divergence > "logs/kld-$name.log" 2>&1
t2=$(date +%s)

L=logs/kld-$name.log
bytes=$(stat -f %z "$OUT")
bpw=$(sed -n 's/.*quant size.*(\([0-9.]*\) BPW).*/\1/p' "logs/q-$name.log")
ppl=$(awk '/^ *[0-9]+ +[0-9.]+ ±/ {p=$2} END {print p}' "$L")
mk=$(awk '/Mean +KLD:/ {print $3"\t"$5}' "$L")
p99=$(awk '/99.0% +KLD:/ {print $3}' "$L")
mx=$(awk '/Maximum KLD:/ {print $3}' "$L")
top=$(awk '/Same top p:/ {print $4}' "$L")
[ -n "$mk" ] || { echo "KLD parse failed for $name, see $L" >&2; exit 1; }
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$base" "$*" "${IMATRIX:-}" "$ev" "$CTX" "$CHUNKS" \
  "$bytes" "$bpw" "$ppl" "$mk" "$p99" "$mx" "$top" "$((t1-t0))" "$((t2-t1))" >> "$TSV"
[ -n "${KEEP:-}${PREBUILT:-}" ] || rm -f "$OUT"
tail -1 "$TSV"
