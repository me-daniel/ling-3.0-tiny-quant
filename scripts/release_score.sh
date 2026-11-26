#!/bin/sh
# Release gate scoring (registered 25 Nov). usage: sh scripts/release_score.sh base | score NAME GGUF
#   Sets: zh512 / code512 / tool512 = ../ling-3.2-air-quant/data/{zh,code,tool}-test.txt at 128 x 512;
#   pg19-BOOK = long-pg19-BOOK.txt at 8 x 4096, books 10321 10356 10762 22424. Special-token parsing on
#   (patch 0013, PPL_PARSE_SPECIAL=1) so tool-test's chat tokens match the stored Air ids (checked after base).
#   Per-token KL dumps (patch 0011) to runs/release/NAME-SET.f32, logs logs/rg-kld-NAME-SET.log.
set -eu
cd "$(dirname "$0")/.."
BIN=vendor/llama.cpp/build/bin; BF16=models/ling-3.0-tiny-BF16.gguf; D=../ling-3.2-air-quant/data; K=runs/release
mkdir -p "$K" logs
SETS="zh512 code512 tool512 pg19-10321 pg19-10356 pg19-10762 pg19-22424"
args() { case $1 in
    zh512) echo "$D/zh-test.txt 512 128" ;; code512) echo "$D/code-test.txt 512 128" ;; tool512) echo "$D/tool-test.txt 512 128" ;;
    pg19-*) echo "$D/long-${1}.txt 4096 8" ;;
esac; }
export PPL_PARSE_SPECIAL=1
case $1 in
base)
    for s in $SETS; do
        set -- $(args "$s"); f="$K/base-$s.kld"
        [ -f "$f" ] || /usr/bin/time -l "$BIN/llama-perplexity" -m "$BF16" -f "$1" -c "$2" --chunks "$3" -t 10 \
            --kl-divergence-base "$f" > "logs/rg-base-$s.log" 2>&1
    done ;;
score)
    name=$2; gg=$3
    for s in $SETS; do
        set -- $(args "$s"); log="logs/rg-kld-$name-$s.log"
        grep -q "Mean    KLD" "$log" 2>/dev/null && continue
        KLD_DUMP="$K/$name-$s.f32" /usr/bin/time -l "$BIN/llama-perplexity" -m "$gg" -f "$1" -c "$2" --chunks "$3" -t 10 \
            --kl-divergence-base "$K/base-$s.kld" --kl-divergence > "$log" 2>&1
        grep -E "Mean +KLD" "$log" | tr '\n' ' '; echo " $name $s"
    done ;;
esac
