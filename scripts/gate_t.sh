#!/bin/sh
# Gate T builds and scores. usage: sh scripts/gate_t.sh build U3|U2  |  splice R3|R2  |  score NAME GGUF  |  base
#   build: base Q8_0, routed experts gate/up/down at IQ3_XXS (U3) or IQ2_XXS (U2), imatrix from the Gram capture.
#   splice NAME [START]: NAME = START (default models/tmp/U<n>.gguf) with every models/refined/NAME/blk.L.gguf
#     taken byte for byte (scripts/splice.py --verify).
#   base: BF16 --kl-divergence-base files (wiki 64 x 512, eval-ood 64 x 512, wiki 16 x 4096).
#   score: KLD vs each base, per-token KL dumped to runs/gate-t/NAME-SET.f32 (patch 0011, KLD_DUMP).
set -eu
cd "$(dirname "$0")/.."
BIN=vendor/llama.cpp/build/bin; BF16=models/ling-3.0-tiny-BF16.gguf
IM=models/grams/mixc-c512-n400/imatrix-p1.gguf; K=runs/gate-t; T=models/tmp
mkdir -p "$K" "$T" logs
SETS="wiki512 ood512 wiki4096"
set_args() { case $1 in
    wiki512) echo "data/wiki.test.raw 512 64" ;; ood512) echo "data/eval-ood.txt 512 64" ;; wiki4096) echo "data/wiki.test.raw 4096 16" ;;
esac; }
case $1 in
build)
    case $2 in U3) ty=iq3_xxs ;; U2) ty=iq2_xxs ;; *) exit 2 ;; esac
    [ -f "$T/$2.gguf" ] && { echo "have $T/$2.gguf"; exit 0; }
    /usr/bin/time -l "$BIN/llama-quantize" --imatrix "$IM" \
        --tensor-type "^blk\.[0-9]+\.ffn_(gate|up|down)_exps\.weight\$=$ty" "$BF16" "$T/$2.gguf.part" Q8_0 > "logs/gt-q-$2.log" 2>&1
    mv "$T/$2.gguf.part" "$T/$2.gguf"; grep -E "quant size|real|maximum resident" "logs/gt-q-$2.log" ;;
splice)
    # splice NAME [START [DIR ...]]: refined tensors from each DIR (default models/refined/NAME): blk.L.gguf files
    # (experts of layer L) and dense.gguf (every tensor in it, by exact name)
    n=$2; U=${3:-$T/$(echo "$n" | sed 's/^R/U/').gguf}; shift 2; [ $# -gt 0 ] && shift
    [ $# -gt 0 ] || set -- "models/refined/$n"
    args=""
    for d in "$@"; do
        for f in "$d"/blk.*.gguf; do
            [ -f "$f" ] || continue
            L=$(basename "$f" .gguf | sed 's/blk\.//'); args="$args --take $f ^blk\\.$L\\.ffn_(gate|up|down)_exps\\.weight\$"
        done
        if [ -f "$d/dense.gguf" ]; then
            rx=$(.venv/bin/python -I -c "import json,re,sys; print('^(' + '|'.join(re.escape(t['name']) for t in json.load(open(sys.argv[1]))['tensors']) + ')\$')" "$d/dense.json")
            args="$args --take $d/dense.gguf $rx"
        fi
    done
    # shellcheck disable=SC2086
    .venv/bin/python -I scripts/splice.py "$U" "$T/$n.gguf" $args --verify > "logs/gt-splice-$n.json" ;;
base)
    for s in $SETS; do
        set -- $(set_args "$s"); f="$K/base-$s.kld"
        [ -f "$f" ] || /usr/bin/time -l "$BIN/llama-perplexity" -m "$BF16" -f "$1" -c "$2" --chunks "$3" -t 10 \
            --kl-divergence-base "$f" > "logs/gt-base-$s.log" 2>&1
    done ;;
score)
    name=$2; gg=$3
    for s in $SETS; do
        set -- $(set_args "$s"); log="logs/gt-kld-$name-$s.log"
        grep -q "Mean    KLD" "$log" 2>/dev/null && continue
        KLD_DUMP="$K/$name-$s.f32" /usr/bin/time -l "$BIN/llama-perplexity" -m "$gg" -f "$1" -c "$2" --chunks "$3" -t 10 \
            --kl-divergence-base "$K/base-$s.kld" --kl-divergence > "$log" 2>&1
        grep -E "Mean +KLD|Same top p" "$log" | tr '\n' ' '; echo " $name $s"
    done ;;
esac
