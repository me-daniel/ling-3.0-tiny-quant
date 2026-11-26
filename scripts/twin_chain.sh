#!/bin/sh
# Twin / mix / cheap-mode chain (CHANGELOG 2026-11-25 registration). Sequential, one heavy step at a time; resumable.
set -eu
cd "$(dirname "$0")/.."
P=models/public/bloomer010/Ling-3.0-tiny-IQ3_XXS.gguf; M2=models/ling-3.0-tiny-m2.gguf; T=models/tmp
PY=".venv/bin/python -I"; mkdir -p runs/twin
L=$(seq 1 23)
step() { echo "[$(date '+%H:%M:%S')] $*"; }
ref() { # name start-gguf extra-args...
    n=$1; q=$2; shift 2
    [ -f "models/refined/$n/blk.23.gguf" ] && return 0
    step "refine $n"; /usr/bin/time -l $PY scripts/refine_codes.py --quant "$q" --layers $L --workers 10 --out-dir "models/refined/$n" "$@" > "logs/refine-$n.log" 2>&1
    grep -E "all layers|real" "logs/refine-$n.log"
}
sc() { # name start-gguf  (splice, check, score, delete)
    n=$1; q=$2
    grep -q "Mean    KLD" "logs/gt-kld-$n-wiki4096.log" 2>/dev/null && return 0
    step "splice + score $n"; sh scripts/gate_t.sh splice "$n" "$q"
    $PY scripts/check_refined.py "$q" "$T/$n.gguf" "models/refined/$n" --ggml-layers 12 | tail -4
    sh scripts/gate_t.sh score "$n" "$T/$n.gguf"; rm -f "$T/$n.gguf" "$T/$n.gguf.json"
}
ref twin "$P"
sh scripts/gate_t.sh base
sh scripts/gate_t.sh score bloomer "$P"; sc twin "$P"
ref m2r "$M2"; sh scripts/gate_t.sh score m2 "$M2"; sc m2r "$M2"
ref twin-s1 "$P" --sweeps 1; sc twin-s1 "$P"
[ -f runs/twin/select-half.json ] || $PY scripts/select_half.py models/refined/twin > runs/twin/select-half.json
ref twin-half "$P" --select runs/twin/select-half.json; sc twin-half "$P"
[ -f models/grams/mixc-c512-n100/done-p4 ] || { step "capture n100"; CHUNKS=100 sh scripts/gram-capture.sh; }
ref twin-g100 "$P" --grams models/grams/mixc-c512-n100; sc twin-g100 "$P"
step "chain done"
