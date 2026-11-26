#!/bin/sh
# Gate T step 2: full per-expert input Grams of the routed experts (patch 0010, patches/0010-imatrix-gram-capture.diff),
# gate (= up input) and down, layers 1-23, BF16 forward, data/calib-mixc.txt 400 x 512, --parse-special.
# The patch holds every matched Gram in RAM (1.2 GB per gate layer), so the 23 layers go in 4 passes to stay under
# ~30 GB with the 15.8 GB model. Each pass also writes its imatrix (imatrix-pN.gguf); pass 1's is the one used.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
CHUNKS=${CHUNKS:-400}; D="$ROOT/models/grams/mixc-c512-n$CHUNKS"; mkdir -p "$D"  # CHUNKS=100 for the 1/4-token Grams
BIN="$ROOT/vendor/llama.cpp/build/bin/llama-imatrix"
n=0
for RX in '1|2|3|4|5|6' '7|8|9|10|11|12' '13|14|15|16|17|18' '19|20|21|22|23'; do
    n=$((n + 1))
    [ -f "$D/done-p$n" ] && continue
    echo "[$(date '+%H:%M:%S')] pass $n layers $RX"
    GRAM_DIR="$D" GRAM_MATCH="^blk\\.($RX)\\.ffn_(gate|down)_exps\\.weight\$" /usr/bin/time -l "$BIN" \
        -m "$ROOT/models/ling-3.0-tiny-BF16.gguf" -f "$ROOT/data/calib-mixc.txt" -c 512 --chunks "$CHUNKS" --parse-special \
        -o "$D/imatrix-p$n.gguf" > "$ROOT/logs/gram-capture-n$CHUNKS-p$n.log" 2>&1
    touch "$D/done-p$n"
    grep -E "real|maximum resident|peak memory" "$ROOT/logs/gram-capture-n$CHUNKS-p$n.log"
done
echo "[$(date '+%H:%M:%S')] done"
