#!/bin/sh
# Build scripts/gguf_logits.cpp against the vendored llama.cpp build (libllama + ggml dylibs).
set -eu
cd "$(dirname "$0")/.."
L=vendor/llama.cpp
mkdir -p build
clang++ -std=c++17 -O2 -Wno-deprecated-declarations \
  -I$L/include -I$L/ggml/include scripts/gguf_logits.cpp \
  -L$L/build/bin -lllama -lggml -lggml-base -Wl,-rpath,"$PWD/$L/build/bin" -o build/gguf_logits
echo built build/gguf_logits
