// Dequantize one tensor of a GGUF with ggml's own CPU to_float (the reference decoder llama.cpp uses), to raw f32.
//   build/ggml_dequant FILE.gguf TENSOR_NAME OUT.f32
#include "ggml.h"
#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char ** argv) {
    if (argc != 4) { fprintf(stderr, "usage: %s FILE.gguf TENSOR OUT.f32\n", argv[0]); return 2; }
    struct ggml_context * meta = NULL;
    struct gguf_init_params p = { /*no_alloc*/ true, &meta };
    struct gguf_context * g = gguf_init_from_file(argv[1], p);
    if (!g) { fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }
    int64_t id = gguf_find_tensor(g, argv[2]);
    if (id < 0) { fprintf(stderr, "no tensor %s\n", argv[2]); return 1; }
    struct ggml_tensor * t = ggml_get_tensor(meta, argv[2]);
    size_t nb = ggml_nbytes(t);
    int64_t n = ggml_nelements(t);
    void * buf = malloc(nb);
    float * out = malloc((size_t) n * sizeof(float));
    FILE * f = fopen(argv[1], "rb");
    fseeko(f, (off_t) (gguf_get_data_offset(g) + gguf_get_tensor_offset(g, id)), SEEK_SET);
    if (fread(buf, 1, nb, f) != nb) { fprintf(stderr, "short read\n"); return 1; }
    fclose(f);
    const struct ggml_type_traits * tr = ggml_get_type_traits(t->type);
    if (!tr->to_float) { fprintf(stderr, "no to_float for %s\n", ggml_type_name(t->type)); return 1; }
    tr->to_float(buf, out, n);
    FILE * o = fopen(argv[3], "wb");
    fwrite(out, sizeof(float), (size_t) n, o);
    fclose(o);
    fprintf(stderr, "%s %s %lld values\n", argv[2], ggml_type_name(t->type), (long long) n);
    return 0;
}
