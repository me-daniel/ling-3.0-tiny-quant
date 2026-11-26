// Feed fixed token ids to a GGUF through libllama, write logits (and optionally every l_out-N
// hidden state) at chosen positions. No tokenizer involved: ids come from the HF reference.
//
// build: sh scripts/build_gguf_logits.sh
// usage: gguf_logits <model.gguf> <ids.i32> <pos.i32> <out_logits.f32> [out_hidden.f32]
//        env: GL_CPU=1 (no Metal offload), GL_UBATCH=n (default: whole prompt in one ubatch),
//             GL_KV=f16|f32 (KV cache type, default f16 = llama.cpp default), GL_THREADS=n,
//             GL_NAMES=a,b (exact graph tensor names, e.g. kda_out-0) GL_DIR=d: dump those whole to d/<name>.f32
// out_logits: [n_pos, n_vocab] f32. out_hidden: [n_layer, n_pos, n_embd] f32 (l_out-0..N-1).
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

static std::vector<int32_t> read_i32(const char * p) {
    FILE * f = fopen(p, "rb");
    if (!f) { perror(p); exit(1); }
    std::vector<int32_t> v;
    int32_t x;
    while (fread(&x, 4, 1, f) == 1) v.push_back(x);
    fclose(f);
    return v;
}

struct dump_state {
    std::map<int, std::vector<float>> layers;  // layer -> concatenated [n_tokens_so_far, n_embd]
    int64_t n_embd = 0;
    std::vector<std::string> extra;  // GL_NAMES: exact tensor names dumped whole to <dir>/<name>.f32
    std::string dir;
};

static bool cb(struct ggml_tensor * t, bool ask, void * ud) {
    auto * st = (dump_state *) ud;
    const char * name = ggml_get_name(t);
    if (ask && getenv("GL_LIST")) fprintf(stderr, "name %s %s\n", name, ggml_op_desc(t));
    for (auto & e : st->extra) {
        if (e == name) {
            if (ask) return true;
            std::vector<char> buf(ggml_nbytes(t));
            ggml_backend_tensor_get(t, buf.data(), 0, buf.size());
            std::string fn = st->dir + "/" + name + ".f32";
            FILE * f = fopen(fn.c_str(), "ab");
            fwrite(buf.data(), 1, buf.size(), f);
            fclose(f);
            fprintf(stderr, "dump %s type=%s ne=[%lld,%lld,%lld,%lld] contig=%d\n", name, ggml_type_name(t->type),
                    (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
                    (int) ggml_is_contiguous(t));
            return true;
        }
    }
    if (strncmp(name, "l_out-", 6) != 0) return false;
    if (ask) return true;
    int il = atoi(name + 6);
    GGML_ASSERT(t->type == GGML_TYPE_F32 && ggml_is_contiguous(t));
    st->n_embd = t->ne[0];
    auto & v = st->layers[il];
    size_t old = v.size();
    v.resize(old + ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data() + old, 0, ggml_nbytes(t));
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s model ids pos out_logits [out_hidden]\n", argv[0]); return 1; }
    auto ids = read_i32(argv[2]);
    auto pos = read_i32(argv[3]);
    const int T = (int) ids.size();
    const bool cpu = getenv("GL_CPU") != nullptr;
    const int ub = getenv("GL_UBATCH") ? atoi(getenv("GL_UBATCH")) : T;
    const bool kv32 = getenv("GL_KV") && strcmp(getenv("GL_KV"), "f32") == 0;
    const bool dump = argc > 5;

    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = cpu ? 0 : 999;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;

    dump_state st;
    auto cp = llama_context_default_params();
    cp.n_ctx = T < 256 ? 256 : T;
    cp.n_batch = T;
    cp.n_ubatch = ub;
    cp.n_seq_max = 1;
    if (getenv("GL_THREADS")) cp.n_threads = cp.n_threads_batch = atoi(getenv("GL_THREADS"));
    if (kv32) { cp.type_k = GGML_TYPE_F32; cp.type_v = GGML_TYPE_F32; }
    if (getenv("GL_NAMES")) {
        std::string all = getenv("GL_NAMES"), cur;
        for (char c : all + ",") { if (c == ',') { if (!cur.empty()) st.extra.push_back(cur); cur.clear(); } else cur += c; }
        st.dir = getenv("GL_DIR") ? getenv("GL_DIR") : ".";
    }
    if (dump || !st.extra.empty()) { cp.cb_eval = cb; cp.cb_eval_user_data = &st; }
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) return 1;

    llama_batch b = llama_batch_init(T, 0, 1);
    for (int i = 0; i < T; i++) {
        b.token[i] = ids[i]; b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1;
    }
    b.n_tokens = T;
    if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed\n"); return 1; }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    FILE * f = fopen(argv[4], "wb");
    for (int p : pos) fwrite(llama_get_logits_ith(ctx, p), sizeof(float), n_vocab, f);
    fclose(f);
    fprintf(stderr, "T=%d n_vocab=%d ubatch=%d cpu=%d kv=%s wrote %zu positions\n",
            T, n_vocab, ub, cpu, kv32 ? "f32" : "f16", pos.size());

    if (dump) {
        FILE * g = fopen(argv[5], "wb");
        for (auto & [il, v] : st.layers) {
            if ((int64_t) v.size() != (int64_t) T * st.n_embd) {
                // last layer is cropped to output rows; with all logits requested it is still T rows
                fprintf(stderr, "layer %d has %zu values, expected %lld\n", il, v.size(), (long long) T * st.n_embd);
                return 1;
            }
            for (int p : pos) fwrite(v.data() + (size_t) p * st.n_embd, sizeof(float), st.n_embd, g);
        }
        fclose(g);
        fprintf(stderr, "dumped %zu layers, n_embd=%lld\n", st.layers.size(), (long long) st.n_embd);
    }
    llama_batch_free(b);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
