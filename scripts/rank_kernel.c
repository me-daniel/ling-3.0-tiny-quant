// Exhaustive candidate ranking for scripts/refine_codes.py (float32 scores, best `top` per row; the caller rescores
// those exactly in float64). Candidates are stored transposed, CT[i * nc + c] = component i of candidate c.
//   score[r, c] = qc[c] + sum_i bp[r, i] * CT[i, c]; with uw != NULL only c with sum_i uw[r, i] * CT[i, c] >= th[r].
// cls[r] picks the candidate set (set_off[cls] .. + set_n[cls]); cls < 0 skips the row. out[r, j] = index within
// the set, or -1 where fewer than `top` candidates are feasible. Tiled over candidates so a tile stays in L1.
#include <float.h>
#include <stdlib.h>

#define TILE 512

static void scan(const float * s, int n, int base, int top, float * bv, int * bi, int * m) {
    for (int c = 0; c < n; ++c) {
        const float v = s[c];
        if (*m < top || v < bv[*m - 1]) {
            int j = *m < top ? (*m)++ : top - 1;
            while (j > 0 && bv[j - 1] > v) { bv[j] = bv[j - 1]; bi[j] = bi[j - 1]; --j; }
            bv[j] = v; bi[j] = base + c;
        }
    }
}

void rank_top(int rows, int k, int top, const float * CT, int nc_total, const int * set_off, const int * set_n,
              const float * qc, const float * bp, const float * uw, const float * th, const int * cls, int * out) {
    float s[TILE], bv[16]; int bi[16];
    for (int r = 0; r < rows; ++r) {
        int * o = out + (size_t) r * top;
        for (int j = 0; j < top; ++j) o[j] = -1;
        if (cls[r] < 0) continue;
        const int off = set_off[cls[r]], n = set_n[cls[r]];
        const float * b = bp + (size_t) r * k;
        const float * u = uw ? uw + (size_t) r * k : NULL;
        int m = 0;
        for (int t0 = 0; t0 < n; t0 += TILE) {
            const int nt = n - t0 < TILE ? n - t0 : TILE;
            const float * q = qc + off + t0;
            const float * C0 = CT + off + t0;
#define SCORE(K) \
            if (!u) { \
                for (int c = 0; c < nt; ++c) { \
                    float a = q[c]; \
                    for (int i = 0; i < K; ++i) a += b[i] * C0[(size_t) i * nc_total + c]; \
                    s[c] = a; \
                } \
            } else { \
                const float tr = th[r]; \
                for (int c = 0; c < nt; ++c) { \
                    float a = q[c], l = 0.f; \
                    for (int i = 0; i < K; ++i) { const float x = C0[(size_t) i * nc_total + c]; a += b[i] * x; l += u[i] * x; } \
                    s[c] = l >= tr ? a : FLT_MAX; \
                } \
            }
            if (k == 8) { SCORE(8) } else if (k == 4) { SCORE(4) } else { return; }
            if (m == top) { // skip the scalar scan unless something in the tile beats the current top-th best
                const float lim = bv[top - 1]; float mn = FLT_MAX;
                for (int c = 0; c < nt; ++c) mn = s[c] < mn ? s[c] : mn;
                if (!(mn < lim)) continue;
            }
            scan(s, nt, t0, top, bv, bi, &m);
        }
        for (int j = 0; j < m; ++j) o[j] = bv[j] < FLT_MAX ? bi[j] : -1;
    }
}
