/*
 * ggml-turboquant.c — TurboQuant quantization for GGML/llama.cpp
 *
 * Drop this into ggml/src/ alongside ggml-quants.c
 *
 * Algorithm: WHT rotation + Lloyd-Max optimal scalar quantization
 * with norm correction (TheTom/spiritbuun optimization).
 *
 * Community-validated findings incorporated:
 *   - WHT >> random rotation (59x better at 4-bit, Arclabs001)
 *   - MSE-only >> QJL (80.4% vs 69.6% top-1, Arclabs001)
 *   - Block-32 optimal for FA parallelism (TheTom, Aaryan-Kapoor)
 *   - Norm correction: -0.36% PPL at zero decode cost (TheTom/spiritbuun)
 *   - K/V norm disparity: K needs more bits than V (scos-lab)
 */

#include "ggml-turboquant.h"
#include <math.h>
#include <string.h>
#include <assert.h>

/* ─── Precomputed Lloyd-Max codebooks (N(0,1) scaled by 1/sqrt(32)) ── */

static const float TURBO_SCALE = 0.17677669f; /* 1/sqrt(32) */

/* 2-bit (4 levels) */
static const float LM2_CENTROIDS[4] = {
    -1.5104f * 0.17677669f, -0.4528f * 0.17677669f,
     0.4528f * 0.17677669f,  1.5104f * 0.17677669f
};
static const float LM2_BOUNDARIES[3] = {
    -0.9816f * 0.17677669f, 0.0f, 0.9816f * 0.17677669f
};

/* 3-bit (8 levels) */
static const float LM3_CENTROIDS[8] = {
    -2.1519f * 0.17677669f, -1.3439f * 0.17677669f,
    -0.7560f * 0.17677669f, -0.2451f * 0.17677669f,
     0.2451f * 0.17677669f,  0.7560f * 0.17677669f,
     1.3439f * 0.17677669f,  2.1519f * 0.17677669f
};
static const float LM3_BOUNDARIES[7] = {
    -1.7479f * 0.17677669f, -1.0500f * 0.17677669f,
    -0.5006f * 0.17677669f,  0.0f,
     0.5006f * 0.17677669f,  1.0500f * 0.17677669f,
     1.7479f * 0.17677669f
};

/* 4-bit (16 levels) */
static const float LM4_CENTROIDS[16] = {
    -2.7326f * 0.17677669f, -2.0690f * 0.17677669f,
    -1.6180f * 0.17677669f, -1.2562f * 0.17677669f,
    -0.9424f * 0.17677669f, -0.6568f * 0.17677669f,
    -0.3881f * 0.17677669f, -0.1284f * 0.17677669f,
     0.1284f * 0.17677669f,  0.3881f * 0.17677669f,
     0.6568f * 0.17677669f,  0.9424f * 0.17677669f,
     1.2562f * 0.17677669f,  1.6180f * 0.17677669f,
     2.0690f * 0.17677669f,  2.7326f * 0.17677669f
};
static const float LM4_BOUNDARIES[15] = {
    -2.4008f * 0.17677669f, -1.8435f * 0.17677669f,
    -1.4370f * 0.17677669f, -1.0993f * 0.17677669f,
    -0.7996f * 0.17677669f, -0.5224f * 0.17677669f,
    -0.2582f * 0.17677669f,  0.0f,
     0.2582f * 0.17677669f,  0.5224f * 0.17677669f,
     0.7996f * 0.17677669f,  1.0993f * 0.17677669f,
     1.4370f * 0.17677669f,  1.8435f * 0.17677669f,
     2.4008f * 0.17677669f
};

/* ─── FP16 helpers ───────────────────────────────────────────────────── */

static uint16_t f32_to_f16(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    uint32_t sign = (u >> 16) & 0x8000;
    int32_t  exp  = ((u >> 23) & 0xFF) - 127;
    uint32_t mant = u & 0x7FFFFF;
    if (exp > 15)  return (uint16_t)(sign | 0x7C00);
    if (exp < -14) return (uint16_t)sign;
    return (uint16_t)(sign | ((exp + 15) << 10) | (mant >> 13));
}

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x03FF;
    uint32_t u;
    if (exp == 0) {
        u = (mant == 0) ? sign : sign | ((127 - 14) << 23) | (mant << 13);
    } else if (exp == 31) {
        u = sign | 0x7F800000 | (mant << 13);
    } else {
        u = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

/* ─── Fast Walsh-Hadamard Transform ──────────────────────────────────── */

static void wht32(float * x) {
    /* Optimized in-place WHT for n=32, O(n log n) */
    for (int len = 1; len < 32; len <<= 1) {
        for (int i = 0; i < 32; i += len << 1) {
            for (int j = 0; j < len; j++) {
                float u = x[i + j];
                float v = x[i + j + len];
                x[i + j]       = u + v;
                x[i + j + len] = u - v;
            }
        }
    }
    /* 1/sqrt(32) normalization — self-inverse */
    for (int i = 0; i < 32; i++) {
        x[i] *= TURBO_SCALE;
    }
}

/* ─── L2 norm ────────────────────────────────────────────────────────── */

static float vec_norm32(const float * x) {
    float sum = 0.0f;
    for (int i = 0; i < 32; i++) sum += x[i] * x[i];
    return sqrtf(sum);
}

/* ─── Scalar quantization (binary search on boundaries) ──────────────── */

static inline int quantize_scalar(float val, const float * boundaries, int n_levels) {
    int lo = 0, hi = n_levels - 2;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        if (val < boundaries[mid]) hi = mid - 1;
        else lo = mid + 1;
    }
    return lo;
}

/* ─── 4-bit packing ──────────────────────────────────────────────────── */

static void pack_4bit(const uint8_t * idx, uint8_t * packed) {
    for (int i = 0; i < 16; i++) {
        packed[i] = (uint8_t)(idx[2*i] | (idx[2*i + 1] << 4));
    }
}

static void unpack_4bit(const uint8_t * packed, uint8_t * idx) {
    for (int i = 0; i < 16; i++) {
        idx[2*i]     = packed[i] & 0x0F;
        idx[2*i + 1] = (packed[i] >> 4) & 0x0F;
    }
}

/* ─── 3-bit packing: 8 x 3-bit values → 3 bytes, x4 for 32 elements ── */

static void pack_3bit(const uint8_t * idx, uint8_t * packed) {
    for (int g = 0; g < 4; g++) {
        const uint8_t * s = idx + g * 8;
        uint8_t * d = packed + g * 3;
        d[0] = (uint8_t)(s[0] | (s[1] << 3) | (s[2] << 6));
        d[1] = (uint8_t)((s[2] >> 2) | (s[3] << 1) | (s[4] << 4) | (s[5] << 7));
        d[2] = (uint8_t)((s[5] >> 1) | (s[6] << 2) | (s[7] << 5));
    }
}

static void unpack_3bit(const uint8_t * packed, uint8_t * idx) {
    for (int g = 0; g < 4; g++) {
        const uint8_t * s = packed + g * 3;
        uint8_t * d = idx + g * 8;
        d[0] = s[0] & 0x07;
        d[1] = (s[0] >> 3) & 0x07;
        d[2] = ((s[0] >> 6) | (s[1] << 2)) & 0x07;
        d[3] = (s[1] >> 1) & 0x07;
        d[4] = (s[1] >> 4) & 0x07;
        d[5] = ((s[1] >> 7) | (s[2] << 1)) & 0x07;
        d[6] = (s[2] >> 2) & 0x07;
        d[7] = (s[2] >> 5) & 0x07;
    }
}

/* ─── 2-bit packing ──────────────────────────────────────────────────── */

static void pack_2bit(const uint8_t * idx, uint8_t * packed) {
    for (int i = 0; i < 8; i++) {
        packed[i] = (uint8_t)(idx[4*i] | (idx[4*i+1] << 2) |
                              (idx[4*i+2] << 4) | (idx[4*i+3] << 6));
    }
}

static void unpack_2bit(const uint8_t * packed, uint8_t * idx) {
    for (int i = 0; i < 8; i++) {
        idx[4*i]   =  packed[i]       & 0x03;
        idx[4*i+1] = (packed[i] >> 2) & 0x03;
        idx[4*i+2] = (packed[i] >> 4) & 0x03;
        idx[4*i+3] = (packed[i] >> 6) & 0x03;
    }
}

/* ─── Generic quantize one block with norm correction ────────────────── */

static void quantize_block_turbo(
    const float * src, uint8_t * indices, uint16_t * norm_out,
    const float * centroids, const float * boundaries, int n_levels
) {
    float work[32];

    /* 1. L2 norm */
    float norm = vec_norm32(src);

    /* 2. Normalize */
    float inv_norm = (norm > 1e-12f) ? (1.0f / norm) : 0.0f;
    for (int i = 0; i < 32; i++) work[i] = src[i] * inv_norm;

    /* 3. WHT rotation */
    wht32(work);

    /* 4. Lloyd-Max quantization */
    for (int i = 0; i < 32; i++) {
        indices[i] = (uint8_t)quantize_scalar(work[i], boundaries, n_levels);
    }

    /* 5. Norm correction: store original_norm / ||reconstruction|| */
    float recon[32];
    for (int i = 0; i < 32; i++) recon[i] = centroids[indices[i]];
    wht32(recon);
    float recon_norm = vec_norm32(recon);
    float corrected = (recon_norm > 1e-12f) ? (norm / recon_norm) : 0.0f;
    *norm_out = f32_to_f16(corrected);
}

/* ─── Generic dequantize one block ───────────────────────────────────── */

static void dequantize_block_turbo(
    const uint8_t * indices, uint16_t norm_fp16, float * dst,
    const float * centroids
) {
    float norm = f16_to_f32(norm_fp16);
    for (int i = 0; i < 32; i++) dst[i] = centroids[indices[i]];
    wht32(dst);
    for (int i = 0; i < 32; i++) dst[i] *= norm;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  turbo4_0 — 4-bit TurboQuant
 * ═══════════════════════════════════════════════════════════════════════ */

void quantize_row_turbo4_0_ref(const float * x, block_turbo4_0 * y, int64_t k) {
    assert(k % QK_TQ == 0);
    const int64_t nb = k / QK_TQ;
    uint8_t indices[32];
    for (int64_t i = 0; i < nb; i++) {
        quantize_block_turbo(x + i*32, indices, &y[i].d,
                            LM4_CENTROIDS, LM4_BOUNDARIES, 16);
        pack_4bit(indices, y[i].qs);
    }
}

void dequantize_row_turbo4_0(const block_turbo4_0 * x, float * y, int64_t k) {
    assert(k % QK_TQ == 0);
    const int64_t nb = k / QK_TQ;
    uint8_t indices[32];
    for (int64_t i = 0; i < nb; i++) {
        unpack_4bit(x[i].qs, indices);
        dequantize_block_turbo(indices, x[i].d, y + i*32, LM4_CENTROIDS);
    }
}

size_t quantize_turbo4_0(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    (void)imatrix;
    size_t row_size = (n_per_row / QK_TQ) * sizeof(block_turbo4_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo4_0_ref(src + row * n_per_row,
                                  (block_turbo4_0 *)((char *)dst + row * row_size),
                                  n_per_row);
    }
    return nrows * row_size;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  turbo3_0 — 3-bit TurboQuant
 * ═══════════════════════════════════════════════════════════════════════ */

void quantize_row_turbo3_0_ref(const float * x, block_turbo3_0 * y, int64_t k) {
    assert(k % QK_TQ == 0);
    const int64_t nb = k / QK_TQ;
    uint8_t indices[32];
    for (int64_t i = 0; i < nb; i++) {
        quantize_block_turbo(x + i*32, indices, &y[i].d,
                            LM3_CENTROIDS, LM3_BOUNDARIES, 8);
        pack_3bit(indices, y[i].qs);
    }
}

void dequantize_row_turbo3_0(const block_turbo3_0 * x, float * y, int64_t k) {
    assert(k % QK_TQ == 0);
    const int64_t nb = k / QK_TQ;
    uint8_t indices[32];
    for (int64_t i = 0; i < nb; i++) {
        unpack_3bit(x[i].qs, indices);
        dequantize_block_turbo(indices, x[i].d, y + i*32, LM3_CENTROIDS);
    }
}

size_t quantize_turbo3_0(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    (void)imatrix;
    size_t row_size = (n_per_row / QK_TQ) * sizeof(block_turbo3_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo3_0_ref(src + row * n_per_row,
                                  (block_turbo3_0 *)((char *)dst + row * row_size),
                                  n_per_row);
    }
    return nrows * row_size;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  turbo2_0 — 2-bit TurboQuant
 * ═══════════════════════════════════════════════════════════════════════ */

void quantize_row_turbo2_0_ref(const float * x, block_turbo2_0 * y, int64_t k) {
    assert(k % QK_TQ == 0);
    const int64_t nb = k / QK_TQ;
    uint8_t indices[32];
    for (int64_t i = 0; i < nb; i++) {
        quantize_block_turbo(x + i*32, indices, &y[i].d,
                            LM2_CENTROIDS, LM2_BOUNDARIES, 4);
        pack_2bit(indices, y[i].qs);
    }
}

void dequantize_row_turbo2_0(const block_turbo2_0 * x, float * y, int64_t k) {
    assert(k % QK_TQ == 0);
    const int64_t nb = k / QK_TQ;
    uint8_t indices[32];
    for (int64_t i = 0; i < nb; i++) {
        unpack_2bit(x[i].qs, indices);
        dequantize_block_turbo(indices, x[i].d, y + i*32, LM2_CENTROIDS);
    }
    return;
}

size_t quantize_turbo2_0(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    (void)imatrix;
    size_t row_size = (n_per_row / QK_TQ) * sizeof(block_turbo2_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo2_0_ref(src + row * n_per_row,
                                  (block_turbo2_0 *)((char *)dst + row * row_size),
                                  n_per_row);
    }
    return nrows * row_size;
}
