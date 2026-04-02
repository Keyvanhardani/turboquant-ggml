/*
 * turboquant.c — Core TurboQuant algorithm
 *
 * WHT rotation + Lloyd-Max optimal scalar quantization + bit-packing.
 * Pure C, no dependencies beyond math.h and string.h.
 */

#include "turboquant.h"
#include <math.h>
#include <string.h>

/* ─── FP16 conversion helpers ────────────────────────────────────────── */

static uint16_t fp32_to_fp16(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));

    uint32_t sign = (u >> 16) & 0x8000;
    int32_t  exp  = ((u >> 23) & 0xFF) - 127;
    uint32_t mant = u & 0x7FFFFF;

    if (exp > 15) {
        return (uint16_t)(sign | 0x7C00); /* inf */
    } else if (exp < -14) {
        return (uint16_t)sign; /* zero / denorm */
    }

    return (uint16_t)(sign | ((exp + 15) << 10) | (mant >> 13));
}

static float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x03FF;

    uint32_t u;
    if (exp == 0) {
        if (mant == 0) {
            u = sign;
        } else {
            /* denormalized */
            exp = 1;
            while (!(mant & 0x0400)) { mant <<= 1; exp--; }
            mant &= 0x03FF;
            u = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        u = sign | 0x7F800000 | (mant << 13); /* inf/nan */
    } else {
        u = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }

    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

/* ─── Vector utilities ───────────────────────────────────────────────── */

float tq_vec_norm(const float *x, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += x[i] * x[i];
    }
    return sqrtf(sum);
}

float tq_mse(const float *a, const float *b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum / (float)n;
}

float tq_cosine_sim(const float *a, const float *b, int n) {
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (int i = 0; i < n; i++) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    if (na < 1e-12f || nb < 1e-12f) return 0.0f;
    return dot / (sqrtf(na) * sqrtf(nb));
}

/* ─── Walsh-Hadamard Transform ───────────────────────────────────────── */

void tq_wht(float *x, int n) {
    /* Fast in-place WHT, O(n log n) */
    for (int len = 1; len < n; len <<= 1) {
        for (int i = 0; i < n; i += len << 1) {
            for (int j = 0; j < len; j++) {
                float u = x[i + j];
                float v = x[i + j + len];
                x[i + j]       = u + v;
                x[i + j + len] = u - v;
            }
        }
    }
    /* Normalize: 1/sqrt(n) makes WHT self-inverse */
    float scale = 1.0f / sqrtf((float)n);
    for (int i = 0; i < n; i++) {
        x[i] *= scale;
    }
}

/* ─── Scalar quantization ────────────────────────────────────────────── */

/* Find the nearest centroid index for a single value using binary search
 * on interior boundaries. boundaries has (n_levels - 1) elements. */
static inline int quantize_scalar(float val, const float *boundaries, int n_levels) {
    int lo = 0, hi = n_levels - 2;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        if (val < boundaries[mid]) {
            hi = mid - 1;
        } else {
            lo = mid + 1;
        }
    }
    return lo;
}

/* ─── 3-bit packing: 32 indices (0..7) → 12 bytes ────────────────────── */
/* Pack 8 x 3-bit values into 3 bytes, repeat 4 times for 32 elements */

static void pack_3bit(const uint8_t *indices, uint8_t *packed) {
    for (int g = 0; g < 4; g++) {
        const uint8_t *s = indices + g * 8;
        uint8_t *d = packed + g * 3;

        /* byte 0: [idx0(3)] [idx1(3)] [idx2_lo(2)] */
        d[0] = (uint8_t)(s[0] | (s[1] << 3) | (s[2] << 6));
        /* byte 1: [idx2_hi(1)] [idx3(3)] [idx4(3)] [idx5_lo(1)] */
        d[1] = (uint8_t)((s[2] >> 2) | (s[3] << 1) | (s[4] << 4) | (s[5] << 7));
        /* byte 2: [idx5_hi(2)] [idx6(3)] [idx7(3)] */
        d[2] = (uint8_t)((s[5] >> 1) | (s[6] << 2) | (s[7] << 5));
    }
}

static void unpack_3bit(const uint8_t *packed, uint8_t *indices) {
    for (int g = 0; g < 4; g++) {
        const uint8_t *s = packed + g * 3;
        uint8_t *d = indices + g * 8;

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

/* ─── 4-bit packing: 32 indices (0..15) → 16 bytes ───────────────────── */

static void pack_4bit(const uint8_t *indices, uint8_t *packed) {
    for (int i = 0; i < 16; i++) {
        packed[i] = (uint8_t)(indices[2*i] | (indices[2*i + 1] << 4));
    }
}

static void unpack_4bit(const uint8_t *packed, uint8_t *indices) {
    for (int i = 0; i < 16; i++) {
        indices[2*i]     = packed[i] & 0x0F;
        indices[2*i + 1] = (packed[i] >> 4) & 0x0F;
    }
}

/* ─── 2-bit packing: 32 indices (0..3) → 8 bytes ─────────────────────── */

static void pack_2bit(const uint8_t *indices, uint8_t *packed) {
    for (int i = 0; i < 8; i++) {
        packed[i] = (uint8_t)(
            indices[4*i]           |
            (indices[4*i + 1] << 2) |
            (indices[4*i + 2] << 4) |
            (indices[4*i + 3] << 6)
        );
    }
}

static void unpack_2bit(const uint8_t *packed, uint8_t *indices) {
    for (int i = 0; i < 8; i++) {
        indices[4*i]     =  packed[i]       & 0x03;
        indices[4*i + 1] = (packed[i] >> 2) & 0x03;
        indices[4*i + 2] = (packed[i] >> 4) & 0x03;
        indices[4*i + 3] = (packed[i] >> 6) & 0x03;
    }
}

/* ─── Generic quantize block ─────────────────────────────────────────── */

static void quantize_block(const float *src, float *work,
                           uint8_t *indices, uint16_t *norm_out,
                           const tq_codebook *cb) {
    /* 1. Compute L2 norm */
    float norm = tq_vec_norm(src, TQ_BLOCK_SIZE);
    *norm_out = fp32_to_fp16(norm);

    /* 2. Normalize + copy to work buffer */
    float inv_norm = (norm > 1e-12f) ? (1.0f / norm) : 0.0f;
    for (int i = 0; i < TQ_BLOCK_SIZE; i++) {
        work[i] = src[i] * inv_norm;
    }

    /* 3. WHT rotation (in-place on work buffer) */
    tq_wht(work, TQ_BLOCK_SIZE);

    /* 4. Scalar quantization */
    for (int i = 0; i < TQ_BLOCK_SIZE; i++) {
        indices[i] = (uint8_t)quantize_scalar(work[i], cb->boundaries, cb->n_levels);
    }
}

static void dequantize_block(const uint8_t *indices, uint16_t norm_fp16,
                             float *dst, const tq_codebook *cb) {
    float norm = fp16_to_fp32(norm_fp16);

    /* 1. Codebook lookup */
    for (int i = 0; i < TQ_BLOCK_SIZE; i++) {
        dst[i] = cb->centroids[indices[i]];
    }

    /* 2. Inverse WHT (WHT is self-inverse with normalization) */
    tq_wht(dst, TQ_BLOCK_SIZE);

    /* 3. Scale by norm */
    for (int i = 0; i < TQ_BLOCK_SIZE; i++) {
        dst[i] *= norm;
    }
}

/* ─── TQ3_0: 3-bit quantize/dequantize ───────────────────────────────── */

void quantize_row_tq3_0(const float *src, void *dst, int64_t n, int head_dim) {
    /* Codebook is scaled by 1/sqrt(block_size) since WHT operates on blocks */
    (void)head_dim;
    const tq_codebook *cb = tq_get_codebook(3, TQ_BLOCK_SIZE);

    block_tq3_0 *blocks = (block_tq3_0 *)dst;
    int64_t nb = n / TQ_BLOCK_SIZE;

    float work[TQ_BLOCK_SIZE];
    uint8_t indices[TQ_BLOCK_SIZE];

    for (int64_t b = 0; b < nb; b++) {
        quantize_block(src + b * TQ_BLOCK_SIZE, work, indices, &blocks[b].d, cb);
        pack_3bit(indices, blocks[b].qs);
    }
}

void dequantize_row_tq3_0(const void *src, float *dst, int64_t n, int head_dim) {
    (void)head_dim;
    const tq_codebook *cb = tq_get_codebook(3, TQ_BLOCK_SIZE);

    const block_tq3_0 *blocks = (const block_tq3_0 *)src;
    int64_t nb = n / TQ_BLOCK_SIZE;

    uint8_t indices[TQ_BLOCK_SIZE];

    for (int64_t b = 0; b < nb; b++) {
        unpack_3bit(blocks[b].qs, indices);
        dequantize_block(indices, blocks[b].d, dst + b * TQ_BLOCK_SIZE, cb);
    }
}

/* ─── TQ4_0: 4-bit quantize/dequantize ───────────────────────────────── */

void quantize_row_tq4_0(const float *src, void *dst, int64_t n, int head_dim) {
    (void)head_dim;
    const tq_codebook *cb = tq_get_codebook(4, TQ_BLOCK_SIZE);

    block_tq4_0 *blocks = (block_tq4_0 *)dst;
    int64_t nb = n / TQ_BLOCK_SIZE;

    float work[TQ_BLOCK_SIZE];
    uint8_t indices[TQ_BLOCK_SIZE];

    for (int64_t b = 0; b < nb; b++) {
        quantize_block(src + b * TQ_BLOCK_SIZE, work, indices, &blocks[b].d, cb);
        pack_4bit(indices, blocks[b].qs);
    }
}

void dequantize_row_tq4_0(const void *src, float *dst, int64_t n, int head_dim) {
    (void)head_dim;
    const tq_codebook *cb = tq_get_codebook(4, TQ_BLOCK_SIZE);

    const block_tq4_0 *blocks = (const block_tq4_0 *)src;
    int64_t nb = n / TQ_BLOCK_SIZE;

    uint8_t indices[TQ_BLOCK_SIZE];

    for (int64_t b = 0; b < nb; b++) {
        unpack_4bit(blocks[b].qs, indices);
        dequantize_block(indices, blocks[b].d, dst + b * TQ_BLOCK_SIZE, cb);
    }
}

/* ─── TQ2_0: 2-bit quantize/dequantize ───────────────────────────────── */

void quantize_row_tq2_0(const float *src, void *dst, int64_t n, int head_dim) {
    (void)head_dim;
    const tq_codebook *cb = tq_get_codebook(2, TQ_BLOCK_SIZE);

    block_tq2_0 *blocks = (block_tq2_0 *)dst;
    int64_t nb = n / TQ_BLOCK_SIZE;

    float work[TQ_BLOCK_SIZE];
    uint8_t indices[TQ_BLOCK_SIZE];

    for (int64_t b = 0; b < nb; b++) {
        quantize_block(src + b * TQ_BLOCK_SIZE, work, indices, &blocks[b].d, cb);
        pack_2bit(indices, blocks[b].qs);
    }
}

void dequantize_row_tq2_0(const void *src, float *dst, int64_t n, int head_dim) {
    (void)head_dim;
    const tq_codebook *cb = tq_get_codebook(2, TQ_BLOCK_SIZE);

    const block_tq2_0 *blocks = (const block_tq2_0 *)src;
    int64_t nb = n / TQ_BLOCK_SIZE;

    uint8_t indices[TQ_BLOCK_SIZE];

    for (int64_t b = 0; b < nb; b++) {
        unpack_2bit(blocks[b].qs, indices);
        dequantize_block(indices, blocks[b].d, dst + b * TQ_BLOCK_SIZE, cb);
    }
}
