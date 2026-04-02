/*
 * turboquant.h — TurboQuant GGML-compatible KV cache quantization
 *
 * First real integration of Google's TurboQuant (ICLR 2026, arXiv:2504.19874)
 * for llama.cpp / ollama.
 *
 * Algorithm: WHT rotation + Lloyd-Max optimal scalar quantization
 * Based on community findings: WHT >> random rotation, MSE-only >> QJL,
 * block_size=32 optimal for flash attention.
 *
 * MIT License — Keyvan Hardani
 */

#ifndef TURBOQUANT_H
#define TURBOQUANT_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Block size ─────────────────────────────────────────────────────── */

#define TQ_BLOCK_SIZE 32

/* ─── Block structs (GGML-compatible) ────────────────────────────────── */

/*
 * TQ3_0: 3-bit TurboQuant
 *   - 2 bytes: fp16 L2 norm (scale)
 *   - 12 bytes: 32 x 3-bit packed indices
 *   Total: 14 bytes / 32 elements = 3.5 bits/element
 */
typedef struct {
    uint16_t d;                     /* fp16 L2 norm */
    uint8_t  qs[12];               /* 32 x 3-bit packed */
} block_tq3_0;

/*
 * TQ4_0: 4-bit TurboQuant
 *   - 2 bytes: fp16 L2 norm (scale)
 *   - 16 bytes: 32 x 4-bit packed indices
 *   Total: 18 bytes / 32 elements = 4.5 bits/element
 */
typedef struct {
    uint16_t d;                     /* fp16 L2 norm */
    uint8_t  qs[16];               /* 32 x 4-bit packed */
} block_tq4_0;

/*
 * TQ2_0: 2-bit TurboQuant (aggressive, for values)
 *   - 2 bytes: fp16 L2 norm (scale)
 *   - 8 bytes: 32 x 2-bit packed indices
 *   Total: 10 bytes / 32 elements = 2.5 bits/element
 */
typedef struct {
    uint16_t d;                     /* fp16 L2 norm */
    uint8_t  qs[8];                /* 32 x 2-bit packed */
} block_tq2_0;

/* ─── Codebook structure ─────────────────────────────────────────────── */

typedef struct {
    int         n_levels;           /* 2^bits */
    int         dim;                /* head dimension (for scaling) */
    const float *centroids;         /* n_levels centroids */
    const float *boundaries;        /* n_levels - 1 interior boundaries */
} tq_codebook;

/* ─── Codebook access ────────────────────────────────────────────────── */

/* Get precomputed Lloyd-Max codebook for given bits and head dimension.
 * Supported: bits={2,3,4}, dim={64,128,256}
 * Returns NULL if combination is unsupported. */
const tq_codebook *tq_get_codebook(int bits, int dim);

/* ─── Walsh-Hadamard Transform ───────────────────────────────────────── */

/* In-place normalized WHT. n must be power of 2, n >= 2.
 * Applies 1/sqrt(n) normalization (self-inverse). */
void tq_wht(float *x, int n);

/* ─── Quantize / Dequantize — GGML-compatible signatures ─────────────── */

/* Quantize n floats from src into dst blocks.
 * n must be a multiple of TQ_BLOCK_SIZE.
 * head_dim is the model's head dimension (for codebook selection). */
void quantize_row_tq2_0(const float *src, void *dst, int64_t n, int head_dim);
void quantize_row_tq3_0(const float *src, void *dst, int64_t n, int head_dim);
void quantize_row_tq4_0(const float *src, void *dst, int64_t n, int head_dim);

/* Dequantize n elements from src blocks into dst floats.
 * n must be a multiple of TQ_BLOCK_SIZE. */
void dequantize_row_tq2_0(const void *src, float *dst, int64_t n, int head_dim);
void dequantize_row_tq3_0(const void *src, float *dst, int64_t n, int head_dim);
void dequantize_row_tq4_0(const void *src, float *dst, int64_t n, int head_dim);

/* ─── Utility ────────────────────────────────────────────────────────── */

/* Compute L2 norm of a vector */
float tq_vec_norm(const float *x, int n);

/* Compute MSE between two vectors */
float tq_mse(const float *a, const float *b, int n);

/* Compute cosine similarity between two vectors */
float tq_cosine_sim(const float *a, const float *b, int n);

/* ─── Block sizes for GGML registration ──────────────────────────────── */

static inline int tq2_0_block_size(void) { return TQ_BLOCK_SIZE; }
static inline int tq3_0_block_size(void) { return TQ_BLOCK_SIZE; }
static inline int tq4_0_block_size(void) { return TQ_BLOCK_SIZE; }

static inline int tq2_0_type_size(void) { return (int)sizeof(block_tq2_0); }
static inline int tq3_0_type_size(void) { return (int)sizeof(block_tq3_0); }
static inline int tq4_0_type_size(void) { return (int)sizeof(block_tq4_0); }

/* ─── Head dimension detection (AmesianX P1→P5 cascade) ──────────────── */

/* Validate head_dim is power-of-2 and supported.
 * Returns 1 if valid, 0 if unsupported (e.g., head_dim=80). */
static inline int tq_head_dim_valid(int head_dim) {
    return (head_dim == 32 || head_dim == 64 ||
            head_dim == 128 || head_dim == 256);
}

/* ─── Asymmetric K/V recommendation (scos-lab findings) ──────────────── */

/* Recommend bits for K and V based on K/V norm ratio.
 * K/V < 10x  → 3-bit K, 2-bit V
 * K/V 10-60x → 4-bit K, 3-bit V
 * K/V > 100x → 4-bit K, 3-bit V (or mixed precision) */
typedef struct {
    int k_bits;
    int v_bits;
} tq_kv_config;

static inline tq_kv_config tq_recommend_kv_bits(float k_norm, float v_norm) {
    tq_kv_config cfg;
    float ratio = (v_norm > 1e-6f) ? (k_norm / v_norm) : 1.0f;
    if (ratio < 10.0f)       { cfg.k_bits = 3; cfg.v_bits = 2; }
    else if (ratio < 60.0f)  { cfg.k_bits = 4; cfg.v_bits = 3; }
    else                     { cfg.k_bits = 4; cfg.v_bits = 3; }
    return cfg;
}

/* ─── Compression ratios ─────────────────────────────────────────────── */

/* Effective bits per element including norm overhead */
static inline float tq_bits_per_elem(int bits) {
    switch (bits) {
        case 2: return 2.5f;  /* 10 bytes / 32 elem */
        case 3: return 3.5f;  /* 14 bytes / 32 elem */
        case 4: return 4.5f;  /* 18 bytes / 32 elem */
        default: return 0.0f;
    }
}

/* Compression ratio vs FP16 */
static inline float tq_compression_ratio(int bits) {
    float bpe = tq_bits_per_elem(bits);
    return (bpe > 0.0f) ? (16.0f / bpe) : 0.0f;
}

#ifdef __cplusplus
}
#endif

#endif /* TURBOQUANT_H */
