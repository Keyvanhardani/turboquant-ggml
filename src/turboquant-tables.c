/*
 * turboquant-tables.c — Precomputed Lloyd-Max codebooks for TurboQuant
 *
 * After WHT rotation of a unit vector in R^d, each coordinate follows
 * a distribution approximating N(0, 1/sqrt(d)). The Lloyd-Max algorithm
 * finds the MSE-optimal scalar quantizer for this distribution.
 *
 * Values are precomputed for standard N(0,1) and scaled by 1/sqrt(dim).
 */

#include "turboquant.h"
#include <math.h>

/* ─── Standard N(0,1) Lloyd-Max codebooks ────────────────────────────── */

/* 2-bit (4 levels) Lloyd-Max for N(0,1) */
static const float lm_centroids_2bit[] = {
    -1.5104f, -0.4528f, 0.4528f, 1.5104f
};
static const float lm_boundaries_2bit[] = {
    -0.9816f, 0.0f, 0.9816f
};

/* 3-bit (8 levels) Lloyd-Max for N(0,1) */
static const float lm_centroids_3bit[] = {
    -2.1519f, -1.3439f, -0.7560f, -0.2451f,
     0.2451f,  0.7560f,  1.3439f,  2.1519f
};
static const float lm_boundaries_3bit[] = {
    -1.7479f, -1.0500f, -0.5006f, 0.0f,
     0.5006f,  1.0500f,  1.7479f
};

/* 4-bit (16 levels) Lloyd-Max for N(0,1) */
static const float lm_centroids_4bit[] = {
    -2.7326f, -2.0690f, -1.6180f, -1.2562f,
    -0.9424f, -0.6568f, -0.3881f, -0.1284f,
     0.1284f,  0.3881f,  0.6568f,  0.9424f,
     1.2562f,  1.6180f,  2.0690f,  2.7326f
};
static const float lm_boundaries_4bit[] = {
    -2.4008f, -1.8435f, -1.4370f, -1.0993f,
    -0.7996f, -0.5224f, -0.2582f,  0.0f,
     0.2582f,  0.5224f,  0.7996f,  1.0993f,
     1.4370f,  1.8435f,  2.4008f
};

/* ─── Scaled codebooks per dimension ─────────────────────────────────── */
/* We store them lazily: scale = 1/sqrt(dim) applied at lookup time */

/* Supported dimensions */
#define TQ_NUM_DIMS 4
static const int tq_supported_dims[TQ_NUM_DIMS] = { 32, 64, 128, 256 };

/* Scaled codebook storage (filled on first access) */
#define TQ_MAX_LEVELS 16
#define TQ_MAX_BOUNDARIES 15

typedef struct {
    int   initialized;
    float centroids[TQ_MAX_LEVELS];
    float boundaries[TQ_MAX_BOUNDARIES];
} tq_scaled_cb;

/* [bits_idx][dim_idx] where bits_idx: 0=2bit, 1=3bit, 2=4bit */
static tq_scaled_cb scaled_cbs[3][TQ_NUM_DIMS];

static tq_codebook codebook_result;

static int find_dim_idx(int dim) {
    for (int i = 0; i < TQ_NUM_DIMS; i++) {
        if (tq_supported_dims[i] == dim) return i;
    }
    return -1;
}

static void init_scaled_cb(int bits_idx, int dim_idx) {
    tq_scaled_cb *cb = &scaled_cbs[bits_idx][dim_idx];
    if (cb->initialized) return;

    float scale = 1.0f / sqrtf((float)tq_supported_dims[dim_idx]);

    const float *src_c;
    const float *src_b;
    int n_levels;
    int n_boundaries;

    switch (bits_idx) {
        case 0:
            src_c = lm_centroids_2bit;
            src_b = lm_boundaries_2bit;
            n_levels = 4;
            n_boundaries = 3;
            break;
        case 1:
            src_c = lm_centroids_3bit;
            src_b = lm_boundaries_3bit;
            n_levels = 8;
            n_boundaries = 7;
            break;
        case 2:
            src_c = lm_centroids_4bit;
            src_b = lm_boundaries_4bit;
            n_levels = 16;
            n_boundaries = 15;
            break;
        default:
            return;
    }

    for (int i = 0; i < n_levels; i++) {
        cb->centroids[i] = src_c[i] * scale;
    }
    for (int i = 0; i < n_boundaries; i++) {
        cb->boundaries[i] = src_b[i] * scale;
    }

    cb->initialized = 1;
}

const tq_codebook *tq_get_codebook(int bits, int dim) {
    int bits_idx;
    switch (bits) {
        case 2: bits_idx = 0; break;
        case 3: bits_idx = 1; break;
        case 4: bits_idx = 2; break;
        default: return NULL;
    }

    int dim_idx = find_dim_idx(dim);
    if (dim_idx < 0) return NULL;

    init_scaled_cb(bits_idx, dim_idx);

    tq_scaled_cb *cb = &scaled_cbs[bits_idx][dim_idx];

    int n_levels;
    int n_boundaries;
    switch (bits) {
        case 2: n_levels = 4;  n_boundaries = 3;  break;
        case 3: n_levels = 8;  n_boundaries = 7;  break;
        case 4: n_levels = 16; n_boundaries = 15; break;
        default: return NULL;
    }

    codebook_result.n_levels   = n_levels;
    codebook_result.dim        = dim;
    codebook_result.centroids  = cb->centroids;
    codebook_result.boundaries = cb->boundaries;

    return &codebook_result;
}
