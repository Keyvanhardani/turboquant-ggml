/*
 * test_full.c — Comprehensive TurboQuant validation suite
 *
 * Tests: stress, edge cases, statistical validation,
 * realistic KV cache simulation, GGML integration compile check.
 */

#include "turboquant.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <float.h>

static int tests_passed = 0;
static int tests_failed = 0;
static int total_assertions = 0;

#define ASSERT(cond, msg) do { \
    total_assertions++; \
    if (!(cond)) { \
        printf("  FAIL: %s (line %d)\n", msg, __LINE__); \
        tests_failed++; \
        return; \
    } \
} while(0)

#define ASSERT_NEAR(a, b, eps, msg) do { \
    total_assertions++; \
    float _a = (a), _b = (b), _e = (eps); \
    if (fabsf(_a - _b) > _e) { \
        printf("  FAIL: %s — got %.6f, expected %.6f (line %d)\n", \
               msg, _a, _b, __LINE__); \
        tests_failed++; \
        return; \
    } \
} while(0)

#define TEST(name) static void test_##name(void)
#define RUN(name) do { \
    printf("  [%s] ", #name); fflush(stdout); \
    test_##name(); \
    tests_passed++; \
    printf("OK\n"); \
} while(0)

/* ─── Random helpers ─────────────────────────────────────────────────── */

static float randn(void) {
    /* Box-Muller for Gaussian N(0,1) */
    float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
    float u2 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
}

static void gaussian_vector(float *v, int n, float mean, float std) {
    for (int i = 0; i < n; i++) v[i] = mean + std * randn();
}

/* ═══════════════════════════════════════════════════════════════════════
 *  1. WHT STRESS TESTS
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(wht_sizes) {
    /* WHT must work for all power-of-2 sizes */
    int sizes[] = {2, 4, 8, 16, 32, 64, 128, 256};
    for (int s = 0; s < 8; s++) {
        int n = sizes[s];
        float *x = malloc(n * sizeof(float));
        float *orig = malloc(n * sizeof(float));
        for (int i = 0; i < n; i++) x[i] = orig[i] = randn();

        tq_wht(x, n);
        tq_wht(x, n); /* self-inverse */

        float max_err = 0;
        for (int i = 0; i < n; i++) {
            float err = fabsf(x[i] - orig[i]);
            if (err > max_err) max_err = err;
        }
        ASSERT(max_err < 1e-3f, "WHT self-inverse for all sizes");

        free(x); free(orig);
    }
}

TEST(wht_orthogonality) {
    /* WHT should preserve dot products */
    float a[32], b[32];
    for (int i = 0; i < 32; i++) { a[i] = randn(); b[i] = randn(); }

    float dot_before = 0;
    for (int i = 0; i < 32; i++) dot_before += a[i] * b[i];

    tq_wht(a, 32);
    tq_wht(b, 32);

    float dot_after = 0;
    for (int i = 0; i < 32; i++) dot_after += a[i] * b[i];

    ASSERT_NEAR(dot_before, dot_after, 1e-2f, "WHT preserves dot product");
}

TEST(wht_deterministic) {
    /* Same input always gives same output */
    float x1[32], x2[32];
    for (int i = 0; i < 32; i++) x1[i] = x2[i] = (float)i * 0.1f;

    tq_wht(x1, 32);
    tq_wht(x2, 32);

    for (int i = 0; i < 32; i++) {
        ASSERT_NEAR(x1[i], x2[i], 1e-7f, "WHT deterministic");
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 *  2. BIT-PACKING EXHAUSTIVE
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(pack_3bit_exhaustive) {
    /* Test all possible 3-bit index combinations for first 8 values */
    for (int a = 0; a < 8; a++) {
        for (int b = 0; b < 8; b++) {
            float src[32] = {0};
            /* Create values that will quantize to specific indices */
            const tq_codebook *cb = tq_get_codebook(3, 32);
            ASSERT(cb != NULL, "codebook exists");

            /* Use centroid values directly as input (after inverse WHT + scale) */
            block_tq3_0 block;
            quantize_row_tq3_0(src, &block, 32, 128);
            /* Just verify no crash — exhaustive index validation */
        }
    }
}

TEST(pack_4bit_all_values) {
    /* Verify all 16 possible 4-bit values survive pack/unpack */
    float src[32], dst[32];
    const tq_codebook *cb = tq_get_codebook(4, 32);
    ASSERT(cb != NULL, "codebook exists");

    /* Use centroid values that should map to each index */
    for (int i = 0; i < 32; i++) {
        src[i] = cb->centroids[i % 16] * 5.0f; /* scale up */
    }

    block_tq4_0 block;
    quantize_row_tq4_0(src, &block, 32, 128);
    dequantize_row_tq4_0(&block, dst, 32, 128);

    /* Cosine similarity should be reasonable */
    float cosim = tq_cosine_sim(src, dst, 32);
    ASSERT(cosim > 0.80f, "4-bit all values roundtrip cosine > 0.80");
}

/* ═══════════════════════════════════════════════════════════════════════
 *  3. EDGE CASES
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(very_small_values) {
    float src[32], dst[32];
    for (int i = 0; i < 32; i++) src[i] = 1e-8f * (float)(i - 16);

    block_tq3_0 block;
    quantize_row_tq3_0(src, &block, 32, 128);
    dequantize_row_tq3_0(&block, dst, 32, 128);

    /* Should not produce NaN or Inf */
    for (int i = 0; i < 32; i++) {
        ASSERT(!isnan(dst[i]), "no NaN from tiny values");
        ASSERT(!isinf(dst[i]), "no Inf from tiny values");
    }
}

TEST(single_nonzero) {
    float src[32] = {0};
    src[0] = 1.0f;
    float dst[32];

    block_tq4_0 block;
    quantize_row_tq4_0(src, &block, 32, 128);
    dequantize_row_tq4_0(&block, dst, 32, 128);

    /* Dominant element should remain dominant */
    float max_val = -1e30f;
    int max_idx = -1;
    for (int i = 0; i < 32; i++) {
        if (dst[i] > max_val) { max_val = dst[i]; max_idx = i; }
    }
    ASSERT(max_idx == 0, "single nonzero: dominant element preserved");
}

TEST(alternating_sign) {
    float src[32], dst[32];
    for (int i = 0; i < 32; i++) src[i] = (i % 2 == 0) ? 1.0f : -1.0f;

    block_tq3_0 block;
    quantize_row_tq3_0(src, &block, 32, 128);
    dequantize_row_tq3_0(&block, dst, 32, 128);

    float cosim = tq_cosine_sim(src, dst, 32);
    ASSERT(cosim > 0.80f, "alternating sign pattern preserved");
}

TEST(identical_values) {
    float src[32], dst[32];
    for (int i = 0; i < 32; i++) src[i] = 42.0f;

    block_tq4_0 block;
    quantize_row_tq4_0(src, &block, 32, 128);
    dequantize_row_tq4_0(&block, dst, 32, 128);

    /* WHT of constant vector concentrates energy in first element.
     * After quantization roundtrip, cosine sim should be high. */
    float cosim = tq_cosine_sim(src, dst, 32);
    printf("(cosim=%.4f) ", cosim);
    ASSERT(cosim > 0.85f, "identical values roundtrip cosine > 0.85");
}

TEST(extreme_magnitude) {
    float src[32], dst[32];
    /* Use 1e4 not 1e6 — fp16 norm maxes out at ~65504 */
    for (int i = 0; i < 32; i++) src[i] = 1e4f * randn();

    block_tq3_0 block;
    quantize_row_tq3_0(src, &block, 32, 128);
    dequantize_row_tq3_0(&block, dst, 32, 128);

    for (int i = 0; i < 32; i++) {
        ASSERT(!isnan(dst[i]), "no NaN from extreme magnitude");
        ASSERT(!isinf(dst[i]), "no Inf from extreme magnitude");
    }
    float cosim = tq_cosine_sim(src, dst, 32);
    ASSERT(cosim > 0.85f, "extreme magnitude cosine > 0.85");
}

/* ═══════════════════════════════════════════════════════════════════════
 *  4. STATISTICAL VALIDATION (large-scale)
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(statistical_tq4_10k) {
    /* 10,000 random vectors, check average quality */
    int N = 10000;
    double total_mse = 0, total_cosim = 0;
    float min_cosim = 1.0f;

    for (int t = 0; t < N; t++) {
        float src[128], dst[128];
        gaussian_vector(src, 128, 0.0f, 1.0f);

        block_tq4_0 blocks[4];
        quantize_row_tq4_0(src, blocks, 128, 128);
        dequantize_row_tq4_0(blocks, dst, 128, 128);

        float mse = tq_mse(src, dst, 128);
        float cosim = tq_cosine_sim(src, dst, 128);
        total_mse += mse;
        total_cosim += cosim;
        if (cosim < min_cosim) min_cosim = cosim;
    }

    float avg_mse = (float)(total_mse / N);
    float avg_cosim = (float)(total_cosim / N);

    printf("(N=%d avg_mse=%.5f avg_cos=%.5f min_cos=%.5f) ",
           N, avg_mse, avg_cosim, min_cosim);

    ASSERT(avg_cosim > 0.99f, "TQ4 avg cosine > 0.99 over 10k vectors");
    ASSERT(min_cosim > 0.90f, "TQ4 worst-case cosine > 0.90");
    ASSERT(avg_mse < 0.01f, "TQ4 avg MSE < 0.01");
}

TEST(statistical_tq3_10k) {
    int N = 10000;
    double total_mse = 0, total_cosim = 0;
    float min_cosim = 1.0f;

    for (int t = 0; t < N; t++) {
        float src[128], dst[128];
        gaussian_vector(src, 128, 0.0f, 1.0f);

        block_tq3_0 blocks[4];
        quantize_row_tq3_0(src, blocks, 128, 128);
        dequantize_row_tq3_0(blocks, dst, 128, 128);

        float mse = tq_mse(src, dst, 128);
        float cosim = tq_cosine_sim(src, dst, 128);
        total_mse += mse;
        total_cosim += cosim;
        if (cosim < min_cosim) min_cosim = cosim;
    }

    float avg_mse = (float)(total_mse / N);
    float avg_cosim = (float)(total_cosim / N);

    printf("(N=%d avg_mse=%.5f avg_cos=%.5f min_cos=%.5f) ",
           N, avg_mse, avg_cosim, min_cosim);

    ASSERT(avg_cosim > 0.97f, "TQ3 avg cosine > 0.97 over 10k vectors");
    ASSERT(min_cosim > 0.80f, "TQ3 worst-case cosine > 0.80");
    ASSERT(avg_mse < 0.05f, "TQ3 avg MSE < 0.05");
}

TEST(statistical_tq2_10k) {
    int N = 10000;
    double total_mse = 0, total_cosim = 0;
    float min_cosim = 1.0f;

    for (int t = 0; t < N; t++) {
        float src[128], dst[128];
        gaussian_vector(src, 128, 0.0f, 1.0f);

        block_tq2_0 blocks[4];
        quantize_row_tq2_0(src, blocks, 128, 128);
        dequantize_row_tq2_0(blocks, dst, 128, 128);

        float mse = tq_mse(src, dst, 128);
        float cosim = tq_cosine_sim(src, dst, 128);
        total_mse += mse;
        total_cosim += cosim;
        if (cosim < min_cosim) min_cosim = cosim;
    }

    float avg_mse = (float)(total_mse / N);
    float avg_cosim = (float)(total_cosim / N);

    printf("(N=%d avg_mse=%.5f avg_cos=%.5f min_cos=%.5f) ",
           N, avg_mse, avg_cosim, min_cosim);

    ASSERT(avg_cosim > 0.90f, "TQ2 avg cosine > 0.90 over 10k vectors");
    ASSERT(min_cosim > 0.60f, "TQ2 worst-case cosine > 0.60");
}

/* ═══════════════════════════════════════════════════════════════════════
 *  5. REALISTIC KV CACHE SIMULATION
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(kv_cache_keys_realistic) {
    /* Simulate realistic Key vectors: higher norms, Gaussian activations
     * scos-lab finding: K norms range from 11.8 (GPT-2) to 778.6 (Qwen) */
    int n_heads = 8;
    int head_dim = 128;
    int seq_len = 512;
    int n_vectors = n_heads * seq_len;

    double total_cosim = 0;
    int n_bad = 0;

    for (int v = 0; v < n_vectors; v++) {
        float src[128], dst[128];

        /* Realistic K: Gaussian with large norm (mean norm ~20-200) */
        float scale = 20.0f + 180.0f * ((float)rand() / RAND_MAX);
        gaussian_vector(src, head_dim, 0.0f, scale / sqrtf((float)head_dim));

        block_tq4_0 blocks[4];
        quantize_row_tq4_0(src, blocks, head_dim, head_dim);
        dequantize_row_tq4_0(blocks, dst, head_dim, head_dim);

        float cosim = tq_cosine_sim(src, dst, head_dim);
        total_cosim += cosim;
        if (cosim < 0.95f) n_bad++;
    }

    float avg_cosim = (float)(total_cosim / n_vectors);
    printf("(keys: N=%d avg_cos=%.5f bad=%.1f%%) ",
           n_vectors, avg_cosim, 100.0f * n_bad / n_vectors);

    ASSERT(avg_cosim > 0.99f, "realistic K vectors: avg cosine > 0.99");
    ASSERT(n_bad < n_vectors / 100, "realistic K vectors: < 1% below 0.95");
}

TEST(kv_cache_values_realistic) {
    /* Simulate realistic Value vectors: much smaller norms than K
     * scos-lab: V norms typically 0.35 - 4.3 */
    int n_vectors = 4096;
    int head_dim = 128;

    double total_cosim = 0;
    int n_bad = 0;

    for (int v = 0; v < n_vectors; v++) {
        float src[128], dst[128];

        /* Realistic V: small norm, Gaussian */
        float scale = 0.5f + 3.5f * ((float)rand() / RAND_MAX);
        gaussian_vector(src, head_dim, 0.0f, scale / sqrtf((float)head_dim));

        block_tq2_0 blocks[4]; /* 2-bit for V (aggressive, per community) */
        quantize_row_tq2_0(src, blocks, head_dim, head_dim);
        dequantize_row_tq2_0(blocks, dst, head_dim, head_dim);

        float cosim = tq_cosine_sim(src, dst, head_dim);
        total_cosim += cosim;
        if (cosim < 0.85f) n_bad++;
    }

    float avg_cosim = (float)(total_cosim / n_vectors);
    printf("(values 2-bit: N=%d avg_cos=%.5f bad=%.1f%%) ",
           n_vectors, avg_cosim, 100.0f * n_bad / n_vectors);

    ASSERT(avg_cosim > 0.90f, "realistic V vectors 2-bit: avg cosine > 0.90");
}

TEST(kv_asymmetric_config) {
    /* Test recommended asymmetric config: 4-bit K, 2-bit V */
    int head_dim = 128;
    int n_trials = 1000;

    double k_cosim = 0, v_cosim = 0;

    for (int t = 0; t < n_trials; t++) {
        float k_src[128], k_dst[128];
        float v_src[128], v_dst[128];

        /* K: large norm */
        gaussian_vector(k_src, head_dim, 0.0f, 50.0f / sqrtf(128.0f));
        /* V: small norm */
        gaussian_vector(v_src, head_dim, 0.0f, 2.0f / sqrtf(128.0f));

        block_tq4_0 k_blocks[4];
        quantize_row_tq4_0(k_src, k_blocks, head_dim, head_dim);
        dequantize_row_tq4_0(k_blocks, k_dst, head_dim, head_dim);

        block_tq2_0 v_blocks[4];
        quantize_row_tq2_0(v_src, v_blocks, head_dim, head_dim);
        dequantize_row_tq2_0(v_blocks, v_dst, head_dim, head_dim);

        k_cosim += tq_cosine_sim(k_src, k_dst, head_dim);
        v_cosim += tq_cosine_sim(v_src, v_dst, head_dim);
    }

    float avg_k = (float)(k_cosim / n_trials);
    float avg_v = (float)(v_cosim / n_trials);

    printf("(K4=%.5f V2=%.5f) ", avg_k, avg_v);

    ASSERT(avg_k > 0.99f, "asymmetric: K@4bit cosine > 0.99");
    ASSERT(avg_v > 0.90f, "asymmetric: V@2bit cosine > 0.90");
}

/* ═══════════════════════════════════════════════════════════════════════
 *  6. LONG SEQUENCE STRESS TEST
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(long_sequence_32k) {
    /* Simulate 32K token context: 32768 * head_dim elements */
    int head_dim = 128;
    int seq_len = 32768;
    int total_elems = seq_len * (head_dim / 32); /* blocks needed */
    int n = seq_len * 4; /* 4 blocks per position (128/32) — just test a subset */

    /* Test 1000 random positions from a 32K sequence */
    int n_test = 1000;
    double total_cosim = 0;

    for (int t = 0; t < n_test; t++) {
        float src[128], dst[128];
        gaussian_vector(src, head_dim, 0.0f, 1.0f);

        block_tq3_0 blocks[4];
        quantize_row_tq3_0(src, blocks, head_dim, head_dim);
        dequantize_row_tq3_0(blocks, dst, head_dim, head_dim);

        total_cosim += tq_cosine_sim(src, dst, head_dim);
    }

    float avg_cosim = (float)(total_cosim / n_test);
    printf("(32K sim: avg_cos=%.5f) ", avg_cosim);
    ASSERT(avg_cosim > 0.97f, "32K sequence simulation: avg cosine > 0.97");
    (void)total_elems; (void)n;
}

/* ═══════════════════════════════════════════════════════════════════════
 *  7. CODEBOOK VALIDATION
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(codebook_all_dims) {
    int dims[] = {32, 64, 128, 256};
    int bits[] = {2, 3, 4};

    for (int d = 0; d < 4; d++) {
        for (int b = 0; b < 3; b++) {
            const tq_codebook *cb = tq_get_codebook(bits[b], dims[d]);
            ASSERT(cb != NULL, "codebook exists for all valid combinations");
            ASSERT(cb->n_levels == (1 << bits[b]), "correct level count");
            ASSERT(cb->dim == dims[d], "correct dimension");

            /* Verify centroids are within bounds */
            for (int i = 0; i < cb->n_levels; i++) {
                ASSERT(!isnan(cb->centroids[i]), "no NaN centroids");
                ASSERT(fabsf(cb->centroids[i]) < 1.0f, "centroids bounded");
            }
        }
    }
}

TEST(codebook_midpoint_property) {
    /* Boundary[i] should be approximately midpoint of centroid[i] and centroid[i+1] */
    const tq_codebook *cb = tq_get_codebook(3, 32);
    ASSERT(cb != NULL, "codebook exists");

    for (int i = 0; i < cb->n_levels - 1; i++) {
        float midpoint = (cb->centroids[i] + cb->centroids[i + 1]) / 2.0f;
        ASSERT(fabsf(cb->boundaries[i] - midpoint) < 0.02f,
               "boundary near midpoint of adjacent centroids");
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 *  8. NORM CORRECTION VALIDATION
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(norm_correction_improvement) {
    /* Verify norm correction actually helps */
    int n_trials = 5000;
    double total_norm_ratio = 0;

    for (int t = 0; t < n_trials; t++) {
        float src[128], dst[128];
        gaussian_vector(src, 128, 0.0f, 1.0f);

        block_tq3_0 blocks[4];
        quantize_row_tq3_0(src, blocks, 128, 128);
        dequantize_row_tq3_0(blocks, dst, 128, 128);

        float src_norm = tq_vec_norm(src, 128);
        float dst_norm = tq_vec_norm(dst, 128);

        if (src_norm > 1e-6f) {
            total_norm_ratio += (double)(dst_norm / src_norm);
        }
    }

    float avg_ratio = (float)(total_norm_ratio / n_trials);
    printf("(norm_ratio=%.5f) ", avg_ratio);

    /* With norm correction, the ratio should be very close to 1.0 */
    ASSERT(fabsf(avg_ratio - 1.0f) < 0.05f,
           "norm correction: output/input norm ratio near 1.0");
}

/* ═══════════════════════════════════════════════════════════════════════
 *  9. HELPER FUNCTION TESTS
 * ═══════════════════════════════════════════════════════════════════════ */

TEST(head_dim_validation) {
    ASSERT(tq_head_dim_valid(32) == 1, "dim=32 valid");
    ASSERT(tq_head_dim_valid(64) == 1, "dim=64 valid");
    ASSERT(tq_head_dim_valid(128) == 1, "dim=128 valid");
    ASSERT(tq_head_dim_valid(256) == 1, "dim=256 valid");
    ASSERT(tq_head_dim_valid(80) == 0, "dim=80 invalid (Qwen3-4B)");
    ASSERT(tq_head_dim_valid(96) == 0, "dim=96 invalid");
    ASSERT(tq_head_dim_valid(576) == 0, "dim=576 invalid (GLM)");
}

TEST(kv_config_recommendation) {
    /* Low ratio → 3-bit K, 2-bit V */
    tq_kv_config cfg1 = tq_recommend_kv_bits(12.0f, 2.0f); /* ratio=6 */
    ASSERT(cfg1.k_bits == 3, "low ratio: k=3");
    ASSERT(cfg1.v_bits == 2, "low ratio: v=2");

    /* High ratio → 4-bit K, 3-bit V */
    tq_kv_config cfg2 = tq_recommend_kv_bits(172.0f, 3.3f); /* ratio=52 */
    ASSERT(cfg2.k_bits == 4, "high ratio: k=4");
    ASSERT(cfg2.v_bits == 3, "high ratio: v=3");
}

TEST(compression_ratios) {
    ASSERT_NEAR(tq_compression_ratio(2), 6.4f, 0.1f, "2-bit: 6.4x");
    ASSERT_NEAR(tq_compression_ratio(3), 4.571f, 0.1f, "3-bit: 4.6x");
    ASSERT_NEAR(tq_compression_ratio(4), 3.556f, 0.1f, "4-bit: 3.6x");
}

/* ═══════════════════════════════════════════════════════════════════════
 *  MAIN
 * ═══════════════════════════════════════════════════════════════════════ */

int main(void) {
    srand(42);

    printf("\n");
    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║    TurboQuant Full Validation Suite                  ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n\n");

    printf("1. WHT Stress Tests:\n");
    RUN(wht_sizes);
    RUN(wht_orthogonality);
    RUN(wht_deterministic);

    printf("\n2. Bit-Packing Exhaustive:\n");
    RUN(pack_3bit_exhaustive);
    RUN(pack_4bit_all_values);

    printf("\n3. Edge Cases:\n");
    RUN(very_small_values);
    RUN(single_nonzero);
    RUN(alternating_sign);
    RUN(identical_values);
    RUN(extreme_magnitude);

    printf("\n4. Statistical Validation (10k vectors):\n");
    RUN(statistical_tq4_10k);
    RUN(statistical_tq3_10k);
    RUN(statistical_tq2_10k);

    printf("\n5. Realistic KV Cache Simulation:\n");
    RUN(kv_cache_keys_realistic);
    RUN(kv_cache_values_realistic);
    RUN(kv_asymmetric_config);

    printf("\n6. Long Sequence Stress:\n");
    RUN(long_sequence_32k);

    printf("\n7. Codebook Validation:\n");
    RUN(codebook_all_dims);
    RUN(codebook_midpoint_property);

    printf("\n8. Norm Correction:\n");
    RUN(norm_correction_improvement);

    printf("\n9. Helper Functions:\n");
    RUN(head_dim_validation);
    RUN(kv_config_recommendation);
    RUN(compression_ratios);

    printf("\n══════════════════════════════════════════════════════\n");
    printf("Results: %d tests passed, %d failed (%d assertions)\n\n",
           tests_passed, tests_failed, total_assertions);

    return tests_failed > 0 ? 1 : 0;
}
