/*
 * test_turboquant.c — Comprehensive tests for TurboQuant
 */

#include "turboquant.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("  FAIL: %s (line %d)\n", msg, __LINE__); \
        tests_failed++; \
        return; \
    } \
} while(0)

#define ASSERT_NEAR(a, b, eps, msg) do { \
    float _a = (a), _b = (b), _e = (eps); \
    if (fabsf(_a - _b) > _e) { \
        printf("  FAIL: %s — got %.6f, expected %.6f (eps=%.6f, line %d)\n", \
               msg, _a, _b, _e, __LINE__); \
        tests_failed++; \
        return; \
    } \
} while(0)

#define TEST(name) static void test_##name(void)
#define RUN(name) do { \
    printf("  [%s] ", #name); \
    test_##name(); \
    tests_passed++; \
    printf("OK\n"); \
} while(0)

/* ─── Random vector generation ───────────────────────────────────────── */

static float randf(void) {
    return (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
}

static void random_vector(float *v, int n) {
    for (int i = 0; i < n; i++) v[i] = randf();
}

/* ─── WHT Tests ──────────────────────────────────────────────────────── */

TEST(wht_self_inverse) {
    float x[32], orig[32];
    random_vector(x, 32);
    memcpy(orig, x, sizeof(x));

    tq_wht(x, 32);
    tq_wht(x, 32); /* apply twice = identity */

    for (int i = 0; i < 32; i++) {
        ASSERT_NEAR(x[i], orig[i], 1e-4f, "WHT self-inverse");
    }
}

TEST(wht_preserves_norm) {
    float x[64];
    random_vector(x, 64);
    float norm_before = tq_vec_norm(x, 64);
    tq_wht(x, 64);
    float norm_after = tq_vec_norm(x, 64);
    ASSERT_NEAR(norm_before, norm_after, 1e-3f, "WHT preserves L2 norm");
}

TEST(wht_known_values) {
    /* WHT of [1, 1, 1, 1] with 1/sqrt(4) normalization = [1, 0, 0, 0] */
    float x[] = { 0.5f, 0.5f, 0.5f, 0.5f };
    tq_wht(x, 4);
    ASSERT_NEAR(x[0], 1.0f, 1e-5f, "WHT known: x[0]");
    ASSERT_NEAR(x[1], 0.0f, 1e-5f, "WHT known: x[1]");
    ASSERT_NEAR(x[2], 0.0f, 1e-5f, "WHT known: x[2]");
    ASSERT_NEAR(x[3], 0.0f, 1e-5f, "WHT known: x[3]");
}

/* ─── Codebook Tests ─────────────────────────────────────────────────── */

TEST(codebook_exists) {
    ASSERT(tq_get_codebook(2, 128) != NULL, "2-bit dim=128 exists");
    ASSERT(tq_get_codebook(3, 128) != NULL, "3-bit dim=128 exists");
    ASSERT(tq_get_codebook(4, 128) != NULL, "4-bit dim=128 exists");
    ASSERT(tq_get_codebook(3, 64)  != NULL, "3-bit dim=64 exists");
    ASSERT(tq_get_codebook(3, 256) != NULL, "3-bit dim=256 exists");
    ASSERT(tq_get_codebook(5, 128) == NULL, "5-bit returns NULL");
    ASSERT(tq_get_codebook(3, 80)  == NULL, "dim=80 returns NULL");
}

TEST(codebook_symmetry) {
    const tq_codebook *cb = tq_get_codebook(3, 128);
    ASSERT(cb != NULL, "codebook exists");

    /* Centroids should be symmetric around 0 */
    for (int i = 0; i < cb->n_levels / 2; i++) {
        int j = cb->n_levels - 1 - i;
        ASSERT_NEAR(cb->centroids[i], -cb->centroids[j], 1e-5f, "centroid symmetry");
    }

    /* Boundaries should be symmetric around 0 */
    int nb = cb->n_levels - 1;
    for (int i = 0; i < nb / 2; i++) {
        int j = nb - 1 - i;
        ASSERT_NEAR(cb->boundaries[i], -cb->boundaries[j], 1e-5f, "boundary symmetry");
    }
}

TEST(codebook_ordering) {
    const tq_codebook *cb = tq_get_codebook(4, 128);
    ASSERT(cb != NULL, "codebook exists");

    /* Centroids must be strictly increasing */
    for (int i = 1; i < cb->n_levels; i++) {
        ASSERT(cb->centroids[i] > cb->centroids[i-1], "centroids increasing");
    }
    /* Boundaries must be strictly increasing */
    for (int i = 1; i < cb->n_levels - 1; i++) {
        ASSERT(cb->boundaries[i] > cb->boundaries[i-1], "boundaries increasing");
    }
}

/* ─── Bit-packing Tests ──────────────────────────────────────────────── */

TEST(pack_unpack_3bit) {
    uint8_t indices[32], recovered[32];
    uint8_t packed[12];

    for (int i = 0; i < 32; i++) indices[i] = (uint8_t)(i % 8);

    /* Use the quantize/dequantize to test packing indirectly,
       but we can also test by doing a roundtrip through a block. */

    /* Direct test: pack and unpack should be identity */
    /* We need to access pack/unpack — they're static, so test through
       quantize roundtrip instead. */

    float src[32], dst[32];
    for (int i = 0; i < 32; i++) src[i] = 0.1f * (float)(i - 16);

    block_tq3_0 block;
    quantize_row_tq3_0(src, &block, 32, 128);
    dequantize_row_tq3_0(&block, dst, 32, 128);

    /* Result should be close to original */
    float mse = tq_mse(src, dst, 32);
    printf("(mse=%.4f) ", mse);
    ASSERT(mse < 0.5f, "3-bit roundtrip MSE reasonable");
    (void)indices; (void)recovered; (void)packed;
}

TEST(pack_unpack_4bit) {
    float src[32], dst[32];
    for (int i = 0; i < 32; i++) src[i] = 0.1f * (float)(i - 16);

    block_tq4_0 block;
    quantize_row_tq4_0(src, &block, 32, 128);
    dequantize_row_tq4_0(&block, dst, 32, 128);

    float mse = tq_mse(src, dst, 32);
    printf("(mse=%.4f) ", mse);
    ASSERT(mse < 0.2f, "4-bit roundtrip MSE reasonable");
}

TEST(pack_unpack_2bit) {
    float src[32], dst[32];
    for (int i = 0; i < 32; i++) src[i] = 0.1f * (float)(i - 16);

    block_tq2_0 block;
    quantize_row_tq2_0(src, &block, 32, 128);
    dequantize_row_tq2_0(&block, dst, 32, 128);

    float mse = tq_mse(src, dst, 32);
    ASSERT(mse < 0.5f, "2-bit roundtrip MSE reasonable");
}

/* ─── Roundtrip Quality Tests ────────────────────────────────────────── */

TEST(roundtrip_tq3_cosine) {
    /* Test with realistic KV cache-like vectors */
    float src[128], dst[128];
    random_vector(src, 128);

    /* Normalize to realistic magnitude */
    float norm = tq_vec_norm(src, 128);
    for (int i = 0; i < 128; i++) src[i] /= norm;

    block_tq3_0 blocks[4]; /* 128 / 32 = 4 blocks */
    quantize_row_tq3_0(src, blocks, 128, 128);
    dequantize_row_tq3_0(blocks, dst, 128, 128);

    float cosim = tq_cosine_sim(src, dst, 128);
    ASSERT(cosim > 0.95f, "TQ3 cosine similarity > 0.95");
}

TEST(roundtrip_tq4_cosine) {
    float src[128], dst[128];
    random_vector(src, 128);

    float norm = tq_vec_norm(src, 128);
    for (int i = 0; i < 128; i++) src[i] /= norm;

    block_tq4_0 blocks[4];
    quantize_row_tq4_0(src, blocks, 128, 128);
    dequantize_row_tq4_0(blocks, dst, 128, 128);

    float cosim = tq_cosine_sim(src, dst, 128);
    ASSERT(cosim > 0.98f, "TQ4 cosine similarity > 0.98");
}

TEST(roundtrip_tq4_mse) {
    /* Average MSE over many random vectors */
    float total_mse = 0.0f;
    int n_trials = 100;

    for (int t = 0; t < n_trials; t++) {
        float src[128], dst[128];
        random_vector(src, 128);

        block_tq4_0 blocks[4];
        quantize_row_tq4_0(src, blocks, 128, 128);
        dequantize_row_tq4_0(blocks, dst, 128, 128);

        total_mse += tq_mse(src, dst, 128);
    }

    float avg_mse = total_mse / (float)n_trials;
    /* Paper claims MSE ~ 0.009 for 4-bit at dim=128 */
    printf("(avg_mse=%.4f) ", avg_mse);
    ASSERT(avg_mse < 0.1f, "TQ4 average MSE reasonable");
}

TEST(roundtrip_tq3_mse) {
    float total_mse = 0.0f;
    int n_trials = 100;

    for (int t = 0; t < n_trials; t++) {
        float src[128], dst[128];
        random_vector(src, 128);

        block_tq3_0 blocks[4];
        quantize_row_tq3_0(src, blocks, 128, 128);
        dequantize_row_tq3_0(blocks, dst, 128, 128);

        total_mse += tq_mse(src, dst, 128);
    }

    float avg_mse = total_mse / (float)n_trials;
    printf("(avg_mse=%.4f) ", avg_mse);
    ASSERT(avg_mse < 0.5f, "TQ3 average MSE reasonable");
}

/* ─── Multi-block Tests ──────────────────────────────────────────────── */

TEST(multiblock_roundtrip) {
    int n = 256; /* 8 blocks */
    float src[256], dst[256];
    random_vector(src, n);

    block_tq3_0 blocks[8];
    quantize_row_tq3_0(src, blocks, n, 128);
    dequantize_row_tq3_0(blocks, dst, n, 128);

    float cosim = tq_cosine_sim(src, dst, n);
    ASSERT(cosim > 0.90f, "multi-block cosine sim > 0.90");
}

/* ─── Edge Cases ─────────────────────────────────────────────────────── */

TEST(zero_vector) {
    float src[32] = {0};
    float dst[32];

    block_tq3_0 block;
    quantize_row_tq3_0(src, &block, 32, 128);
    dequantize_row_tq3_0(&block, dst, 32, 128);

    for (int i = 0; i < 32; i++) {
        ASSERT_NEAR(dst[i], 0.0f, 1e-3f, "zero vector preserved");
    }
}

TEST(constant_vector) {
    float src[32], dst[32];
    for (int i = 0; i < 32; i++) src[i] = 1.0f;

    block_tq4_0 block;
    quantize_row_tq4_0(src, &block, 32, 128);
    dequantize_row_tq4_0(&block, dst, 32, 128);

    float cosim = tq_cosine_sim(src, dst, 32);
    ASSERT(cosim > 0.90f, "constant vector cosine > 0.90");
}

TEST(large_magnitude) {
    float src[32], dst[32];
    for (int i = 0; i < 32; i++) src[i] = 1000.0f * randf();

    block_tq3_0 block;
    quantize_row_tq3_0(src, &block, 32, 128);
    dequantize_row_tq3_0(&block, dst, 32, 128);

    float cosim = tq_cosine_sim(src, dst, 32);
    ASSERT(cosim > 0.90f, "large magnitude cosine > 0.90");
}

/* ─── Block Size Tests ───────────────────────────────────────────────── */

TEST(block_sizes) {
    ASSERT(tq2_0_block_size() == 32, "TQ2 block size = 32");
    ASSERT(tq3_0_block_size() == 32, "TQ3 block size = 32");
    ASSERT(tq4_0_block_size() == 32, "TQ4 block size = 32");
    ASSERT(tq2_0_type_size() == 10,  "TQ2 type size = 10");
    ASSERT(tq3_0_type_size() == 14,  "TQ3 type size = 14");
    ASSERT(tq4_0_type_size() == 18,  "TQ4 type size = 18");
}

/* ─── Compression Ratio Tests ────────────────────────────────────────── */

TEST(compression_ratio) {
    /* FP16: 32 elements * 2 bytes = 64 bytes per block */
    float fp16_bytes = 64.0f;

    float tq2_ratio = fp16_bytes / (float)sizeof(block_tq2_0);
    float tq3_ratio = fp16_bytes / (float)sizeof(block_tq3_0);
    float tq4_ratio = fp16_bytes / (float)sizeof(block_tq4_0);

    printf("(TQ2=%.1fx TQ3=%.1fx TQ4=%.1fx) ", tq2_ratio, tq3_ratio, tq4_ratio);

    ASSERT(tq2_ratio > 6.0f, "TQ2 compression > 6x");
    ASSERT(tq3_ratio > 4.0f, "TQ3 compression > 4x");
    ASSERT(tq4_ratio > 3.0f, "TQ4 compression > 3x");
}

/* ─── Main ───────────────────────────────────────────────────────────── */

int main(void) {
    srand(42);

    printf("\n=== TurboQuant Test Suite ===\n\n");

    printf("WHT:\n");
    RUN(wht_self_inverse);
    RUN(wht_preserves_norm);
    RUN(wht_known_values);

    printf("\nCodebooks:\n");
    RUN(codebook_exists);
    RUN(codebook_symmetry);
    RUN(codebook_ordering);

    printf("\nBit-packing roundtrip:\n");
    RUN(pack_unpack_2bit);
    RUN(pack_unpack_3bit);
    RUN(pack_unpack_4bit);

    printf("\nQuality (cosine similarity):\n");
    RUN(roundtrip_tq3_cosine);
    RUN(roundtrip_tq4_cosine);

    printf("\nQuality (MSE, 100 trials):\n");
    RUN(roundtrip_tq3_mse);
    RUN(roundtrip_tq4_mse);

    printf("\nMulti-block:\n");
    RUN(multiblock_roundtrip);

    printf("\nEdge cases:\n");
    RUN(zero_vector);
    RUN(constant_vector);
    RUN(large_magnitude);

    printf("\nBlock sizes & compression:\n");
    RUN(block_sizes);
    RUN(compression_ratio);

    printf("\n─────────────────────────────\n");
    printf("Results: %d passed, %d failed\n\n", tests_passed, tests_failed);

    return tests_failed > 0 ? 1 : 0;
}
