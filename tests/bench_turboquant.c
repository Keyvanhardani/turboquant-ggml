/*
 * bench_turboquant.c — Performance benchmark for TurboQuant
 *
 * Measures throughput (vectors/sec), MSE, and cosine similarity
 * across all quantization types and dimensions.
 */

#include "turboquant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define N_WARMUP   100
#define N_BENCH   10000
#define N_QUALITY  1000

static float randf(void) {
    return (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
}

static void random_vector(float *v, int n) {
    for (int i = 0; i < n; i++) v[i] = randf();
}

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ─── Naive min/max quantization (simulates Q4_0) for comparison ─────── */

typedef struct {
    float scale;
    float min;
    uint8_t qs[16]; /* 32 x 4-bit packed */
} block_naive_q4;

static void quantize_naive_q4(const float *src, block_naive_q4 *dst, int n) {
    int nb = n / 32;
    for (int b = 0; b < nb; b++) {
        const float *s = src + b * 32;
        float vmin = s[0], vmax = s[0];
        for (int i = 1; i < 32; i++) {
            if (s[i] < vmin) vmin = s[i];
            if (s[i] > vmax) vmax = s[i];
        }
        float scale = (vmax - vmin) / 15.0f;
        float inv_scale = (scale > 1e-12f) ? 1.0f / scale : 0.0f;
        dst[b].scale = scale;
        dst[b].min = vmin;
        for (int i = 0; i < 16; i++) {
            int lo = (int)((s[2*i] - vmin) * inv_scale + 0.5f);
            int hi = (int)((s[2*i+1] - vmin) * inv_scale + 0.5f);
            if (lo < 0) lo = 0; if (lo > 15) lo = 15;
            if (hi < 0) hi = 0; if (hi > 15) hi = 15;
            dst[b].qs[i] = (uint8_t)(lo | (hi << 4));
        }
    }
}

static void dequantize_naive_q4(const block_naive_q4 *src, float *dst, int n) {
    int nb = n / 32;
    for (int b = 0; b < nb; b++) {
        float scale = src[b].scale;
        float vmin = src[b].min;
        for (int i = 0; i < 16; i++) {
            dst[b*32 + 2*i]     = (src[b].qs[i] & 0x0F) * scale + vmin;
            dst[b*32 + 2*i + 1] = (src[b].qs[i] >> 4) * scale + vmin;
        }
    }
}

/* ─── Benchmark runner ───────────────────────────────────────────────── */

typedef struct {
    const char *name;
    float       bits_per_elem;
    float       compression;
    float       avg_mse;
    float       avg_cosim;
    double      quant_vecs_per_sec;
    double      dequant_vecs_per_sec;
} bench_result;

static bench_result run_bench_tq(const char *name, int bits, int dim, int vec_dim) {
    bench_result r;
    r.name = name;

    int n_blocks = vec_dim / 32;
    size_t block_sz;
    switch (bits) {
        case 2: block_sz = sizeof(block_tq2_0); r.bits_per_elem = 2.5f; break;
        case 3: block_sz = sizeof(block_tq3_0); r.bits_per_elem = 3.5f; break;
        case 4: block_sz = sizeof(block_tq4_0); r.bits_per_elem = 4.5f; break;
        default: r.bits_per_elem = 0; return r;
    }
    r.compression = 16.0f / r.bits_per_elem;

    void *compressed = malloc(n_blocks * block_sz);
    float *src = malloc(vec_dim * sizeof(float));
    float *dst = malloc(vec_dim * sizeof(float));

    /* Quality measurement */
    r.avg_mse = 0; r.avg_cosim = 0;
    for (int i = 0; i < N_QUALITY; i++) {
        random_vector(src, vec_dim);
        switch (bits) {
            case 2: quantize_row_tq2_0(src, compressed, vec_dim, dim);
                    dequantize_row_tq2_0(compressed, dst, vec_dim, dim); break;
            case 3: quantize_row_tq3_0(src, compressed, vec_dim, dim);
                    dequantize_row_tq3_0(compressed, dst, vec_dim, dim); break;
            case 4: quantize_row_tq4_0(src, compressed, vec_dim, dim);
                    dequantize_row_tq4_0(compressed, dst, vec_dim, dim); break;
        }
        r.avg_mse += tq_mse(src, dst, vec_dim);
        r.avg_cosim += tq_cosine_sim(src, dst, vec_dim);
    }
    r.avg_mse /= N_QUALITY;
    r.avg_cosim /= N_QUALITY;

    /* Throughput: quantize */
    for (int i = 0; i < N_WARMUP; i++) {
        random_vector(src, vec_dim);
        switch (bits) {
            case 2: quantize_row_tq2_0(src, compressed, vec_dim, dim); break;
            case 3: quantize_row_tq3_0(src, compressed, vec_dim, dim); break;
            case 4: quantize_row_tq4_0(src, compressed, vec_dim, dim); break;
        }
    }

    random_vector(src, vec_dim);
    double t0 = get_time_ms();
    for (int i = 0; i < N_BENCH; i++) {
        switch (bits) {
            case 2: quantize_row_tq2_0(src, compressed, vec_dim, dim); break;
            case 3: quantize_row_tq3_0(src, compressed, vec_dim, dim); break;
            case 4: quantize_row_tq4_0(src, compressed, vec_dim, dim); break;
        }
    }
    double dt = get_time_ms() - t0;
    r.quant_vecs_per_sec = N_BENCH / (dt / 1000.0);

    /* Throughput: dequantize */
    t0 = get_time_ms();
    for (int i = 0; i < N_BENCH; i++) {
        switch (bits) {
            case 2: dequantize_row_tq2_0(compressed, dst, vec_dim, dim); break;
            case 3: dequantize_row_tq3_0(compressed, dst, vec_dim, dim); break;
            case 4: dequantize_row_tq4_0(compressed, dst, vec_dim, dim); break;
        }
    }
    dt = get_time_ms() - t0;
    r.dequant_vecs_per_sec = N_BENCH / (dt / 1000.0);

    free(compressed); free(src); free(dst);
    return r;
}

static bench_result run_bench_naive_q4(int vec_dim) {
    bench_result r;
    r.name = "Q4_0 (naive)";
    r.bits_per_elem = 4.5f; /* 4-bit + scale/min overhead */
    r.compression = 16.0f / r.bits_per_elem;

    int n_blocks = vec_dim / 32;
    block_naive_q4 *compressed = malloc(n_blocks * sizeof(block_naive_q4));
    float *src = malloc(vec_dim * sizeof(float));
    float *dst = malloc(vec_dim * sizeof(float));

    r.avg_mse = 0; r.avg_cosim = 0;
    for (int i = 0; i < N_QUALITY; i++) {
        random_vector(src, vec_dim);
        quantize_naive_q4(src, compressed, vec_dim);
        dequantize_naive_q4(compressed, dst, vec_dim);
        r.avg_mse += tq_mse(src, dst, vec_dim);
        r.avg_cosim += tq_cosine_sim(src, dst, vec_dim);
    }
    r.avg_mse /= N_QUALITY;
    r.avg_cosim /= N_QUALITY;

    random_vector(src, vec_dim);
    double t0 = get_time_ms();
    for (int i = 0; i < N_BENCH; i++) {
        quantize_naive_q4(src, compressed, vec_dim);
    }
    double dt = get_time_ms() - t0;
    r.quant_vecs_per_sec = N_BENCH / (dt / 1000.0);

    t0 = get_time_ms();
    for (int i = 0; i < N_BENCH; i++) {
        dequantize_naive_q4(compressed, dst, vec_dim);
    }
    dt = get_time_ms() - t0;
    r.dequant_vecs_per_sec = N_BENCH / (dt / 1000.0);

    free(compressed); free(src); free(dst);
    return r;
}

static void print_bar(float val, float max_val, int width) {
    int filled = (int)(val / max_val * width);
    if (filled > width) filled = width;
    for (int i = 0; i < filled; i++) printf("#");
    for (int i = filled; i < width; i++) printf(" ");
}

int main(void) {
    srand(42);

    int vec_dim = 128;

    printf("\n");
    printf("  ╔══════════════════════════════════════════════════════════════╗\n");
    printf("  ║          TurboQuant Benchmark — dim=%d, %dk trials          ║\n", vec_dim, N_BENCH/1000);
    printf("  ╚══════════════════════════════════════════════════════════════╝\n");
    printf("\n");

    bench_result results[5];
    results[0] = run_bench_naive_q4(vec_dim);
    results[1] = run_bench_tq("TQ4_0", 4, 128, vec_dim);
    results[2] = run_bench_tq("TQ3_0", 3, 128, vec_dim);
    results[3] = run_bench_tq("TQ2_0", 2, 128, vec_dim);

    /* ─── Quality comparison ─────────────────────────────────────────── */

    printf("  ┌─────────────┬───────┬────────────┬──────────┬────────────┐\n");
    printf("  │ Type        │ Bits  │ Compress.  │ MSE      │ Cosine Sim │\n");
    printf("  ├─────────────┼───────┼────────────┼──────────┼────────────┤\n");

    for (int i = 0; i < 4; i++) {
        bench_result *r = &results[i];
        printf("  │ %-11s │ %4.1f  │    %4.1fx   │ %8.5f │   %7.5f  │\n",
               r->name, r->bits_per_elem, r->compression, r->avg_mse, r->avg_cosim);
    }

    printf("  └─────────────┴───────┴────────────┴──────────┴────────────┘\n");
    printf("\n");

    /* ─── MSE visual comparison ──────────────────────────────────────── */

    printf("  MSE (lower is better):\n\n");
    float max_mse = 0;
    for (int i = 0; i < 4; i++) {
        if (results[i].avg_mse > max_mse) max_mse = results[i].avg_mse;
    }

    for (int i = 0; i < 4; i++) {
        bench_result *r = &results[i];
        printf("  %-11s │", r->name);
        print_bar(r->avg_mse, max_mse * 1.1f, 40);
        printf("│ %.5f\n", r->avg_mse);
    }

    printf("\n");

    /* ─── Throughput ─────────────────────────────────────────────────── */

    printf("  ┌─────────────┬────────────────┬──────────────────┐\n");
    printf("  │ Type        │ Quant (vec/s)  │ Dequant (vec/s)  │\n");
    printf("  ├─────────────┼────────────────┼──────────────────┤\n");

    for (int i = 0; i < 4; i++) {
        bench_result *r = &results[i];
        printf("  │ %-11s │ %12.0f   │ %14.0f   │\n",
               r->name, r->quant_vecs_per_sec, r->dequant_vecs_per_sec);
    }

    printf("  └─────────────┴────────────────┴──────────────────┘\n");

    /* ─── VRAM savings projection ────────────────────────────────────── */

    printf("\n");
    printf("  Projected KV cache VRAM savings (8B model, 32K context):\n\n");

    float fp16_bytes = 2.0f * 128 * 32 * 32768; /* 2 bytes * head_dim * n_heads * seq_len */
    float fp16_gb = fp16_bytes / (1024*1024*1024);

    const char *names[] = { "FP16", "Q8_0", "Q4_0 (naive)", "TQ4_0", "TQ3_0", "TQ2_0" };
    float ratios[] = { 1.0f, 1.9f, 3.6f, 3.6f, 4.6f, 6.4f };

    for (int i = 0; i < 6; i++) {
        float gb = fp16_gb / ratios[i];
        int bar_len = (int)(gb / fp16_gb * 50);
        printf("  %-13s ", names[i]);
        for (int j = 0; j < bar_len; j++) printf("█");
        for (int j = bar_len; j < 50; j++) printf("░");
        printf(" %.2f GB\n", gb);
    }

    printf("\n");
    printf("  TQ3 saves %.1f GB vs FP16, %.1f GB vs Q8_0\n",
           fp16_gb - fp16_gb/4.6f, fp16_gb/1.9f - fp16_gb/4.6f);

    printf("\n");

    return 0;
}
