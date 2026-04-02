/*
 * test_cuda.cu — GPU roundtrip test for TurboQuant CUDA kernels
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* Include the CPU implementation for quantize (host-side) */
extern "C" {
#include "turboquant.h"
}

/* CUDA kernel wrappers */
extern "C" void dequantize_row_turbo4_0_cuda(const void *vx, float *y, int64_t k, cudaStream_t stream);
extern "C" void dequantize_row_turbo3_0_cuda(const void *vx, float *y, int64_t k, cudaStream_t stream);
extern "C" void dequantize_row_turbo2_0_cuda(const void *vx, float *y, int64_t k, cudaStream_t stream);

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

static float randn() {
    float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
    float u2 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 1.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
}

static float cosine_sim(const float *a, const float *b, int n) {
    float dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; i++) { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
    return dot / (sqrtf(na) * sqrtf(nb));
}

template<typename Block>
static void test_gpu_roundtrip(
    const char *name,
    int bits,
    void (*quantize_fn)(const float*, void*, int64_t, int),
    void (*dequant_cuda_fn)(const void*, float*, int64_t, cudaStream_t),
    size_t block_size_bytes
) {
    const int dim = 128;
    const int n_blocks = dim / 32;
    const int n_trials = 1000;
    const size_t data_bytes = n_blocks * block_size_bytes;

    float *h_src = (float *)malloc(dim * sizeof(float));
    float *h_dst_cpu = (float *)malloc(dim * sizeof(float));
    float *h_dst_gpu = (float *)malloc(dim * sizeof(float));
    void *h_compressed = malloc(data_bytes);

    void *d_compressed;
    float *d_output;
    CHECK_CUDA(cudaMalloc(&d_compressed, data_bytes));
    CHECK_CUDA(cudaMalloc(&d_output, dim * sizeof(float)));

    double total_cosim_cpu = 0, total_cosim_gpu = 0;
    double max_cpu_gpu_diff = 0;

    for (int t = 0; t < n_trials; t++) {
        /* Generate random vector */
        for (int i = 0; i < dim; i++) h_src[i] = randn();

        /* CPU quantize */
        quantize_fn(h_src, h_compressed, dim, 128);

        /* CPU dequantize (for comparison) */
        if (bits == 4) dequantize_row_tq4_0(h_compressed, h_dst_cpu, dim, 128);
        else if (bits == 3) dequantize_row_tq3_0(h_compressed, h_dst_cpu, dim, 128);
        else dequantize_row_tq2_0(h_compressed, h_dst_cpu, dim, 128);

        /* GPU dequantize */
        CHECK_CUDA(cudaMemcpy(d_compressed, h_compressed, data_bytes, cudaMemcpyHostToDevice));
        dequant_cuda_fn(d_compressed, d_output, dim, 0);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(h_dst_gpu, d_output, dim * sizeof(float), cudaMemcpyDeviceToHost));

        /* Compare */
        float cosim_cpu = cosine_sim(h_src, h_dst_cpu, dim);
        float cosim_gpu = cosine_sim(h_src, h_dst_gpu, dim);
        total_cosim_cpu += cosim_cpu;
        total_cosim_gpu += cosim_gpu;

        /* CPU vs GPU should be identical */
        float diff = 0;
        for (int i = 0; i < dim; i++) {
            float d = fabsf(h_dst_cpu[i] - h_dst_gpu[i]);
            if (d > diff) diff = d;
        }
        if (diff > max_cpu_gpu_diff) max_cpu_gpu_diff = diff;
    }

    float avg_cpu = (float)(total_cosim_cpu / n_trials);
    float avg_gpu = (float)(total_cosim_gpu / n_trials);

    printf("  [%s] CPU_cos=%.5f  GPU_cos=%.5f  max_diff=%.2e  %s\n",
           name, avg_cpu, avg_gpu,
           max_cpu_gpu_diff,
           max_cpu_gpu_diff < 1e-3f ? "OK" : "MISMATCH!");

    cudaFree(d_compressed);
    cudaFree(d_output);
    free(h_src); free(h_dst_cpu); free(h_dst_gpu); free(h_compressed);
}

int main() {
    srand(42);

    /* Print GPU info */
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("\n  GPU: %s (SM %d.%d, %d MB)\n\n", prop.name, prop.major, prop.minor,
           (int)(prop.totalGlobalMem / (1024*1024)));

    printf("  TurboQuant CUDA Roundtrip Tests (1000 trials, dim=128):\n\n");

    test_gpu_roundtrip<block_tq4_0>(
        "turbo4_0", 4,
        quantize_row_tq4_0, dequantize_row_turbo4_0_cuda,
        sizeof(block_tq4_0));

    test_gpu_roundtrip<block_tq3_0>(
        "turbo3_0", 3,
        quantize_row_tq3_0, dequantize_row_turbo3_0_cuda,
        sizeof(block_tq3_0));

    test_gpu_roundtrip<block_tq2_0>(
        "turbo2_0", 2,
        quantize_row_tq2_0, dequantize_row_turbo2_0_cuda,
        sizeof(block_tq2_0));

    printf("\n  Done.\n\n");
    return 0;
}
