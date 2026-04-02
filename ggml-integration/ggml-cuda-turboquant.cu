/*
 * ggml-cuda-turboquant.cu — CUDA kernels for TurboQuant KV cache quantization
 *
 * Author: Keyvan Hardani (https://github.com/Keyvanhardani)
 *
 * Provides GPU-accelerated dequantize for turbo2_0/turbo3_0/turbo4_0.
 * Phase 4a approach: dequant → FP16 before Flash Attention (zero FA changes).
 *
 * Drop into ggml/src/ggml-cuda/ alongside existing dequantize kernels.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

/* ─── Block structs (must match ggml-turboquant.h) ───────────────────── */

#define QK_TQ 32

struct block_turbo4_0 {
    uint16_t d;        /* fp16 corrected norm */
    uint8_t  qs[16];   /* 32 x 4-bit packed */
};

struct block_turbo3_0 {
    uint16_t d;        /* fp16 corrected norm */
    uint8_t  qs[12];   /* 32 x 3-bit packed */
};

struct block_turbo2_0 {
    uint16_t d;        /* fp16 corrected norm */
    uint8_t  qs[8];    /* 32 x 2-bit packed */
};

/* ─── Constexpr Lloyd-Max centroids (register-allocated) ─────────────── */

static __device__ __constant__ float LM4_CENTROIDS[16] = {
    -2.7326f * 0.17677669f, -2.0690f * 0.17677669f,
    -1.6180f * 0.17677669f, -1.2562f * 0.17677669f,
    -0.9424f * 0.17677669f, -0.6568f * 0.17677669f,
    -0.3881f * 0.17677669f, -0.1284f * 0.17677669f,
     0.1284f * 0.17677669f,  0.3881f * 0.17677669f,
     0.6568f * 0.17677669f,  0.9424f * 0.17677669f,
     1.2562f * 0.17677669f,  1.6180f * 0.17677669f,
     2.0690f * 0.17677669f,  2.7326f * 0.17677669f
};

static __device__ __constant__ float LM3_CENTROIDS[8] = {
    -2.1519f * 0.17677669f, -1.3439f * 0.17677669f,
    -0.7560f * 0.17677669f, -0.2451f * 0.17677669f,
     0.2451f * 0.17677669f,  0.7560f * 0.17677669f,
     1.3439f * 0.17677669f,  2.1519f * 0.17677669f
};

static __device__ __constant__ float LM2_CENTROIDS[4] = {
    -1.5104f * 0.17677669f, -0.4528f * 0.17677669f,
     0.4528f * 0.17677669f,  1.5104f * 0.17677669f
};

/* ─── FP16 conversion ────────────────────────────────────────────────── */

static __device__ __forceinline__ float fp16_to_float(uint16_t h) {
    __half hv;
    memcpy(&hv, &h, sizeof(hv));
    return __half2float(hv);
}

/* ─── In-place WHT for 32 elements (per-thread, registers) ──────────── */

static __device__ __forceinline__ void wht32_device(float * __restrict__ x) {
    /* Fast butterfly WHT, 5 stages for n=32 */
    #pragma unroll
    for (int len = 1; len < 32; len <<= 1) {
        #pragma unroll
        for (int i = 0; i < 32; i += len << 1) {
            #pragma unroll
            for (int j = 0; j < len; j++) {
                float u = x[i + j];
                float v = x[i + j + len];
                x[i + j]       = u + v;
                x[i + j + len] = u - v;
            }
        }
    }
    /* 1/sqrt(32) normalization */
    const float scale = 0.17677669f;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        x[i] *= scale;
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 *  turbo4_0 dequantize kernel
 * ═══════════════════════════════════════════════════════════════════════ */

static __global__ void dequantize_block_turbo4_0(
    const void * __restrict__ vx,
    float * __restrict__ y,
    const int64_t k
) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nb = k / QK_TQ;

    if (i >= nb) return;

    const block_turbo4_0 * x = (const block_turbo4_0 *)vx + i;
    float * out = y + i * QK_TQ;

    const float norm = fp16_to_float(x->d);

    /* Unpack 4-bit indices + centroid lookup */
    float vals[32];
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        uint8_t byte = x->qs[j];
        vals[2*j]     = LM4_CENTROIDS[byte & 0x0F];
        vals[2*j + 1] = LM4_CENTROIDS[(byte >> 4) & 0x0F];
    }

    /* Inverse WHT */
    wht32_device(vals);

    /* Scale by corrected norm */
    #pragma unroll
    for (int j = 0; j < 32; j++) {
        out[j] = vals[j] * norm;
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 *  turbo3_0 dequantize kernel
 * ═══════════════════════════════════════════════════════════════════════ */

static __global__ void dequantize_block_turbo3_0(
    const void * __restrict__ vx,
    float * __restrict__ y,
    const int64_t k
) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nb = k / QK_TQ;

    if (i >= nb) return;

    const block_turbo3_0 * x = (const block_turbo3_0 *)vx + i;
    float * out = y + i * QK_TQ;

    const float norm = fp16_to_float(x->d);

    /* Unpack 3-bit indices: 8 values from 3 bytes, x4 groups */
    float vals[32];
    #pragma unroll
    for (int g = 0; g < 4; g++) {
        const uint8_t b0 = x->qs[g*3];
        const uint8_t b1 = x->qs[g*3 + 1];
        const uint8_t b2 = x->qs[g*3 + 2];

        vals[g*8 + 0] = LM3_CENTROIDS[ b0       & 0x07];
        vals[g*8 + 1] = LM3_CENTROIDS[(b0 >> 3) & 0x07];
        vals[g*8 + 2] = LM3_CENTROIDS[((b0 >> 6) | (b1 << 2)) & 0x07];
        vals[g*8 + 3] = LM3_CENTROIDS[(b1 >> 1) & 0x07];
        vals[g*8 + 4] = LM3_CENTROIDS[(b1 >> 4) & 0x07];
        vals[g*8 + 5] = LM3_CENTROIDS[((b1 >> 7) | (b2 << 1)) & 0x07];
        vals[g*8 + 6] = LM3_CENTROIDS[(b2 >> 2) & 0x07];
        vals[g*8 + 7] = LM3_CENTROIDS[(b2 >> 5) & 0x07];
    }

    /* Inverse WHT */
    wht32_device(vals);

    /* Scale by corrected norm */
    #pragma unroll
    for (int j = 0; j < 32; j++) {
        out[j] = vals[j] * norm;
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 *  turbo2_0 dequantize kernel
 * ═══════════════════════════════════════════════════════════════════════ */

static __global__ void dequantize_block_turbo2_0(
    const void * __restrict__ vx,
    float * __restrict__ y,
    const int64_t k
) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nb = k / QK_TQ;

    if (i >= nb) return;

    const block_turbo2_0 * x = (const block_turbo2_0 *)vx + i;
    float * out = y + i * QK_TQ;

    const float norm = fp16_to_float(x->d);

    /* Unpack 2-bit indices: 4 values per byte, 8 bytes */
    float vals[32];
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        uint8_t byte = x->qs[j];
        vals[4*j]     = LM2_CENTROIDS[ byte       & 0x03];
        vals[4*j + 1] = LM2_CENTROIDS[(byte >> 2) & 0x03];
        vals[4*j + 2] = LM2_CENTROIDS[(byte >> 4) & 0x03];
        vals[4*j + 3] = LM2_CENTROIDS[(byte >> 6) & 0x03];
    }

    /* Inverse WHT */
    wht32_device(vals);

    /* Scale by corrected norm */
    #pragma unroll
    for (int j = 0; j < 32; j++) {
        out[j] = vals[j] * norm;
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 *  Host-callable wrappers
 * ═══════════════════════════════════════════════════════════════════════ */

extern "C" {

void dequantize_row_turbo4_0_cuda(
    const void * vx, float * y, const int64_t k, cudaStream_t stream
) {
    const int64_t nb = k / QK_TQ;
    const int block_size = 256;
    const int grid_size = (int)((nb + block_size - 1) / block_size);
    dequantize_block_turbo4_0<<<grid_size, block_size, 0, stream>>>(vx, y, k);
}

void dequantize_row_turbo3_0_cuda(
    const void * vx, float * y, const int64_t k, cudaStream_t stream
) {
    const int64_t nb = k / QK_TQ;
    const int block_size = 256;
    const int grid_size = (int)((nb + block_size - 1) / block_size);
    dequantize_block_turbo3_0<<<grid_size, block_size, 0, stream>>>(vx, y, k);
}

void dequantize_row_turbo2_0_cuda(
    const void * vx, float * y, const int64_t k, cudaStream_t stream
) {
    const int64_t nb = k / QK_TQ;
    const int block_size = 256;
    const int grid_size = (int)((nb + block_size - 1) / block_size);
    dequantize_block_turbo2_0<<<grid_size, block_size, 0, stream>>>(vx, y, k);
}

} /* extern "C" */
