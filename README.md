# turboquant-ggml

**First real implementation of Google's TurboQuant (ICLR 2026) for llama.cpp and Ollama.**

Compress your LLM's KV cache by **4.6x at 3-bit** with near-zero quality loss. Run bigger models, longer contexts, less VRAM.

```
TQ4: 3.6x compression — MSE 0.003 — cosine sim > 0.99
TQ3: 4.6x compression — MSE 0.011 — cosine sim > 0.95
TQ2: 6.4x compression — aggressive, best for values
```

## What is this?

TurboQuant ([arXiv:2504.19874](https://arxiv.org/abs/2504.19874)) is Google Research's algorithm for extreme KV cache compression. Instead of naive min/max quantization, it:

1. **Rotates** vectors with Walsh-Hadamard Transform (spreads information evenly)
2. **Quantizes** each coordinate with mathematically optimal Lloyd-Max codebooks
3. **Packs** indices into 2/3/4 bits per element

The result: near-optimal distortion at extreme compression ratios. Your 8B model's 32K context KV cache drops from **2GB to 430MB**.

## Why this matters for Ollama / llama.cpp

| Current (`q8_0`) | With TurboQuant (`tq3`) | Savings |
|---|---|---|
| 8 bits/element | 3.5 bits/element | **4.6x less VRAM** |
| 2.0 GB KV cache (8B, 32K) | 430 MB | **1.6 GB freed** |
| 4.0 GB KV cache (8B, 64K) | 870 MB | **3.1 GB freed** |

This means: run **longer contexts** on the same GPU, or run **bigger models** that didn't fit before.

## Build

```bash
git clone https://github.com/Keyvanhardani/turboquant-ggml.git
cd turboquant-ggml
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/test_turboquant   # 19/19 tests passing
```

## Architecture

```
include/turboquant.h       # GGML-compatible API + block structs
src/turboquant.c           # Core: WHT, quantize, dequantize, bit-packing
src/turboquant-tables.c    # Precomputed Lloyd-Max codebooks
tests/test_turboquant.c    # Comprehensive test suite
```

### Block formats (GGML-compatible)

```c
// TQ3_0: 14 bytes per 32 elements (3.5 bits/elem)
typedef struct {
    uint16_t d;        // fp16 L2 norm
    uint8_t  qs[12];   // 32 x 3-bit packed
} block_tq3_0;

// TQ4_0: 18 bytes per 32 elements (4.5 bits/elem)
typedef struct {
    uint16_t d;        // fp16 L2 norm
    uint8_t  qs[16];   // 32 x 4-bit packed
} block_tq4_0;
```

### API

```c
#include "turboquant.h"

float kv_data[128];  // one head's KV vector

// Quantize
block_tq3_0 compressed[4];  // 128/32 = 4 blocks
quantize_row_tq3_0(kv_data, compressed, 128, 128);

// Dequantize
float reconstructed[128];
dequantize_row_tq3_0(compressed, reconstructed, 128, 128);
// cosine_similarity(kv_data, reconstructed) > 0.95
```

## Design decisions (backed by community research)

| Paper says | We do | Why |
|---|---|---|
| Random orthogonal rotation | **Walsh-Hadamard Transform** | 59x better quality (community finding) |
| QJL residual correction | **MSE-only** | QJL degrades quality from 80.4% to 69.6% |
| Block size 128 | **Block size 32** | Better flash attention parallelism |
| Symmetric K/V bits | **Asymmetric support** | Keys need more precision than values |

Sources: [llama.cpp Discussion #20969](https://github.com/ggml-org/llama.cpp/discussions/20969), [ik_llama.cpp #1509](https://github.com/ikawrakow/ik_llama.cpp/issues/1509)

## llama.cpp integration roadmap

This library is designed as a drop-in for llama.cpp's quantization system:

- [x] **Phase 1**: Block structs + type registration interfaces
- [x] **Phase 2**: CPU quantize/dequantize (pure C, zero dependencies)
- [x] **Phase 3**: Comprehensive test suite (19/19 passing)
- [ ] **Phase 4**: GGML type registration (`GGML_TYPE_TQ3`, `GGML_TYPE_TQ4`)
- [ ] **Phase 5**: KV cache write/read path integration
- [ ] **Phase 6**: Flash attention dequant (non-fused, zero-risk)
- [ ] **Phase 7**: CLI flags (`--cache-type-k tq3 --cache-type-v tq4`)
- [ ] **Phase 8**: CUDA kernels (fused FA for peak performance)

## How it works

```
Input vector (FP16)
    |
    v
[Store L2 norm] ─── norm stored as FP16 (2 bytes)
    |
    v
[Normalize to unit sphere]
    |
    v
[Walsh-Hadamard Transform] ─── O(n log n), self-inverse
    |                           spreads info evenly across coords
    v
[Lloyd-Max quantization] ─── optimal codebook for Beta distribution
    |                         precomputed, zero per-block overhead
    v
[Bit-pack to 2/3/4 bits] ─── compact storage
    |
    v
Compressed block (14 bytes for 32 FP16 elements)
```

## Benchmarks

Measured on random vectors (100 trials, dim=128):

| Type | Bits/elem | Compression | MSE | Storage/block |
|------|-----------|------------|-----|--------------|
| FP16 | 16.0 | 1.0x | 0.000 | 64 bytes |
| Q8_0 | 8.5 | 1.9x | ~0.001 | 34 bytes |
| Q4_0 | 4.5 | 3.6x | ~0.01 | 18 bytes |
| **TQ4_0** | **4.5** | **3.6x** | **0.003** | **18 bytes** |
| **TQ3_0** | **3.5** | **4.6x** | **0.011** | **14 bytes** |
| **TQ2_0** | **2.5** | **6.4x** | — | **10 bytes** |

TQ4 achieves **3x lower MSE** than Q4_0 at identical storage cost, thanks to the optimal codebook.

## License

MIT

## Citation

```bibtex
@inproceedings{zandieh2026turboquant,
  title={TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate},
  author={Zandieh, Amir and Daliri, Majid and Hadian, Ali and Mirrokni, Vahab},
  booktitle={ICLR},
  year={2026}
}
```
