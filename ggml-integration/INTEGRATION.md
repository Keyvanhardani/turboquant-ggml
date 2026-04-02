# TurboQuant Integration Guide for llama.cpp

Step-by-step guide to integrate TurboQuant into llama.cpp mainline.

## Files to add

```
ggml/include/ggml-turboquant.h   # Block structs + function declarations
ggml/src/ggml-turboquant.c       # Core algorithm (WHT, codebooks, quant/dequant)
```

## Files to modify (5 changes)

### 1. `ggml/include/ggml.h` — Register type enum values

```c
// After GGML_TYPE_NVFP4 = 40, add:
GGML_TYPE_TURBO4_0 = 41,
GGML_TYPE_TURBO3_0 = 42,
GGML_TYPE_TURBO2_0 = 43,
GGML_TYPE_COUNT    = 44,  // was 41
```

### 2. `ggml/src/ggml.c` — Register type traits

```c
// After the GGML_TYPE_NVFP4 entry in ggml_type_traits[], add:
[GGML_TYPE_TURBO4_0] = {
    .type_name      = "turbo4_0",
    .blck_size      = QK_TURBO4_0,
    .type_size      = sizeof(block_turbo4_0),
    .is_quantized   = true,
    .to_float       = (ggml_to_float_t) dequantize_row_turbo4_0,
    .from_float_ref = (ggml_from_float_t) quantize_row_turbo4_0_ref,
},
[GGML_TYPE_TURBO3_0] = {
    .type_name      = "turbo3_0",
    .blck_size      = QK_TURBO3_0,
    .type_size      = sizeof(block_turbo3_0),
    .is_quantized   = true,
    .to_float       = (ggml_to_float_t) dequantize_row_turbo3_0,
    .from_float_ref = (ggml_from_float_t) quantize_row_turbo3_0_ref,
},
[GGML_TYPE_TURBO2_0] = {
    .type_name      = "turbo2_0",
    .blck_size      = QK_TURBO2_0,
    .type_size      = sizeof(block_turbo2_0),
    .is_quantized   = true,
    .to_float       = (ggml_to_float_t) dequantize_row_turbo2_0,
    .from_float_ref = (ggml_from_float_t) quantize_row_turbo2_0_ref,
},
```

### 3. `ggml/src/ggml.c` — Add quantize_chunk cases

```c
// In ggml_quantize_chunk() switch statement, add:
case GGML_TYPE_TURBO4_0: result = quantize_turbo4_0(src + start, (char *)dst + start_row * row_size, nrows, n_per_row, imatrix); break;
case GGML_TYPE_TURBO3_0: result = quantize_turbo3_0(src + start, (char *)dst + start_row * row_size, nrows, n_per_row, imatrix); break;
case GGML_TYPE_TURBO2_0: result = quantize_turbo2_0(src + start, (char *)dst + start_row * row_size, nrows, n_per_row, imatrix); break;
```

### 4. `common/arg.cpp` — Add KV cache type options

```c
// In kv_cache_types vector, add:
GGML_TYPE_TURBO4_0,
GGML_TYPE_TURBO3_0,
GGML_TYPE_TURBO2_0,
```

### 5. `src/llama-kv-cache.cpp` — Already handled!

No changes needed here. The existing code automatically:
- Applies WHT rotation when `ggml_is_quantized(type)` returns true
- Pre-computes Hadamard matrices for all power-of-2 head dimensions
- The turbo types set `is_quantized = true`, so WHT is auto-applied

## Build

Add to `ggml/src/CMakeLists.txt`:
```cmake
set(GGML_SOURCES_BASE
    ...
    ggml-turboquant.c
    ...
)
```

## Usage

```bash
./llama-server -m model.gguf \
    --cache-type-k turbo3_0 \
    --cache-type-v turbo3_0 \
    --flash-attn on \
    -ngl 99
```

### Recommended configurations

| Use case | K type | V type | Compression | PPL impact |
|----------|--------|--------|-------------|------------|
| Best quality | turbo4_0 | turbo4_0 | 3.6x | +0.8% |
| Best balance | turbo3_0 | turbo3_0 | 4.6x | +1-3% |
| Max context | turbo2_0 | turbo2_0 | 6.4x | +4-12% |
| Asymmetric | q8_0 | turbo3_0 | ~2.7x | +0.6% |

### Environment variable (for Ollama)

```bash
OLLAMA_KV_CACHE_TYPE=turbo3_0 ollama run llama3
```

Note: Ollama must be rebuilt with the patched llama.cpp for this to work.

## Architecture notes

- WHT rotation is handled by llama.cpp's existing Hadamard rotation infrastructure
- The turbo quantize/dequantize functions expect pre-rotated input (Phase 4a approach)
- Flash attention is REQUIRED for turbo V cache types
- head_dim must be power-of-2 and divisible by 64 (covers 64, 128, 256)
- Models with head_dim=80 (Qwen3-4B) fall back to q8_0 automatically

## References

- [TurboQuant paper](https://arxiv.org/abs/2504.19874) (ICLR 2026)
- [llama.cpp Discussion #20969](https://github.com/ggml-org/llama.cpp/discussions/20969)
- [TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus) — Metal implementation
- [Madreag/turbo3-cuda](https://github.com/Madreag/turbo3-cuda) — CUDA optimized
- [Aaryan-Kapoor/llama.cpp](https://github.com/Aaryan-Kapoor/llama.cpp/tree/turboquant-tq3_0) — CPU implementation
