# Windows GPU Test Instructions

Gib dem Windows-Claude diese Anweisungen:

---

## Schritt 1: Clone + Build

```powershell
cd ~\Desktop\KV-NEw
git clone -b feature/turboquant-kv-cache https://github.com/Keyvanhardani/llama.cpp-turboquant.git .
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release --target llama-perplexity llama-bench -j
```

## Schritt 2: Wikitext-2 Dataset holen

```powershell
pip install huggingface_hub pyarrow
python -c "
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
path = hf_hub_download('Salesforce/wikitext', 'wikitext-2-raw-v1/test-00000-of-00001.parquet', repo_type='dataset')
table = pq.read_table(path)
texts = table.column('text').to_pylist()
with open('wiki.test.raw', 'w', encoding='utf-8') as f:
    f.write('\n'.join(texts))
print(f'OK: {len(texts)} lines')
"
```

## Schritt 3: PPL Benchmark (GPU)

Alle 4 Configs nacheinander laufen lassen:

```powershell
# Baseline f16
.\build\bin\Release\llama-perplexity.exe -m D:\models\qwen2.5-3b-instruct-q4_k_m.gguf -f wiki.test.raw --ctx-size 512 --chunks 5 -ngl 99 --cache-type-k f16 --cache-type-v f16

# Baseline q8_0
.\build\bin\Release\llama-perplexity.exe -m D:\models\qwen2.5-3b-instruct-q4_k_m.gguf -f wiki.test.raw --ctx-size 512 --chunks 5 -ngl 99 --cache-type-k q8_0 --cache-type-v q8_0

# TurboQuant 4-bit
.\build\bin\Release\llama-perplexity.exe -m D:\models\qwen2.5-3b-instruct-q4_k_m.gguf -f wiki.test.raw --ctx-size 512 --chunks 5 -ngl 99 --cache-type-k turbo4_0 --cache-type-v turbo4_0

# TurboQuant 3-bit
.\build\bin\Release\llama-perplexity.exe -m D:\models\qwen2.5-3b-instruct-q4_k_m.gguf -f wiki.test.raw --ctx-size 512 --chunks 5 -ngl 99 --cache-type-k turbo3_0 --cache-type-v turbo3_0
```

## Schritt 4: Ergebnisse

Schreib die Ergebnisse in eine Datei:
```
D:\models\qwen2.5-3b-instruct-q4_k_m.gguf
GPU: [GPU Name]
ctx=512, chunks=5

f16:      PPL = ???
q8_0:     PPL = ???
turbo4_0: PPL = ???
turbo3_0: PPL = ???
```

## WICHTIG

- Falls turbo-Types mit ngl=99 crashen (SET_ROWS error): versuche ngl=0 (CPU-only)
- Falls GPU OOM: reduziere --chunks auf 3
- Die KV cache size sollte in den Logs stehen: `K (turbo4_0): X MiB, V (turbo4_0): X MiB`
