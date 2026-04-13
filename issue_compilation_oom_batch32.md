**Title:** [vLLM] Compilation OOM at batch=32 with Llama-3.1-8B on single BH chip

---

## Summary

Precompilation of Llama-3.1-8B-Instruct OOMs on a single Blackhole chip (P150, 32GB DRAM) when `max_num_seqs=32` and `max_model_len=4096`. The OOM occurs during graph compilation (warmup), not during inference. Batch sizes 1-16 compile successfully at the same seq len.

## Reproduction

Branch: `kmabee/vllm_perf_high_batch_seq_len_debug` on tt-xla  
Machine: QB2 (P300X2), single P150 chip  
Model: `meta-llama/Llama-3.1-8B-Instruct`

```bash
cd tt-xla
source venv/activate

# This works (batch=16, len=4096)
pytest -sv tests/benchmark/test_vllm_benchmarks.py::test_vllm_benchmark[llama-3.1-8b-instruct-batch16] 

# This OOMs during compilation (batch=32, len=4096)
pytest -sv tests/benchmark/test_vllm_benchmarks.py::test_vllm_benchmark[llama-3.1-8b-instruct-batch32-len4096]
```

Benchmark config:
```python
VLLMBenchmarkConfig(
    model="meta-llama/Llama-3.1-8B-Instruct",
    batch_size=32,
    max_model_len=4096,
    max_num_batched_tokens=4096,  # capped to avoid token-dimension OOM
    gpu_memory_utilization=0.1,
    additional_config={
        "enable_const_eval": True,
        "cpu_sampling": False,
        "experimental_weight_dtype": "bfp_bf8",
        "optimization_level": 1,
    },
)
```

Note: `max_num_batched_tokens` is capped to 4096 (not the default 4096*32=131072) to enable chunked prefill. This required relaxing the assertion in `model_runner.py` that previously required `max_num_batched_tokens >= max_model_len * max_num_seqs`. Even with this cap, batch=32 still OOMs.

## Error

```
TT_FATAL: Out of Memory: Not enough space to allocate 3758096384 B DRAM buffer
across 8 banks, where each bank needs to store 469762048 B,
but bank size is 4273390016 B (allocated: 3078993536 B, free: 1194396480 B,
largest free block: 402653184 B)
```

This occurs ~15 minutes into warmup during precompilation of the model backbone.

## Analysis

The OOM is caused by the **batch dimension** in compiled tensors, not the token dimension:

- Input tensors: `(max_num_reqs, padded_tokens)` = `(32, 4096)`
- Page table: `(32, 128)` (128 = max_model_len / block_size)
- Attention intermediate buffers scale with batch dimension

Evidence — what compiles and what doesn't:

| Batch | max_model_len | Tensor shape | max_num_batched_tokens | Compiles? |
|---|---|---|---|---|
| 1 | 4096 | (1, 4096) | 4096 | Yes |
| 16 | 4096 | (16, 4096) | 65536 | Yes |
| 32 | 128 | (32, 128) | 4096 | Yes |
| 32 | 4096 | (32, 4096) | 4096 (capped) | OOM |
| 32 | 4096 | (32, 4096) | 131072 (default) | OOM |

The max compilable `batch * max_model_len` product appears to be between 65K (batch=16 * 4096) and 131K (batch=32 * 4096).

Note: `gpu_memory_utilization=0.1` allocates ~3.2GB for KV cache. Lowering this further might free enough DRAM for compilation to succeed, but would leave insufficient KV cache for 32 concurrent users at runtime.

## Benchmark Results (what does work)

All passing runs at gpu_mem=0.1, len=4096, greedy, device sampling, bfp_bf8:

| Batch | Decode tok/s per user | TTFT (ms) |
|---|---|---|
| 1 | 22.4 | 69 |
| 2 | 21.6 | 90 |
| 4 | 20.4 | — |
| 8 | 19.4 | — |
| 16 | 16.8 | 429 |

At len=128 (smaller compiled tensors), batch=32 works at 14.8 tok/s.

## Impact

- Cannot serve 32 concurrent users with 4K context on a single BH chip
- Limits practical deployment batch size to 16 at 4096 context
- Higher seq lens (8K, 16K, 64K) will have even lower batch limits

## Possible Solutions

1. **Reduce compilation intermediate DRAM** — compiler/TTNN could tile or stream large attention computations instead of allocating full buffers
2. **Lower gpu_memory_utilization during compilation** — free more DRAM for compilation, then allocate KV cache after warmup (not currently supported by vLLM)
3. **TP across 2 chips** — split attention computation across devices, halving per-chip DRAM requirements
4. **Find exact batch limit** — test batch=20, 24, 28 to find the maximum that compiles at len=4096

