# P150 vLLM Benchmark Results

**Date**: 2026-06-05
**Machine**: bh-38-special (P150b Blackhole, 32 GB DRAM)
**tt-xla branch**: kmabee-setup (= origin/kmabee/llm_kv_cache_seq_len_work @ c8cc0e739)
**tt-mlir**: kmabee/mlir_525_opt1_improvements @ 66d2edc34
**Config**: opt-1, BFP8 weights, BFP8 KV cache, cpu_sampling=True, enable_trace=True

## Results Matrix (8k seq len)

| Model | Batch | Seq Len | Status | Time | TTFT (ms) | Decode TPS/req | Notes |
|-------|-------|---------|--------|------|-----------|----------------|-------|
| Llama-3.2-3B-Instruct | 1 | 128 | PASS | 3:34 | 51 | 28.5 | baseline sanity |
| Llama-3.2-3B-Instruct | 8 | 8192 | PASS | 5:48 | 31,020 | 20.4 | |
| Llama-3.2-3B-Instruct | 16 | 8192 | PASS | 8:07 | 60,202 | 17.2 | |
| Llama-3.2-3B-Instruct | 32 | 8192 | **OOM** | 4:44 | - | - | 8 GB activation, 1.6 GB free/bank |
| Llama-3.1-8B-Instruct | 8 | 8192 | PASS | 9:08 | 57,181 | 15.3 | |
| Llama-3.1-8B-Instruct | 16 | 8192 | **OOM** | 5:15 | - | - | 3.7 GB needed, 0.5 GB free |
| Falcon3-7B-Base | 8 | 8192 | PASS | 8:37 | 57,031 | 14.1 | |
| Falcon3-7B-Base | 16 | 8192 | **OOM** | 4:36 | - | - | 6 GB needed, 0.5 GB free |
| Qwen3-4B | 8 | 8192 | PASS | 8:26 | 48,086 | 16.7 | |
| Qwen3-4B | 16 | 8192 | PASS | 11:51 | 93,630 | 13.5 | |
| Qwen3-8B | 8 | 8192 | PASS | 10:05 | 57,217 | 14.0 | |
| Qwen3-8B | 16 | 8192 | **OOM** | 6:08 | - | - | 3.2 GB needed, 0.7 GB free |

## Results Matrix (16k seq len)

| Model | Batch | Seq Len | Status | Time | TTFT (ms) | Decode TPS/req | Notes |
|-------|-------|---------|--------|------|-----------|----------------|-------|
| Llama-3.2-3B-Instruct | 8 | 16384 | PASS | 9:27 | 88,454 | 22.0 | |
| Llama-3.2-3B-Instruct | 16 | 16384 | | | | | |
| Llama-3.1-8B-Instruct | 8 | 16384 | | | | | |
| Falcon3-7B-Base | 8 | 16384 | | | | | |
| Qwen3-4B | 8 | 16384 | | | | | |
| Qwen3-8B | 8 | 16384 | | | | | |

## Summary

**Max batch at 8k seq len (single P150, opt-1, BFP8):**
| Model | Max Batch @ 8k |
|-------|---------------|
| Llama-3.2-3B (3B params) | 16 |
| Qwen3-4B (4B params) | 16 |
| Falcon3-7B (7B params) | 8 |
| Llama-3.1-8B (8B params) | 8 |
| Qwen3-8B (8B params) | 8 |

**Pattern**: 3-4B models fit batch=16, 7-8B models max at batch=8. batch=32 OOMs on ALL models at 8k seq (activation memory exceeds free DRAM).

## OOM Root Cause

The bottleneck is **prefill activation memory**, not KV cache. batch × seq_len tokens are processed in one matmul during prefill. For batch=32 × 8192:
- Token matrix: 262,144 × hidden_dim → multi-GB intermediate buffer
- P150 has 8 DRAM banks × 4.27 GB/bank = 34.2 GB total
- After model weights (~4-8 GB BFP8) + KV cache (gpu_memory_utilization × 32 GB), only 0.5-1.7 GB/bank remains
- The single matmul activation exceeds available per-bank DRAM

Kyle's upcoming "prefill simplification" changes (splitting prefill into smaller chunks) should fix this by reducing peak activation memory.

## Batch=32 Trace Error (with enable_trace=True)

Even before OOM, batch=32 hits a compiler error with trace:
```
error: 'ttnn.capture_or_execute_trace' op All output tensors of trace function must be on device.
%2 = "ttnn.from_device"(%1) : (tensor<32x8192xui32, ...>) -> tensor<32x8192xui32, ..., system_memory>
```
Kyle confirmed: "trace doesn't work with opt-level=1 and cpu_sampling=False anyways (a preferred combo) I care less right now"

## Environment Notes
- Board was hung from a prior crashed process; `tt-smi -r` fixed it
- `tt-smi` installed in venv: `pip install tt-smi`
- Each test must run in its own pytest invocation (InitializeComputationClient crash)
- gpu_memory_utilization: 0.3 for 3B, 0.25 for 4B, 0.2 for 7B/8B
- Fixed `_p150_config()` to respect `TT_BENCHMARK_TRACE` env var

## Runner Commands
```bash
source venv/activate
# batch=8, 8k seq, opt-1
_BENCH_OPTIMIZATION_LEVEL=1 TT_BENCHMARK_P150_MAX_MODEL_LEN=8192 pytest -sv \
  "tests/benchmark/test_vllm_benchmarks.py::test_vllm_p150_benchmark[p150-llama-3.2-3b-instruct-batch8]"

# batch=8, 16k seq, opt-1
_BENCH_OPTIMIZATION_LEVEL=1 TT_BENCHMARK_P150_MAX_MODEL_LEN=16384 pytest -sv \
  "tests/benchmark/test_vllm_benchmarks.py::test_vllm_p150_benchmark[p150-llama-3.2-3b-instruct-batch8]"
```
