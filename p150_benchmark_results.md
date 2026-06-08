# P150 vLLM Benchmark Results

**Dates**: 2026-06-05 (initial 8k sweep on bh-38-special), 2026-06-08 (16k sweep, GMU experiments, trace investigation on bh-30-special)
**Hardware**: P150b Blackhole, 32 GB DRAM, single chip
**tt-xla branch**: `akhan/p150-vllm-benchmarks` (= origin/kmabee/llm_kv_cache_seq_len_work @ c8cc0e739 + benchmark commits)
**tt-mlir**: target pin is `kmabee/mlir_525_opt1_improvements @ 66d2edc34`; today's session built against `c5f398432a` because the branch-pinned SHA's transitive `tt-umd` dep (`8aa688168e1b`) has been force-pushed off origin. BFP8 KV cache path still functional on `c5f398432a`.
**Config**: opt-1, BFP8 weights, BFP8 KV cache, cpu_sampling=True, enable_trace=True (unless noted otherwise)

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
| Llama-3.2-3B-Instruct | 16 | 16384 | **TRACE ERROR** | 3:47 | - | - | `from_device` on tensor[16, 16384] (262K tokens) — **same trigger as b32×s8k** (262K tokens). Trace fails by total-token-count threshold, not seq-length alone |
| Llama-3.1-8B-Instruct | 8 | 16384 | **OOM** | 5:52 | - | - | 3.75 GB needed, 0.54 GB free/bank, 0.35 GB largest (fragmented) |
| Falcon3-7B-Base | 8 | 16384 | **OOM** | 5:21 | - | - | 6.04 GB needed, 0.56 GB free/bank, 0.27 GB largest |
| Qwen3-4B | 8 | 16384 | PASS | 15:12 | 148,769 | 17.1 | req 0 outlier (TTFT 77.8s, decode_tps 1.4); reqs 1-7 uniform |
| Qwen3-8B | 8 | 16384 | **OOM** | 6:27 | - | - | 3.22 GB needed, 0.72 GB free/bank, 0.35 GB largest |

## Results Matrix (8k seq len, batch=16, GMU=0.05)

| Model | Batch | Seq Len | Status | Time | TTFT (ms) | Decode TPS/req | Notes |
|-------|-------|---------|--------|------|-----------|----------------|-------|
| Llama-3.1-8B-Instruct | 16 | 8192 | **OOM** | 8:05 | - | - | 3.76 GB needed, 1.01 GB free/bank (2× default), 0.40 GB largest — **fragmentation**, not capacity |
| Falcon3-7B-Base | 16 | 8192 | **OOM** | 5:03 | - | - | 6.04 GB needed, 0.49 GB free/bank — **lower GMU didn't help Falcon** (vs 2× help for Llama-3.1-8B) |
| Qwen3-8B | 16 | 8192 | **PASS** | 14:49 | 128,233 | 11.2 | **GMU=0.05 unlocked b16 for Qwen3-8B** (aggregate TPS 180 vs 112 at b8 — +60%) |

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

**With GMU=0.05 tuning**: Qwen3-8B unlocks b16/s8k (+60% aggregate TPS over b8). Llama-3.1-8B and Falcon3-7B still OOM at b16 even with GMU=0.05 — Llama-3.1-8B fits in total free DRAM but the heap is fragmented; Falcon3-7B doesn't free meaningful DRAM by lowering GMU. Different memory profiles across the 7-8B class — per-model tuning is required.

**Max batch at 16k seq (b8 only)**: only 3-4B models (Llama-3.2-3B, Qwen3-4B) fit. All 7-8B models OOM.

## OOM Root Cause

The bottleneck is **prefill activation memory**, not KV cache. batch × seq_len tokens are processed in one matmul during prefill. For batch=32 × 8192:
- Token matrix: 262,144 × hidden_dim → multi-GB intermediate buffer
- P150 has 8 DRAM banks × 4.27 GB/bank = 34.2 GB total
- After model weights (~4-8 GB BFP8) + KV cache (gpu_memory_utilization × 32 GB), only 0.5-1.7 GB/bank remains
- The single matmul activation exceeds available per-bank DRAM

Kyle's upcoming "prefill simplification" changes (splitting prefill into smaller chunks) should fix this by reducing peak activation memory.

## Trace Error Investigation (enable_trace=True + cpu_sampling=True)

**Filed as tt-xla #5130.**

The compiler emits `from_device` on the model's `[batch, seq]` uint32 output tensor in `_precompile_backbone`, which violates trace's "all outputs on device" rule. **Failure depends on total token count (batch × seq), not seq length alone.**

| Config | b × s tokens | Trace |
|---|---|---|
| b8 × s16k | 131,072 | PASS |
| b16 × s8k | 131,072 | PASS |
| b32 × s4k | 131,072 | **PASS** |
| b16 × s16k | 262,144 | **FAIL** (this session) |
| b32 × s8k | 262,144 | **FAIL** (original) |

Threshold sits somewhere in (131K, 262K). Mid-point bracket attempt b32 × s6k (= 196,608) hit a different error first (`paged_fill_cache` validation: `input_shape[2] <= effective_block_size * page_table_shape[1]`), so s=6144 max_model_len is likely page-size-misaligned — does not narrow the trace boundary.

Error format:
```
error: 'ttnn.capture_or_execute_trace' op All output tensors of trace function must be on device.
%2 = "ttnn.from_device"(%1) : tensor<B×S×ui32> → ..., system_memory
```

Distinct from tt-xla #4570 (`cpu_sampling=False` path fails in the sampler op `OpModel<SamplingOp>::getOpConstraints` regardless of size). Same end symptom (`from_device` → trace verifier reject), different op and different trigger. The `cpu_sampling=True` combination broken here is the workaround #4570 currently recommends.

## Environment Notes
- Board was hung from a prior crashed process; `tt-smi -r` fixed it
- `tt-smi` installed in venv: `pip install tt-smi`
- Each test must run in its own pytest invocation (InitializeComputationClient crash)
- Per-model default `gpu_memory_utilization`: 0.3 for 3B, 0.25 for 4B, 0.2 for 7B/8B
- `_p150_config()` respects `TT_BENCHMARK_TRACE` (0/1 override) and `TT_BENCHMARK_GMU` (float override). The GMU env var was wired through in this session's commit (`49b53f4f2`).

## Runner Commands
```bash
source venv/activate

# batch=8, 8k seq, opt-1
_BENCH_OPTIMIZATION_LEVEL=1 TT_BENCHMARK_P150_MAX_MODEL_LEN=8192 pytest -sv \
  "tests/benchmark/test_vllm_benchmarks.py::test_vllm_p150_benchmark[p150-llama-3.2-3b-instruct-batch8]"

# batch=8, 16k seq, opt-1
_BENCH_OPTIMIZATION_LEVEL=1 TT_BENCHMARK_P150_MAX_MODEL_LEN=16384 pytest -sv \
  "tests/benchmark/test_vllm_benchmarks.py::test_vllm_p150_benchmark[p150-llama-3.2-3b-instruct-batch8]"

# batch=16 with lowered GMU=0.05 (e.g. Qwen3-8B which only fits this way)
_BENCH_OPTIMIZATION_LEVEL=1 TT_BENCHMARK_P150_MAX_MODEL_LEN=8192 TT_BENCHMARK_GMU=0.05 pytest -sv \
  "tests/benchmark/test_vllm_benchmarks.py::test_vllm_p150_benchmark[p150-qwen3-8b-batch16]"
```
