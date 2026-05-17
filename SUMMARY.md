loader_path: third_party.tt_forge_models.bartowski_olmo_2_1124_7b_instruct_gguf.causal_lm.pytorch.loader
variant_id: OLMo_2_1124_7B_Instruct_GGUF
arch: n150
status: DONE_FAIL
test_function: test_bartowski_olmo_2_1124_7b_instruct_gguf
samples_per_second: null
ttft_ms: null
prefill_pcc: null
first_decode_pcc: null
top_perf_samples_per_sec: 23.3858
pct_of_target: null
roofline_bound: dram
optimization_level: 0
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: "OOM during PCC benchmark at opt_level=0 (device DRAM ~12GB exhausted after perf model compilation; cannot fit logits model); opt_level>=1 fails with SDPA type mismatch: 'ttnn.scaled_dot_product_attention' op Query and result must have same element type"

# Benchmark added: test_bartowski_olmo_2_1124_7b_instruct_gguf

## Test
tests/benchmark/test_llms.py::test_bartowski_olmo_2_1124_7b_instruct_gguf

## Model
- HF name:    bartowski/OLMo-2-1124-7B-Instruct-GGUF
- Loader:     third_party.tt_forge_models.bartowski_olmo_2_1124_7b_instruct_gguf.causal_lm.pytorch.loader
- Variant:    OLMo_2_1124_7B_Instruct_GGUF

## Test config landed
- optimization_level:        0
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8 (default from benchmark infrastructure)
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Failure details
Two blocking issues prevent a full benchmark pass:

### 1. Compiler bug at optimization_level >= 1
Both optimization_level=1 and optimization_level=2 fail during model compilation with:
```
loc("dot.391"): error: 'ttnn.scaled_dot_product_attention' op Query and result must have the same element type
ValueError: Error code: 13
```
This is a bug in tt-mlir's SDPA lowering for the OLMo-2 attention kernel.

### 2. OOM during PCC benchmark at optimization_level=0
The performance benchmark (no logits returned) completes successfully at opt_level=0,
but immediately after, when the benchmark tries to compile the logits version of the model
for PCC verification, the device DRAM (~12 GB on n150) runs out:
```
TT_FATAL: Out of Memory: Not enough space to allocate 205520896 B DRAM buffer across 12 banks,
where each bank needs to store 17127424 B, but bank size is 1071821792 B
(allocated: 1061253728 B, free: 10568064 B, largest free block: 4897824 B)
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```
After the performance model compilation, only ~10 MB of DRAM remains free (out of ~12 GB total).
The PCC phase needs to compile a second version of the model (with logits returned), which
exhausts the available DRAM.

Performance benchmark DID run successfully before the OOM; estimated decode throughput
from iteration times in log: ~1.82 steps/sec (110 decode steps at ~548 ms each), TTFT ~3217 ms.
This is ~7.8% of the 23.39 samples/sec roofline ceiling — consistent with opt_level=0 (DRAM
bound, no SRAM optimization).

## Infrastructure fix also landed
`tests/benchmark/benchmarks/llm_benchmark.py`: Added `hasattr` guard around
`model_loader.get_weight_dtype_config_path()` call (matching the pattern already used in
`tests/runner/testers/torch/dynamic_torch_model_tester.py`) to fix an
`AttributeError` for loaders that don't implement this optional method.

## Measured (full model, defaults)
- Sample per second:  null (PCC phase OOM before print_benchmark_results is reached)
- TTFT (ms):          null (same reason; estimated ~3217 ms from log iteration times)
- Prefill PCC:        null
- First decode PCC:   null
- Wall clock:         ~9 min (model load + compilation + 111 perf iterations + OOM)
- Hardware:           n150 (wormhole_b0, 12 GB DRAM)

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_bartowski_olmo_2_1124_7b_instruct_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: N/A (test did not complete PCC)

### System
- arch:                        wormhole_b0
- chip_count_in_system_desc:   2
- single_chip_assumption:      True
- worker_grid_cores:           64
- dram_bandwidth_bytes_per_sec: 288000000000

### Peak FLOPs
- lofi:  256000000000000
- hifi2: 128000000000000
- hifi3: 85333333333333
- hifi4: 64000000000000

### Compute
- total_flops:             442918502528
- breakdown.matmul:        442918502528
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        1073741824
- memory_bytes: 2147483648
- memory_gb:    2

### Params
- count:                  7298617542
- effective_count:        6887575750
- memory_bytes:           8140628756
- memory_gb:              7.58
- effective_memory_bytes: 7318545172
- effective_memory_gb:    6.82
- embedding_count:        411041792
- embedding_memory_bytes: 822083584

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 23.3858
- top_perf_time_ms:         42.7610
- dram_time_ms:             28.5073
- compute_time_ms_lofi:     1.7302
- compute_time_ms_hifi2:    3.4603
- compute_time_ms_hifi3:    5.1905
- compute_time_ms_hifi4:    6.9206

## Files changed
- tests/benchmark/test_llms.py (new test function added)
- .github/workflows/perf-bench-matrix.json (new matrix entry added)
- tests/benchmark/benchmarks/llm_benchmark.py (general hasattr fix for get_weight_dtype_config_path)
- SUMMARY.md

## tt-forge-models submodule
no change
