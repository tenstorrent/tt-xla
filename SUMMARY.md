loader_path: third_party.tt_forge_models.llama_3_1_8b_instruct_bnb_nf4.causal_lm.pytorch.loader
variant_id: 3.1_8B_Instruct_BNB_NF4
arch: p150
status: DONE_PASS
test_function: test_llama_3_1_8b_instruct_bnb_nf4
samples_per_second: 33.077
ttft_ms: 314.196
prefill_pcc: 0.999004
first_decode_pcc: 0.998620
top_perf_samples_per_sec: 42.5800
pct_of_target: 77.7
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: llama_3_1_8b_instruct_bnb_nf4

## Test
tests/benchmark/test_llms.py::test_llama_3_1_8b_instruct_bnb_nf4

## Model
- HF name:    hugging-quants/Meta-Llama-3.1-8B-Instruct-BNB-NF4
- Loader:     third_party.tt_forge_models.llama_3_1_8b_instruct_bnb_nf4.causal_lm.pytorch.loader
- Variant:    ModelVariant.LLAMA_3_1_8B_INSTRUCT_BNB_NF4 = "3.1_8B_Instruct_BNB_NF4"

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  33.077
- TTFT (ms):          314.196
- Prefill PCC:        0.999004
- First decode PCC:   0.998620
- Wall clock:         0:08:53
- Hardware:           p150

## Notes on bring-up
This model uses BNB (bitsandbytes) NF4 4-bit quantization. Two general-purpose
fixes were applied to the benchmark infrastructure (llm_benchmark.py):
1. Added `_dequantize_bnb_model()` helper that replaces Linear4bit/Linear8bitLt
   layers with regular bf16 Linear layers before the CPU reference run. This
   ensures both the CPU golden and TT device see identical float weights.
2. Added `getattr` guard for `get_weight_dtype_config_path` (not all loaders
   implement this method).
The bitsandbytes (>=0.46.1) and kernels packages are required and added to
perf-bench-matrix.json pyreq.

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_llama_3_1_8b_instruct_bnb_nf4_perf_metrics_3.json
Achieved vs top_perf_samples_per_sec: 77.7% (33.077 / 42.58)

### System
- arch:                        blackhole
- chip_count_in_system_desc:   1
- single_chip_assumption:      True
- worker_grid_cores:           110
- dram_bandwidth_bytes_per_sec: 512000000000

### Peak FLOPs
- lofi:  880000000000000
- hifi2: 440000000000000
- hifi3: 293333333333333
- hifi4: 220000000000000

### Compute
- total_flops:             480298139776
- breakdown.matmul:        480298139776
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        268435456
- memory_bytes: 536870912
- memory_gb:    0.5

### Params
- count:                  8030261443
- effective_count:        7504924867
- memory_bytes:           9024905992
- memory_gb:              8.405
- effective_memory_bytes: 7974232840
- effective_memory_gb:    7.427
- embedding_count:        525336576
- embedding_memory_bytes: 1050673152

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 42.5800
- top_perf_time_ms:         23.4852
- dram_time_ms:             15.6568
- compute_time_ms_lofi:     0.5458
- compute_time_ms_hifi2:    1.0916
- compute_time_ms_hifi3:    1.6374
- compute_time_ms_hifi4:    2.1832

## Files changed
- tests/benchmark/test_llms.py (added test_llama_3_1_8b_instruct_bnb_nf4)
- tests/benchmark/benchmarks/llm_benchmark.py (added _dequantize_bnb_model helper, getattr guard for get_weight_dtype_config_path)
- .github/workflows/perf-bench-matrix.json (added llama_3_1_8b_instruct_bnb_nf4 entry with bitsandbytes+kernels pyreq)

## tt-forge-models submodule
no change
