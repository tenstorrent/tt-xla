loader_path: third_party.tt_forge_models.llama_2_7b_chat_hf_q4f16_1_mlc.causal_lm.pytorch.loader
variant_id: llama_2_7b_chat_hf_q4f16_1_mlc
arch: p150
status: DONE_PASS
test_function: test_llama_2_7b_chat_hf_q4f16_1_mlc
samples_per_second: 20.52
ttft_ms: 413.45
prefill_pcc: 0.965109
first_decode_pcc: 0.946570
top_perf_samples_per_sec: 43.0916
pct_of_target: 47.6
roofline_bound: dram
optimization_level: 1
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_llama_2_7b_chat_hf_q4f16_1_mlc

## Test
tests/benchmark/test_llms.py::test_llama_2_7b_chat_hf_q4f16_1_mlc

## Model
- HF name:    mlc-ai/Llama-2-7b-chat-hf-q4f16_1-MLC
- Loader:     third_party.tt_forge_models.llama_2_7b_chat_hf_q4f16_1_mlc.causal_lm.pytorch.loader
- Variant:    LLAMA_2_7B_CHAT_HF_Q4F16_1_MLC

## Test config landed
- optimization_level:        1
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

Note: optimization_level=2 with bfp_bf8 causes first decode PCC to drop to 0.911
(below the 0.94 threshold) on the full 32-layer model. Reduced to optimization_level=1
which passes PCC at 0.946570.

## Measured (full model, defaults)
- Sample per second:  20.52
- TTFT (ms):          413.45
- Prefill PCC:        0.965109
- First decode PCC:   0.946570
- Wall clock:         0:13:13
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_llama_2_7b_chat_hf_q4f16_1_mlc_perf_metrics_3.json
Achieved vs top_perf_samples_per_sec: 47.6% (20.52 / 43.09)

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
- total_flops:             422852952192
- breakdown.matmul:        422852952192
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
- count:                  6738415811
- effective_count:        6607343811
- memory_bytes:           7282696968
- memory_gb:              6.782540090382099
- effective_memory_bytes: 7020552968
- effective_memory_gb:    6.538399465382099
- embedding_count:        131072000
- embedding_memory_bytes: 262144000

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 43.0916
- top_perf_time_ms:         23.2064
- dram_time_ms:             15.4709
- compute_time_ms_lofi:     0.4805
- compute_time_ms_hifi2:    0.9610
- compute_time_ms_hifi3:    1.4415
- compute_time_ms_hifi4:    1.9221

## Files changed
- tests/benchmark/test_llms.py — added test_llama_2_7b_chat_hf_q4f16_1_mlc
- tests/benchmark/benchmarks/llm_benchmark.py — fixed hasattr guard for get_weight_dtype_config_path (general infrastructure fix)
- .github/workflows/perf-bench-matrix.json — added llama_2_7b_chat_hf_q4f16_1_mlc matrix entry

## tt-forge-models submodule
no change
