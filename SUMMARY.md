loader_path: third_party.tt_forge_models.huihui_qwen3_5_9b_abliterated_grimoire_dpo_i1_gguf.causal_lm.pytorch.loader
variant_id: 9B_Abliterated_Grimoire_DPO_i1_GGUF
arch: p150
status: DONE_PASS
test_function: test_huihui_qwen3_5_9b_abliterated_grimoire_dpo_i1_gguf
samples_per_second: 34.45
ttft_ms: 317.47
prefill_pcc: 0.997924
first_decode_pcc: 0.997594
top_perf_samples_per_sec: 49.739
pct_of_target: 69.3
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_huihui_qwen3_5_9b_abliterated_grimoire_dpo_i1_gguf

## Test
tests/benchmark/test_llms.py::test_huihui_qwen3_5_9b_abliterated_grimoire_dpo_i1_gguf

## Model
- HF name:    mradermacher/Huihui-Qwen3.5-9B-abliterated-Grimoire-DPO-i1-GGUF
- Loader:     third_party.tt_forge_models.huihui_qwen3_5_9b_abliterated_grimoire_dpo_i1_gguf.causal_lm.pytorch.loader
- Variant:    9B_Abliterated_Grimoire_DPO_i1_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  34.45
- TTFT (ms):          317.47
- Prefill PCC:        0.997924
- First decode PCC:   0.997594
- Wall clock:         0:10:12
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_huihui_qwen3_5_9b_abliterated_grimoire_dpo_i1_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 69.3% (34.45 / 49.74)

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
- total_flops:             417282916480
- breakdown.matmul:        417282916480
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        134217728
- memory_bytes: 268435456
- memory_gb:    0.25

### Params
- count:                  7537438915
- effective_count:        6520320195
- memory_bytes:           8962335496
- memory_gb:              8.346825368702412
- effective_memory_bytes: 6928098056
- effective_memory_gb:    6.452294118702412
- embedding_count:        1017118720
- embedding_memory_bytes: 2034237440

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 49.7390
- top_perf_time_ms:         20.1050
- dram_time_ms:             13.4033
- compute_time_ms_lofi:     0.4742
- compute_time_ms_hifi2:    0.9484
- compute_time_ms_hifi3:    1.4226
- compute_time_ms_hifi4:    1.8967

## Files changed
- tests/benchmark/test_llms.py (new test function added)
- tests/benchmark/benchmarks/llm_benchmark.py (general infrastructure fix: guard get_weight_dtype_config_path with hasattr)
- .github/workflows/perf-bench-matrix.json (new entry added)

## tt-forge-models submodule
no change
