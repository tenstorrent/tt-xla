loader_path: third_party.tt_forge_models.ziadky_velora_ai_supported_models_gguf.causal_lm.pytorch.loader
variant_id: CRYSTAL_THINK_V2_Q4
arch: p150
status: DONE_PASS
test_function: test_ziadky_velora_crystal_think_v2_q4
samples_per_second: 31.544
ttft_ms: 130.972
prefill_pcc: 0.997637
first_decode_pcc: 0.997533
top_perf_samples_per_sec: 76.539
pct_of_target: 41.2
roofline_bound: dram
optimization_level: 2
trace_enabled: false
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_ziadky_velora_crystal_think_v2_q4

## Test
tests/benchmark/test_llms.py::test_ziadky_velora_crystal_think_v2_q4

## Model
- HF name:    ZiADKY/VeloraAI_SupportedModels
- Loader:     third_party.tt_forge_models.ziadky_velora_ai_supported_models_gguf.causal_lm.pytorch.loader
- Variant:    CRYSTAL_THINK_V2_Q4

## Test config landed
- optimization_level:        2
- trace_enabled:             false
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  31.544
- TTFT (ms):          130.972
- Prefill PCC:        0.997637
- First decode PCC:   0.997533
- Wall clock:         0:09:01
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_ziadky_velora_crystal_think_v2_q4_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 41.2% (31.544 / 76.539)

Note: gap from roofline is explained by trace_enabled=False (required due to
TTIRToTTNNCommon pipeline failure when compiling logits model for full 32-layer
model with trace_enabled=True on p150).

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
- total_flops:             257425408128
- breakdown.matmul:        257425408128
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        301989888
- memory_bytes: 603979776
- memory_gb:    0.5625

### Params
- count:                  4411424320
- effective_count:        4022468160
- memory_bytes:           8822848640
- memory_gb:              8.217
- effective_memory_bytes: 8044936320
- effective_memory_gb:    7.492
- embedding_count:        388956160
- embedding_memory_bytes: 777912320

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 76.5390
- top_perf_time_ms:         13.0652
- dram_time_ms:             8.7102
- compute_time_ms_lofi:     0.2925
- compute_time_ms_hifi2:    0.5851
- compute_time_ms_hifi3:    0.8776
- compute_time_ms_hifi4:    1.1701

## Files changed
- tests/benchmark/test_llms.py (added test_ziadky_velora_crystal_think_v2_q4)
- tests/benchmark/benchmarks/llm_benchmark.py (guard get_weight_dtype_config_path with hasattr)
- third_party/tt_forge_models (submodule updated: 93218a34fc → 951272ecd2)

## tt-forge-models submodule
93218a34fc → 951272ecd2: Updated to include ziadky_velora_ai_supported_models_gguf loader (added in arch-c-36-tt-xla-dev/nsmith/hf-bringup-2 branch)
