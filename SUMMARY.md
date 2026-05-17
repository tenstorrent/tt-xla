loader_path: third_party.tt_forge_models.bartowski_alamios_mistral_small_3_1_draft_0_5b_gguf.causal_lm.pytorch.loader
variant_id: Mistral_Small_3_1_DRAFT_0_5B_GGUF
arch: n150
status: DONE_PASS
test_function: test_bartowski_alamios_mistral_small_3_1_draft_0_5b_gguf
samples_per_second: 80.96
ttft_ms: 250.16
prefill_pcc: 0.995354
first_decode_pcc: 0.997356
top_perf_samples_per_sec: 165.0211
pct_of_target: 49.1
roofline_bound: compute
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_bartowski_alamios_mistral_small_3_1_draft_0_5b_gguf

## Test
tests/benchmark/test_llms.py::test_bartowski_alamios_mistral_small_3_1_draft_0_5b_gguf

## Model
- HF name:    bartowski/alamios_Mistral-Small-3.1-DRAFT-0.5B-GGUF
- Loader:     third_party.tt_forge_models.bartowski_alamios_mistral_small_3_1_draft_0_5b_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.MISTRAL_SMALL_3_1_DRAFT_0_5B_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  80.96
- TTFT (ms):          250.16
- Prefill PCC:        0.995354
- First decode PCC:   0.997356
- Wall clock:         0:07:39
- Hardware:           n150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_bartowski_alamios_mistral_small_3_1_draft_0_5b_gguf_perf_metrics_0.json
Achieved vs top_perf_samples_per_sec: 49.1% (80.96 / 165.02)

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
- total_flops:             517105615936
- breakdown.matmul:        490137977920
- breakdown.linear:        26967638016
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        561
- memory_bytes: 2244

### KV cache
- count:        25165824
- memory_bytes: 50331648
- memory_gb:    0.046875

### Params
- count:                  592779299
- effective_count:        475338787
- memory_bytes:           739996040
- memory_gb:              0.6891750171780586
- effective_memory_bytes: 505115016
- effective_memory_gb:    0.4704250171780586
- embedding_count:        117440512
- embedding_memory_bytes: 234881024

### Roofline
- bound:                    compute
- top_perf_samples_per_sec: 165.0211
- top_perf_time_ms:         6.0598
- dram_time_ms:             1.7922
- compute_time_ms_lofi:     2.0199
- compute_time_ms_hifi2:    4.0399
- compute_time_ms_hifi3:    6.0598
- compute_time_ms_hifi4:    8.0798

## Files changed
- tests/benchmark/test_llms.py (added test_bartowski_alamios_mistral_small_3_1_draft_0_5b_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (fix: guard get_weight_dtype_config_path with hasattr)

## tt-forge-models submodule
no change
