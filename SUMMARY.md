loader_path: third_party.tt_forge_models.babylm_mop_10m_gpt2.causal_lm.pytorch.loader
variant_id: 10M
arch: p150
status: DONE_PASS
test_function: test_babylm_mop_10m_gpt2
samples_per_second: 167.66388279832586
ttft_ms: 84.55838
prefill_pcc: 0.999648
first_decode_pcc: 0.999488
top_perf_samples_per_sec: 1853.9876
pct_of_target: 9.0
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: babylm_mop_10m_gpt2

## Test
tests/benchmark/test_llms.py::test_babylm_mop_10m_gpt2

## Model
- HF name:    NeTS-lab/babylm-mop-10m-gpt2
- Loader:     third_party.tt_forge_models.babylm_mop_10m_gpt2.causal_lm.pytorch.loader
- Variant:    ModelVariant.BABYLM_MOP_10M_GPT2 (10M)

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  167.66
- TTFT (ms):          84.56
- Prefill PCC:        0.999648
- First decode PCC:   0.999488
- Wall clock:         0:02:07
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_babylm_mop_10m_gpt2_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 9.0%

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
- total_flops:             6588874752
- breakdown.matmul:        1150402560
- breakdown.linear:        5438472192
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        75497472
- memory_bytes: 150994944
- memory_gb:    0.140625

### Params
- count:                  121792644
- effective_count:        103031172
- memory_bytes:           147273596
- memory_gb:              0.13715922459959984
- effective_memory_bytes: 109750652
- effective_memory_gb:    0.10221325978636742
- embedding_count:        18761472
- embedding_memory_bytes: 37522944

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 1853.9876
- top_perf_time_ms:         0.5394
- dram_time_ms:             0.3596
- compute_time_ms_lofi:     0.0075
- compute_time_ms_hifi2:    0.0150
- compute_time_ms_hifi3:    0.0225
- compute_time_ms_hifi4:    0.0299

## Files changed
- tests/benchmark/test_llms.py (added test_babylm_mop_10m_gpt2)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: fallback tokenizer for non-batching tokenizers; graceful getattr for get_weight_dtype_config_path)

## tt-forge-models submodule
no change
