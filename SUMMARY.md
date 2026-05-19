loader_path: third_party.tt_forge_models.pythia_6_9b_deduped_sft_tldr_gguf.causal_lm.pytorch.loader
variant_id: 6_9B_DEDUPED_SFT_TLDR_Q4_K_M_GGUF
arch: p150
status: DONE_PASS
test_function: test_pythia_6_9b_deduped_sft_tldr_gguf
samples_per_second: 16.5948
ttft_ms: 472.621
prefill_pcc: 0.997488
first_decode_pcc: 0.996692
top_perf_samples_per_sec: 42.8495
pct_of_target: 38.7
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_pythia_6_9b_deduped_sft_tldr_gguf

## Test
tests/benchmark/test_llms.py::test_pythia_6_9b_deduped_sft_tldr_gguf

## Model
- HF name:    RichardErkhov/HuggingFaceH4_-_EleutherAI_pythia-6.9b-deduped__sft__tldr-gguf
- Loader:     third_party.tt_forge_models.pythia_6_9b_deduped_sft_tldr_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.PYTHIA_6_9B_DEDUPED_SFT_TLDR_Q4_K_M_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  16.5948
- TTFT (ms):          472.621
- Prefill PCC:        0.997488
- First decode PCC:   0.996692
- Wall clock:         0:21:04
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_pythia_6_9b_deduped_sft_tldr_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 38.7% (16.5948 / 42.8495)

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
- total_flops:             425575055392
- breakdown.matmul:        13220446240
- breakdown.linear:        412354609152
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
- count:                  6857302163
- effective_count:        6650732691
- memory_bytes:           7481147976
- memory_gb:              6.967361994087696
- effective_memory_bytes: 7068009032
- effective_memory_gb:    6.582596369087696
- embedding_count:        206569472
- embedding_memory_bytes: 413138944

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 42.8495
- top_perf_time_ms:         23.3375
- dram_time_ms:             15.5583
- compute_time_ms_lofi:     0.4836
- compute_time_ms_hifi2:    0.9672
- compute_time_ms_hifi3:    1.4508
- compute_time_ms_hifi4:    1.9344

## Files changed
- tests/benchmark/test_llms.py (added test_pythia_6_9b_deduped_sft_tldr_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (infrastructure fix: graceful get_weight_dtype_config_path)
- .github/workflows/perf-bench-matrix.json (added pythia_6_9b_deduped_sft_tldr_gguf entry)

## tt-forge-models submodule
no change
