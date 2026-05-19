loader_path: third_party.tt_forge_models.mistral_7b_claude_chat_gguf.causal_lm.pytorch.loader
variant_id: 7B_GGUF
arch: p150
status: DONE_PASS
test_function: test_mistral_7b_claude_chat_gguf
samples_per_second: 35.896
ttft_ms: 296.723
prefill_pcc: 0.9996
first_decode_pcc: 0.9959
top_perf_samples_per_sec: 44.8551
pct_of_target: 80.0
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_mistral_7b_claude_chat_gguf

## Test
tests/benchmark/test_llms.py::test_mistral_7b_claude_chat_gguf

## Model
- HF name:    TheBloke/Mistral-7B-Claude-Chat-GGUF
- Loader:     third_party.tt_forge_models.mistral_7b_claude_chat_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.MISTRAL_7B_CLAUDE_CHAT_GGUF (7B_GGUF)

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  35.896
- TTFT (ms):          296.723
- Prefill PCC:        0.9996
- First decode PCC:   0.9959
- Wall clock:         0:09:02
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_mistral_7b_claude_chat_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 80.0% (35.896 / 44.8551)

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
- total_flops:             455065206912
- breakdown.matmul:        455065206912
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
- count:                  7241732291
- effective_count:        7110660291
- memory_bytes:           7817470728
- memory_gb:              7.28
- effective_memory_bytes: 7555326728
- effective_memory_gb:    7.04
- embedding_count:        131072000
- embedding_memory_bytes: 262144000

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 44.8551
- top_perf_time_ms:         22.2940
- dram_time_ms:             14.8627
- compute_time_ms_lofi:     0.5171
- compute_time_ms_hifi2:    1.0342
- compute_time_ms_hifi3:    1.5514
- compute_time_ms_hifi4:    2.0685

## Files changed
- tests/benchmark/test_llms.py (added test_mistral_7b_claude_chat_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (guard get_weight_dtype_config_path with hasattr)
- .github/workflows/perf-bench-matrix.json (added mistral_7b_claude_chat_gguf entry)

## tt-forge-models submodule
no change
