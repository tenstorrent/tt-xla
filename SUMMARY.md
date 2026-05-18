loader_path: third_party.tt_forge_models.estopian_maid_gguf.causal_lm.pytorch.loader
variant_id: KatyTheCutie_13B_Q4_K_S_GGUF
arch: p150
status: DONE_PASS
test_function: test_estopian_maid_gguf_katy_the_cutie_13b_q4_k_s
samples_per_second: 12.475011481897564
ttft_ms: 650.711739
prefill_pcc: 0.999323
first_decode_pcc: 0.986472
top_perf_samples_per_sec: 22.7802
pct_of_target: 54.8
roofline_bound: dram
optimization_level: 1
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: estopian_maid_gguf_katy_the_cutie_13b_q4_k_s

## Test
tests/benchmark/test_llms.py::test_estopian_maid_gguf_katy_the_cutie_13b_q4_k_s

## Model
- HF name:    KatyTheCutie/EstopianMaid-13B-GGUF
- Loader:     third_party.tt_forge_models.estopian_maid_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.KATYTHECUTIE_ESTOPIAN_MAID_13B_Q4_K_S_GGUF

## Test config landed
- optimization_level:        1
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  12.475011481897564
- TTFT (ms):          650.711739
- Prefill PCC:        0.999323
- First decode PCC:   0.986472
- Wall clock:         0:06:57
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_estopian_maid_gguf_katy_the_cutie_13b_q4_k_s_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 54.8% (12.475 / 22.7802)

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
- total_flops:             822503014528
- breakdown.matmul:        822503014528
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        1677721600
- memory_bytes: 3355443200
- memory_gb:    3.125

### Params
- count:                  13015864515
- effective_count:        12852024515
- memory_bytes:           13983345416
- memory_gb:              13.023
- effective_memory_bytes: 13655665416
- effective_memory_gb:    12.718
- embedding_count:        163840000
- embedding_memory_bytes: 327680000

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 22.7802
- top_perf_time_ms:         43.8979
- dram_time_ms:             29.2652
- compute_time_ms_lofi:     0.9347
- compute_time_ms_hifi2:    1.8693
- compute_time_ms_hifi3:    2.8040
- compute_time_ms_hifi4:    3.7387

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py
- .github/workflows/perf-bench-matrix.json

## tt-forge-models submodule
no change
