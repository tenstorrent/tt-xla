loader_path: third_party.tt_forge_models.deepseek_prover_v1_5_sft.causal_lm.pytorch.loader
variant_id: 7B
arch: p150
status: DONE_PASS
test_function: test_deepseek_prover_v1_5_sft_7b
samples_per_second: 25.876062896480317
ttft_ms: 323.230292
prefill_pcc: 0.996873
first_decode_pcc: 0.988454
top_perf_samples_per_sec: 44.1463
pct_of_target: 58.6
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_deepseek_prover_v1_5_sft_7b

## Test
tests/benchmark/test_llms.py::test_deepseek_prover_v1_5_sft_7b

## Model
- HF name:    deepseek-ai/DeepSeek-Prover-V1.5-SFT
- Loader:     third_party.tt_forge_models.deepseek_prover_v1_5_sft.causal_lm.pytorch.loader
- Variant:    ModelVariant.DEEPSEEK_PROVER_V1_5_SFT_7B

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  25.876062896480317
- TTFT (ms):          323.230292
- Prefill PCC:        0.996873
- First decode PCC:   0.988454
- Wall clock:         0:06:37
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_deepseek_prover_v1_5_sft_7b_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 58.6%

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
- total_flops:             415403868288
- breakdown.matmul:        415403868288
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        1006632960
- memory_bytes: 2013265920
- memory_gb:    1.875

### Params
- count:                  6910365891
- effective_count:        6490935491
- memory_bytes:           7735714568
- memory_gb:              7.204445607960224
- effective_memory_bytes: 6896853768
- effective_memory_gb:    6.423195607960224
- embedding_count:        419430400
- embedding_memory_bytes: 838860800

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 44.1463
- top_perf_time_ms:         22.6520
- dram_time_ms:             15.1013
- compute_time_ms_lofi:     0.4720
- compute_time_ms_hifi2:    0.9441
- compute_time_ms_hifi3:    1.4161
- compute_time_ms_hifi4:    1.8882

## Files changed
- tests/benchmark/test_llms.py (added test_deepseek_prover_v1_5_sft_7b)
- tests/benchmark/benchmarks/llm_benchmark.py (fixed hasattr guard for get_weight_dtype_config_path)
- .github/workflows/perf-bench-matrix.json (added deepseek_prover_v1_5_sft_7b entry)

## tt-forge-models submodule
no change
