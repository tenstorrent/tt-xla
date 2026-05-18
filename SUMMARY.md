loader_path: third_party.tt_forge_models.codellama_13b_instruct_gguf.causal_lm.pytorch.loader
variant_id: 13B_Instruct_GGUF
arch: p150
status: DONE_FAIL
test_function: test_codellama_13b_instruct_gguf
samples_per_second: 14.636045274181038
ttft_ms: 562.461919
prefill_pcc: 0.999090
first_decode_pcc: 0.866670
top_perf_samples_per_sec: 22.78
pct_of_target: 64.2
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: "First decode PCC=0.867 consistently below required 0.94 across opt_level=1 and 2, with and without bfp_bf8 weight quantization, and with fp32_dest_acc_en=True; likely numerical precision degradation from Q4_K_M GGUF quantization accumulating across 40 transformer layers"

# Benchmark added: test_codellama_13b_instruct_gguf

## Test
tests/benchmark/test_llms.py::test_codellama_13b_instruct_gguf

## Model
- HF name:    TheBloke/CodeLlama-13B-Instruct-GGUF (GGUF file: codellama-13b-instruct.Q4_K_M.gguf)
- Loader:     third_party.tt_forge_models.codellama_13b_instruct_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.CODELLAMA_13B_INSTRUCT_GGUF (13B_Instruct_GGUF)

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8" (harness default)
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  14.636045274181038
- TTFT (ms):          562.461919
- Prefill PCC:        0.999090
- First decode PCC:   0.866670 (FAILED — required 0.94)
- Wall clock:         0:13:40
- Hardware:           p150 (blackhole, 4x PCI 0xb140)

## PCC Failure Investigation
Configurations tried — all fail first decode PCC at ~0.867:

| Config                                          | First Decode PCC |
|-------------------------------------------------|-----------------|
| optimization_level=2, bfp_bf8 (default)        | 0.866670        |
| optimization_level=1, bfp_bf8                  | 0.866526        |
| optimization_level=1, no weight dtype           | 0.873039        |
| optimization_level=1, bfp_bf8, fp32_dest_acc_en=True | 0.866526  |

Single-layer runs (num_layers=1) pass easily (PCC≥0.992), indicating
the degradation accumulates across all 40 transformer layers with
Q4_K_M GGUF quantized weights.

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_codellama_13b_instruct_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 14.636 / 22.78 = 64.2%

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
- total_flops:             822508257408
- breakdown.matmul:        822508257408
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
- count:                  13016028355
- effective_count:        12852106435
- memory_bytes:           13983596296
- memory_gb:              13.023238904774189
- effective_memory_bytes: 13655752456
- effective_memory_gb:    12.717910535633564
- embedding_count:        163921920
- embedding_memory_bytes: 327843840

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 22.7800
- top_perf_time_ms:         43.8981
- dram_time_ms:             29.2654
- compute_time_ms_lofi:     0.9347
- compute_time_ms_hifi2:    1.8693
- compute_time_ms_hifi3:    2.8040
- compute_time_ms_hifi4:    3.7387

## Files changed
- tests/benchmark/test_llms.py (added test_codellama_13b_instruct_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: use getattr for get_weight_dtype_config_path to handle loaders that don't implement it)

## tt-forge-models submodule
no change
