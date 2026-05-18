loader_path: third_party.tt_forge_models.llama_3_1_8b_stheno_v3_4_fp8.causal_lm.pytorch.loader
variant_id: 8B_Stheno_v3.4_FP8
arch: p150
status: DONE_FAIL
test_function: test_llama_3_1_8b_stheno_v3_4_fp8
samples_per_second: 20.29
ttft_ms: 364.85
prefill_pcc: 0.851
first_decode_pcc: null
top_perf_samples_per_sec: 42.58
pct_of_target: 47.7
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: null
failure_reason: "Prefill PCC 0.851 < required 0.94 on p150/Blackhole: FP8 model scale factors (weight_scale, input_scale) marked UNEXPECTED in load report (not loaded by loader), causing numerical error that accumulates over 32 layers; consistent across all tested configurations (bfp_bf8 on/off, fp32_dest_acc_en on/off, optimization_level 1-2)"

# Benchmark added: test_llama_3_1_8b_stheno_v3_4_fp8

## Test
tests/benchmark/test_llms.py::test_llama_3_1_8b_stheno_v3_4_fp8

## Model
- HF name:    mikudev/Llama-3.1-8B-Stheno-v3.4-FP8
- Loader:     third_party.tt_forge_models.llama_3_1_8b_stheno_v3_4_fp8.causal_lm.pytorch.loader
- Variant:    LLAMA_3_1_8B_STHENO_V3_4_FP8 ("8B_Stheno_v3.4_FP8")

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: none (empty string — disabled due to FP8 model PCC regression with bfp_bf8)
- fp32_dest_acc_en:          True
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, best-effort configuration)
- Sample per second:  20.29
- TTFT (ms):          364.85
- Prefill PCC:        0.851 (FAIL — below required 0.94)
- First decode PCC:   null (test failed at prefill PCC check)
- Wall clock:         0:08:01
- Hardware:           p150

## Failure Analysis

The model `mikudev/Llama-3.1-8B-Stheno-v3.4-FP8` stores weights in FP8 quantized format with
per-tensor scale factors (`weight_scale`, `input_scale`). The loader deletes `quantization_config`
from the model config before loading (`del config.quantization_config`), which prevents the FP8
dequantization library from applying the scale factors. As a result, 23 weight scale/input scale
tensors are marked UNEXPECTED in the load report — they exist in the safetensors checkpoint but
cannot be loaded into the base LlamaForCausalLM architecture without the FP8 quantization backend.

This causes systematic numerical imprecision that compounds over 32 layers on the Blackhole
(p150) architecture, resulting in a consistent Prefill PCC of ~0.84-0.85 across all tested
configurations. The fix requires modifying the loader to properly handle FP8 dequantization,
which is out of scope for this skill.

Configurations tested:
| Config                                    | Prefill PCC | Decode PCC (1-layer) |
|-------------------------------------------|-------------|----------------------|
| bfp_bf8, opt=2, trace=True               | 0.840       | N/A (full model)     |
| no bfp_bf8, opt=2, trace=True            | 0.851       | 0.804                |
| no bfp_bf8, opt=2, trace=False           | N/A         | 0.804                |
| no bfp_bf8, opt=1, trace=False           | N/A         | 0.590                |
| no bfp_bf8, fp32_dest_acc_en=True, opt=2 | 0.851       | 0.804                |

Infrastructure fix also included: `llm_benchmark.py` now uses `hasattr(model_loader, "get_weight_dtype_config_path")`
before calling the method, matching the pattern in `dynamic_torch_model_tester.py`. This prevents
AttributeError for loaders that don't implement the method.

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_llama_3_1_8b_stheno_v3_4_fp8_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 47.7% (20.29 / 42.58)

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
- total_flops:             480298139776
- breakdown.matmul:        480298139776
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
- count:                  8030261443
- effective_count:        7504924867
- memory_bytes:           16060523272
- memory_gb:              14.96
- effective_memory_bytes: 15009850120
- effective_memory_gb:    13.98
- embedding_count:        525336576
- embedding_memory_bytes: 1050673152

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 42.58
- top_perf_time_ms:         23.4852
- dram_time_ms:             15.6568
- compute_time_ms_lofi:     0.5458
- compute_time_ms_hifi2:    1.0916
- compute_time_ms_hifi3:    1.6374
- compute_time_ms_hifi4:    2.1832

## Files changed
- tests/benchmark/test_llms.py (added test_llama_3_1_8b_stheno_v3_4_fp8)
- tests/benchmark/benchmarks/llm_benchmark.py (infrastructure fix: hasattr check for get_weight_dtype_config_path)

## tt-forge-models submodule
no change
