loader_path: third_party.tt_forge_models.gemma_2b_it_gguf.causal_lm.pytorch.loader
variant_id: 2B_IT_GGUF
arch: p150
status: DONE_FAIL
test_function: test_gemma_2b_it_gguf
samples_per_second: null
ttft_ms: null
prefill_pcc: null
first_decode_pcc: null
top_perf_samples_per_sec: 130.0967
pct_of_target: null
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: "First decode PCC 0.936944 < 0.94 threshold at optimization_level=1 (best achieved); GGUF Q4_K_M quantization errors compound across 18 decoder layers. optimization_level=2 gives PCC=0.881, optimization_level=0 gives PCC=0.906, optimization_level=1 gives PCC=0.937 (closest to threshold)."

# Benchmark added: gemma_2b_it_gguf

## Test
tests/benchmark/test_llms.py::test_gemma_2b_it_gguf

## Model
- HF name:    lmstudio-ai/gemma-2b-it-GGUF
- Loader:     third_party.tt_forge_models.gemma_2b_it_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.GEMMA_2B_IT_GGUF (value: "2B_IT_GGUF")

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A (PCC failure)
- TTFT (ms):          N/A (PCC failure)
- Prefill PCC:        N/A (PCC failure before this assertion)
- First decode PCC:   0.936944 (optimization_level=1); 0.881257 (optimization_level=2)
- Wall clock:         0:02:44 (optimization_level=1 run)
- Hardware:           p150

## PCC Failure Analysis
The Gemma 2B IT GGUF model uses Q4_K_M (4-bit) quantization. When loaded via
transformers, weights are dequantized to bfloat16. The bfp_bf8 compile-time
weight quantization (via experimental_weight_dtype) additionally re-quantizes
these already-lossy weights. The compound quantization error accumulates across
all 18 decoder layers of the full model:
- num_layers=1:  decode PCC=0.9992 (excellent — errors don't accumulate)
- Full model:    decode PCC=0.937 at optimization_level=1 (just below 0.94 threshold)

Tested combinations (all fail 0.94 threshold on full model):
- optimization_level=2, bfp_bf8, trace=True:   prefill=0.950, decode=0.881
- optimization_level=1, bfp_bf8, trace=True:   prefill=0.975, decode=0.937
- optimization_level=1, bfp_bf8, trace=False:  prefill=0.975, decode=0.937
- optimization_level=0, bfp_bf8, trace=True:   prefill=0.958, decode=0.906
- optimization_level=2, no bfp_bf8, trace=True: prefill=0.894 (failed prefill)
- optimization_level=2, fp32_dest_acc_en=True:  decode=0.881 (no improvement)

## Infrastructure Fix
Fixed `tests/benchmark/benchmarks/llm_benchmark.py` to guard the
`get_weight_dtype_config_path()` call with `hasattr()` (the same pattern used
in `tests/runner/testers/torch/dynamic_torch_model_tester.py`). Without this
fix, any loader that does not implement this optional method raises
`AttributeError` at benchmark runtime, blocking even basic bring-up.

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_gemma_2b_it_gguf_perf_metrics_2.json
Achieved vs top_perf_samples_per_sec: N/A (test fails PCC)

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
- total_flops:             160406962432
- breakdown.matmul:        160406962432
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        37748736
- memory_bytes: 75497472
- memory_gb:    0.0703125

### Params
- count:                  3030984833
- effective_count:        2506434689
- memory_bytes:           6061969666
- memory_gb:              5.645649196580052
- effective_memory_bytes: 5012869378
- effective_memory_gb:    4.668598415330052
- embedding_count:        524550144
- embedding_memory_bytes: 1049100288

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 130.0967
- top_perf_time_ms:         7.6866
- dram_time_ms:             5.1244
- compute_time_ms_lofi:     0.1823
- compute_time_ms_hifi2:    0.3646
- compute_time_ms_hifi3:    0.5468
- compute_time_ms_hifi4:    0.7291

## Files changed
- tests/benchmark/test_llms.py (added test_gemma_2b_it_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (infrastructure fix: guard get_weight_dtype_config_path with hasattr)

## tt-forge-models submodule
no change
