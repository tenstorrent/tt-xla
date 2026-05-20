loader_path: third_party.tt_forge_models.llamantino_2_chat_13b_hf_ultrachat_ita.causal_lm.pytorch.loader
variant_id: llamantino_2_chat_13b_hf_ultrachat_ita
arch: p150
status: DONE_FAIL
test_function: test_llamantino_2_chat_13b
samples_per_second: null
ttft_ms: null
prefill_pcc: 0.983690
first_decode_pcc: 0.493522
top_perf_samples_per_sec: 500.1099
pct_of_target: null
roofline_bound: compute
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: "First decode PCC consistently fails (PCC=0.493522 at optimization_level=2 full model, degenerates to ValueError at optimization_level=0); prefill PCC=0.983690 passes. Decoder step produces incorrect outputs across all settings tried (bfp_bf8, no weight dtype, optimization_level=0,1,2, num_layers=1 and full model)."

# Benchmark added: test_llamantino_2_chat_13b

## Test
tests/benchmark/test_llms.py::test_llamantino_2_chat_13b

## Model
- HF name:    swap-uniba/LLaMAntino-2-chat-13b-hf-UltraChat-ITA
- Loader:     third_party.tt_forge_models.llamantino_2_chat_13b_hf_ultrachat_ita.causal_lm.pytorch.loader
- Variant:    LLAMANTINO_2_CHAT_13B_HF_ULTRACHAT_ITA

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  14.273 (decode throughput measured but test FAILED on PCC)
- TTFT (ms):          567.451
- Prefill PCC:        0.983690 (PASSED)
- First decode PCC:   0.493522 (FAILED, required 0.94)
- Wall clock:         ~11 min (model load + compile + run)
- Hardware:           p150

## Decode investigation
All configurations tried failed first decode PCC:
- optimization_level=2, bfp_bf8, full model (40L): PCC=0.493522 FAIL
- optimization_level=2, no weight dtype, full model: PCC=0.493522 FAIL
- optimization_level=2, bfp_bf8, num_layers=1: PCC=0.783482 FAIL
- optimization_level=2, no weight dtype, num_layers=1: PCC=0.789670 FAIL
- optimization_level=0, bfp_bf8, num_layers=1: ValueError (denominator=0, degenerate output)

The decode step consistently produces incorrect outputs. Prefill passes in all cases.
Infrastructure fix also applied: `llm_benchmark.py` now uses `hasattr` check before
calling `get_weight_dtype_config_path()` (general fix, not model-specific).

## Decode roofline (first decode graph, single-chip — from num_layers=1 exploratory run)
Source JSON: tt_xla_llamantino_2_chat_13b_perf_metrics_0.json
Achieved vs top_perf_samples_per_sec: N/A (test failed)

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
- total_flops:             586537699712
- breakdown.matmul:        586537699712
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        627
- memory_bytes: 2508

### KV cache
- count:        41943040
- memory_bytes: 83886080
- memory_gb:    0.078125

### Params
- count:                  644900038
- effective_count:        481054918
- memory_bytes:           838826068
- memory_gb:              0.7812176533043385
- effective_memory_bytes: 511135828
- effective_memory_gb:    0.4760323353111744
- embedding_count:        163845120
- embedding_memory_bytes: 327690240

### Roofline
- bound:                    compute
- top_perf_samples_per_sec: 500.1099
- top_perf_time_ms:         1.9996
- dram_time_ms:             1.0534
- compute_time_ms_lofi:     0.6665
- compute_time_ms_hifi2:    1.3330
- compute_time_ms_hifi3:    1.9996
- compute_time_ms_hifi4:    2.6661

## Files changed
- tests/benchmark/test_llms.py (added test_llamantino_2_chat_13b)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: hasattr check for get_weight_dtype_config_path)

## tt-forge-models submodule
no change
