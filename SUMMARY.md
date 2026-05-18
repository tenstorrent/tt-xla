loader_path: third_party.tt_forge_models.bielik.causal_lm.pytorch.loader
variant_id: 7B_Instruct_v0.1
arch: p150
status: DONE_FAIL
test_function: test_bielik_7b_instruct_v0_1
samples_per_second: null
ttft_ms: null
prefill_pcc: null
first_decode_pcc: null
top_perf_samples_per_sec: null
pct_of_target: null
roofline_bound: null
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: "compiler error: failed to legalize operation 'ttir.paged_update_cache' in TTIR->TTNN lowering for full 32-layer model at all optimization levels (0,1,2); num_layers=1 passes"

# Benchmark added: test_bielik_7b_instruct_v0_1

## Test
tests/benchmark/test_llms.py::test_bielik_7b_instruct_v0_1

## Model
- HF name:    speakleash/Bielik-7B-Instruct-v0.1
- Loader:     third_party.tt_forge_models.bielik.causal_lm.pytorch.loader
- Variant:    ModelVariant.BIELIK_7B_INSTRUCT_V0_1

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A (test failed)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         ~11 min (failed)
- Hardware:           p150 (Blackhole p300c)

## Failure details
The full 32-layer model fails to compile at all optimization levels (0, 1, 2) with:
  loc("scatter.11541"): error: failed to legalize operation 'ttir.paged_update_cache'
  RuntimeError: Error code: 13

The num_layers=1 test PASSES at optimization_level=2 (PCC=0.999) and optimization_level=1,
indicating the paged_update_cache legalization failure is specific to larger KV cache
sizes produced by the full 32-layer model. This is a compiler bug in the TTIR→TTNN
lowering pass.

Also applied a general infrastructure fix to tests/benchmark/benchmarks/llm_benchmark.py:
guarded `model_loader.get_weight_dtype_config_path()` with `hasattr` check (same pattern
already used in tests/runner/testers/torch/dynamic_torch_model_tester.py). This fixes
ModelLoaders that don't implement get_weight_dtype_config_path.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (test failed before producing perf metrics)
Achieved vs top_perf_samples_per_sec: N/A

### System
- arch:                        N/A
- chip_count_in_system_desc:   N/A
- single_chip_assumption:      N/A
- worker_grid_cores:           N/A
- dram_bandwidth_bytes_per_sec: N/A

### Peak FLOPs
- lofi:  N/A
- hifi2: N/A
- hifi3: N/A
- hifi4: N/A

### Compute
- total_flops:             N/A
- breakdown.matmul:        N/A
- breakdown.linear:        N/A
- breakdown.conv2d:        N/A
- breakdown.sparse_matmul: N/A

### Inputs
- count:        N/A
- memory_bytes: N/A

### KV cache
- count:        N/A
- memory_bytes: N/A
- memory_gb:    N/A

### Params
- count:                  N/A
- effective_count:        N/A
- memory_bytes:           N/A
- memory_gb:              N/A
- effective_memory_bytes: N/A
- effective_memory_gb:    N/A
- embedding_count:        N/A
- embedding_memory_bytes: N/A

### Roofline
- bound:                    N/A
- top_perf_samples_per_sec: N/A
- top_perf_time_ms:         N/A
- dram_time_ms:             N/A
- compute_time_ms_lofi:     N/A
- compute_time_ms_hifi2:    N/A
- compute_time_ms_hifi3:    N/A
- compute_time_ms_hifi4:    N/A

## Files changed
- tests/benchmark/test_llms.py (added test_bielik_7b_instruct_v0_1)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: guard get_weight_dtype_config_path with hasattr)

## tt-forge-models submodule
no change — submodule stays at 1e47cb75d6
