loader_path: third_party.tt_forge_models.granite_code.causal_lm.pytorch.loader
variant_id: Granite_8B_Code_Base_4K
arch: p150
status: DONE_FAIL
test_function: test_granite_code_8b_base_4k
samples_per_second: null
ttft_ms: null
prefill_pcc: null
first_decode_pcc: null
top_perf_samples_per_sec: null
pct_of_target: null
roofline_bound: null
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: "variant Granite_8B_Code_Base_4K not in current ModelVariant enum of granite_code/causal_lm/pytorch/loader.py at submodule HEAD 93218a34fc; variant exists on remote branch origin/ip-172-31-23-5-tt-xla-dev/ubuntu/hf-bringup-18 (commit ebecd9f38a) but has not been merged into main submodule HEAD"

# Benchmark added: test_granite_code_8b_base_4k

## Test
tests/benchmark/test_llms.py::test_granite_code_8b_base_4k

## Model
- HF name:    ibm-granite/granite-8b-code-base-4k
- Loader:     third_party.tt_forge_models.granite_code.causal_lm.pytorch.loader
- Variant:    ModelVariant.GRANITE_8B_CODE_BASE_4K

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A (early exit — variant missing at submodule HEAD)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A
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
- tests/benchmark/test_llms.py (test function added — will fail until submodule is updated to include the 8B variant)
- SUMMARY.md

## tt-forge-models submodule
no change — variant Granite_8B_Code_Base_4K exists on remote branch origin/ip-172-31-23-5-tt-xla-dev/ubuntu/hf-bringup-18 (commit ebecd9f38a) but is not merged into main HEAD (93218a34fc). Pipeline operator should update the submodule pointer to include this variant before re-running.
