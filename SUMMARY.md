loader_path: third_party.tt_forge_models.medgemma.causal_lm.pytorch.loader
variant_id: 4B_Instruct_Crimson
arch: p150
status: DONE_FAIL
test_function: test_medgemma_4b_instruct_crimson
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
failure_reason: "Gemma3ForCausalLM.__init__() does not accept use_cache kwarg: loader passes use_cache=False to from_pretrained(), incompatible with transformers==5.2.0 (TypeError: Gemma3ForCausalLM.__init__() got an unexpected keyword argument 'use_cache')"

# Benchmark added: test_medgemma_4b_instruct_crimson

## Test
tests/benchmark/test_llms.py::test_medgemma_4b_instruct_crimson

## Model
- HF name:    rajpurkarlab/medgemma-4b-it-crimson
- Loader:     third_party.tt_forge_models.medgemma.causal_lm.pytorch.loader
- Variant:    ModelVariant.MEDGEMMA_4B_IT_CRIMSON

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           p150

## Failure
The test failed at Step 3 (bring-up) due to a loader incompatibility with transformers==5.2.0.

The loader (`third_party/tt_forge_models/medgemma/causal_lm/pytorch/loader.py`) passes
`use_cache=False` as a model kwarg to `AutoModelForCausalLM.from_pretrained()`. In
transformers 5.2, `Gemma3ForCausalLM.__init__()` accepts only `config` and rejects
`use_cache` with:

    TypeError: Gemma3ForCausalLM.__init__() got an unexpected keyword argument 'use_cache'

This is a loader-level bug (analogous to the `cache_position` issue in test_phi3_mini).
Fixing it requires modifying the loader under `third_party/tt_forge_models/`, which is
out of scope for this skill. The fix belongs in the tt-forge-models repo.

Submodule HEAD at time of failure: 986e808f12

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (test did not reach compilation)
Achieved vs top_perf_samples_per_sec: N/A

### System
- arch:                        p150 (Blackhole)
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
- tests/benchmark/test_llms.py
- .github/workflows/perf-bench-matrix.json

## tt-forge-models submodule
no change — submodule HEAD: 986e808f12
