loader_path: third_party.tt_forge_models.galactica.causal_lm.pytorch.loader
variant_id: 1.3b
arch: p150
status: DONE_FAIL
test_function: test_galactica_1_3b
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
failure_reason: "loader bug: OPTForCausalLM.__init__() got an unexpected keyword argument 'use_cache' in third_party/tt_forge_models/galactica/causal_lm/pytorch/loader.py:load_model — transformers API no longer accepts use_cache as constructor kwarg"

# Benchmark added: test_galactica_1_3b

## Test
tests/benchmark/test_llms.py::test_galactica_1_3b

## Model
- HF name:    facebook/galactica-1.3b
- Loader:     third_party.tt_forge_models.galactica.causal_lm.pytorch.loader
- Variant:    ModelVariant.GALACTICA_1_3B ("1.3b")

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A (test failed)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           p150

## Failure Details
The loader (`third_party/tt_forge_models/galactica/causal_lm/pytorch/loader.py`) passes
`use_cache=False` as a keyword argument to `OPTForCausalLM.from_pretrained()`, which
in turn forwards it to `OPTForCausalLM.__init__()`. The current version of transformers
installed in the venv (transformers>=5.x) no longer accepts `use_cache` as a constructor
argument. This causes:

    TypeError: OPTForCausalLM.__init__() got an unexpected keyword argument 'use_cache'

This is a loader bug that must be fixed in the tt-forge-models repo.
Editing `third_party/tt_forge_models/` is out of scope for this skill.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (test did not reach compilation)
Achieved vs top_perf_samples_per_sec: N/A

### System
- arch:                        p150
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
- tests/benchmark/test_llms.py (test_galactica_1_3b added)
- SUMMARY.md

## tt-forge-models submodule
no change
