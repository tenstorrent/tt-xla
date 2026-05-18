loader_path: third_party.tt_forge_models.iconn_1_mini_beta.causal_lm.pytorch.loader
variant_id: iconn_1_mini_beta
arch: p150
status: DONE_FAIL
test_function: test_iconn_1_mini_beta
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
failure_reason: "TypeError: IconnForCausalLM.__init__() got an unexpected keyword argument 'use_cache' — the loader unconditionally sets model_kwargs[\"use_cache\"] = False (loader.py:88) and passes it to AutoModelForCausalLM.from_pretrained(), but the custom ICONN architecture's __init__ does not accept this argument; this is a loader-level bug in third_party/tt_forge_models that cannot be fixed in the benchmark test"

# Benchmark added: test_iconn_1_mini_beta

## Test
tests/benchmark/test_llms.py::test_iconn_1_mini_beta

## Model
- HF name:    ICONNAI/ICONN-1-Mini-Beta
- Loader:     third_party.tt_forge_models.iconn_1_mini_beta.causal_lm.pytorch.loader
- Variant:    ModelVariant.ICONN_1_MINI_BETA

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A (test failed before reaching benchmark)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           p150

## Failure Details
The test fails at model load time with:

    TypeError: IconnForCausalLM.__init__() got an unexpected keyword argument 'use_cache'

The loader at `third_party/tt_forge_models/iconn_1_mini_beta/causal_lm/pytorch/loader.py:88`
unconditionally sets `model_kwargs["use_cache"] = False` and forwards it to
`AutoModelForCausalLM.from_pretrained()`. The custom ICONN architecture's `__init__`
does not accept this kwarg. This is a loader-level bug in `tt-forge-models` that
must be fixed there (e.g., by removing or guarding the `use_cache=False` assignment,
or by adding `use_cache` support to `IconnForCausalLM`).

No compiler or hardware was reached; the crash occurs entirely in Python/transformers.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (test failed before compilation)
Achieved vs top_perf_samples_per_sec: N/A

### System
- arch:                        p150 / blackhole
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

## tt-forge-models submodule
no change
