loader_path: third_party.tt_forge_models.ministral_8b_gguf.causal_lm.pytorch.loader
variant_id: 8B_Instruct_2512_GGUF
arch: p150
status: DONE_FAIL
test_function: test_ministral_8b_gguf
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
failure_reason: "NotImplementedError: Unknown gguf model_type: ministral3 in gguf-py - loader monkey-patching in third_party/tt_forge_models/ministral_8b_gguf/causal_lm/pytorch/loader.py patches transformers but not the gguf-py package itself; needs gguf-py package update or loader fix"

# Benchmark added: test_ministral_8b_gguf

## Test
tests/benchmark/test_llms.py::test_ministral_8b_gguf

## Model
- HF name:    unsloth/Ministral-3-8B-Instruct-2512-GGUF
- Loader:     third_party.tt_forge_models.ministral_8b_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.MINISTRAL_8B_INSTRUCT_2512_GGUF

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

## Failure details
The test fails at model loading with:
```
NotImplementedError: Unknown gguf model_type: ministral3 in gguf-py.
This might because you're using an outdated version of gguf-py package,
you can install `gguf` package from source refer to
https://github.com/ggerganov/llama.cpp/tree/master/gguf-py#development
```

The loader in `third_party/tt_forge_models/ministral_8b_gguf/causal_lm/pytorch/loader.py`
monkey-patches `transformers.modeling_gguf_pytorch_utils` to add `mistral3` architecture
support, but the underlying `gguf-py` package's `get_gguf_hf_weights_map` function in
`transformers/modeling_gguf_pytorch_utils.py:369` still raises NotImplementedError because
`ministral3` is not registered in the `gguf-py` package's architecture mappings.

Fix required: either upgrade `gguf-py` to a version that supports `ministral3`, or extend
the loader's monkey-patching to also patch the `gguf-py` package's architecture registry.
This fix belongs in `tt-forge-models`, not in `tt-xla`.

## Files changed
- tests/benchmark/test_llms.py (test function test_ministral_8b_gguf added at line ~1058)
- SUMMARY.md

## tt-forge-models submodule
no change
