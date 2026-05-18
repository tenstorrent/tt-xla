loader_path: third_party.tt_forge_models.falcon_h1r_7b_gguf.causal_lm.pytorch.loader
variant_id: Q4_K_M
arch: p150
status: DONE_FAIL
test_function: test_falcon_h1r_7b_gguf
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
failure_reason: "NotImplementedError: Unknown gguf model_type: falcon_h1 in gguf-py — the loader's monkey-patch patches transformers but not gguf-py's internal registry; fix belongs in the loader or gguf-py dependency"

# Benchmark added: falcon_h1r_7b_gguf

## Test
tests/benchmark/test_llms.py::test_falcon_h1r_7b_gguf

## Model
- HF name:    mradermacher/Falcon-H1R-7B-i1-GGUF
- Loader:     third_party.tt_forge_models.falcon_h1r_7b_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.FALCON_H1R_7B_Q4_K_M (value: "Q4_K_M")

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A (DONE_FAIL)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           p150

## Failure Details
```
NotImplementedError: Unknown gguf model_type: falcon_h1 in gguf-py.
This might because you're using an outdated version of gguf-py package,
you can install `gguf` package from source refer to
https://github.com/ggerganov/llama.cpp/tree/master/gguf-py#development
```

The loader in `third_party/tt_forge_models/falcon_h1r_7b_gguf/causal_lm/pytorch/loader.py`
contains a `_patch_transformers_falcon_h1_gguf()` function that monkey-patches
`transformers` to add `falcon-h1` GGUF architecture support. However, the error
occurs inside `gguf-py`'s `get_gguf_hf_weights_map()` function which has its own
internal registry that the monkey-patch does not update. Fixing this requires
either updating `gguf-py` to a version that includes `falcon_h1` support, or
extending the monkey-patch in the loader to also register the architecture
in `gguf-py`'s registry. Both fixes belong in the `tt-forge-models` repo.

Traceback (abbreviated):
    tests/benchmark/benchmarks/llm_benchmark.py:72: in setup_model_and_tokenizer
        model = model_loader.load_model(dtype_override=torch.bfloat16)
    third_party/tt_forge_models/falcon_h1r_7b_gguf/causal_lm/pytorch/loader.py:178: in load_model
        model = AutoModelForCausalLM.from_pretrained(...)
    transformers/modeling_gguf_pytorch_utils.py:369: in get_gguf_hf_weights_map
        raise NotImplementedError(...)
    NotImplementedError: Unknown gguf model_type: falcon_h1 in gguf-py.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (model failed to load)

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
- tests/benchmark/test_llms.py (added test_falcon_h1r_7b_gguf)
- SUMMARY.md

## tt-forge-models submodule
no change (submodule at 7cf0e9b8df)
