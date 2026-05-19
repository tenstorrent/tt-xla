loader_path: third_party.tt_forge_models.ministral_3_3b_instruct_2512.causal_lm.pytorch.loader
variant_id: Ministral_3_3B_Instruct_2512
arch: p150
status: DONE_FAIL
test_function: test_ministral_3_3b_instruct_2512
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
failure_reason: "loader bug at submodule HEAD 706546ab8e: AutoModelForCausalLM.from_pretrained fails for unsloth/Ministral-3-3B-Instruct-2512 which uses Mistral3Config (multimodal); loader must use AutoModelForImageTextToText instead"

# Benchmark added: test_ministral_3_3b_instruct_2512

## Test
tests/benchmark/test_llms.py::test_ministral_3_3b_instruct_2512

## Model
- HF name:    unsloth/Ministral-3-3B-Instruct-2512
- Loader:     third_party.tt_forge_models.ministral_3_3b_instruct_2512.causal_lm.pytorch.loader
- Variant:    ModelVariant.MINISTRAL_3_3B_INSTRUCT_2512

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

## Failure Details

The loader at submodule HEAD (706546ab8e) calls `AutoModelForCausalLM.from_pretrained(pretrained_model_name)` but the HuggingFace checkpoint `unsloth/Ministral-3-3B-Instruct-2512` uses `Mistral3Config` (from `transformers.models.mistral3`, a multimodal architecture). `AutoModelForCausalLM` does not support `Mistral3Config`; it only supports `Ministral3Config` (from `transformers.models.ministral3`, the text-only architecture).

An earlier version of this loader (visible in git history) used `AutoModelForImageTextToText.from_pretrained` and extracted the language model component via `full_model.model.language_model`, which correctly handled the multimodal checkpoint. The current loader was changed to use `AutoModelForCausalLM.from_pretrained` directly, which is incompatible with this checkpoint. The fix belongs in the tt-forge-models repository.

Error:
```
ValueError: Unrecognized configuration class <class 'transformers.models.mistral3.configuration_mistral3.Mistral3Config'> for this kind of AutoModel: AutoModelForCausalLM.
```

## Infrastructure fix also included
`tests/benchmark/benchmarks/llm_benchmark.py`: Added `hasattr` guard for `get_weight_dtype_config_path` to prevent `AttributeError` for loaders that do not implement this method. (Back-port of the same fix from `odjuricic/ai-benchmark-pipeline-test_mergekit_ties_xvxbphx_gguf_8b-p150`.)

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
- tests/benchmark/test_llms.py (added test_ministral_3_3b_instruct_2512)
- tests/benchmark/benchmarks/llm_benchmark.py (fixed missing hasattr guard for get_weight_dtype_config_path)

## tt-forge-models submodule
no change
