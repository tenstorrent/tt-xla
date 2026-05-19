loader_path: third_party.tt_forge_models.fla_hub_transformer.causal_lm.pytorch.loader
variant_id: 1.3B_100B
arch: p150
status: DONE_FAIL
test_function: test_fla_hub_transformer_1_3b_100b
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
failure_reason: "loader incompatible with transformers 5.2.0: AutoModelForCausalLM.from_pretrained raises ValueError: model type 'transformer' not in CONFIG_MAPPING; trust_remote_code=True does not resolve this with transformers==5.2.0"

# Benchmark added: test_fla_hub_transformer_1_3b_100b

## Test
tests/benchmark/test_llms.py::test_fla_hub_transformer_1_3b_100b

## Model
- HF name:    fla-hub/transformer-1.3B-100B
- Loader:     third_party.tt_forge_models.fla_hub_transformer.causal_lm.pytorch.loader
- Variant:    ModelVariant.TRANSFORMER_1_3B_100B

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
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

The loader uses AutoModelForCausalLM.from_pretrained("fla-hub/transformer-1.3B-100B", trust_remote_code=True).
The model config.json specifies "model_type": "transformer", which is not in transformers built-in
CONFIG_MAPPING. Even with trust_remote_code=True, transformers 5.2.0 fails to register the custom
config class and raises:

  ValueError: The checkpoint you are trying to load has model type 'transformer' but Transformers does
  not recognize this architecture.

This is a loader incompatibility with transformers==5.2.0. The fix belongs in tt-forge-models:
either update the loader to load weights via LlamaForCausalLM with key remapping (as was done
in the hf-bringup branch worktree-aus-wh-01-tt-xla-dev+nsmith+hf-bringup-start65-1), or
register the custom config before calling from_pretrained.

Also fixed a general benchmarking infrastructure issue: llm_benchmark.py called
model_loader.get_weight_dtype_config_path() unconditionally, but many loaders (including this one)
don't implement this method. Fixed with a hasattr guard (general fix, applies to all loaders).

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
- tests/benchmark/test_llms.py (added test_fla_hub_transformer_1_3b_100b)
- tests/benchmark/benchmarks/llm_benchmark.py (hasattr guard for get_weight_dtype_config_path, included in prior dolphin commit)
- SUMMARY.md

## tt-forge-models submodule
no change (submodule HEAD: 6cb56d720b)
