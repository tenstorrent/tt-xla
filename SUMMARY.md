loader_path: third_party.tt_forge_models.academic_ds.causal_lm.pytorch.loader
variant_id: 9B
arch: p150
status: DONE_FAIL
test_function: test_academic_ds_9b
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
failure_reason: "DeepSeek V3 MoE layers (layers 1-15) fail on p150: all three expert dispatch modes are broken in the current stack — 'dense' is not registered in transformers 5.2 ExpertsInterface, 'batched_mm' segfaults during torch.compile graph extraction (exit 139), 'grouped_mm' fails with ttnn.sort op type mismatch; layer 0 (dense MLP) compiles and passes PCC with --num-layers 1 (prefill PCC 0.999, decode PCC 0.999)"

# Benchmark added: test_academic_ds_9b

## Test
tests/benchmark/test_llms.py::test_academic_ds_9b

## Model
- HF name:    ByteDance-Seed/academic-ds-9B
- Loader:     third_party.tt_forge_models.academic_ds.causal_lm.pytorch.loader
- Variant:    ModelVariant.ACADEMIC_DS_9B

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  null (full model failed to compile)
- TTFT (ms):          null
- Prefill PCC:        null
- First decode PCC:   null
- Wall clock:         N/A
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_academic_ds_9b_perf_metrics_0.json (num_layers=1 only — full model did not produce metrics)
Achieved vs top_perf_samples_per_sec: N/A (full model failed)

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

### Compute (layer 0 only — not representative)
- total_flops:             353663714304
- breakdown.matmul:        353663714304
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        528
- memory_bytes: 2112

### KV cache
- count:        20971520
- memory_bytes: 41943040
- memory_gb:    0.039

### Params (layer 0 + embeddings only — not representative)
- count:                  609492646
- effective_count:        344727206
- memory_bytes:           895811220
- memory_gb:              0.834
- effective_memory_bytes: 366280340
- effective_memory_gb:    0.341
- embedding_count:        264765440
- embedding_memory_bytes: 529530880

### Roofline (layer 0 only)
- bound:                    compute
- top_perf_samples_per_sec: 829.4131
- top_perf_time_ms:         1.2057
- dram_time_ms:             0.7366
- compute_time_ms_lofi:     0.4019
- compute_time_ms_hifi2:    0.8038
- compute_time_ms_hifi3:    1.2057
- compute_time_ms_hifi4:    1.6076

## Files changed
- tests/benchmark/test_llms.py (added test_academic_ds_9b; fixed use_mla_cache incompatibility)
- tests/benchmark/llm_utils/decode_utils.py (fixed init_static_cache for DeepSeek V3 K/V head dims)
- tests/benchmark/benchmarks/llm_benchmark.py (fixed missing hasattr for get_weight_dtype_config_path; fixed _experts_implementation dense fallback to batched_mm)

## Infrastructure fixes (general, not model-specific)
Three general infrastructure fixes were required to get as far as the compilation step:

1. **decode_utils.py / init_static_cache**: For DeepSeek V3 models, `config.head_dim=64` is the
   RoPE head dim, not the actual key head dim (`qk_nope_head_dim + qk_rope_head_dim = 192`).
   The `early_initialization` call was pre-allocating cache tensors with wrong shape, causing
   `index_copy_` to fail on the first prefill. Fixed by detecting DeepSeek V3 configs via
   `hasattr(config, 'qk_nope_head_dim')` and initialising each cache layer directly with the
   correct K (192) and V (128) head dims.

2. **llm_benchmark.py / get_weight_dtype_config_path**: The `setup_model_and_tokenizer` function
   called `model_loader.get_weight_dtype_config_path()` without checking `hasattr` first. The
   academic-ds-9B ModelLoader doesn't implement this method. Fixed by guarding with `hasattr`.

3. **llm_benchmark.py / _experts_implementation**: Setting `_experts_implementation = "dense"`
   blindly for all models with the attribute caused a `KeyError` for DeepSeek V3 models under
   transformers 5.2, because `"dense"` is not registered in `ExpertsInterface._global_mapping`.
   Fixed by probing `ALL_EXPERTS_FUNCTIONS.get_interface("dense", None)` first and falling back
   to `"batched_mm"` on `KeyError`. (Note: `"batched_mm"` still fails for this specific model
   due to a segfault during graph extraction — see failure_reason above.)

## tt-forge-models submodule
no change
