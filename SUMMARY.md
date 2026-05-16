loader_path: third_party.tt_forge_models.academic_ds.causal_lm.pytorch.loader
variant_id: 9B
arch: n150
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
trace_enabled: null
experimental_weight_dtype: null
failure_reason: "DeepSeek V3 MoE layers cannot compile on TT backend: grouped_mm_experts_forward (transformers 5.2 default) fails with ttnn.sort op type mismatch; batched_mm fallback segfaults in extract_compiled_graph_helper; dense mode not registered in transformers 5.2 ExpertsInterface"

# Benchmark added: test_academic_ds_9b

## Test
tests/benchmark/test_llms.py::test_academic_ds_9b

## Model
- HF name:    ByteDance-Seed/academic-ds-9B
- Loader:     third_party.tt_forge_models.academic_ds.causal_lm.pytorch.loader
- Variant:    ModelVariant.ACADEMIC_DS_9B

## Test config landed
- optimization_level:        2
- trace_enabled:             (not overridden — default True)
- experimental_weight_dtype: none
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Architecture notes
The `academic-ds-9B` model uses the DeepSeek V3 architecture (`model_type = deepseek_v3`)
with Multi-head Latent Attention (MLA) and Mixture of Experts (MoE):
- 16 layers total: layer 0 is dense MLP, layers 1–15 are MoE with 64 routed experts, 8 active
- Keys have head_dim = qk_nope_head_dim + qk_rope_head_dim = 128 + 64 = 192
- Values have head_dim = v_head_dim = 128
- Standard StaticCache cannot be initialized with a single head_dim for both K and V

## Infrastructure fixes (general, not model-specific)

### 1. `tests/benchmark/llm_utils/decode_utils.py` — `init_static_cache`
Added detection for DeepSeek V3-style models (via `qk_nope_head_dim` / `v_head_dim`
config attributes). When present, calls `layer.lazy_initialization(fake_k, fake_v)` per
cache layer with correct per-axis shapes instead of `early_initialization` (which takes
a single `head_dim` for both K and V). Fixes KV cache shape mismatch for all models
with MLA-style expanded K/V.

### 2. `tests/benchmark/benchmarks/llm_benchmark.py` — `hasattr` guard
Added `elif hasattr(model_loader, "get_weight_dtype_config_path"):` guard before calling
the method. Prevents `AttributeError` for loaders that don't implement the method
(matches the pattern already used in `dynamic_torch_model_tester.py`).

### 3. `tests/benchmark/benchmarks/llm_benchmark.py` — experts_implementation guard
Wrapped the `model.config._experts_implementation = "dense"` assignment in a try/except
`KeyError`. In transformers ≥ 5.2 the `ExpertsInterface._global_mapping` only registers
`"batched_mm"` and `"grouped_mm"`; `"dense"` is gone. On `KeyError` the fallback now
sets `"batched_mm"` (which avoids the sort/argsort lowering to `ttnn.sort`).

## Failure analysis

### 1-layer run (layer 0 = dense MLP, no MoE) — PASSES
- Prefill PCC:      0.999663
- First decode PCC: 0.999595
- Roofline (1-layer decode graph):
  - arch: wormhole_b0 → n150
  - top_perf_samples_per_sec: 241.28 (compute-bound, hifi3)
  - Source: tt_xla_academic_ds_9b_perf_metrics_0.json

### Full model (layers 1–15 include MoE) — FAILS

Three expert dispatch modes were attempted; all fail on the TT backend:

| Mode | Symptom |
|---|---|
| `grouped_mm` (transformers 5.2 default) | `'ttnn.sort' op Sorted tensor type does not match with input tensor` — `grouped_mm_experts_forward` calls `torch.argsort`, which lowers to `ttnn.sort` with a type incompatibility |
| `dense` (original llm_benchmark.py workaround) | `KeyError: 'dense' is not a valid experts implementation registered in the ExpertsInterface` — `"dense"` was removed in transformers 5.2 |
| `batched_mm` (fallback after above fixes) | `Fatal Python error: Segmentation fault` in `tt_torch.backend.backend._call_experimental_compile` → `dynamo_bridge.extract_compiled_graph_helper` after ~45 min of CPU compilation |

Recommended next steps:
1. Fix `ttnn.sort` type mismatch so `grouped_mm` compiles (preferred — it is the
   upstream-recommended MoE dispatch path for static-shape backends).
2. Alternatively fix the segfault in `extract_compiled_graph_helper` for `batched_mm`.
3. Either fix unblocks this model immediately; the test function and infrastructure
   patches are already in place.

## Measured (full model, defaults)
- Sample per second:  N/A (full model could not complete)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           n300 (Wormhole, dual-chip card; single-chip n150 benchmark)

## Decode roofline (1-layer proxy only — NOT full model)
Source JSON: tt_xla_academic_ds_9b_perf_metrics_0.json
Note: these numbers reflect layer 0 (dense MLP, no MoE) and are NOT
representative of the full model's decode graph.

### System
- arch:                        wormhole_b0
- chip_count_in_system_desc:   2
- single_chip_assumption:      True
- worker_grid_cores:           64
- dram_bandwidth_bytes_per_sec: 288000000000
