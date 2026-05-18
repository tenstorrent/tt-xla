loader_path: third_party.tt_forge_models.huginn_0125.causal_lm.pytorch.loader
variant_id: 0125
arch: p150
status: DONE_FAIL
test_function: test_huginn_0125
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
failure_reason: "compiler error: AssertionError in torch.fx partition_fx_graph_for_cpu_fallback -> fuse_by_partitions -> insert_subgm (last_output_node is None); Huginn's depth-recurrent architecture uses block_idx (torch.Tensor scalar) to index into StaticCache.layers, causing a graph break that the TT XLA dynamo backend's CPU fallback partitioner cannot handle at all optimization levels (0, 1, 2)"

# Benchmark added: huginn_0125

## Test
tests/benchmark/test_llms.py::test_huginn_0125

## Model
- HF name:    tomg-group-umd/huginn-0125
- Loader:     third_party.tt_forge_models.huginn_0125.causal_lm.pytorch.loader
- Variant:    ModelVariant.HUGINN_0125 ("0125")

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

## Failure Analysis

### Bring-up fixes applied
Two general infrastructure fixes were required and applied:

1. **`tests/benchmark/llm_utils/decode_utils.py` — `init_static_cache`**:
   Huginn (depth-recurrent) has `n_layers=8` physical layers but needs 132 KV cache
   slots (`effective_expected_depth=132`). The `StaticCache` was being initialized with
   8 slots → `IndexError: list index out of range` in the KV cache update. Fix: check
   for `config.effective_expected_depth > config.num_hidden_layers` and temporarily
   override `num_hidden_layers` before constructing `StaticCache`. General fix — benefits
   any depth-recurrent model exposing this attribute.

2. **`tests/benchmark/benchmarks/llm_benchmark.py` — `benchmark_llm_torch_xla`**:
   Called `model_loader.get_weight_dtype_config_path()` unconditionally. Huginn's
   `ModelLoader` doesn't implement this optional method. Fix: guard with `hasattr`,
   matching the existing pattern in `tests/runner/testers/torch/dynamic_torch_model_tester.py`.

### Terminal failure (compiler bug — not fixable here)
After the two infrastructure fixes, compilation fails at all optimization levels (0, 1, 2):

```
torch_xla._dynamo.dynamo_bridge.partition_fx_graph_for_cpu_fallback
  → partitioner.fuse_partitions
    → fuse_by_partitions
      → insert_subgm
        AssertionError: assert last_output_node is not None
```

Root cause: Huginn's `raven_modeling_minimal.py` passes `block_idx` (a `torch.Tensor`
scalar) to `StaticCache.update(k, v, layer_idx)`. `StaticCache.update` does
`self.layers[layer_idx]` — indexing an `nn.ModuleList` with a tensor. torch.dynamo
creates a graph break at this dynamic indexing point. The TT XLA dynamo bridge's
`partition_fx_graph_for_cpu_fallback` then fails to produce a valid graph partition
(`last_output_node is None`).

This is a compiler/model-architecture incompatibility. Fixing it would require either:
- Making `block_idx` a static Python int during traced execution (model change in tt-forge-models), OR
- Fixing the TT XLA dynamo bridge's CPU fallback partitioner to handle this graph topology.

Both are out of scope for this skill.

## Decode roofline (first decode graph, single-chip)
N/A — test did not complete compilation.

## Files changed
- tests/benchmark/test_llms.py (added test_huginn_0125)
- tests/benchmark/llm_utils/decode_utils.py (general fix: init_static_cache with effective_expected_depth)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: guard get_weight_dtype_config_path with hasattr)
- .github/workflows/perf-bench-matrix.json (added huginn_0125 entry)
- SUMMARY.md

## tt-forge-models submodule
no change
