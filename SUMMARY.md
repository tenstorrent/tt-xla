loader_path: third_party.tt_forge_models.kg_ner_qwen3_14b_w4a16_v3.causal_lm.pytorch.loader
variant_id: KG_NER_Qwen3_14B_W4A16_V3
arch: p150
status: DONE_FAIL
test_function: test_kg_ner_qwen3_14b_w4a16_v3
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
failure_reason: "compressed_tensors W4A16 quantization incompatible with TT torch.compile backend: AssertionError in torch.fx fuser_utils.py:235 (last_output_node is None) triggered by compressed_tensors quantized_forward accessing weight.data via tt_torch __torch_function__ override"

# Benchmark added: test_kg_ner_qwen3_14b_w4a16_v3

## Test
tests/benchmark/test_llms.py::test_kg_ner_qwen3_14b_w4a16_v3

## Model
- HF name:    rhx1234/kg-ner-qwen3-14b-w4a16-v3
- Loader:     third_party.tt_forge_models.kg_ner_qwen3_14b_w4a16_v3.causal_lm.pytorch.loader
- Variant:    KG_NER_Qwen3_14B_W4A16_V3

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

## Failure details
The model uses W4A16 quantization via the `compressed_tensors` library (neuralmagic/nm format).
When torch.compile with the TT backend attempts to compile the model, the `compressed_tensors`
quantized forward pass in `forward.py:273` calls `weight.data`, which triggers `tt_torch`'s
`__torch_function__` override. This causes the torch.fx graph partitioner to fail:

    AssertionError: assert last_output_node is not None
    (torch/fx/passes/utils/fuser_utils.py:235, in insert_subgm)

This is a fundamental incompatibility between the compressed_tensors quantization scheme and the
TT XLA torch.compile backend. The fix requires either:
1. Support for compressed_tensors ops in the TT compiler pipeline, or
2. Dequantization of weights before compilation (pre-processing step in the benchmark harness).

Infrastructure fix applied: Added `hasattr` guard to `llm_benchmark.py:480` for
`get_weight_dtype_config_path()` method (models without this method previously raised
`AttributeError` — now gracefully skipped, matching existing pattern in
`tests/runner/testers/torch/dynamic_torch_model_tester.py:82`).

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (test did not reach compilation)
Achieved vs top_perf_samples_per_sec: N/A

### System
- arch: N/A
- chip_count_in_system_desc: N/A
- single_chip_assumption: N/A
- worker_grid_cores: N/A
- dram_bandwidth_bytes_per_sec: N/A

### Peak FLOPs
- lofi:  N/A
- hifi2: N/A
- hifi3: N/A
- hifi4: N/A

### Compute
- total_flops: N/A
- breakdown.matmul: N/A
- breakdown.linear: N/A
- breakdown.conv2d: N/A
- breakdown.sparse_matmul: N/A

### Inputs
- count: N/A
- memory_bytes: N/A

### KV cache
- count: N/A
- memory_bytes: N/A
- memory_gb: N/A

### Params
- count: N/A
- effective_count: N/A
- memory_bytes: N/A
- memory_gb: N/A
- effective_memory_bytes: N/A
- effective_memory_gb: N/A
- embedding_count: N/A
- embedding_memory_bytes: N/A

### Roofline
- bound: N/A
- top_perf_samples_per_sec: N/A
- top_perf_time_ms: N/A
- dram_time_ms: N/A
- compute_time_ms_lofi: N/A
- compute_time_ms_hifi2: N/A
- compute_time_ms_hifi3: N/A
- compute_time_ms_hifi4: N/A

## Files changed
- tests/benchmark/test_llms.py (added test_kg_ner_qwen3_14b_w4a16_v3)
- tests/benchmark/benchmarks/llm_benchmark.py (added hasattr guard for get_weight_dtype_config_path)
- .github/workflows/perf-bench-matrix.json (added kg_ner_qwen3_14b_w4a16_v3 entry with compressed-tensors pyreq)
- SUMMARY.md

## tt-forge-models submodule
no change (ebcfe743a1)
