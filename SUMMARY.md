loader_path: third_party.tt_forge_models.document_validation_qwen2_5_vl_simple_v2_i1_gguf.causal_lm.pytorch.loader
variant_id: Simple_V2_i1_GGUF
arch: p150
status: DONE_PASS
test_function: test_document_validation_qwen2_5_vl_simple_v2_i1_gguf
samples_per_second: 4.19515704026459
ttft_ms: 1241.081805
prefill_pcc: 0.992996
first_decode_pcc: 0.997859
top_perf_samples_per_sec: 46.0471
pct_of_target: 9.1
roofline_bound: dram
optimization_level: 2
trace_enabled: false
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark Summary: test_document_validation_qwen2_5_vl_simple_v2_i1_gguf (p150)

## Model
- **Loader**: `third_party.tt_forge_models.document_validation_qwen2_5_vl_simple_v2_i1_gguf.causal_lm.pytorch.loader`
- **Variant**: `Simple_V2_i1_GGUF`
- **Architecture**: p150 (blackhole)

## Performance Results (trace_enabled=False)
- **Samples/sec**: 4.195 (9.1% of roofline)
- **TTFT**: 1241 ms
- **Roofline**: 46.0471 samples/sec (DRAM-bound)

## Accuracy (PCC)
- **Prefill PCC**: 0.992996 ✓ (required: 0.94)
- **First Decode PCC**: 0.997859 ✓ (required: 0.94)

## Configuration
- **optimization_level**: 2
- **trace_enabled**: false (trace=True causes prefill PCC regression: 0.799 < required 0.94)
- **experimental_weight_dtype**: bfp_bf8
- **batch_size**: 32

## Notes
- `trace_enabled=False` required: trace=True causes prefill PCC to drop from ~0.993 to 0.799 (below required 0.94), confirmed on full-model run (May 19 2026, 6h39m run).
- Performance (4.195 s/s, 9.1% of roofline) is significantly below target due to trace being disabled.
