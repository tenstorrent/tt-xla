loader_path: third_party.tt_forge_models.mungert_glyph_gguf.causal_lm.pytorch.loader
variant_id: Glyph_Q4_K_M_GGUF
arch: p150
status: DONE_FAIL
test_function: test_mungert_glyph_gguf
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
failure_reason: "GGUF architecture glm4 not supported by transformers 5.2.0: ValueError: GGUF model with architecture glm4 is not supported yet."

# Benchmark added: mungert_glyph_gguf

## Test
tests/benchmark/test_llms.py::test_mungert_glyph_gguf

## Model
- HF name:    Mungert/Glyph-GGUF
- Loader:     third_party.tt_forge_models.mungert_glyph_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.GLYPH_Q4_K_M_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A (test failed before benchmark)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           p150 (Blackhole p300c)

## Failure detail
The test failed during model/tokenizer loading with:

    ValueError: GGUF model with architecture glm4 is not supported yet.

The loader calls `AutoTokenizer.from_pretrained("Mungert/Glyph-GGUF", gguf_file="Glyph-q4_k_m.gguf")`, but the installed transformers==5.2.0 does not support loading GGUF files whose architecture is `glm4`. This is a library compatibility issue — not fixable by changes to the test or benchmark harness. The fix requires either a newer version of transformers that supports the `glm4` GGUF architecture or an updated loader that loads the model differently.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (test did not reach compilation)

### System
- arch: N/A
- chip_count_in_system_desc: N/A
- single_chip_assumption: N/A
- worker_grid_cores: N/A
- dram_bandwidth_bytes_per_sec: N/A

### Roofline
- bound: N/A
- top_perf_samples_per_sec: N/A
- top_perf_time_ms: N/A

## Files changed
- tests/benchmark/test_llms.py (added test_mungert_glyph_gguf)
- .github/workflows/perf-bench-matrix.json (added mungert_glyph_q4_k_m_gguf entries)

## tt-forge-models submodule
no change
