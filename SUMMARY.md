loader_path: third_party.tt_forge_models.bat_venom.causal_lm.pytorch.loader
variant_id: BatVenom
arch: n150
status: DONE_FAIL
test_function: test_bat_venom
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
failure_reason: "OOM: model is ~24B params (Mistral-Small-3.1-24B architecture: 40 hidden layers, hidden_size=5120), exceeds n150 single-chip DRAM capacity"

# Benchmark added: test_bat_venom

## Test
tests/benchmark/test_llms.py::test_bat_venom

## Model
- HF name:    BrainDelay/BatVenom
- Loader:     third_party.tt_forge_models.bat_venom.causal_lm.pytorch.loader
- Variant:    ModelVariant.BAT_VENOM ("BatVenom")

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A (OOM)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           n150

## Failure analysis
The model is based on the Mistral-Small-3.1-24B architecture:
- num_hidden_layers: 40
- hidden_size: 5120
- num_attention_heads: 32
- vocab_size: 131072

At ~24B parameters, the model consistently OOMs during the decode warmup step
on a single n150 chip. The error occurs at DRAM allocation:

  Out of Memory: Not enough space to allocate 146800640 B DRAM buffer across
  12 banks... free: 12122848 B

This happens with both optimization_level=2 and optimization_level=1, and with
experimental_weight_dtype="bfp_bf8" already applied. The model passes with
--num-layers 1 but the full 40-layer model exceeds DRAM capacity.

The test was added to test_llms.py with a # FAILED: OOM comment (following the
established pattern from test_gemma_1_1_7b etc.) and is NOT added to the
perf-bench-matrix.json CI matrix.

## Infrastructure fix
A general fix was applied to tests/benchmark/benchmarks/llm_benchmark.py:
the get_weight_dtype_config_path() call was made optional (using getattr with
a None fallback) to support loaders that don't implement this method.

## Decode roofline (first decode graph, single-chip)
N/A — test failed before reaching decode

## Files changed
- tests/benchmark/test_llms.py (added test_bat_venom with # FAILED: OOM comment)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: graceful handling of missing get_weight_dtype_config_path)

## tt-forge-models submodule
no change
