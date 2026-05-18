loader_path: third_party.tt_forge_models.cosail_knu_llama2_2_7b_8bit.causal_lm.pytorch.loader
variant_id: 2.7B_8bit
arch: p150
status: DONE_FAIL
test_function: test_cosail_knu_llama2_2_7b_8bit
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
failure_reason: "hqq package missing: model cosail-knu/llama2-2.7b_8 uses HQQ 8-bit quantization and requires hqq>=0.2.1 which is not installed in the benchmark venv (ImportError: A valid HQQ version (>=0.2.1) is not available)"

# Benchmark added: test_cosail_knu_llama2_2_7b_8bit

## Test
tests/benchmark/test_llms.py::test_cosail_knu_llama2_2_7b_8bit

## Model
- HF name:    cosail-knu/llama2-2.7b_8
- Loader:     third_party.tt_forge_models.cosail_knu_llama2_2_7b_8bit.causal_lm.pytorch.loader
- Variant:    ModelVariant.LLAMA2_2_7B_8BIT ("2.7B_8bit")

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A (test failed before inference)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           p150

## Failure details

The bring-up run failed during model loading with:

```
ImportError: A valid HQQ version (>=0.2.1) is not available.
Please follow the instructions to install it: https://github.com/mobiusml/hqq/
```

The model `cosail-knu/llama2-2.7b_8` was saved in HQQ 8-bit quantized format. When
`transformers` loads its `config.json`, it finds a `quantization_config` of type `hqq`
and attempts to instantiate the HQQ quantizer, which raises `ImportError` because
`hqq>=0.2.1` is not present in the benchmark venv. This is a model-level dependency
issue — the loader calls `AutoModelForCausalLM.from_pretrained()` which unconditionally
triggers the HQQ code path. No benchmark-infrastructure fix is possible without either:
1. Installing the `hqq` package (environment change, not in scope), or
2. Modifying the loader to skip quantization (not in scope — loaders are read-only).

## Decode roofline (first decode graph, single-chip)
N/A — test did not reach inference

## Files changed
- tests/benchmark/test_llms.py (test function added)

## tt-forge-models submodule
no change
