loader_path: third_party.tt_forge_models.pytorch_qwen3_4b_int8_int4.causal_lm.pytorch.loader
variant_id: 4B_INT8_INT4
arch: p150
status: DONE_FAIL
test_function: test_pytorch_qwen3_4b_int8_int4
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
failure_reason: "loader torchao fallback broken: ValueError from transformers is 'TorchAoConfig requires torchao to be installed' but loader only catches errors containing 'Failed to find class'; fix needed in third_party/tt_forge_models/pytorch_qwen3_4b_int8_int4/causal_lm/pytorch/loader.py"

# Benchmark added: pytorch_qwen3_4b_int8_int4

## Test
tests/benchmark/test_llms.py::test_pytorch_qwen3_4b_int8_int4

## Model
- HF name:    pytorch/Qwen3-4B-INT8-INT4
- Loader:     third_party.tt_forge_models.pytorch_qwen3_4b_int8_int4.causal_lm.pytorch.loader
- Variant:    ModelVariant.QWEN3_4B_INT8_INT4

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

### Root cause
The model `pytorch/Qwen3-4B-INT8-INT4` uses TorchAO INT8-INT4 quantization.
`torchao` is not installed in the tt-xla venv, so `AutoModelForCausalLM.from_pretrained`
raises:

    ValueError: TorchAoConfig requires torchao to be installed. Install with `pip install torchao`

The loader has a fallback for exactly this case (creates a random-weight unquantized model
for compile-only runs) but the fallback only activates when the error message contains
`"Failed to find class"`. The actual transformers error has a different message, so the
`except ValueError` block re-raises.

### Fix needed (in tt-forge-models)
Update the `except ValueError` guard in
`third_party/tt_forge_models/pytorch_qwen3_4b_int8_int4/causal_lm/pytorch/loader.py`
to also catch this error, e.g.:

    if "Failed to find class" not in str(e) and "requires torchao to be installed" not in str(e):
        raise

or simply:

    if "torchao" not in str(e).lower() and "Failed to find class" not in str(e):
        raise

## Decode roofline (first decode graph, single-chip)
N/A — test did not reach compilation stage

## Files changed
- tests/benchmark/test_llms.py (test function added but fails at model loading)

## tt-forge-models submodule
no change
