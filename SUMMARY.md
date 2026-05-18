loader_path: third_party.tt_forge_models.chatgpt1_model.causal_lm.pytorch.loader
variant_id: model
arch: n150
status: DONE_FAIL
test_function: test_chatgpt1_model
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
failure_reason: "HuggingFace model ChatGpt1/model only has GGUF files (unsloth.F16.gguf, unsloth.Q4_K_M.gguf, unsloth.Q8_0.gguf) — no pytorch_model.bin or model.safetensors; loader uses AutoModelForCausalLM.from_pretrained which cannot load GGUF format; fix required in loader under third_party/tt_forge_models"

# Benchmark added: test_chatgpt1_model

## Test
tests/benchmark/test_llms.py::test_chatgpt1_model

## Model
- HF name:    ChatGpt1/model
- Loader:     third_party.tt_forge_models.chatgpt1_model.causal_lm.pytorch.loader
- Variant:    ModelVariant.CHATGPT1_MODEL ("model")

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Failure
The HuggingFace repository `ChatGpt1/model` only contains GGUF quantized weight
files:

- unsloth.F16.gguf
- unsloth.Q4_K_M.gguf
- unsloth.Q8_0.gguf

There is no `pytorch_model.bin` or `model.safetensors` file. The loader calls
`AutoModelForCausalLM.from_pretrained("ChatGpt1/model")` which requires standard
HuggingFace format weights and raises:

```
OSError: ChatGpt1/model does not appear to have a file named pytorch_model.bin
or model.safetensors.
```

This is a loader bug that must be fixed in `third_party/tt_forge_models`
(out of scope for this skill). The model is a Llama 3 8B Instruct fine-tune
based on `unsloth/llama-3-8b-Instruct-bnb-4bit` per the loader docstring.
The repo was likely published as GGUF-only.

## Measured (full model, defaults)
- Sample per second:  N/A
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           n150 (Wormhole n300)

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A — test did not reach compilation
Achieved vs top_perf_samples_per_sec: N/A

### System
N/A

## Files changed
- tests/benchmark/test_llms.py  (test_chatgpt1_model added)

## tt-forge-models submodule
no change
