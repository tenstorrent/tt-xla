loader_path: third_party.tt_forge_models.chatgpt1_model.causal_lm.pytorch.loader
variant_id: model
arch: p150
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
failure_reason: "loader uses AutoModelForCausalLM.from_pretrained without gguf_file= but ChatGpt1/model only has GGUF files (unsloth.F16.gguf, unsloth.Q4_K_M.gguf, unsloth.Q8_0.gguf) — OSError: ChatGpt1/model does not appear to have a file named pytorch_model.bin or model.safetensors; fix requires updating loader in third_party/tt_forge_models to add gguf_file support"

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

## Measured (full model, defaults)
- Sample per second:  N/A
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           p150

## Failure Details
The HuggingFace model `ChatGpt1/model` is a Llama 3 8B Instruct fine-tune that ships
only GGUF weight files (`unsloth.F16.gguf`, `unsloth.Q4_K_M.gguf`, `unsloth.Q8_0.gguf`).
There is no `pytorch_model.bin` or `model.safetensors` in the repository.

The loader at `third_party/tt_forge_models/chatgpt1_model/causal_lm/pytorch/loader.py`
uses `AutoModelForCausalLM.from_pretrained("ChatGpt1/model")` without a `gguf_file=`
argument, which causes an `OSError` on model load. Other GGUF-aware loaders in
`tt_forge_models` (e.g. `a0l_12b_heretic_i1_gguf`) correctly pass `gguf_file=` to both
`AutoModelForCausalLM.from_pretrained` and `AutoTokenizer.from_pretrained`.

Fix required: update the loader in tt-forge-models to add `_GGUF_FILES` support and pass
`gguf_file=` when loading. This is out of scope for this skill.

Confirmed error at submodule HEAD 1e47cb75d6:
```
OSError: ChatGpt1/model does not appear to have a file named pytorch_model.bin or model.safetensors.
```

## Decode roofline (first decode graph, single-chip)
N/A — model failed to load

## Files changed
- tests/benchmark/test_llms.py (added test_chatgpt1_model stub)

## tt-forge-models submodule
no change — submodule at 1e47cb75d6
