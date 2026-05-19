loader_path: third_party.tt_forge_models.mradermacher_kai_3b_instruct_i1_gguf.causal_lm.pytorch.loader
variant_id: 3B_INSTRUCT_I1_GGUF
arch: p150
status: DONE_FAIL
test_function: test_mradermacher_kai_3b_instruct_i1_gguf
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
failure_reason: "loader tokenizer init fails: ValueError: add_bos_token = True but bos_token = None in AutoTokenizer.from_pretrained with GGUF file (transformers CodeLlama tokenizer)"

# Benchmark added: test_mradermacher_kai_3b_instruct_i1_gguf

## Test
tests/benchmark/test_llms.py::test_mradermacher_kai_3b_instruct_i1_gguf

## Model
- HF name:    mradermacher/Kai-3B-Instruct-i1-GGUF
- Loader:     third_party.tt_forge_models.mradermacher_kai_3b_instruct_i1_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.KAI_3B_INSTRUCT_I1_GGUF ("3B_INSTRUCT_I1_GGUF")

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A (test did not complete)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           p150 (blackhole p300c)

## Failure Details

The test failed at the tokenizer loading step — before any compilation or
device execution. The loader calls `AutoTokenizer.from_pretrained` with the
GGUF file `Kai-3B-Instruct.i1-Q4_K_M.gguf`, which resolves to a CodeLlama
tokenizer. The CodeLlama tokenizer's `__init__` sets `add_bos_token=True`
but the GGUF's tokenizer config provides no `bos_token`, causing:

```
ValueError: add_bos_token = True but bos_token = None
```

Traceback (key frames):
```
loader.py:190  → AutoTokenizer.from_pretrained(pretrained_model_name, gguf_file=..., ...)
tokenization_auto.py:736 → tokenizer_class.from_pretrained(...)
tokenization_code_llama.py:167 → super().__init__(...)
tokenization_utils_tokenizers.py:428 → raise ValueError("add_bos_token = True but bos_token = None")
```

This is a bug in the loader or an incompatibility between the GGUF file's
tokenizer metadata and transformers 5.2.0's CodeLlama tokenizer. The fix
belongs in `third_party/tt_forge_models` — this skill does not modify
loader code.

## Decode roofline (first decode graph, single-chip)
N/A — test did not reach compilation or device execution.

## Files changed
- tests/benchmark/test_llms.py (test function added)
- .github/workflows/perf-bench-matrix.json (matrix entry added)

## tt-forge-models submodule
no change
