loader_path: third_party.tt_forge_models.granite_3_1_2b_instruct_gguf.causal_lm.pytorch.loader
variant_id: Granite_3_1_2B_Instruct_Q4_K_M_GGUF
arch: p150
status: DONE_FAIL
test_function: test_granite_3_1_2b_instruct_q4_k_m_gguf
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
failure_reason: "NameError in loader _patched_load_gguf_checkpoint: 'gguf_path' is not defined — loader bug in third_party/tt_forge_models/granite_3_1_2b_instruct_gguf/causal_lm/pytorch/loader.py line 58; fix needed in tt-forge-models repo"

# Benchmark added: test_granite_3_1_2b_instruct_q4_k_m_gguf

## Test
tests/benchmark/test_llms.py::test_granite_3_1_2b_instruct_q4_k_m_gguf

## Model
- HF name:    bartowski/granite-3.1-2b-instruct-GGUF
- Loader:     third_party.tt_forge_models.granite_3_1_2b_instruct_gguf.causal_lm.pytorch.loader
- Variant:    Granite_3_1_2B_Instruct_Q4_K_M_GGUF

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
The loader at submodule HEAD fde82c3752 contains a bug in `_patched_load_gguf_checkpoint`
(third_party/tt_forge_models/granite_3_1_2b_instruct_gguf/causal_lm/pytorch/loader.py, line 58):

```python
def _patched_load_gguf_checkpoint(*args, **kwargs):
    _patch_granite_gguf_support()
    return _orig_load_gguf_checkpoint(gguf_path, return_tensors=return_tensors)
```

The variables `gguf_path` and `return_tensors` are referenced but never defined in the
function body — they should be `args[0]` and `kwargs.get("return_tensors")` respectively
(or simply `*args, **kwargs`). This raises `NameError: name 'gguf_path' is not defined`
on the very first call to `AutoTokenizer.from_pretrained(...)`.

The fix belongs in the tt-forge-models repo. Per skill policy, no files under
`third_party/tt_forge_models/` may be edited here.

## Decode roofline (first decode graph, single-chip)
N/A — test did not reach compilation or execution

## Files changed
- tests/benchmark/test_llms.py (test function added)

## tt-forge-models submodule
no change
