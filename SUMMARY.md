loader_path: third_party.tt_forge_models.audiogemma_3n_finetune_gguf.causal_lm.pytorch.loader
variant_id: FINETUNE_GGUF
arch: p150
status: DONE_FAIL
test_function: test_audiogemma_3n_finetune_gguf
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
failure_reason: "KeyError: 'sliding_attention' in modeling_gemma3n.py:1725 — loader's _patch_transformers_gemma3n_gguf() sets layer_types=['sliding_attention',...] but position_embeddings dict is not populated with a 'sliding_attention' key; bug in third_party/tt_forge_models loader GGUF patch logic"

# Benchmark added: test_audiogemma_3n_finetune_gguf

## Test
tests/benchmark/test_llms.py::test_audiogemma_3n_finetune_gguf

## Model
- HF name:    mradermacher/Audiogemma-3N-finetune-GGUF
- Loader:     third_party.tt_forge_models.audiogemma_3n_finetune_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.AUDIOGEMMA_3N_FINETUNE_GGUF

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
The test fails immediately during CPU golden generation (before any TT device execution) with:

```
KeyError: 'sliding_attention'
```

Traceback (abbreviated):
```
transformers/models/gemma3n/modeling_gemma3n.py:1725: in forward
    position_embeddings[decoder_layer.attention_type],
KeyError: 'sliding_attention'
```

The loader's `_patch_transformers_gemma3n_gguf()` function constructs `layer_types`
from the GGUF metadata's `_sliding_window_pattern` field, producing entries like
`['sliding_attention', 'full_attention', ...]`. However, the patched
`load_gguf_checkpoint` doesn't populate the `position_embeddings` dict with a
`'sliding_attention'` key. The Gemma 3N forward method then fails when it tries
to look up RoPE embeddings for a sliding-attention layer.

This is a bug inside `third_party/tt_forge_models/audiogemma_3n_finetune_gguf/causal_lm/pytorch/loader.py`
(specifically in `patched_load_gguf_checkpoint`). Modifying files under
`third_party/tt_forge_models/` is out of scope for this skill. The fix belongs
in the tt-forge-models repo.

## Decode roofline (first decode graph, single-chip)
N/A — test did not reach device execution.

## Files changed
- tests/benchmark/test_llms.py (test function added)
- SUMMARY.md

## tt-forge-models submodule
no change
