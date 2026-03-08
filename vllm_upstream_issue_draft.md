# `record_metadata_for_reloading` causes host memory regression during `torch.compile` tracing

## Summary

`record_metadata_for_reloading()` (introduced in PR #32133) runs unconditionally during model initialization for all users, even though it only benefits the QeRL layerwise weight reloading use case. On `torch_xla` backends, this causes a 2-3x host memory regression during `torch.compile` tracing. The regression scales with model size.

## Impact

Measured with Qwen3 models on Tenstorrent hardware via `torch_xla` + PJRT backend:

| Model | v0.15.1 (before PR #32133) | v0.16.0 (after) | v0.16.0 + workaround |
|-------|---------------------------|-----------------|---------------------|
| Qwen3-0.6B | 3.7 GB | 8.5 GB (2.3x) | 3.7 GB |
| Qwen3-1.7B | 7.5 GB | 22 GB (2.9x) | 8.0 GB |
| Qwen3-4B | 16.5 GB | 49.6 GB (3.0x) | ~17 GB |
| Qwen3-32B | ~150 GB | ~435 GB (2.9x) | ~128 GB |

The regression occurs during `torch._dynamo` tracing of the model forward pass, before any backend-specific compilation. Host RSS spikes from ~6.5 GB to ~20.6 GB for Qwen3-1.7B during the `self.model()` call in the compilation warmup path.

## Root Cause

`record_metadata_for_reloading()` is called from `initialize_model()` in `model_executor/model_loader/utils.py`. It iterates over every module and calls `capture_layer_to_meta()`, which:

1. Calls `tensor.data.to("meta")` on every parameter
2. Copies `tensor.__dict__` (containing vLLM parameter attributes like `weight_loader`, `output_dim`, etc.) to the meta tensor
3. Stores these meta tensor copies in `LAYERWISE_INFO` (a `WeakKeyDictionary`)

On `torch_xla`, these additional tensor references and `__dict__` copies cause the XLA dynamo bridge to create significantly more tensor copies during graph tracing. The effect scales linearly with model size.

On GPU with `eager` or `aot_eager` backends, we did not observe a measurable difference — the regression appears specific to `torch_xla`'s graph capture mechanism.

## Bisect

Git bisect across 471 commits between v0.15.1 and v0.16.0 identified the regression:

- **Last good**: `74898a701` — `[BugFix][LoRA] TritonExperts` (3.7 GB)
- **First bad**: `f857a03f6` — `[QeRL] Layerwise Reloading (#32133)` (8.5 GB)

## Workaround

Monkey-patching `record_metadata_for_reloading` to a no-op before model loading fully resolves the regression:

```python
import vllm.model_executor.model_loader.utils as loader_utils
loader_utils.record_metadata_for_reloading = lambda model: None
```

## Suggested Fix

Make `record_metadata_for_reloading` lazy or opt-in. The metadata is only needed when `reload_weights()` is actually called, so it could be captured on-demand at that point rather than unconditionally at model initialization. This would avoid the overhead for all users who don't use layerwise weight reloading.

## Environment

- vllm: 0.16.0 (also present in 0.17.0)
- torch: 2.9.1
- torch_xla: custom build (Tenstorrent PJRT backend)
- Hardware: Tenstorrent QuietBox (Wormhole n300-llmbox)
- The regression is observed during `torch.compile` with XLA backend; not reproduced on CPU with `eager`/`aot_eager` backends

