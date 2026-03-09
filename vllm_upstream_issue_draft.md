# [Bug]: `record_metadata_for_reloading` causes ~3x host memory regression during torch.compile on XLA backends

## Your current environment

- vLLM Version: 0.16.0 (also present in 0.17.0)
- PyTorch version: 2.9.1
- torch-xla: 2.9.0+git8ee513e
- Python version: 3.12.12
- OS: Ubuntu 22.04.5 LTS (x86_64)
- Hardware: Tenstorrent Wormhole (n300) via torch_xla PJRT backend

<details>
<summary>Full output of <code>python collect_env.py</code></summary>

```text
Collecting environment information...
==============================
        System Info
==============================
OS                           : Ubuntu 22.04.5 LTS (x86_64)
GCC version                  : (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0
Clang version                : 17.0.6 (++20231209124227+6009708b4367-1~exp1~20231209124336.77)
CMake version                : version 4.2.1
Libc version                 : glibc-2.35

==============================
       PyTorch Info
==============================
PyTorch version              : 2.9.1+cu128
Is debug build               : False
CUDA used to build PyTorch   : 12.8
ROCM used to build PyTorch   : N/A

==============================
      Python Environment
==============================
Python version               : 3.12.12 (main, Oct 10 2025, 08:52:57) [GCC 11.4.0] (64-bit runtime)
Python platform              : Linux-5.15.0-141-generic-x86_64-with-glibc2.35

==============================
       CUDA / GPU Info
==============================
Is CUDA available            : False
CUDA runtime version         : No CUDA
CUDA_MODULE_LOADING set to   : N/A
GPU models and configuration : No CUDA
Nvidia driver version        : No CUDA
cuDNN version                : No CUDA
HIP runtime version          : N/A
MIOpen runtime version       : N/A
Is XNNPACK available         : True

==============================
Versions of relevant libraries
==============================
[pip3] numpy==2.2.6
[pip3] torch==2.9.1
[pip3] torch-xla==2.9.0+git8ee513e
[pip3] transformers==4.57.6
[pip3] triton==3.5.1

==============================
         vLLM Info
==============================
vLLM Version                 : 0.16.0
vLLM Build Flags:
  CUDA Archs: Not Set; ROCm: Disabled

==============================
     Environment Variables
==============================
VLLM_TARGET_DEVICE=empty
TORCHINDUCTOR_COMPILE_THREADS=1
```

</details>

## Describe the bug

`record_metadata_for_reloading()` (introduced in PR #32133) runs unconditionally during `initialize_model()` for all users, even though it only benefits the QeRL layerwise weight reloading use case. On `torch_xla` backends, this causes a 2-3x host memory regression during `torch.compile` tracing. The regression scales with model size.

Even outside of the XLA-specific memory impact, `record_metadata_for_reloading` does unnecessary work at model initialization for the vast majority of users who never call `reload_weights()`. It iterates every module, creates meta tensor copies via `tensor.data.to("meta")`, and copies `__dict__` on every parameter — all eagerly, with no way to opt out.

### Impact

Measured with Qwen3 models via `torch_xla` + PJRT backend (peak host RSS):

| Model | v0.15.1 (before PR #32133) | v0.16.0 (after) | v0.16.0 + fix |
|-------|---------------------------|-----------------|---------------|
| Qwen3-0.6B | 3.7 GB | 8.5 GB (2.3x) | 3.7 GB |
| Qwen3-1.7B | 7.5 GB | 22 GB (2.9x) | 8.0 GB |
| Qwen3-4B | 16.5 GB | 49.6 GB (3.0x) | ~17 GB |
| Qwen3-32B | ~150 GB | ~435 GB (2.9x) | ~128 GB |

### Root cause

`record_metadata_for_reloading()` is called from `initialize_model()` in `model_executor/model_loader/utils.py`. It iterates over every module and calls `capture_layer_to_meta()`, which:

1. Calls `tensor.data.to("meta")` on every parameter
2. Copies `tensor.__dict__` (containing vLLM parameter attributes like `weight_loader`, `output_dim`, etc.) to the meta tensor
3. Stores these meta tensor copies in `LAYERWISE_INFO` (a `WeakKeyDictionary`)

On `torch_xla`, these additional tensor references and `__dict__` copies cause the XLA dynamo bridge to create significantly more tensor copies during graph tracing. The effect scales linearly with model size.

On GPU with `eager` or `aot_eager` backends, we did not observe a measurable memory difference during compilation — the severe memory impact appears specific to `torch_xla`'s graph capture mechanism. However, the unconditional metadata capture is still unnecessary overhead for all non-QeRL users.

### Bisect

Git bisect across 471 commits between v0.15.1 and v0.16.0 identified the regression:

- **Last good**: `74898a701` — `[BugFix][LoRA] TritonExperts` (3.7 GB peak RSS)
- **First bad**: `f857a03f6` — `[QeRL] Layerwise Reloading (#32133)` (8.5 GB peak RSS)

### Reproduction

The regression requires vLLM's full model loading path (which creates `BasevLLMParameter` subclasses with populated `__dict__`) running on a `torch_xla` backend. We were unable to create a standalone GPU/CPU repro because the severe memory impact is specific to `torch_xla`'s graph capture.

The regression can be verified by running vLLM inference on any XLA device and comparing peak host RSS with and without the following workaround:

```python
# Monkey-patch to disable record_metadata_for_reloading
import vllm.model_executor.model_loader.utils as loader_utils
loader_utils.record_metadata_for_reloading = lambda model: None
```

### Suggested fix (tested, confirmed working)

Move `record_metadata_for_reloading(model)` from `initialize_model()` to `initialize_layerwise_reload()`, so metadata is captured on-demand when `reload_weights()` is first called rather than unconditionally at model init.

This is a 2-line change:

**`model_executor/model_loader/utils.py`** — remove both `record_metadata_for_reloading(model)` calls from `initialize_model()`:
```diff
         with set_current_vllm_config(vllm_config, check_compile=True, prefix=prefix):
             model = model_class(vllm_config=vllm_config, prefix=prefix)
-            record_metadata_for_reloading(model)
             return model
```

**`model_executor/model_loader/reload/layerwise.py`** — add it at the top of `initialize_layerwise_reload()`:
```diff
 def initialize_layerwise_reload(model: torch.nn.Module):
+    # Capture metadata on-demand rather than eagerly at model init
+    record_metadata_for_reloading(model)
+
     # disable torchao reloading to avoid infinite recursion
```

We tested this fix on our XLA platform and confirmed it resolves the regression (3.9 GB vs 8.5 GB for Qwen3-0.6B) while preserving QeRL functionality — the metadata is just captured lazily at first use.
