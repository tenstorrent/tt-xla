# DeepSeek-V3.2 (DSA) decode enablement — findings & fixes

**Branch:** `hshah/dsa-vllm`
**Date:** 2026-06-12
**Hardware:** llmbox, 8×Wormhole B0 (GSPMD mesh `(2, 4)`, axes `("batch", "model")`)

## Goal

Get the DeepSeek-V3.2 (DeepSeek Sparse Attention, "DSA") single-layer smoke test
passing in the vLLM plugin:

```
tests/integrations/vllm_plugin/generative/test_tensor_parallel_generation.py::test_tensor_parallel_generation_llmbox_deepseek_v32_single_layer
```

The test compiles only decoder layer 0 (`num_hidden_layers=1`, a dense layer) with
`load_format="dummy"`, runs prefill + 16 decode steps, and asserts only that tokens
are produced (a single dummy-weight layer cannot produce coherent text).

## TL;DR

DSA realizes sparsity as an additive top-k attention mask fed into the existing
dense MLA ops with `is_causal=False`. **Prefill already worked. Decode did not**, and
fixing it took three changes across three layers of the stack:

| # | Layer | Problem | Fix |
|---|-------|---------|-----|
| 1 | tt-metal kernel | MLA decode op hard-asserts `is_causal` | Relax assert to allow non-causal **when an `attn_mask` is supplied** |
| 2 | tt-xla (`attention_mla.py`) | Decode mask needs a head per row (`[users,1,num_heads,max_seq]`), not prefill's head-broadcast `[users,1,1,S]` | `expand` the head-independent mask across query heads |
| 3 | tt-xla (`attention_mla.py` + `model_runner.py`) | Under TP, q heads are sharded over `"model"` but the mask was replicated → 128-head mask vs 32-head q | Shard the mask head dim with the Shardy `sharding_constraint_tensor` op |

Result: **`1 passed in 99.99s`**. End-to-end prefill + DSA decode compile, run on
device, and emit tokens.

---

## Background: how DSA decode is wired

DSA = Multi-head Latent Attention (MLA) + a "lightning indexer" that ranks past
tokens per query and keeps the top-k. On TT this is realized as an **additive
attention mask** (causal + top-k → `0`/`-inf`) handed to the *existing* dense MLA
kernels with `is_causal=False`:

- **Prefill:** `torch.ops.tt.flash_mla_prefill(..., attn_mask=mask, is_causal=False)`
  → ttnn `sdpa` (chunked prefill). Mask shape `[users, 1, S, S]` (broadcast over heads).
- **Decode:** `torch.ops.tt.paged_flash_mla_decode(..., attn_mask=mask, is_causal=False)`
  → ttnn `paged_flash_multi_latent_attention_decode` → `ttnn::prim::sdpa_decode`
  (`use_mla=True`).

The mask is built in `attention_dsa.compute_dsa_sparse_mask` and threaded through
`attention_mla.TTMLAAttentionBackendImpl`. The indexer is **head-independent** (it has
its own heads, separate from the 128 attention heads), so the mask is the same for
every attention head.

---

## Issue 1 — tt-metal MLA decode asserts causal-only

### Error

```
TT_FATAL @ .../sdpa_decode/device/sdpa_decode_device_operation.cpp:28: operation_attributes.is_causal
Multi-latent attention decode only tested for causal!
```

Surfaced during `compile_or_warm_up_model` → `capture_model` → `_precompile_backbone`,
wrapped on the Python side as `RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13`.

### Root cause

The decode device op validation has a hard guard for MLA:

```cpp
if (use_mla) {
    TT_FATAL(operation_attributes.head_dim_v.has_value(), "...");
    TT_FATAL(operation_attributes.is_causal, "Multi-latent attention decode only tested for causal!");
}
```

DSA decode runs `is_causal=False` with an explicit mask, which trips the assert.

Crucially, this is a **conservative "not tested" guard, not a real kernel
limitation**:

- MLA decode uses the **same kernels** as regular SDPA decode
  (`reader_decode_all.cpp`, `compute/sdpa_flash_decode.cpp`,
  `writer_decode_all.cpp`).
- Those kernels gate masking on a `use_attention_mask` compile-time arg that is
  **independent of `use_mla`**. The generic non-causal + paged + mask machinery
  already exists and is exercised by the non-MLA path.
- `k_chunk_size` (required `> 0` for the non-causal masked path) is **already
  supplied** as `32` by the tt-mlir runtime handler — see
  `runtime/lib/ttnn/operations/transformer/paged_flash_multi_latent_attention_decode.cpp:54`
  (`programConfig->k_chunk_size = 32;`). 32 is a power-of-2 multiple of 32 and divides
  any paged sequence length (which is always a multiple of `block_size=32`), so all
  the downstream chunk/mask-alignment checks pass.

### Fix

Relax the assert to permit non-causal MLA decode **only when an explicit mask is
present** (so the kernel always has a defined source of masking):

`third_party/tt-mlir/.../tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_device_operation.cpp`

```cpp
     if (use_mla) {
         TT_FATAL(
             operation_attributes.head_dim_v.has_value(), "Must provide head_dim_v for multi-latent attention decode");
-        TT_FATAL(operation_attributes.is_causal, "Multi-latent attention decode only tested for causal!");
+        // Multi-latent attention decode runs the shared sdpa_decode kernels, which
+        // gate masking on use_attention_mask independently of MLA. Causal mode is
+        // the well-trodden path; non-causal MLA decode is allowed only when an
+        // explicit additive attn_mask is supplied (e.g. DeepSeek Sparse Attention's
+        // top-k mask), so the kernel still has a defined source of masking.
+        TT_FATAL(
+            operation_attributes.is_causal || tensor_args.attn_mask.has_value(),
+            "Multi-latent attention decode requires causal mode or an explicit attn_mask "
+            "(got non-causal without a mask).");
     } else {
         TT_FATAL(tensor_args.v.has_value(), "Must have 3 input tensors and mask");
     }
```

> ⚠️ **This change lives in the tt-metal sub-submodule** (`third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal`), which is a **clean upstream checkout not tracked by the tt-xla repo**. It is a working-tree edit plus a hand-installed `_ttnncpp.so`, so it is **ephemeral** — `clean_build.sh` or a tt-mlir uplift will wipe it. For durability it must go onto a tt-metal branch pinned by the tt-mlir branch (`hshah/mla-rebased-464a5`). See the build/install steps below.

### Rebuild & install

```bash
cd third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/build_Release
ninja _ttnncpp.so                       # incremental: one TU + relink, warm ccache (~minutes)

# the runtime loads from the tt-mlir install dir, which is a *copy*, so reinstall:
DEST=/localdev/hshah/tt-xla/third_party/tt-mlir/install/lib/_ttnncpp.so
cp -p "$DEST" "${DEST}.prepatch.bak"    # backup
cp -p ttnn/_ttnncpp.so "$DEST"          # install the freshly-built lib
```

(`ninja _ttnncpp.so` updates `build_Release/ttnn/_ttnncpp.so`; the install dir
`third_party/tt-mlir/install/lib/_ttnncpp.so` is a separate copy that the runtime
`dlopen`s, so it must be overwritten explicitly.)

---

## Issue 2 — decode mask has the wrong head layout

### Error (after Issue 1 fixed)

```
TT_FATAL @ .../sdpa_decode/device/sdpa_decode_device_operation.cpp:130: mask_shape[2] == q_shape[2]
Expect same number of padded heads in mask as in Q, got 128 and 32
```

### Root cause

The decode kernel's reader treats the mask as `(B, PNHt, PSt)` — i.e.
`[users, 1, num_heads, max_seq]` — because in decode `S == 1`, so the **query heads
occupy the kernel's row dimension** (the rows of `QK^T`). The validator therefore
requires `mask_shape[2] == q_heads`.

This is **different from prefill**, where the mask is `[users, 1, S, S]` and the head
dimension (`dim 1`) is broadcast (`== 1`). The DSA mask was being built head-broadcast
(`[users, 1, 1, max_seq]`) for both paths; that works for prefill but not decode.

(The `128` vs `32` is explained in Issue 3 — `128` was the full mask, `32` the
per-shard q heads.)

### Fix

The indexer mask is head-independent, so broadcast the single computed row across the
query heads in the decode branch of `attention_mla.TTMLAAttentionBackendImpl._forward_decode`:

```python
decode_mask = attn_mask.expand(-1, -1, self.num_heads, -1)
```

(Full diff in the appendix.)

---

## Issue 3 — decode mask not sharded to match Q under tensor parallelism

### Error (after Issue 2 fixed)

```
module_builder.cc:1034 ERR| Failed to run TTIRToTTNNCommon pipeline
```
wrapped as `ValueError: Error code: 13`, raised from `torch_xla._XLAC._xla_warm_up_cache`
while compiling the decode graph.

(Before getting the right fix, an intermediate attempt used `xs.mark_sharding` on the
mask, which produced exactly this compile failure — see below.)

### Root cause — two parts

**(a) The mask must be head-sharded like Q.** Under GSPMD tensor parallelism the
query heads are sharded over the `"model"` mesh axis (`q_b_proj` is column-parallel:
`safe_mark_sharding(self.q_weight, mesh, ("model", "batch"))`). On the `(2,4)` mesh the
`"model"` axis is size 4, so per device q has `128 / 4 = 32` heads. The mask, built at
full size (`128` heads) and **not** sharded, stayed replicated → the per-device kernel
saw a `128`-head mask against `32`-head q. **That is the `128 and 32` mismatch from
Issue 2.**

**(b) Use the right sharding primitive.** The decode mask is an **intermediate tensor
created inside the compiled forward**, not a graph parameter/input. For those:

- `xs.mark_sharding` (GSPMD, used for weights/inputs) injects a sharding annotation
  that the tt-mlir Shardy pipeline could not reconcile with the custom MLA-decode op →
  `TTIRToTTNNCommon` failed to lower.
- `tt_torch.sharding.sharding_constraint_tensor` (which emits the Shardy
  `torch.ops.tt.sharding_constraint` op) is the supported way to reshard an
  intermediate, and is what `model_runner` already uses for hidden states / logits.

Because the mask is head-independent, any 32-head slice each device receives is
identical, so sharding it is semantically free.

### Fix

1. Publish the runner's mesh globally so the attention forward (which has no runner
   handle) can reach it — `model_runner.py`, right after the mesh is created:

   ```python
   self.mesh = xs.Mesh(device_ids, mesh_shape, ("batch", "model"))
   xs.set_global_mesh(self.mesh)
   ```

2. Shard the decode mask's head dim over `"model"` with the Shardy op —
   `attention_mla.py`:

   ```python
   decode_mask = attn_mask.expand(-1, -1, self.num_heads, -1)
   mesh = xs.get_global_mesh()
   if mesh is not None:
       decode_mask = sharding_constraint_tensor(
           decode_mask, mesh, (None, None, "model", None)
       )
   ```

After this the decode graph compiles (`PNHt=1`, i.e. 32 per-shard heads = one tile),
the kernels JIT-build, and device execution succeeds.

---

## Result

```
prompt: I like taking walks in the, output: 'ностями Mutterностями Mutter...'
PASSED
================== 1 passed, 20 warnings in 99.99s (0:01:39) ===================
```

Gibberish output is expected and explicitly allowed by the test (single dummy-weight
layer; it only asserts `len(output.token_ids) > 0`).

### Blast radius

- The tt-xla changes only affect the DSA decode path (guarded by `attn_mask is not
  None`). Plain MLA decode (`attn_mask is None`) and non-MLA models are unchanged.
- `xs.set_global_mesh` is additive; nothing else in the plugin read the global mesh
  before this change.
- The tt-metal change only **loosens** a validation constraint (causal is still
  permitted); it cannot regress existing causal MLA decode.

---

## Follow-ups / caveats

- **Durability of the tt-metal patch.** It lives in an untracked sub-submodule. To
  survive a clean build / uplift it must be committed to a tt-metal branch referenced
  by the tt-mlir `hshah/mla-rebased-464a5` branch (and ideally upstreamed). Backup of
  the pre-patch lib is at `third_party/tt-mlir/install/lib/_ttnncpp.so.prepatch.bak`.
- **Numerics are unvalidated.** This is a dummy-weight smoke test; it proves the graph
  compiles/runs and emits tokens, not that the DSA math is correct. Real
  DeepSeek-V3.2 weights are needed to check PCC.
- **Not exercised:** multi-layer / MoE experts, FP8-quantized indexer checkpoints.
- **Prefill indexer cost.** Prefill builds a full `[users, S, NH, S]` indexer score
  tensor (O(S²·NH)) — memory-heavy at long context; a fused indexer kernel is future
  work.
- **Regression check not run.** Plain-MLA TP tests were not re-run after these changes
  (the changes are guarded), but a quick MLA TP smoke test would confirm no regression.

---

## Appendix — full code changes

### A. tt-metal — `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_device_operation.cpp`

*(sub-submodule: `third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal`)*

```diff
@@ -25,7 +25,15 @@ void SdpaDecodeDeviceOperation::validate_on_program_cache_miss(
     if (use_mla) {
         TT_FATAL(
             operation_attributes.head_dim_v.has_value(), "Must provide head_dim_v for multi-latent attention decode");
-        TT_FATAL(operation_attributes.is_causal, "Multi-latent attention decode only tested for causal!");
+        // Multi-latent attention decode runs the shared sdpa_decode kernels, which
+        // gate masking on use_attention_mask independently of MLA. Causal mode is
+        // the well-trodden path; non-causal MLA decode is allowed only when an
+        // explicit additive attn_mask is supplied (e.g. DeepSeek Sparse Attention's
+        // top-k mask), so the kernel still has a defined source of masking.
+        TT_FATAL(
+            operation_attributes.is_causal || tensor_args.attn_mask.has_value(),
+            "Multi-latent attention decode requires causal mode or an explicit attn_mask "
+            "(got non-causal without a mask).");
     } else {
         TT_FATAL(tensor_args.v.has_value(), "Must have 3 input tensors and mask");
     }
```

### B. tt-xla — `integrations/vllm_plugin/vllm_tt/attention_mla.py`

```diff
@@ -9,12 +9,15 @@ from typing import TYPE_CHECKING, Optional

 import torch
 import torch.nn as nn
+import torch_xla.distributed.spmd as xs
 from vllm.forward_context import get_forward_context
 from vllm.model_executor.custom_op import PluggableLayer
 from vllm.model_executor.layers.attention.mla_attention import MLAAttention
 from vllm.model_executor.layers.mla import MultiHeadLatentAttentionWrapper
 from vllm.v1.attention.backend import AttentionBackend, AttentionLayer, MLAAttentionImpl

+from tt_torch.sharding import sharding_constraint_tensor
+
 from .attention import TTAttentionMetadataBuilder, TTMetadata
 from .logger import tt_init_logger

@@ -385,7 +388,26 @@ class TTMLAAttentionBackendImpl(MLAAttentionImpl):
             # run the kernel uncausal with the explicit mask. cur_pos is still
             # forwarded (the decode kernel ignores it for masking when not causal).
             is_causal = False
-            decode_mask = attn_mask
+            # The paged MLA decode kernel lays the mask out as
+            # [users, 1, num_heads, max_seq]: with S == 1 the query heads occupy the
+            # kernel's row dimension, so the mask validation requires a head per row
+            # (mask_shape[2] == q heads), unlike prefill's head-broadcast
+            # [users, 1, S, S]. The indexer's top-k selection is head-independent, so
+            # broadcast the single computed row across all query heads.
+            decode_mask = attn_mask.expand(-1, -1, self.num_heads, -1)
+            # Under tensor parallelism the query heads are sharded over the "model"
+            # mesh axis (q_b_proj is column-parallel), so per device q has
+            # num_heads / model_axis heads. Constrain the mask's head dim the same
+            # way or the kernel sees a 128-head mask against 32-head q. The mask rows
+            # are identical across heads, so any head-shard slice is correct. Use the
+            # Shardy sharding-constraint op (the supported way to reshard an
+            # intermediate inside the compiled graph), not xs.mark_sharding which is
+            # for graph params/inputs.
+            mesh = xs.get_global_mesh()
+            if mesh is not None:
+                decode_mask = sharding_constraint_tensor(
+                    decode_mask, mesh, (None, None, "model", None)
+                )
         else:
             is_causal = attn_metadata.is_causal if attn_metadata is not None else True
             decode_mask = None if is_causal else attn_metadata.attn_mask
```

### C. tt-xla — `integrations/vllm_plugin/vllm_tt/model_runner.py`

```diff
             self.mesh = xs.Mesh(device_ids, mesh_shape, ("batch", "model"))
+            # Publish the mesh globally so code that runs inside the compiled
+            # forward (and has no runner handle) can match its sharding — e.g. the
+            # DeepSeek Sparse Attention decode path shards its additive mask's head
+            # dim over "model" to line up with the query heads (see
+            # attention_mla.TTMLAAttentionBackendImpl._forward_decode).
+            xs.set_global_mesh(self.mesh)
             # Updating the config to reflect the actual mesh shape used.
             if self.use_2d_mesh and 1 in mesh_shape:
                 self.use_2d_mesh = False
```

---

## Reproduce

```bash
source venv/activate
python -m pytest -svv \
  "tests/integrations/vllm_plugin/generative/test_tensor_parallel_generation.py::test_tensor_parallel_generation_llmbox_deepseek_v32_single_layer"
```

(Requires the patched + reinstalled `_ttnncpp.so` from Issue 1.)
