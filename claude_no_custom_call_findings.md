# Investigation: Missing MLA Custom Calls in StableHLO

## Context

Running `test_tensor_parallel_generation_llmbox_small` with DeepSeek-V3 (MLA attention) produces 4 StableHLO modules in the log, none of which contain the expected `tt.flash_mla_prefill` or `tt.paged_flash_multi_latent_attention_decode` custom_calls. The TT MLA backend IS selected (log line 190: "Using TT MLA Attention layer").

## Root Cause

The MLA custom_calls are absent from the logged StableHLO because **the attention module (which would contain them) crashes before its StableHLO is ever logged**, due to a device mesh mismatch.

## Detailed Analysis

### 1. Graph Breaks Split the Model Into 5+ Sub-graphs

Torch.dynamo traces the model and splits it at graph break points (likely at `get_forward_context()` calls and other untraceable operations). The 4 successfully compiled modules are all **pre-attention** computations:

| Module | Mesh | Contents |
|--------|------|----------|
| #1 (`SyncTensorsGraph.16`) | [2,4] (8 devices) | Token embedding: gather + all_gather |
| #2 (`SyncTensorsGraph.42`) | [1,1] → overridden to [2,4] | Input LayerNorm (RMS norm) |
| #3 (`SyncTensorsGraph.249`) | [2,4] (8 devices) | QKV projections + RoPE + q_a/kv_a LayerNorms |
| #4 (`SyncTensorsGraph.7`) | [1,1] (**NOT overridden**) | Zero tensor initialization (1x16384 bf16) |

The MLA custom_calls would be in **Module #5** (the attention computation), which is never logged.

### 2. Module #5 Exists But Crashes

Evidence that module #5 (attention) IS being compiled:
- **Log line 6538**: `_validate_demangling: Failed to demangle 1 argument name(s): ['L__self___W_UK_T']` — `W_UK_T` is the absorbed key-nope weight matrix used exclusively in the MLA decode path (`forward_impl` → `torch.bmm(mqa_q_nope, self.W_UK_T)`)
- The error traceback confirms `mla_attention.py:471 forward_impl` → `backend.py:225 __call__` → `extract_compiled_graph` → `torch_xla.sync` → crash

### 3. The Crash: Device Mesh Mismatch

The crash sequence (log lines 6508-6534):

```
[22:03:55.851] ClientInstance::getOrCreateMeshDevice - reshaping mesh device - [2, 4] -> [1, 1]
[22:03:55.851] Moving tensors to host.
[22:03:56.641] Closing parent mesh.
[22:03:56.947] ClientInstance::openMeshDevice - setting fabric config: DISABLED
...
[22:03:57.390] LoadedExecutableInstance::PJRT_LoadedExecutable_Execute
[22:03:57.390] FlatbufferLoadedExecutableInstance::Execute
[22:03:57.390] ERR| Device count mismatch: 8 vs 1
```

**What happens:**
1. Module #4 is compiled with mesh [1,1] (trivial — it's just a zero tensor)
2. The PJRT plugin **reshapes the physical device mesh from [2,4] to [1,1]** to execute module #4
3. When a module compiled for mesh [2,4] (8 devices) is subsequently executed on the [1,1] mesh (1 device), it fails with `Device count mismatch: 8 vs 1`
4. This becomes `INTERNAL: Error code: 13`, preventing module #5's StableHLO from ever being generated/logged

### 4. Why Module #4's Mesh Isn't Overridden (But Module #2's Was)

Both modules #2 and #4 start with trivial mesh `<["x"=1, "y"=1]>`. But they're handled differently:

- **Module #2** (log line 3150): `"SPMD-enabled mesh has trivial size [1, 1], reusing already opened mesh shape"` — the trivial mesh is detected and overridden to [2,4]. No mesh reshape occurs.
- **Module #4**: This message does **NOT** appear. The [1,1] mesh is taken literally, triggering the destructive mesh reshape.

The logic in `shlo_set_proper_sdy_mes` (C++ side, in `module_builder.cc`) decides whether to reuse the existing mesh or reshape. Something between module #2 and module #4 changes the state such that module #4 no longer qualifies for the override.

### 5. What Module #5 Would Contain

Based on the code path (`forward_impl` decode path in `mla_attention.py`), module #5 would contain:
- `_tt_concat_and_cache_mla` — KV cache write via `index_put` (skipped during profile run since kv_cache is empty 1D tensor)
- `torch.bmm(mqa_q_nope, self.W_UK_T)` — query absorption into latent space (confirmed by `W_UK_T` demangling at line 6538)
- **`torch.ops.tt.paged_flash_multi_latent_attention_decode(...)`** — the missing custom_call (registered in `custom_ops.py:1464`, called from `attention.py:1075`)
- `self._v_up_proj(attn_out)` — value up-projection

These ops are registered with `torch.compiler.allow_in_graph()` (`custom_ops.py:1631`) and have `register_fake` implementations, so dynamo CAN trace them. The `stablehlo_custom_call.stablehlo_custom_call()` in the real implementation (`custom_ops.py:1528`) would generate the StableHLO `custom_call` node with name `"tt.paged_flash_multi_latent_attention_decode"`.

## Key Files

| File | Relevance |
|------|-----------|
| `integrations/vllm_plugin/vllm_tt/attention.py:803-1086` | TTMLAAttentionBackend + TTMLAAttentionBackendImpl (forward_mha/forward_mqa) |
| `python_package/tt_torch/custom_ops.py:1376-1631` | MLA custom op definitions + stablehlo_custom_call generation |
| `integrations/vllm_plugin/vllm_tt/model_runner.py:928-1010` | `_build_mla_metadata()` for dummy/profile run |
| `src/common/module_builder.cc` | `shlo_set_proper_sdy_mes` — trivial mesh override logic |
| `src/common/client_instance.cc` | `getOrCreateMeshDevice` — mesh reshape logic |

## Summary

The MLA custom_calls are NOT absent because they fail to generate — they're absent because **the compilation never reaches the attention module**. The device mesh is corrupted by module #4 (a trivial zero-tensor with [1,1] mesh that triggers a mesh reshape from [2,4] to [1,1]), and when a previously-compiled [2,4] module is executed on the [1,1] mesh, the "Device count mismatch: 8 vs 1" error halts the entire pipeline before module #5 can be compiled.

## Precise Root Cause (Code Level)

File: `pjrt_implementation/src/api/module_builder/frontend_passes/shlo_set_proper_sdy_mesh_attribute.cc`

The `setProperSdyMeshAttributeInSpmdMode()` function had this early return:
```cpp
if (!internal::isSpmdMode(mlir_module)) {
    return tt_pjrt_status::kSuccess;  // <-- Module #4 exits here
}
```

`isSpmdMode()` checks if the first function argument has the `mhlo.sharding` attribute. Module #4 is `func.func @main() -> tensor<1x16384xbf16>` — it has **zero arguments**. So `func.getNumArguments()` returns 0, `isSpmdMode` returns false, and the trivial mesh override is completely skipped.

Module #2, by contrast, has arguments with `mhlo.sharding = "{replicated}"`, so `isSpmdMode` returns true and the mesh override applies.

## Fix Applied

Modified `setProperSdyMeshAttributeInSpmdMode()` to:
1. Remove the early return for non-SPMD modules
2. Apply the "reuse current mesh" logic for **all** modules with trivial meshes (both SPMD and non-SPMD)
3. Only apply the 1xN fallback for SPMD modules (preserving the original behavior when no current mesh is available)
4. Guard the `setMeshAttr` call with `!new_axes.empty()` for the case where no override is needed
