# `ttnn.transformer.flash_multi_latent_attention_decode` — silent hang on Galaxy-4×8 / Blackhole MLA decode layout

## TL;DR

Calling `ttnn.transformer.flash_multi_latent_attention_decode` (the unpaged MLA decode kernel) with the **per-chip layout that falls out of a `(batch=128, n_heads=128) → mesh(4,8)=(batch_axis, model_axis)` tensor-parallel split** — i.e. `Q=[1, B=32, NH=16, D=576]` and `K=[B=32, NKV=1, S=128, D=576]` — passes every validator in `sdpa_decode_device_operation.cpp` and then **hangs forever inside the device kernel**, producing zero output. No TT_FATAL, no PCC mismatch, no early exit — the process just sits at 100% CPU with the host blocked waiting for the kernel to return.

The same op runs fine in the existing demo (`models/demos/deepseek_v3/tt/mla/mla1d.py`), but only after an `all_to_all_async_generic` re-shards the activations from `(batch_per_chip=32, heads_per_chip=16)` to `(batch_per_chip=4, heads_per_chip=128)`. We did not realise the kernel **silently requires** that reshard.

This was hit while tuning the `tt-xla` codegen-py emitted graph for `tests/benchmark/test_llms.py::test_deepseek_v3_2_exp_tp_galaxy_2_layers` (tt-xla branch `mvasiljevic/deepseek-router-fuse`, commit `d972fd6f6`, see `deepseek_codegen/TUNING_LOG.md` E39 attempt).

## Environment

| | |
| --- | --- |
| Hardware | Galaxy 6U, 32× Blackhole `tt-galaxy-…` (BDF `0000:01:00.0` … `0000:c8:00.0`) |
| Mesh | `(4, 8)` named `(batch_axis, model_axis)`, `FAKE_DEVICE=TG` |
| tt-metal commit | (see `third_party/tt-mlir/install/tt-metal` in this tree) |
| Op | `ttnn.transformer.flash_multi_latent_attention_decode`, unpaged path |
| Bind name | `flash_multi_latent_attention_decode` in `transformer/sdpa_decode/sdpa_decode_nanobind.cpp` |
| Implementation | `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/` |

## What we called the kernel with

Per-chip tensors. All on `(4, 8)` mesh with `FAKE_DEVICE=TG`.

| Argument | Shape (per chip) | dtype | layout | memory |
| --- | --- | --- | --- | --- |
| `input_tensor_q` | `[1, 32, 16, 576]` | BFLOAT16 | TILE | DRAM interleaved |
| `input_tensor_k` | `[32, 1, 128, 576]` | BFLOAT16 | TILE | DRAM interleaved |
| `input_tensor_v` | `None` | — | — | — |
| `head_dim_v` | `512` | — | — | — |
| `is_causal` | `True` | — | — | — |
| `cur_pos_tensor` | `[32]` per-batch | (came from upstream `paged_update_cache.update_idxs_tensor`; likely UINT32, see note) | ROW_MAJOR | DRAM |
| `scale` | `1.0 / sqrt(576)` ≈ `0.04167` | — | — | — |
| `memory_config` (output) | DRAM interleaved | — | — | — |
| `program_config` | both `None` and an explicit `SDPAProgramConfig(grid=(8,8), q_chunk_size=0, k_chunk_size=128, exp_approx_mode=False, max_cores_per_head_batch=1)` were tried — same hang either way | | | |

Interpretation:

* `B = 32` is `batch_size / mesh.dim(batch_axis) = 128 / 4`.
* `NH = 16` is `n_heads / mesh.dim(model_axis) = 128 / 8`.
* `D = 576` is `kv_lora_rank + qk_rope_head_dim = 512 + 64`.
* `S = 128` is `kv_tokens` (small in our short-decode benchmark; `index_topk = 2048 > 128` makes the V3.2 indexer mask a no-op, which is why we believed it was safe to call the kernel with `is_causal=True` and no `attn_mask`).
* `head_dim_v = 512` is `kv_lora_rank`.

The Q and K tensors were built immediately before the call as:

```python
_sdpa0_q_concat = ttnn.concat(
    [ttnn_reshape_49,  # Q_nope_absorbed [32, 16, 512] BFLOAT16 TILE DRAM
     ttnn_reshape_51], # Q_rope          [32, 16,  64] BFLOAT16 TILE DRAM
    dim=-1,
    memory_config=ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
    ),
)
_sdpa0_q = ttnn.reshape(
    _sdpa0_q_concat,
    [1, 32, 16, 576],
    memory_config=ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
    ),
)

_sdpa0_k_concat = ttnn.concat(
    [ttnn_reshape_48,  # compressed_kv  [32, 128, 512] BFLOAT16 TILE DRAM
     ttnn_reshape_54], # k_pe_cache     [32, 128,  64] BFLOAT16 TILE DRAM
    dim=-1,
    memory_config=ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
    ),
)
_sdpa0_k = ttnn.reshape(
    _sdpa0_k_concat,
    [32, 1, 128, 576],
    memory_config=ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
    ),
)

_sdpa0_out = ttnn.transformer.flash_multi_latent_attention_decode(
    _sdpa0_q,
    _sdpa0_k,
    None,
    head_dim_v=512,
    is_causal=True,
    cur_pos_tensor=ttnn_to_layout_108,
    scale=1.0 / math.sqrt(576),
    memory_config=ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
    ),
    program_config=ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
        q_chunk_size=0,
        k_chunk_size=128,
        exp_approx_mode=False,
        max_cores_per_head_batch=1,
    ),
)
```

## What the validator said

`ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_device_operation.cpp:241..302` (the **unpaged** branch) was the active validation path for our call. Every check in this branch passed for our inputs:

* `TT_FATAL(operation_attributes.is_causal, "Multi-latent attention decode only tested for causal!");` — we set `is_causal=True`. ✓
* `TT_FATAL(input_tensor.dtype() == DataType::BFLOAT16 || ...)` — Q and K are BFLOAT16. ✓
* `TT_FATAL(input_tensors.at(i).layout() == Layout::TILE, ...)` — K is TILE. ✓ (Q is allowed to be either.)
* `TT_FATAL(Q_memcfg.buffer_type() == DRAM ...)` — Q is DRAM interleaved (unsharded path). ✓
* K buffer type DRAM. ✓
* `B = q_shape[1] = 32; TT_FATAL(k_shape[0] == B, ...)` — our K[0]=32. ✓
* `TT_FATAL(q_shape[0] == 1, "Q tensor batch size must be 1 for decode mode")` — Q[0]=1. ✓
* `head_dim_v=512 ≤ q_shape[3]=576`. ✓
* `is_gqa = (k_shape[1] > 1)` — our `NKV=1` → `is_gqa=false`, GQA-specific checks skipped. ✓

The source comment block at line 247 says:
```
// Q: [1, B, NH, D]
// K: [1, B, S, D]
// V: [1, B, S, D]
```
which **contradicts** the actual TT_FATAL at line 256 (`k_shape[0] == B`). The TT_FATAL is authoritative — our `K=[B=32, NKV=1, S=128, D=576]` is what the TT_FATAL requires, even though it doesn't match the comment. This comment-vs-code mismatch is part of why we didn't catch the layout issue from reading the source alone.

## Symptom

* `python3 deepseek_codegen/pcc.py` runs the first decode step.
* The process reaches `flash_multi_latent_attention_decode` and **never returns**.
* No stdout, no log line, no TT_FATAL, no Python traceback.
* `kill -KILL` after ~180s; `tt-smi -glx_reset_auto` restores the board.
* Reproduces with and without the explicit `SDPAProgramConfig`.

## Why the demo doesn't hit it

`models/demos/deepseek_v3/tt/mla/mla1d.py` calls `flash_multi_latent_attention_decode` (well, the **paged** variant — same family) at `_fwd_decode_flash_mla`:

```python
# 1,4,128,576 L1 height sharded 8x9 [32,576]
attn_out = ttnn.transformer.paged_flash_multi_latent_attention_decode(
    tt_q, kvpe_cache,
    page_table_tensor=page_table,
    cur_pos_tensor=position_idxs,
    **cfg["flash_mla"],
)
```

The Q shape there is `[1, 4, 128, 576]` — 4 batches per chip, 128 heads per chip — **not** our `[1, 32, 16, 576]`. The demo gets to that layout by running `ttnn.experimental.all_to_all_async_generic(tt_q, **cfg["wq_a2a_decode"], **cfg["flash_mla_reshard"])` (line `_fwd_decode_all_to_all_pre_flash_mla`, 2278–2282) **immediately before** the SDPA call, inverting the per-chip (batch_per_chip=32, heads_per_chip=16) split to (batch_per_chip=4, heads_per_chip=128).

Hypothesis: the SDPA decode kernel's reduction grid (the `max_cores_per_head_batch` / `compute_with_storage_grid_size` allocation) assumes a maximum-`B`-per-chip somewhere around `4`, and at `B=32` the cores either deadlock waiting for cross-batch reductions that never happen, or are dispatched a workload that exceeds an internal queue / semaphore depth.

The validator does **not** check this — it accepts any `B` matching `q_shape[1] == k_shape[0]` — so the hang is the first signal the user gets.

## Asks for tt-metal

1. **TT_FATAL on the unsupported batch / heads-per-chip combos**, or otherwise loud-fail when the kernel would deadlock. Silent hang during kernel-compile-then-launch is the worst failure mode for downstream debugging.
2. **Either widen support to handle `B=32, NH=16` (and the symmetric "batch hasn't been swapped with heads yet" layouts), OR document explicitly that `flash_multi_latent_attention_decode` requires an upstream `all_to_all_async_generic` to invert the TP split.** If the latter, the docstring at `sdpa_decode_nanobind.cpp` for this op should call this out, and ideally a runnable example linked.
3. **Reconcile the source comment at `sdpa_decode_device_operation.cpp:247` (`K: [1, B, S, D]`) with the actual TT_FATAL on line 256 (`k_shape[0] == B`).** They contradict each other; the comment is misleading.
4. **Clarify whether `cur_pos_tensor` must be INT32 in the unpaged-causal MLA path.** The paged-causal MLA branch at line 167 enforces INT32, but the unpaged-causal branch has no equivalent check. If INT32 is required, add the TT_FATAL.

## Standalone reproduction sketch

This doesn't need the tt-xla codegen — synthetic torch tensors with the same shapes/dtypes/layouts should reproduce on a `(4,8)` Galaxy mesh:

```python
import math
import torch
import ttnn

mesh_device = ttnn.open_mesh_device(
    ttnn.MeshShape(4, 8),
    # ... usual TG init
)

B, NKV, S, D = 32, 1, 128, 576  # per-chip
NH = 16                          # per-chip
HEAD_DIM_V = 512

# Per-chip tensors. Distribute as replicated across the mesh for the repro
# (the real failure is layout-level, not mesh-level).
q_torch = torch.randn(1, B, NH, D, dtype=torch.bfloat16)
k_torch = torch.randn(B, NKV, S, D, dtype=torch.bfloat16)
cur_pos_torch = torch.randint(0, S, (B,), dtype=torch.int32)

tt_q = ttnn.from_torch(
    q_torch, device=mesh_device,
    dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
)
tt_k = ttnn.from_torch(
    k_torch, device=mesh_device,
    dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
)
tt_cur_pos = ttnn.from_torch(
    cur_pos_torch, device=mesh_device,
    dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
)

# This call hangs.
tt_out = ttnn.transformer.flash_multi_latent_attention_decode(
    tt_q,
    tt_k,
    None,                 # V derived from K
    head_dim_v=HEAD_DIM_V,
    is_causal=True,
    cur_pos_tensor=tt_cur_pos,
    scale=1.0 / math.sqrt(D),
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    program_config=ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
        q_chunk_size=0,
        k_chunk_size=128,
        exp_approx_mode=False,
        max_cores_per_head_batch=1,
    ),
)
print(tt_out.shape)  # never reached
```

A control that should NOT hang (per the demo): drop `B` from 32 to 4 and bump `NH` from 16 to 128 (same total Q work). If that returns, it confirms the batch-per-chip ceiling.

## Pointers

* Failed attempt log entry: `deepseek_codegen/TUNING_LOG.md`, row `E39 (attempt)`.
* Branch: `mvasiljevic/deepseek-router-fuse`, commit `d972fd6f6` (the log update for the reverted attempt).
* Demo reference: `models/demos/deepseek_v3/tt/mla/mla1d.py:_fwd_decode_flash_mla` (line 2284) and `_fwd_decode_all_to_all_pre_flash_mla` (line 2277).
* Validator: `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_device_operation.cpp:241..302`.
* Op binding: `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/sdpa_decode_nanobind.cpp`.
