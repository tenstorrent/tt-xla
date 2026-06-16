# transformer_sharded — failure investigation (with encoder fix applied)

Machine: 8-chip LLMBox (wh-lb-46). Submodule @ `ac2212ffdf` (encoder-OOM fix). Mesh `(1, 8)` (8-way), confirmed active.

## What happens now (vs before the fix)
The fix's `(1,8)` mesh DOES shard the transformer 8-way — it now runs **deep into execution**
(672 all_gather/reduce_scatter CCL ops executed, ~889 s) instead of failing early. It still
ultimately **OOMs**, but later and on an activation, not on weight load.

## Exact failure
```
TT_FATAL Out of Memory: cannot allocate 56623104 B (54 MB) DRAM buffer across 12 banks
(each bank needs 4718592 B; allocated 1039355456 B/bank, free 31417728 B, largest free block 3538944 B)
bank_manager.cpp:462
```
Call site: `SliceOp` → `ttnn::slice` → `to_layout` → `tilize_with_val_padding` →
`create_output_tensors` (surfaced to Python as `Bad StatusOr Error code 13`).

Two things are true at the failure point:
- **DRAM is 97.6% full**: 12.47 GB of 12.85 GB (12 banks × 1.07 GB) already allocated.
- **It's also fragmented**: largest free block is 3.5 MB/bank but the op needs 4.7 MB/bank, so even
  the 31 MB total free can't satisfy it.

## Root cause
The 32B transformer in bf16 ≈ 64 GB of weights. 8-way sharding → ~8 GB/device, leaving only
~4 GB on a 12.85 GB device for activations + CCL all-gather reconstruction buffers + allocator
fragmentation headroom. That margin is too thin: a mid-graph slice/tilize tips it over.
This is a genuine capacity problem, not a sharding bug — sharding is working (it wouldn't reach
672 CCL ops otherwise; the broken-navigation case OOMs immediately at load, like the pre-fix encoder).

## Fix options (by effort / likelihood)
1. **Run on 32-chip Galaxy** — `MESH_SHAPES` already maps `32: (1, 32)` → ~2 GB weights/device,
   comfortable headroom. This is almost certainly the intended target (Galaxy bringup logs already
   exist on the branch). Fastest path to green if a Galaxy is available. *Recommended first.*
2. **bfp8_b weight quantization** — halve on-device weight footprint (~4 GB/device on 8 chips),
   leaving ~8 GB for activations. This is a tt-mlir / compiler memory-config change (bfp8 is an
   on-device format, not a torch dtype), and needs PCC re-validation. Keeps it on 8 chips.
3. **Reduce activation / CCL pressure on 8 chips** — earlier deallocation of intermediates,
   sequence-parallel to avoid full-activation all-gathers, or defrag. More involved compiler work.

## Cheap audit worth doing
Confirm no large weight is left replicated in `shard_transformer_specs` (any param NOT in the spec
dict, or marked `(None, None)`, is replicated full-size on every device). `proj_out.weight` is
`(None, None)` — verify it (and the embedders) are small; an accidentally-replicated big weight
would waste device DRAM and is a cheap win if present.
