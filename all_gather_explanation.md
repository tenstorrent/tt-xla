# Why `stablehlo.all_gather` Breaks `tenstorrent.gather` Composite (batch_size=4)

## Context

In `test_deepseek_attention_decode` with `seq_len=32, batch_size=4`, the `tenstorrent.gather`
composite op fails to be reconstructed by `ReoutlineCompositePass` in tt-mlir. Without the
composite, the gather doesn't lower to `ttir.gather`. The symptom is a `stablehlo.all_gather`
appearing at `32_4_failing/shlo.mlir:277` — inside the reoutline group's span but **without**
the `reoutline.group` attribute — which causes `ReoutlineCompositePass::analyzeBoundary` to
return false.

---

## Root Cause: The 5-step failure chain

### Step 1 — Batch sharding is active for batch_size=4

In `test_deepseek_v3_2_exp.py:255`:
```python
batch_axis = "batch" if batch_size >= mesh_batch_axis_size else None  # 4 >= 2 → "batch"
```
This causes:
```python
shard_specs[attention.kv_cache] = ("batch", None, None)  # line 271
shard_specs[attention.pe_cache] = ("batch", None, None)  # line 272
```
The mesh is `<["_axis_0"=2, "_axis_1"=4]>`. Inside `sdy.manual_computation` the local kv_cache
shard shape is `<2x33x512xbf16>` (half of the global batch-4 tensor).

### Step 2 — The composite's stablehlo.gather uses a 3D index including dim 0 (batch)

`composite_gather` in `python_package/tt_torch/composite_ops.py:230` wraps
`torch.gather(input, dim=1, index)`. When lowered to stablehlo, `torch.aten.gather` decomposes
into a `stablehlo.gather` with `start_index_map = [0, 1, 2]`:

```mlir
%iota    = stablehlo.iota dim=0 ... : tensor<2xui32>          ; local batch indices [0, 1]
%batch_i = broadcast_in_dim %iota, dims=[0] : tensor<2x33x512x1xui32>
%seq_i   = <from original gather index>
%feat_i  = stablehlo.iota dim=0 ... : tensor<512xui32>
%indices = concatenate %batch_i, %seq_i, %feat_i  : tensor<2x33x512x3xui32>
stablehlo.gather(%INPUT, %indices)
  <{ start_index_map = [0, 1, 2],   ← ALL dims, including dim 0 (batch)
     collapsed_slice_dims = [0, 1, 2],
     slice_sizes = [1, 1, 1] }>
```

The index explicitly names dim 0 (batch) as a coordinate. The actual batch values are always
`{0, 1}` (from the iota), but the stablehlo.gather type signature does not encode this.

### Step 3 — Shardy sees a gather over a sharded dimension and inserts all_gather

When Shardy's sharding propagation processes this `stablehlo.gather` inside
`sdy.manual_computation`:

- `%INPUT` (the kv_cache slice) is sharded on `_axis_0` → local shape `<2x33x512>`
- `start_index_map = [0, 1, 2]` signals that the gather may access **any** batch coordinate
- Shardy does **not** analyze iota constants to know the indices will always be `{0, 1}`
- Conservative choice: insert `stablehlo.all_gather(%INPUT)` along dim 0 with
  `replica_groups = [[0,4],[1,5],[2,6],[3,7]]`, expanding `<2x33x512>` → `<4x33x512>`

This all_gather is semantically unnecessary. In fact it is also **incorrect**: on devices with
`_axis_0=1` (global batches 2–3), the local iota still generates `[0, 1]`, so after the
all_gather those devices read from global batches 0 and 1 instead of their own 2 and 3.

### Step 4 — The all_gather is inserted without reoutline.group metadata

The `reoutline.group` attribute was attached to decomposition ops by `FlattenCompositePass`
**before** Shardy ran. The newly inserted `stablehlo.all_gather` receives no such attribute.

### Step 5 — ReoutlineCompositePass fails: the group is not contiguous

`ReoutlineComposite.cpp` (`analyzeBoundary`, ~line 68):
```cpp
for (mlir::Operation &it : llvm::make_range(firstOp, lastOp)) {
  // All ops between first and last must be in the group.
  // If new ops were inserted in between, we cannot reoutline.
  if (!inGroup.contains(&it)) { return false; }
}
```

The group spans from the `iota` (line 270) to the `stablehlo.gather` (line 278). The
`all_gather` at line 277 sits between them but is not in the group →
`analyzeBoundary` returns false → composite reconstruction fails → no `stablehlo.composite
"tenstorrent.gather"` → no `ttir.gather`.

---

## What You Can Do

### Option A — Immediate workaround: don't batch-shard kv_cache / pe_cache

**File:** `tests/torch/models/deepseek_v3_2_exp/test_deepseek_v3_2_exp.py` lines 271–272

```python
# Change from:
shard_specs[attention.kv_cache] = (batch_axis, None, None)
shard_specs[attention.pe_cache] = (batch_axis, None, None)

# To:
shard_specs[attention.kv_cache] = (None, None, None)
shard_specs[attention.pe_cache] = (None, None, None)
```

Effect: caches are fully replicated → no sharding on dim 0 → Shardy has no reason to insert
`all_gather` inside the composite → composite preserved → `ttir.gather` generated.

**Trade-off:** Each device holds the full KV cache. For batch_size=4 on 8 devices the caches
are small (4×64×512 bf16 ≈ 256 KB), so memory is not a concern. All other sharding is unchanged.

### Option B — Force replication just before the gather (surgically scoped)

**File:** `tests/torch/models/deepseek_v3_2_exp/modified_model.py` around line 929

After slicing `orig_kv_cache` / `orig_pe_cache` and before the `torch.gather` calls, add
`xs.mark_sharding(..., (None, None, None))` to force replication. This moves the all_gather
**outside** the composite scope so the composite sees fully replicated inputs.

Requires passing the mesh into `modified_decode_flow` (store as `self.mesh` or pass explicitly).

### Option C — Correct long-term fix: use operand_batching_dims in the composite impl

**Why it works:** Replacing `start_index_map = [0, 1, 2]` with
`operand_batching_dims = [0, 2]` + `start_index_map = [1]` in the composite's
`stablehlo.gather` tells Shardy that dims 0 and 2 are **batch dimensions** requiring no
cross-device communication → no all_gather → composite intact.

**Two-part change required in tt-mlir:**

1. Change the lowering of `torch.aten.gather` to emit a batching-dim-aware `stablehlo.gather`
   (this lives upstream in torch-mlir and would need to be patched or overridden in tt-mlir).

2. Update `StableHLOGatherToEmbeddingPattern` in
   `third_party/tt-mlir/src/tt-mlir/lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`
   to **accept** `operand_batching_dims` when lowering to `ttir.gather`. Currently the pattern
   has an explicit `notifyMatchFailure` rejection for non-empty batching dims:
   ```cpp
   if (!dimensionNumbers.getOperandBatchingDims().empty() || ...)
     return rewriter.notifyMatchFailure(srcOp, "Did not satisfy batching = none");
   ```

---

## Recommended Approach

Apply **Option A** immediately (two-line change in the test) to unblock `ttir.gather` generation
and validate correctness. Track **Option C** as the proper fix for production since Option A
costs memory and the correctness bug (wrong batch items accessed on `_axis_0=1` devices) goes
away once the composite is preserved and Shardy never sees the raw sharded input.

---

## Verification

After applying Option A:

1. Dump the stablehlo: confirm no `stablehlo.all_gather` appears inside the
   `reoutline.group = "composite_tenstorrent.gather.impl_0"` span.
2. Confirm `stablehlo.composite "tenstorrent.gather"` appears in the output
   (ReoutlineCompositePass succeeded).
3. Confirm `ttir.gather` appears in the ttir output.
4. Run the test:
   ```
   pytest -svv tests/torch/models/deepseek_v3_2_exp/test_deepseek_v3_2_exp.py::test_deepseek_attention_decode[32-4]
   ```
