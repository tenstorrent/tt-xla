# Investigation: Is the stablehlo.all_gather in test_gather_indices Necessary?

## Context

Running `test_gather_indices` with `batch_size=4`, `seq_len=32` on a 2×4 mesh inserts a
`stablehlo.all_gather` on the data tensor _inside_ the composite body, before the
`stablehlo.gather`. This prevents `ReoutlineCompositePass` from re-matching the
`tenstorrent.gather` composite.

---

## Root Cause: Global Batch Iota in the Decomposition

`composite_gather` in `python_package/tt_torch/composite_ops.py:252` delegates to
`torch.gather`, which PyTorch/XLA lowers to a 3-D point-gather in `@tenstorrent.gather.impl`:

```mlir
%0 = stablehlo.iota dim = 0 : tensor<4xui32>           // global [0,1,2,3]
%1 = broadcast_in_dim %0, dims = [0]                    // [4,16,512,1]
%2 = convert + reshape %index                           // [4,16,512,1]
%3 = stablehlo.iota dim = 0 : tensor<512xui32>          // [0..511]
%4 = broadcast_in_dim %3, dims = [2]                    // [4,16,512,1]
%5 = concatenate %1, %2, %4, dim = 3                    // [4,16,512,3]
%6 = stablehlo.gather(%data, %5)
       start_index_map = [0, 1, 2]                      // indexes ALL 3 dims
       collapsed_slice_dims = [0, 1, 2]
       slice_sizes = [1, 1, 1]
```

The first iota produces **global** indices `[0, 1, 2, 3]` for batch dim 0.

### Pass sequence that introduces the all_gather (log line references for verbose_gather_32_4.log)

| Pass | Log line | Effect |
|---|---|---|
| `InsertExplicitReshardsPass` | 928 | Sees global batch iota [0..3] + data sharded `[{"_axis_0"},{},{}]`. Inserts `sdy.reshard %data [{},{},{}]` |
| `WrapUnderManualComputationPass` | 955 | Wraps everything in `sdy.manual_computation` with `manual_axes={"_axis_0","_axis_1"}`. Inside, `%arg3` becomes LOCAL `tensor<2x32x512xbf16>` |
| `ReshardToCollectivesPass` | 1001 | Converts `sdy.reshard` → `sdy.all_gather [{"_axis_0"},{},{}]` |
| `ReoutlineCompositePass` | 1139 | Sees `all_gather` between index-construction ops and `stablehlo.gather` → pattern does not match original composite → **composite not restored** |

The all_gather gathers `tensor<2x32x512xbf16>` → `tensor<4x32x512xbf16>` so the global
indices `[0,1,2,3]` become valid.

---

## Is the all_gather necessary?

**No, semantically it is not.** `torch.gather(x, 1, index)` is fully batch-parallel:

```
output[b, t, f] = x[b, index[b, t, f], f]
```

Device 0 (holding `x[0:2]`) only ever needs `x[0]` and `x[1]`.
Device 1 (holding `x[2:4]`) only ever needs `x[2]` and `x[3]`.
No cross-device data exchange is required.

The all_gather is introduced **solely** because the decomposition builds a global batch
iota of size 4. This forces Shardy to replicate the full data tensor before the gather
can proceed.

---

## Proposed Fix: Use `operand_batching_dims` in the Gather Decomposition

StableHLO gather supports `operand_batching_dims` and `start_indices_batching_dims`,
which declare that a dimension of the operand and start_indices are paired batch
dimensions (not indexed). This eliminates the global batch iota entirely.

### New decomposition (no batch iota)

```mlir
// index: [B, T, F, 1]  — seq index only
// feat:  [B, T, F, 1]  — feature iota
// concatenated: [B, T, F, 2]

stablehlo.gather(%data, %index_2d)
  operand_batching_dims = [0]          // batch dim of data is paired
  start_indices_batching_dims = [0]    // batch dim of index is paired
  start_index_map = [1, 2]             // index into seq (dim 1) and feat (dim 2)
  collapsed_slice_dims = [1, 2]        // collapse both gathered dims
  slice_sizes = [1, 1, 1]
  index_vector_dim = 3
```

With this formulation:
- Batch dim 0 of data and batch dim 0 of index are declared as paired → Shardy propagates
  the `[{"_axis_0"},{},{}]` sharding through the gather without inserting an all_gather
- Each device computes its local batch gather using only its local data
- Output sharding follows as `[{"_axis_0"},{},{}]` (sharded, not replicated)

### Efficiency comparison

| Approach | Communication | Volume |
|---|---|---|
| Current (all_gather input) | all_gather `2×32×512` bf16 | 64 KB/device |
| all_gather output instead | all_gather `2×16×512` bf16 | 32 KB/device |
| `operand_batching_dims` fix | none | 0 |

---

## Files to Modify

### 1. `python_package/tt_torch/composite_ops.py:230–254`
`composite_gather()` currently calls `torch.gather(input, dim, index)`, which lowers via
PyTorch/XLA to the 3-D global point-gather. The decomposition body must be replaced with
an explicit StableHLO gather using `operand_batching_dims`. This requires bypassing the
`torch.gather` lowering path and emitting the gather via custom StableHLO ops.

### 2. `third_party/tt-mlir` — `ReoutlineCompositePass`
The pass must recognize the new gather form (2-element index vector +
`operand_batching_dims`) to re-outline the `tenstorrent.gather` composite.

---

## Pre-Implementation Checks

1. **StableHLO version support:** Confirm `operand_batching_dims` is available in the
   StableHLO dialect pinned by tt-mlir:
   ```bash
   grep -r "operand_batching_dims\|start_indices_batching_dims" third_party/
   ```

2. **Fallback if unavailable:** Use a local batch iota sized to the LOCAL shard dimension.
   This requires the decomposition to execute inside a `sdy.manual_computation` scope
   where the batch axis is already manual (local shapes are visible). More invasive.

---

## Verification

After implementing the fix:
```bash
TTXLA_LOGGER_LEVEL=VERBOSE pytest -svv tests/torch/ops/test_gather.py::test_gather_indices[32-4] 2>&1 | tee verbose_gather_32_4_fixed.log

# Confirm:
# 1. No stablehlo.all_gather in the shlo_compiler module
# 2. tenstorrent.gather composite IS re-outlined by ReoutlineCompositePass
# 3. Test passes with correct numerics
```
