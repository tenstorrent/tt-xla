# Identity MeshShardOp output type defaults to TILE regardless of actual input layout

## Bug

When an Identity `MeshShardOp` follows a `Conv3dOp` (which produces ROW_MAJOR output), the runtime crashes:

```
FATAL | Layout mismatch, expected TILE, got ROW_MAJOR
DEBUG_ASSERT @ runtime/lib/ttnn/debug/debug_apis.cpp:22
```

## Visual Explanation

A model with graph breaks produces multiple compiled subgraphs. Consider a Conv3d
with SPMD-sharded weights — its subgraph looks like this:

### Unsharded (works)

Conv3d's output is ROW_MAJOR. The return rewriter sees ROW_MAJOR and inserts `ToLayoutOp`:

```
                         Compiler types           Runtime layout
                         ─────────────            ──────────────
  SiLU                   TILE                     TILE
    │
    ▼
  PermuteOp (_to_ndhwc)  TILE → ROW_MAJOR        ROW_MAJOR
    │
    ▼
  Conv3dOp               ROW_MAJOR → ROW_MAJOR   ROW_MAJOR       ← Conv3d always outputs ROW_MAJOR
    │
    ▼
  PermuteOp (_from_ndhwc) ROW_MAJOR              ROW_MAJOR
    │
    ▼
  ToLayoutOp             ROW_MAJOR → TILE         TILE            ← inserted by TTNNLayoutFuncReturnRewriter
    │                                                                because return operand was ROW_MAJOR
    ▼
  return                 TILE                     TILE  ✅
```

### Sharded (crashes)

With SPMD sharding, Shardy inserts an Identity `MeshShardOp` before the return.
Phase 1 of TTNNLayout stamps its output as TILE (default). The return rewriter
sees "already TILE" and inserts nothing. But at runtime, the tensor is ROW_MAJOR:

```
                         Compiler types           Runtime layout
                         ─────────────            ──────────────
  SiLU                   TILE                     TILE
    │
    ▼
  PermuteOp (_to_ndhwc)  TILE → ROW_MAJOR        ROW_MAJOR
    │
    ▼
  Conv3dOp               ROW_MAJOR → ROW_MAJOR   ROW_MAJOR       ← Conv3d always outputs ROW_MAJOR
    │
    ▼
  PermuteOp (_from_ndhwc) ROW_MAJOR              ROW_MAJOR
    │
    ▼
  MeshShardOp (identity)  TILE  ← Phase 1 default  ROW_MAJOR     ← runtime just forwards input!
    │                        ▲                         │
    │               nobody corrects this          actual tensor
    ▼                                                  │
  return                 TILE                     ROW_MAJOR  ❌ CRASH
                           │                         │
                           └── flatbuffer says ──────┘── mismatch!
```

The crash happens inside MeshShardOp: `insertTTNNTensorAndValidate` (types.cpp:179)
compares the flatbuffer's declared output type (TILE) against the actual forwarded
tensor (ROW_MAJOR).

### Why the return rewriter doesn't help

```
TTNNLayout Phase 1:  Stamps ALL tensors as DRAM+TILE (including MeshShardOp output)

TTNNLayout Phase 2:  Three rewriters, all skip Identity MeshShardOp:
                     ┌─ TTNNLayoutRewriter (line 228):         skips MeshShardOp
                     ├─ TTNNLayoutMeshShardRewriter (line 396): skips Identity type
                     └─ TTNNWorkaroundsPass (line 254):        skips Identity type

TTNNLayoutFuncReturnRewriter:
  return operand ← MeshShardOp output
  MeshShardOp output type = TILE  (from Phase 1, uncorrected)
  createToLayoutOp(operand, DRAM, tiled=true)
    → operand already DRAM+TILE → returns nullopt → NO ToLayoutOp inserted

In the unsharded case:
  return operand ← Conv3d _from_ndhwc PermuteOp output
  PermuteOp output type = ROW_MAJOR  (correctly set by shouldTilizeResult=false)
  createToLayoutOp(operand, DRAM, tiled=true)
    → operand is ROW_MAJOR, need TILE → inserts ToLayoutOp ✅
```

## Repro

`res_block.py --mode sharded` — single `MochiResnetBlock3D` from diffusers with
Megatron column-row sharding on 4 devices. Both unsharded and sharded produce 6
subgraphs (same graph breaks). Only the sharded case has `MeshShardOp` at the
subgraph boundary.

```bash
python res_block.py --mode unsharded 2>&1 | tee res_block_unsharded.log  # passes
python res_block.py --mode sharded 2>&1 | tee res_block_sharded.log      # crashes
```

## Suggested Fixes (ordered by generality)

### Fix 1 — Propagate input layout to Identity MeshShardOp output

**`TTNNLayoutMeshShardRewriter` in TTNNLayout.cpp**

Identity MeshShardOp is a passthrough — its output layout should match its input.
Extract the input's layout style (tiled/row-major, buffer type) and create a
matching layout for the output shape. Existing downstream passes
(`TTNNLayoutFuncReturnRewriter`) will then see the true layout and insert
`ToLayoutOp` where needed.

Most general fix — works for any ROW_MAJOR producer before Identity MeshShardOp,
not just Conv3d. Avoids unnecessary tilize when the input is already TILE.

### Fix 2 — Force Identity MeshShardOp to DRAM+TILE

**`TTNNLayoutMeshShardRewriter` in TTNNLayout.cpp**

Treat Identity MeshShardOp as a standard on-device consumer: force its input
and output to DRAM+TILE. Insert `ToLayoutOp` before the MeshShardOp if the
input is ROW_MAJOR.

Simpler but may insert unnecessary tilize/untilize when the consumer after the
subgraph boundary also expects ROW_MAJOR.

### Fix 3 — Force Conv3d to always output TILE

**`Conv3dOpConversionPattern` in TTIRToTTNN.cpp (~line 1988)**

After creating the `_from_ndhwc` permute, insert a `ToLayoutOp(TILE)` so that
Conv3d always produces TILE output regardless of downstream consumer.

Narrowest fix — only addresses Conv3d. Doesn't help if other experimental ops
also produce ROW_MAJOR before an Identity MeshShardOp.
