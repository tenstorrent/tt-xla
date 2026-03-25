# Mochi VAE Decoder — Sharding Strategy

## 1. Problem Statement

### Hardware
- **4 TT devices**, each with **32 GB RAM**
- Total: 128 GB across all devices

### Memory Budget Per Device
- Model weights (replicated): ~700 MB
- XLA/compiler overhead: ~2–4 GB
- **Available for activations: ~27–29 GB per device**

### Memory Demand (Unsharded, bfloat16)
| Stage | Single Activation | Realistic Peak (input + padded + output) |
|-------|------------------|------------------------------------------|
| block_in (768ch, 8×60×106) | 75 MB | ~150 MB |
| up_block_0 (768ch, 8×60×106) | 75 MB | ~150 MB |
| up_block_1 (512ch, 24×120×212) | 596 MB | ~1.2 GB |
| up_block_2 (256ch, 48×240×424) | 2.33 GB | **~7.0 GB** |
| **block_out (128ch, 48×480×848)** | **4.66 GB** | **~14.3 GB** |

**Conclusion**: block_out's peak of ~14.3 GB is manageable on a single 32 GB device, but with 4-way sharding each device only needs ~3.6 GB peak — providing comfortable headroom.

## 2. Candidate Strategies

### Strategy A: Megatron-Style Channel Tensor Parallelism

**Pattern**: Within each ResBlock, pair Conv3d layers as column-parallel (shard C_out) and row-parallel (shard C_in). One all-reduce per ResBlock.

**References**:
- Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism" (arXiv:1909.08053), Sections 3.1–3.2
- Existing implementation: [vae_tt_sharded.py](../../internal-midjourney/vae_tt_sharded.py) (lines 133-187)

**How it works for each ResBlock**:
```
Input x (replicated, [B, C, T, H, W])
  │
  ├── GroupNorm1 (replicated) → SiLU
  │   │
  │   ▼
  │   CausalConv3d_1: COLUMN-PARALLEL (shard C_out)    ← NO CCL
  │   │
  │   ▼
  │   GroupNorm2 (LOCAL — 8 groups/device)               ← NO CCL
  │   │
  │   ▼
  │   SiLU → CausalConv3d_2: ROW-PARALLEL (shard C_in) ← ALL-REDUCE
  │   │
  │   ▼
  │   Output (replicated after all-reduce)
  │
  └── + residual skip (replicated) ─────────────────────→ Block output (replicated)
```

| Metric | Value |
|--------|-------|
| CCL ops per ResBlock | **1 all-reduce** |
| CCL ops total (19 blocks) | **19 all-reduces** |
| GroupNorm CCL | **0** (32 groups ÷ 4 devices = 8 groups/device, all local) |
| Per-device activation memory | **1/4 of unsharded** (for intermediates between conv1 and conv2) |
| Implementation complexity | Low (reference implementation exists) |
| Channel divisibility | All pass: 768/4=192, 512/4=128, 256/4=64, 128/4=32 ✓ |

**Pros**:
- Minimal CCL: 1 all-reduce per block, 19 total
- GroupNorm is free (zero communication) — this is key for this architecture
- Proven pattern in the existing WanVAE codebase
- SPMD/Shardy handles all-reduce insertion automatically
- No halo exchange complexity
- Depth-to-space-time upsampling is compatible (linear proj can be column-parallel)
- All pointwise ops (SiLU) and channel-local ops (GroupNorm) run without communication

**Cons**:
- All-reduce volume can be large at high-resolution stages (~4.7 GB at block_out)
- ResBlock I/O is replicated (full spatial extent on every device)
- Skip connection keeps replicated input in memory during block execution

---

### Strategy B: Spatial Sharding (W-dimension)

**Pattern**: Split width dimension across 4 devices. Each device processes a strip. Halo exchange needed for 3x3 conv boundaries.

**References**:
- Dryden et al., "Channel and Filter Parallelism for Large-Scale CNN Training" (2019)
- Standard technique in HPC for distributed convolutions

| Metric | Value |
|--------|-------|
| CCL ops per ResBlock | **2 halo exchanges + 2 GN all-reduces** = ~4 ops |
| CCL ops total (19 blocks) | **~76 ops** |
| GroupNorm CCL | **38 all-reduces** (partial stats aggregation) |
| Total halo data | ~205 MB (much less than all-reduce data) |
| Per-device activation memory | **1/4 of unsharded** |
| Implementation complexity | **High** (no existing infrastructure in TT-XLA) |
| Divisibility issue | W=106 not divisible by 4 (need padding to 108) |

**Pros**:
- Less total data moved (~205 MB halo vs ~35 GB all-reduce volume)
- Even memory distribution across spatial extent

**Cons**:
- **4x more CCL operations** (76 vs 19)
- GroupNorm requires distributed statistics computation (38 extra all-reduces)
- No existing halo exchange implementation in TT-XLA SPMD
- Causal conv temporal padding + spatial halo = complex boundary handling
- W=106 not divisible by 4 (requires input padding)

---

### Strategy C: Temporal / Context Parallelism

**Pattern**: Split temporal dimension across devices. Causal convolutions enable unidirectional communication (only send to next rank).

**References**:
- Mochi original repo: [cp_conv.py](https://github.com/genmoai/mochi/blob/main/src/genmo/mochi_preview/vae/cp_conv.py)
- Used by Genmo for multi-GPU VAE inference

| Metric | Value |
|--------|-------|
| CCL ops per conv | 1 point-to-point send/recv |
| CCL ops total | **38 point-to-point** |
| GroupNorm CCL | **0** (per-frame operation, each device has full spatial extent) |
| Total halo data | **~2,522 MB** (12x more than spatial) |
| Per-device activation memory | **1/4 of unsharded** |
| Implementation complexity | Medium (reference exists but uses torch.distributed, not SPMD) |

**Pros**:
- Reference implementation exists in Mochi repo
- GroupNorm is naturally compatible (per-frame)
- Unidirectional communication (causal structure)

**Cons**:
- **T=8 at input with 4 devices → T=2 per device** — severe communication-to-computation ratio (1:1)
- Halo data volume is 12x larger than spatial sharding
- 38 CCL ops (2x more than Megatron)
- DepthToSpaceTime temporal expansion creates uneven frame distribution
- Not compatible with SPMD/Shardy (requires explicit send/recv)

---

### Strategy D: Hybrid Selective — Megatron TP Only on Memory-Critical Stages

**Pattern**: Apply Megatron TP only on up_block_2 and block_out (the 6 blocks that need it for memory), run earlier stages replicated.

| Metric | Value |
|--------|-------|
| CCL ops total | **6 all-reduces** (only 6 sharded blocks) |
| + transition cost | 1 scatter (replicated→sharded at up_block_2 entry) |
| Per-device peak memory | ~3.6 GB at block_out (vs 14.3 GB unsharded) |
| Early stage per-device memory | 596 MB at up_block_1 (fits, no sharding needed) |

**Pros**:
- Minimum possible CCL ops (6 vs 19)
- Earlier stages run at full throughput (no communication overhead)

**Cons**:
- Transition from replicated to sharded mid-network adds complexity
- Non-uniform code paths (some blocks sharded, some not)
- Harder to debug and maintain
- Marginal benefit vs full Megatron (13 fewer all-reduces, but early stages have tiny tensors where all-reduce is cheap)

---

### Strategy E: Pipeline Parallelism

**Pattern**: Assign different stages to different devices sequentially.

| Metric | Value |
|--------|-------|
| Verdict | **NOT VIABLE** |

**Reason**: Does not reduce per-device activation memory. Device handling block_out still needs 14.3 GB. With batch=1 inference, pipeline parallelism provides no throughput benefit. Only helps when weights don't fit on one device (not our case — weights are 700 MB).

---

## 3. Strategy Comparison Summary

| Criterion | A: Megatron TP | B: Spatial | C: Temporal | D: Selective TP |
|-----------|---------------|------------|-------------|-----------------|
| **Total CCL ops** | **19** | 76+ | 38 | **6** |
| **CCL data volume** | ~35 GB | ~0.2 GB | ~2.5 GB | ~33 GB |
| **GroupNorm CCL** | **0** | 38 | **0** | **0** |
| **Implementation effort** | **Low** | High | Medium | Medium |
| **Existing reference** | **Yes** | No | Partial | Yes (adapt) |
| **SPMD/Shardy compatible** | **Yes** | No | No | **Yes** |
| **Memory reduction** | 4x | 4x | 4x | 4x (last stages) |
| **Robustness** | **High** | Medium | Low (T=2) | Medium |

## 4. Recommended Strategy: Megatron-Style Channel TP (Strategy A)

### Why Strategy A Wins

1. **Fewest CCL ops with uniform implementation**: 19 all-reduces is only 13 more than the selective approach (Strategy D), but the implementation is uniform — every ResBlock is treated identically. The early-stage all-reduces on small tensors (75 MB) are essentially free.

2. **GroupNorm is zero-cost**: The 32 groups ÷ 4 devices = 8 groups/device arithmetic is the deciding factor. Spatial sharding (Strategy B) would need 38 additional all-reduces just for GroupNorm statistics. This alone eliminates spatial sharding from contention.

3. **Proven pattern**: The [vae_tt_sharded.py](../../internal-midjourney/vae_tt_sharded.py) implements exactly this pattern for the WanVAE decoder on the same TT hardware with SPMD/Shardy. The adaptation to Mochi requires changing attribute access patterns (named attrs vs positional indexing) and handling GroupNorm instead of RMSNorm, but the core sharding logic is identical.

4. **SPMD/Shardy native**: All communication is handled implicitly by the XLA SPMD partitioner. No explicit collective calls needed. Mark tensor sharding specs on weights → compiler inserts all-reduces automatically.

5. **T=2 eliminates temporal**: Context parallelism (Strategy C) is fundamentally problematic with T=8 input across 4 devices. The 1:1 communication-to-computation ratio at early stages makes it worse than channel parallelism.

### Implementation Plan

#### Layers to Shard (Megatron column-row pattern)

**All 19 MochiResnetBlock3D instances**:

| Stage | Block Path | Channels | Sharded Tensors |
|-------|-----------|----------|----------------|
| block_in | `decoder.block_in.resnets[0..2]` | 768 | conv1.weight, conv1.conv.bias, norm2.weight, norm2.bias, conv2.weight |
| up_block_0 | `decoder.up_blocks[0].resnets[0..5]` | 768 | (same pattern) |
| up_block_1 | `decoder.up_blocks[1].resnets[0..3]` | 512 | (same pattern) |
| up_block_2 | `decoder.up_blocks[2].resnets[0..2]` | 256 | (same pattern) |
| block_out | `decoder.block_out.resnets[0..2]` | 128 | (same pattern) |

#### Partition Specs per ResBlock

For `MochiResnetBlock3D` with attrs `norm1, conv1, norm2, conv2`:

```python
# Conv weight shapes: [C_out, C_in, kT, kH, kW] (inside CogVideoXCausalConv3d.conv)
# GroupNorm weight/bias shapes: [C]

# Column-parallel conv1 (shard C_out):
mark_sharding(block.conv1.conv.weight, mesh, (axis, None, None, None, None))
mark_sharding(block.conv1.conv.bias,   mesh, (axis,))

# Intermediate GroupNorm (shard to match partitioned channels):
mark_sharding(block.norm2.norm.weight,  mesh, (axis,))
mark_sharding(block.norm2.norm.bias,    mesh, (axis,))

# Row-parallel conv2 (shard C_in):
mark_sharding(block.conv2.conv.weight,  mesh, (None, axis, None, None, None))
# conv2 bias: NOT sharded (applied after implicit all-reduce)
```

**6 sharding annotations per ResBlock × 19 blocks = 114 total annotations**

#### Layers Left Unsharded (replicated)

| Layer | Reason |
|-------|--------|
| `conv_in`: Conv3d(12→768, 1×1×1) | C_in=12, too small; boundary layer |
| `proj_out`: Linear(128→3) | C_out=3, not divisible by 4; boundary layer |
| Unpatchify `nn.Linear` in each up_block | Operates on replicated data between all-reduced block output and next stage input |
| `norm1` in each ResBlock (first GroupNorm) | Operates on replicated input (before column-parallel conv1) |

#### Mesh Configuration

```python
# For 4 TT devices (QB-like configuration):
mesh_shape = (1, 4)      # (batch=1, model=4)
mesh = Mesh(
    device_ids=np.arange(4),
    mesh_shape=mesh_shape,
    axis_names=("batch", "model"),
)
shard_axis = "model"  # 4-way tensor parallelism
```

#### Expected Communication Profile

| Stage | ResBlocks | All-Reduce Tensor Shape | Per-AR Volume (bf16) | Stage Total |
|-------|-----------|------------------------|---------------------|-------------|
| block_in | 3 | [1, 768, 8, 60, 106] | 75 MB | 225 MB |
| up_block_0 | 6 | [1, 768, 8, 60, 106] | 75 MB | 450 MB |
| up_block_1 | 4 | [1, 512, 24, 120, 212] | 596 MB | 2.4 GB |
| up_block_2 | 3 | [1, 256, 48, 240, 424] | 2.33 GB | 7.0 GB |
| block_out | 3 | [1, 128, 48, 480, 848] | 4.66 GB | 14.0 GB |
| **Total** | **19** | | | **~24 GB** |

Note: Actual wire data for ring all-reduce is `2 × (P-1)/P × tensor_size = 1.5 × tensor_size` for P=4. The block_in and up_block_0 all-reduces are cheap (~112 MB each). The expensive ones are in the last two stages.

#### Key Adaptation from WanVAE Reference

| WanVAE Pattern | Mochi Adaptation |
|----------------|-----------------|
| `block.residual[2].weight` (positional) | `block.conv1.conv.weight` (named attrs) |
| `block.residual[3].gamma` (RMSNorm 4D) | `block.norm2.norm.weight` (GroupNorm 1D) |
| `block.residual[6].weight` (positional) | `block.conv2.conv.weight` (named attrs) |
| `(axis, None, None, None)` for 4D gamma | `(axis,)` for 1D GroupNorm weight/bias |
| Skip AttentionBlock instances | No attention — all blocks are ResBlocks |
| Skip Resample layers | Skip unpatchify Linear layers (similar role) |

### Divisibility Verification

| Channels | ÷ 4 | Groups/device | Channels/group/device | Valid |
|----------|------|--------------|----------------------|-------|
| 768 | 192 | 8 | 24 | ✓ |
| 512 | 128 | 8 | 16 | ✓ |
| 256 | 64 | 8 | 8 | ✓ |
| 128 | 32 | 8 | 4 | ✓ |

### Known Considerations

1. **CausalConv3d padding**: `CogVideoXCausalConv3d` wraps `nn.Conv3d` inside `.conv` attribute and applies padding externally via `F.pad`. The actual weight to shard is `block.conv1.conv.weight` (the inner `nn.Conv3d`), not `block.conv1.weight` (which doesn't exist — `CogVideoXCausalConv3d` is not a subclass of `nn.Conv3d`).

2. **MochiChunkedGroupNorm3D**: Wraps `nn.GroupNorm` inside `.norm` attribute. Sharding target is `block.norm2.norm.weight` and `block.norm2.norm.bias`.

3. **Program cache**: There is a known tt-metal runtime bug where sharded graphs may produce incorrect results with program cache enabled. Workaround: set `TT_RUNTIME_ENABLE_PROGRAM_CACHE=0` (ref: [vae_sharding_plan.md](../../internal-midjourney/vae_sharding_plan.md)).

## 5. Communication Minimization Opportunities

### Fusing All-Reduces

If the SPMD compiler is smart enough, consecutive ResBlocks without interleaved unsharded ops could potentially:
- Keep activations sharded between blocks (avoid materializing replicated tensors)
- Only materialize replicated output at stage boundaries (before unpatchify)

This would change the pattern from "all-reduce after every ResBlock" to "all-reduce only at stage boundaries" (5 all-reduces instead of 19). However, this requires the first GroupNorm (norm1) of each ResBlock to accept sharded input, which means sharding norm1's weight/bias as well.

**Extended sharding (all norms sharded)**:
```python
# Also shard norm1 (first GroupNorm in each ResBlock):
mark_sharding(block.norm1.norm.weight, mesh, (axis,))
mark_sharding(block.norm1.norm.bias,   mesh, (axis,))
```

If both norms are sharded, the activation can remain channel-partitioned through the entire residual path: `sharded_input → sharded_norm1 → column_conv1 → sharded_norm2 → row_conv2 → partial_sums`. The all-reduce is only needed when the residual skip connection requires the replicated original input.

**Whether this optimization is possible depends on the XLA SPMD partitioner's ability to propagate sharding through the residual connection.** If the skip connection forces replicated tensors, we get 19 all-reduces regardless. This is a compiler-level optimization to investigate but not a requirement for correctness.

## 6. Summary

| Decision | Choice |
|----------|--------|
| **Strategy** | Megatron-style channel tensor parallelism |
| **Scope** | All 19 ResBlocks |
| **CCL ops** | 19 all-reduces (implicit via SPMD) |
| **GroupNorm cost** | 0 (local computation, 8 groups/device) |
| **Unsharded layers** | conv_in, proj_out, unpatchify linears |
| **Device mesh** | (1, 4), axis "model" for TP |
| **Dtype** | bfloat16 |
| **Reference impl** | vae_tt_sharded.py (WanVAE) |
