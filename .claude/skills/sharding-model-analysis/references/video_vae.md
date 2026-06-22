## Conv3D Sharding Strategy

### Available Sharding Dimensions

For a Conv3D with input `[B, C_in, D, H, W]`, weight `[C_out, C_in, kD, kH, kW]`, output `[B, C_out, D', H', W']`:

| Strategy | What gets split | Communication needed |
|----------|----------------|---------------------|
| **C_out** (weight dim 0) | Weight split on output filters, bias split | **None** - each output channel is independent |
| **C_in** (weight dim 1) | Input + weight split on input channels | **all_reduce** to sum partial convolutions |
| **D/H/W** (spatial) | Input split along spatial dims | **halo exchange** - neighboring tiles need overlapping kernel-sized borders |
| **B** (batch) | Input split along batch | **None** - pure data parallelism |

### Why depth sharding is not beneficial

The WAN VAE decoder uses **streaming decode** - it processes one temporal frame at a time, feeding `[1, C, 1, H, W]` through the decoder in a loop of 32 iterations, with a `feat_cache` holding the last 1-2 frames of intermediate activations per `CausalConv3d` layer for temporal context. This means **D is always 1 during computation** - there's no depth dimension to shard.

### Spatial sharding

Convolution with a kernel of size k requires each spatial tile to have access to `(k-1)/2` extra elements from its neighbors on each boundary - this is called the **halo**. Before each `conv` layer, devices must exchange these halo strips with their neighbors via collective communication ops (`neighbour_pad`, `slice_reshard`).

Currently we don't support it and we have open [issue](https://github.com/tenstorrent/tt-mlir/issues/8216) in `tt-mlir` (check out issue for more details about spatial sharding and ccls).

The tt-metal takes *this* approach: it shards **activations** along `H` and `W`, and keeps **all weights replicated**. This makes sense for a VAE specifically - the decoder weights are small but activations are large, so the bigger win is splitting the *data* rather than the *parameters*. For each 3×3×3 causal `Conv3d`, neighboring devices exchange 1-pixel halo strips on H and W via a custom `ttnn.experimental.neighbor_pad_async` collective; the only all-gather appears around the single mid-block attention (to materialize a full spatial extent for SDPA) and there are **no all-reduces** in the entire decoder. The halo primitive they rely on is exactly what [issue](https://github.com/tenstorrent/tt-mlir/issues/8216) is tracking - until it's exposed through TTIR/TTNN MLIR, we can't take this route from the compiler side, which is why we fall back to channel sharding below.


### Channel sharding (in Megatron style)

That leaves us with only thing we can shard currently -> channels (`C_in` and `C_out`), so let's see what we can do with it.

The VAE's `ResidualBlock` has this structure:
```python
residual = Sequential(
    RMS_norm(in_dim),
    SiLU(),
    CausalConv3d(in_dim, out_dim, 3),   # Conv A
    RMS_norm(out_dim),
    SiLU(),
    CausalConv3d(out_dim, out_dim, 3),  # Conv B
)
shortcut = CausalConv3d(in_dim, out_dim, 1)
return residual + shortcut
```

The ResidualBlock naturally pairs two consecutive `conv3D` layers - this is a good example of **[Megatron-style tensor parallelism](https://arxiv.org/abs/1909.08053)** for Transformer layers). The idea is to alternate between two complementary sharding strategies across consecutive layers:
- The **first layer** shards its output (column-parallel) - splitting work across devices with no communication
- The **second layer** shards its input (row-parallel) - consuming the already-partitioned output directly, then doing a single all_reduce at the end

This way, only one collective operation (all_reduce) is needed per pair of layers, and the intermediate tensor between the two layers is never gathered - it stays partitioned (given that we know how to deal with `RMS_norm` and `SiLU` in the middle, which we do :).

**Conv A → C_out sharding (column-parallel):**
- Input is replicated across all devices: full `[B, in_dim, 1, H, W]`
- Weight `[out_dim, in_dim, 3, 3, 3]` is sharded on dim 0 (output channels) → each device gets `[out_dim/N, in_dim, 3, 3, 3]`
- Each device independently convolves the full input with its weight slice, producing its portion of output channels
- Output is `[B, out_dim/N, 1, H, W]` - **no communication needed**

**Conv B → C_in sharding (row-parallel):**
- Input comes from Conv A's output, already partitioned: `[B, out_dim/N, 1, H, W]` on each device
- This naturally matches C_in sharding - **no all_gather needed** between Conv A and Conv B
- Weight `[out_dim_b, out_dim, 3, 3, 3]` is sharded on dim 1 (input channels) → each device gets `[out_dim_b, out_dim/N, 3, 3, 3]`
- Each device convolves its `out_dim/N` input channels with its corresponding weight slice. The output has **all** `out_dim` output channels (since weight dim 0 is not split), but each device only computed a **partial sum** over its slice of input channels
- **all_reduce** sums the N partial outputs → correct full result `[B, out_dim_b, 1, H', W']`

The shortcut path (`CausalConv3d(in_dim, out_dim, 1)` or `Identity`) operates on the original replicated input, producing a full `[B, out_dim, 1, H, W]` tensor. The residual add `x + h` works directly since both sides are full after the all_reduce.

**Per ResidualBlock cost: only 1 all_reduce** (at Conv B's output). No all_gather between Conv A→B.

---

## `WanResidualBlock` op sequence under Megatron `(col, row)` C-sharding

> Numbers below are calculated for **480p** at the first **`mid_block.resnets[0]`** (the very first `WanResidualBlock` the decoder hits). At this position the activation has `C = 384` channels and spatial shape `(T, H, W) = (1, 60, 104)` which gives us `[1, 384, 1, 60, 104]` tensor on input. With a 4-device mesh, each device holds `C / N = 96` channels of the C-sharded activation.


> **NOTE**
This only shows us that in Megatron-style sharding we also have to pay attention on what lays between 2 convs that we are sharding, if there wasn't `rms_norm`, we could've gotten away with just one `all_reduce` op per block (one after `conv2`).
```
                              x_in  (replicated)
                                    │
                ┌───────────────────┼───────────────────┐
                │                                       │
                ▼                                       ▼
              norm1                              conv_shortcut
                │                                 (Identity when
                ▼                                  in_dim == out_dim)
              silu                                      │
                │                                       │
                ▼                                       │
              conv1  ── column-parallel                 │
                │       (weight sharded on C_out)       │
                │                                       │
                ▼  ◀── activation is now C-sharded      │
        ┏━━━━━━━━━━━━━━━━━━┓                            │
        ┃     norm2        ┃ ◀── ★ all_reduce  (small)  │
        ┃                  ┃     on (B, 1, T, H, W)     │
        ┗━━━━━━━━━━━━━━━━━━┛                            │
                │                                       │
                ▼                                       │
              silu                                      │
                │                                       │
                ▼                                       │
        ┏━━━━━━━━━━━━━━━━━━┓                            │
        ┃     conv2        ┃ ◀── ★ all_reduce  (big)    │
        ┃ row-parallel     ┃     on (B, C, T, H, W)     │
        ┃ (weight on C_in) ┃                            │
        ┗━━━━━━━━━━━━━━━━━━┛                            │
                │                                       │
                ▼  ◀── back to replicated               │
                │                                       ▼
                └──────────────────► + ◄────────────────┘
                                     │
                                     ▼
                                  x_out  (replicated)
```

### Why the extra small AR inside `norm2`

`norm1` and `norm2` are both **RMS-norms over the channel axis** - they compute `Σ_c x_c²` along C. After `conv1` (column-parallel), the activation is C-sharded, so each device only has `C/N` of the channels and its local sum is just a partial. We need an `all_reduce` to combine the partials into the full sum. (`norm1`'s input is replicated, so its sum is already complete - no AR needed there.)

This AR is small because the channel sum collapses C to 1 *before* the cross-device step. So we're not all-reducing the full activation `(B, C, T, H, W)` - we're all-reducing the per-spatial-position scalar `(B, 1, T, H, W)`, which is `C` times smaller. The activation itself stays C-sharded, gets divided locally by the now-globally-consistent norm, and flows straight into `conv2` as the row-parallel input it expects.


---

For context, here's how the decoder is currently sharded - Megatron col→row pair on every `WanResidualBlock`, everything else replicated:

```
AutoencoderKLWan
├── post_quant_conv      WanCausalConv3d(16 → 16, k=(1,1,1))                            [replicated]
└── decoder              WanDecoder3d
    ├── conv_in          WanCausalConv3d(16 → 384, k=(3,3,3))                           [replicated]
    ├── mid_block        WanMidBlock(dim=384, num_layers=1)
    │   ├── resnets[0]   WanResidualBlock(384 → 384)                                    [SHARDED - Megatron col→row]
    │   ├── attentions[0] WanAttentionBlock(384)                                        [replicated]
    │   └── resnets[1]   WanResidualBlock(384 → 384)                                    [SHARDED - Megatron col→row]
    ├── up_blocks[0]     WanUpBlock(384 → 384, 3 resnets, upsample3d)                   [resnets SHARDED, upsampler replicated]
    ├── up_blocks[1]     WanUpBlock(192 → 384, 3 resnets, upsample3d)                   [resnets SHARDED, upsampler replicated]
    ├── up_blocks[2]     WanUpBlock(192 → 192, 3 resnets, upsample2d)                   [resnets SHARDED, upsampler replicated]
    ├── up_blocks[3]     WanUpBlock(96  → 96,  3 resnets, no upsampler)                 [resnets SHARDED]
    ├── norm_out         WanRMS_norm(96, channel_first=True)                            [replicated]
    └── conv_out         WanCausalConv3d(96 → 3, k=(3,3,3))                             [replicated]
```

## Further sharding consideration: WanResample (upsamplers / downsamplers)

There is additional compute parallelism we leave on the table by keeping `WanResample` replicated. The natural way to shard it is **col-parallel on `resample[1].weight` (split `C_out`)** rather than row-parallel (split `C_in`):

- **Row-parallel (`C_in`)** produces a partial-sum output that needs an `all_reduce`, which decomposes as `reduce_scatter` + `all_gather` - 2 ccls.
- **Col-parallel (`C_out`)** produces a clean C-sharded output that only needs **one `all_gather`** to materialize replicated downstream - 1 ccl.

We don't apply this today because we want every block to be **self-contained: replicated in, replicated out**. If we shard `resample[1]` col-parallel, the C-sharded output leaks into the next downstream block and overrides the sharding strategy we set for it - Shardy re-derives a different (worse) layout for that block, which costs us perf. The way to stop the leak is to explicitly mark the resample output as replicated via `xs.mark_sharding(x, mesh, (None,) * x.ndim)` on the intermediate tensor inside `WanResample.forward`, which we'd rather avoid for now.

If profiling later shows the resample conv is the bottleneck, we can revisit this again.
