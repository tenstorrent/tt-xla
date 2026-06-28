# Compiler support - known limitations and open issues

A living list of sharding patterns the tt-mlir / tt-metal stack does **not** support
yet, so you don't design a strategy around something that can't lower.

> Support changes over time. **Verify each item against current tt-mlir before ruling a
> strategy out**, and append new findings here if you hit them.

## Known limitations

### Spatial sharding of convolutions (halo exchange)
Sharding a Conv3D/Conv2D on spatial dims (D/H/W) requires neighboring devices to
exchange kernel-sized "halo" border strips (`neighbor_pad` / `slice_reshard`). This is
**not yet exposed through TTIR/TTNN MLIR**, so it can't be driven from the compiler
side today.
- Tracking issue: tt-mlir #8216 - https://github.com/tenstorrent/tt-mlir/issues/8216
- Fallback: for conv-heavy models (e.g. VAE), shard channels (C_in/C_out) instead of
  spatial dims. See [video_vae.md](video_vae.md).

### Shardy back-propagating a shard through a reshape
A single `sharding_constraint` placed after a reshape lets Shardy propagate the shard
*back into* the reshape; tt-mlir then fails to update the reshape's static result type
→ "number of elements doesn't match".
- Workaround: apply two constraints back-to-back downstream of the reshape - a
  replicated anchor first, then the sharded spec. See [video_dit.md](video_dit.md) and
  [shardy_sharding.md](shardy_sharding.md).

## Adding an entry
When you hit a new limitation, record it as:
**pattern → symptom → tracking issue/PR → workaround or fallback**, with a source link.
