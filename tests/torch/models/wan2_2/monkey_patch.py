# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


# ---------------------------------------------------------------------------
# DiT monkey patches
# ---------------------------------------------------------------------------


def _patch_apply_lora_scale() -> None:
    """Make `@apply_lora_scale` a pass-through.

    `diffusers.utils.peft_utils.apply_lora_scale` wraps the DiT forward in
    a helper that calls `scale_lora_layers` + `unscale_lora_layers`, each
    of which is a graph break. This test loads plain weights via
    `load_dit()` – no LoRA adapters exist, so the wrapper is pure
    overhead.
    """
    from diffusers.utils import peft_utils

    def noop_decorator(kwargs_name: str = "joint_attention_kwargs"):
        def decorator(forward_fn):
            return forward_fn

        return decorator

    peft_utils.apply_lora_scale = noop_decorator

    # The WanTransformer3DModel.forward in diffusers is decorated at class
    # definition time, so the patch above only affects future imports.
    # Rebind the already-decorated forward to the underlying function.
    from diffusers.models.transformers.transformer_wan import WanTransformer3DModel

    wrapped = WanTransformer3DModel.forward
    underlying = getattr(wrapped, "__wrapped__", None)
    if underlying is not None:
        WanTransformer3DModel.forward = underlying


# ---------------------------------------------------------------------------
# VAE Decoder monkey patches
# ---------------------------------------------------------------------------


from contextlib import contextmanager

import torch

_ORIG_GETITEM = torch.Tensor.__getitem__


def _clamp_slice(s: slice, size: int) -> slice:
    """Canonicalize a slice into ``[0, size]`` (positive step) or ``[-1, size-1]``
    (negative step), like ``slice.indices(size)`` would.

    Hand-written instead of ``slice(...).indices(size)`` because the latter is
    a CPython slot wrapper and dynamo cannot symbolically execute it — it
    graph-breaks at trace time, the resume sub-graph carries a malformed
    ``_guards_fn`` referencing ``L``, and AOT autograd's
    ``PropagateUnbackedSymInts`` then crashes with ``NameError: name 'L' is
    not defined``.  Plain ``max``/``min``/comparisons on concrete ints are
    fully traceable.
    """
    start, stop, step = s.start, s.stop, s.step
    step = 1 if step is None else step

    if step > 0:
        if start is None:
            start = 0
        elif start < 0:
            start = max(0, start + size)
        else:
            start = min(start, size)
        if stop is None:
            stop = size
        elif stop < 0:
            stop = max(0, stop + size)
        else:
            stop = min(stop, size)
    else:
        if start is None:
            start = size - 1
        elif start < 0:
            start = max(-1, start + size)
        else:
            start = min(start, size - 1)
        if stop is None:
            stop = -1
        elif stop < 0:
            stop = max(-1, stop + size)
        else:
            stop = min(stop, size - 1)

    return slice(start, stop, step)


def _normalize_index(idx, shape):
    if not isinstance(idx, tuple):
        idx = (idx,)

    out = []
    dim = 0

    for item in idx:
        if item is Ellipsis:
            remaining_explicit = sum(
                x is not Ellipsis and x is not None for x in idx[idx.index(item) + 1 :]
            )
            fill = len(shape) - dim - remaining_explicit
            out.extend([slice(None)] * fill)
            dim += fill
            continue

        if item is None:
            out.append(item)
            continue

        if isinstance(item, slice):
            out.append(_clamp_slice(item, shape[dim]))
            dim += 1
            continue

        # Leave tensor / bool / advanced indices untouched.
        out.append(item)
        dim += 1

    return tuple(out)


class _SafeSlicingMode(torch.overrides.TorchFunctionMode):
    """Intercept ``Tensor.__getitem__`` via a stack-managed function mode.

    The earlier implementation did ``torch.Tensor.__getitem__ = _safe_getitem``
    on entry and reassigned the slot wrapper back on exit. Restoring the value
    looked correct (identity matched) but the assignment of a Python callable
    to a C-level slot of an extension type permanently flips a CPython flag
    saying "this type has Python overrides". That flag stays set even after
    the slot wrapper is put back, and it disables PyTorch's fast path inside
    ``torch.tensor(list_of_tensors)`` — the fallback then calls ``__len__`` on
    each element, which raises on 0-d tensors. This blew up the diffusers
    UniPC scheduler's ``b = torch.tensor(b, device=device)`` call after the
    first VAE decode in the e2e test.

    A ``TorchFunctionMode`` is the supported per-thread-stack mechanism for
    scoped op interception and doesn't touch ``torch.Tensor``'s class slots,
    so push/pop is properly reversible.
    """

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.Tensor.__getitem__:
            self_, idx = args
            return _ORIG_GETITEM(self_, _normalize_index(idx, self_.shape))
        return func(*args, **kwargs)


@contextmanager
def safe_xla_slicing():
    """
    CPU silently clamps slice `start` / `stop` values lying outside
    `[-size, size]`; torch-xla's lazy backend raises "Value out of range"
    instead. Upstream diffusers' `AutoencoderKLWan` (used by the Wan 2.2
    VAE decoder) relies on the CPU behavior — e.g. `x[:, :, -2:, :, :]`
    on a size-1 temporal dim. Intercept `Tensor.__getitem__` and rewrite
    the index in range before re-dispatching.
    """
    with _SafeSlicingMode():
        yield


def _patch_wan_resample_rep_sentinel() -> None:
    """Replace the `"Rep"` string sentinel in `WanResample.forward` with
    an object-identity sentinel.

    Upstream `diffusers.models.autoencoders.autoencoder_kl_wan.WanResample`
    stores either a tensor or the literal string `"Rep"` in its
    `feat_cache` slot, then branches with `feat_cache[idx] == "Rep"` /
    `!= "Rep"`. When `feat_cache[idx]` is a tensor those comparisons go
    through `Tensor.__eq__(str)` / `Tensor.__ne__(str)`, which dynamo
    cannot trace and triggers a graph break.

    Swap the sentinel for `object()` and use `is` / `is not`, which
    dynamo specializes on without breaking the graph.
    """
    try:
        from diffusers.models.autoencoders import autoencoder_kl_wan as akw
    except ImportError:
        return

    import torch

    cache_t = akw.CACHE_T
    rep = object()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()
        if self.mode == "upsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = rep
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -cache_t:, :, :].clone()
                    if (
                        cache_x.shape[2] < 2
                        and feat_cache[idx] is not None
                        and feat_cache[idx] is not rep
                    ):
                        cache_x = torch.cat(
                            [
                                feat_cache[idx][:, :, -1, :, :]
                                .unsqueeze(2)
                                .to(cache_x.device),
                                cache_x,
                            ],
                            dim=2,
                        )
                    if (
                        cache_x.shape[2] < 2
                        and feat_cache[idx] is not None
                        and feat_cache[idx] is rep
                    ):
                        cache_x = torch.cat(
                            [torch.zeros_like(cache_x).to(cache_x.device), cache_x],
                            dim=2,
                        )
                    if feat_cache[idx] is rep:
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1

                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                    x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.resample(x)
        x = x.view(b, t, x.size(1), x.size(2), x.size(3)).permute(0, 2, 1, 3, 4)

        if self.mode == "downsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -1:, :, :].clone()
                    x = self.time_conv(
                        torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2)
                    )
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x

    akw.WanResample.forward = forward


def _patch_wan_resample_avoid_4d_fold() -> None:
    """Rewrite `WanResample.forward` so the upsample path doesn't lose
    PCC under SPMD channel-parallel sharding.

    Two ops in the original spatial section regress sharded:

      1. The 5D→4D fold via `permute(0,2,1,3,4) → reshape(b*t, c, h, w)`
         loses dim-1 channel sharding through the partitioner.
      2. `WanUpsample` (nn.Upsample with `mode="nearest-exact"`)
         lowers to a tt-mlir kernel that produces wrong values when
         the input is sharded on the channel dim — measured PCC of the
         WanUpsample-only step at ~0.40 vs ~1.0 unsharded on
         `up_blocks[2]` at 480p.

    Both regressions stack into the catastrophic ~0.40 PCC seen on the
    upsample2d branch in `up_blocks[2]` (full block: ~0.94 sharded due
    to partial recovery via the residual `avg_shortcut` add; full
    sharded decoder: ~0.9).

    Fix strategy for upsample modes — for **any T** (including the
    upsample3d non-first-chunk T=2 / T=4 cases that the per-frame
    `_decode` loop triggers): process each temporal slice
    independently via ``unbind(2) → repeat_interleave×2 on H/W →
    Conv2d → stack(2)``. unbind/stack on a non-channel dim and
    repeat_interleave on non-channel dims all preserve dim-1 channel
    sharding cleanly. For exact 2× scale, repeat_interleave is
    bit-equivalent to nearest/nearest-exact upsampling (each input
    pixel → 2×2 block of identical values). The Python loop unrolls
    at compile time, so the trace is fully static.

    Subsumes `_patch_wan_resample_rep_sentinel`: the upsample3d /
    downsample3d cache logic is included verbatim with the same
    object-identity sentinel. Calling the rep-sentinel patch is
    harmless but redundant once this one is applied.
    """
    try:
        from diffusers.models.autoencoders import autoencoder_kl_wan as akw
    except ImportError:
        return

    import torch

    cache_t = akw.CACHE_T
    rep = object()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()
        if self.mode == "upsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = rep
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -cache_t:, :, :].clone()
                    if (
                        cache_x.shape[2] < 2
                        and feat_cache[idx] is not None
                        and feat_cache[idx] is not rep
                    ):
                        cache_x = torch.cat(
                            [
                                feat_cache[idx][:, :, -1, :, :]
                                .unsqueeze(2)
                                .to(cache_x.device),
                                cache_x,
                            ],
                            dim=2,
                        )
                    if (
                        cache_x.shape[2] < 2
                        and feat_cache[idx] is not None
                        and feat_cache[idx] is rep
                    ):
                        cache_x = torch.cat(
                            [torch.zeros_like(cache_x).to(cache_x.device), cache_x],
                            dim=2,
                        )
                    if feat_cache[idx] is rep:
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1

                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                    x = x.reshape(b, c, t * 2, h, w)

        # Spatial resample. T==1 takes the squeeze path that preserves
        # dim-1 channel sharding; T>1 falls back to the original 4D
        # fold (only reached on upsample3d non-first-chunk).
        if self.mode in ("upsample2d", "upsample3d"):
            # Per-slice spatial: unbind T → manual 2x upsample → Conv2d
            # → stack T. SPMD-clean for any T (1, 2, 4 in the per-frame
            # decode loop).
            conv2d = self.resample[1]
            out_slices = []
            for s in torch.unbind(x, dim=2):
                s = s.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
                out_slices.append(conv2d(s))
            x = torch.stack(out_slices, dim=2)
        elif self.mode in ("downsample2d", "downsample3d"):
            t_now = x.shape[2]
            x = x.permute(0, 2, 1, 3, 4).reshape(b * t_now, c, h, w)
            x = self.resample(x)
            x = x.view(b, t_now, x.size(1), x.size(2), x.size(3)).permute(0, 2, 1, 3, 4)

        if self.mode == "downsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -1:, :, :].clone()
                    x = self.time_conv(
                        torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2)
                    )
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x

    akw.WanResample.forward = forward
