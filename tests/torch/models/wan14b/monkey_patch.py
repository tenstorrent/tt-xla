# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


# ---------------------------------------------------------------------------
# DiT monkey patches
# ---------------------------------------------------------------------------


def _patch_pad_seq_len_to_tile_aligned(tile: int = 32) -> None:
    """Pad the DiT sequence length to a multiple of ``tile`` (default 32).

    The raw token-count after patchify is ``post_patch_num_frames * post_patch_height
    * post_patch_width`` — e.g. 21*30*52 = 32760 for 480p Wan2.2. This is not a
    multiple of the 32x32 TT tile size, which forces a pad+reduce_scatter+slice
    dance in every distributed reduction (RMSNorm, etc.) and breaks tile alignment
    for downstream matmuls. Padding seq to 32768 makes all those reductions and
    matmuls tile-aligned at the cost of 8 dummy tokens (~0.024% of work).

    Two surgical edits:
      1. WanTransformer3DModel.forward: pad ``hidden_states`` and ``rotary_emb``
         to the next ``tile`` multiple right after the patchify flatten, slice
         back right before the unpatchify reshape. The unpadded length is
         packed as a 3rd element of the ``rotary_emb`` tuple so the attention
         processor can recover it without globals.
      2. WanAttnProcessor.__call__: on self-attention, slice K/V back to the
         unpadded length before SDPA so softmax normalization stays bit-exact.
         Q stays at the padded length — its padded rows produce garbage that's
         sliced off at the end of the model. Cross-attention is unaffected
         (K/V come from the encoder at length 512, not from hidden_states).
    """
    import torch
    import torch.nn.functional as F
    from diffusers.models.modeling_outputs import Transformer2DModelOutput
    from diffusers.models.transformers.transformer_wan import (
        WanTransformer3DModel,
        WanAttnProcessor,
        _get_qkv_projections,
        _get_added_kv_projections,
    )
    from diffusers.models.attention_dispatch import dispatch_attention_fn

    def patched_model_forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_hidden_states_image=None,
        return_dict=True,
        attention_kwargs=None,
    ):
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        rotary_emb = self.rope(hidden_states)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # Pad seq dim to nearest tile multiple. seq_len is statically known
        # from the shape, so dynamo specializes pad_amount to a Python int.
        unpadded_seq_len = hidden_states.shape[1]
        padded_seq_len = ((unpadded_seq_len + tile - 1) // tile) * tile
        pad_amount = padded_seq_len - unpadded_seq_len

        if pad_amount > 0:
            # hidden_states: (B, S, D) — pad last-but-one dim on the right.
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_amount))
            freqs_cos, freqs_sin = rotary_emb
            # freqs: (1, S, 1, head_dim) — pad dim 1 on the right.
            freqs_cos = F.pad(freqs_cos, (0, 0, 0, 0, 0, pad_amount))
            freqs_sin = F.pad(freqs_sin, (0, 0, 0, 0, 0, pad_amount))
            # Pack unpadded seq_len into the rotary_emb tuple so the patched
            # attention processor can recover it without globals.
            rotary_emb = (freqs_cos, freqs_sin, unpadded_seq_len)

        # ----- everything below is unchanged from the original forward -----
        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
        )
        if ts_seq_len is not None:
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        for block in self.blocks:
            hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

        if temb.ndim == 3:
            shift, scale = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)

        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        # Slice back to the unpadded seq_len before the unpatchify reshape, which
        # requires exactly post_patch_num_frames * pph * ppw tokens.
        if pad_amount > 0:
            hidden_states = hidden_states[:, :unpadded_seq_len, :]

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    def patched_processor_call(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        rotary_emb=None,
    ):
        # Snapshot self-vs-cross BEFORE the add_k_proj branch reassigns
        # encoder_hidden_states. Self-attention is the only path where K/V are
        # derived from the padded hidden_states and need slicing.
        is_self_attention = encoder_hidden_states is None

        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        # Recover unpadded seq_len from the rotary_emb tuple, if the model packed it.
        unpadded_seq_len = None
        if rotary_emb is not None:
            if len(rotary_emb) == 3:
                freqs_cos, freqs_sin, unpadded_seq_len = rotary_emb
            else:
                freqs_cos, freqs_sin = rotary_emb

            def apply_rotary_emb(h, fc, fs):
                x1, x2 = h.unflatten(-1, (-1, 2)).unbind(-1)
                cos = fc[..., 0::2]
                sin = fs[..., 1::2]
                out = torch.empty_like(h)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(h)

            query = apply_rotary_emb(query, freqs_cos, freqs_sin)
            key = apply_rotary_emb(key, freqs_cos, freqs_sin)

        # On self-attention with a padded seq, slice K/V back to the unpadded
        # length so softmax normalization is bit-exact. Q stays at the padded
        # length — padded Q rows produce outputs that are sliced off in the
        # model's final step. Cross-attention has K/V from the encoder,
        # length 512, not padded.
        if is_self_attention and unpadded_seq_len is not None:
            key = key[:, :unpadded_seq_len]
            value = value[:, :unpadded_seq_len]

        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, value_img = _get_added_kv_projections(attn, encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            key_img = key_img.unflatten(2, (attn.heads, -1))
            value_img = value_img.unflatten(2, (attn.heads, -1))
            hidden_states_img = dispatch_attention_fn(
                query,
                key_img,
                value_img,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
                parallel_config=None,
            )
            hidden_states_img = hidden_states_img.flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=(self._parallel_config if encoder_hidden_states is None else None),
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

    WanTransformer3DModel.forward = patched_model_forward
    WanAttnProcessor.__call__ = patched_processor_call


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


def _disable_tt_torch_function_override() -> None:
    """Pop `TorchFunctionOverride` off the global TorchFunctionMode stack.

    `tt_torch/torch_overrides.py` enters a `TorchFunctionMode` at import
    time. Its body is gated by `torch.compiler.is_compiling()` and does
    nothing on the compile path, but the mode still sits on dynamo's
    function-mode stack and forces a `__torch_function__` trace for every
    matmul / linear encountered during tracing.
    """
    try:
        import tt_torch.torch_overrides as overrides
    except ImportError:
        return

    mode = getattr(overrides, "torch_function_override", None)
    if mode is None:
        return

    try:
        mode.__exit__(None, None, None)
    except Exception:
        # Mode wasn't on the stack or was already popped – ignore.
        pass

    
# ---------------------------------------------------------------------------
# VAE Decoder monkey patches
# ---------------------------------------------------------------------------

    
import torch
from contextlib import contextmanager
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
                x is not Ellipsis and x is not None
                for x in idx[idx.index(item) + 1:]
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
            x = x.view(b, t_now, x.size(1), x.size(2), x.size(3)).permute(
                0, 2, 1, 3, 4
            )

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
