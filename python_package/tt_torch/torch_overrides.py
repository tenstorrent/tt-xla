# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn.functional as _torch_F
from torch.overrides import TorchFunctionMode


class TorchFunctionOverride(TorchFunctionMode):
    def __torch_function__(self, func, types, args, kwargs=None):
        if (
            func.__name__ == "matmul" or func.__name__ == "linear"
        ) and not torch.compiler.is_compiling():
            kwargs = kwargs or {}
            if func.__name__ == "linear":
                inp = args[0] if len(args) > 0 else kwargs.get("input")
                weight = args[1] if len(args) > 1 else kwargs.get("weight")
                bias = args[2] if len(args) > 2 else kwargs.get("bias", None)
            else:
                inp = args[0] if len(args) > 0 else kwargs.get("input")
                weight = args[1] if len(args) > 1 else kwargs.get("other")
                bias = None
            if (
                inp is not None
                and weight is not None
                and (len(inp.shape) >= 4 or len(weight.shape) >= 4)
            ):
                if func.__name__ == "linear":
                    res = torch.einsum("...mk,...nk->...mn", inp, weight)
                else:
                    res = torch.einsum("...mk,...kn->...mn", inp, weight)
                if bias is not None:
                    res = res + bias
                return res
        # _upsample_bilinear2d_aa / _upsample_bicubic2d_aa kernels don't support
        # BFloat16 on CPU. Cast to float32, interpolate, cast back.  PyTorch
        # disables this mode for the recursive func() call below, so no infinite
        # recursion occurs.
        if func is _torch_F.interpolate and not torch.compiler.is_compiling():
            kwargs = kwargs or {}
            inp = args[0] if args else kwargs.get("input")
            antialias = kwargs.get("antialias", False)
            mode = args[3] if len(args) > 3 else kwargs.get("mode", "nearest")
            if (
                inp is not None
                and inp.dtype == torch.bfloat16
                and antialias
                and mode in ("bilinear", "bicubic")
            ):
                src_dtype = inp.dtype
                new_args = (inp.to(torch.float32),) + args[1:]
                result = func(*new_args, **(kwargs or {}))
                return result.to(src_dtype)
        return func(*args, **(kwargs or {}))


torch_function_override = TorchFunctionOverride()
torch_function_override.__enter__()


def _router_forward(self, hidden_states):
    """Monkey-patched GptOssTopKRouter.forward that returns full [T, E] sparse
    routing weights (matching 4.57.1 behavior), instead of compact [T, K]."""
    hidden_states = hidden_states.reshape(-1, self.hidden_dim)
    router_logits = torch.nn.functional.linear(hidden_states, self.weight, self.bias)
    router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)
    router_top_value = torch.nn.functional.softmax(
        router_top_value, dim=1, dtype=router_top_value.dtype
    )
    router_scores = torch.zeros_like(router_logits).scatter_(
        1, router_indices, router_top_value
    )
    return router_logits, router_scores, router_indices


def _experts_forward(self, hidden_states, router_indices=None, routing_weights=None):
    """Monkey-patched GptOssExperts.forward matching 4.57.1 behavior.

    CPU path uses per-expert loop (memory-efficient, serves as PCC golden reference).
    Device path uses dense bmm (static graph for torch.compile).

    Args:
        hidden_states: [T, H] or [B, S, H]
        router_indices: [T, K]
        routing_weights: [T, E] full sparse routing weights
    """
    batch_size = hidden_states.shape[0]
    num_experts = routing_weights.shape[1]
    hidden_states = hidden_states.reshape(-1, self.hidden_size)

    if hidden_states.device.type == "cpu":
        next_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(
                router_indices, num_classes=num_experts + 1
            )
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == num_experts:
                continue
            with torch.no_grad():
                _, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate_up = (
                current_state @ self.gate_up_proj[expert_idx]
                + self.gate_up_proj_bias[expert_idx]
            )
            gate, up = gate_up[..., ::2], gate_up[..., 1::2]
            gate = gate.clamp(min=None, max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
            glu = gate * torch.sigmoid(gate * self.alpha)
            out = ((up + 1) * glu) @ self.down_proj[expert_idx] + self.down_proj_bias[
                expert_idx
            ]
            weighted_output = out * routing_weights[token_idx, expert_idx, None]
            next_states.index_add_(
                0, token_idx, weighted_output.to(hidden_states.dtype)
            )
        next_states = next_states.view(batch_size, -1, self.hidden_size)
    else:
        hidden_states = hidden_states.repeat(num_experts, 1)
        hidden_states = hidden_states.view(num_experts, -1, self.hidden_size)
        gate_up = (
            torch.bmm(hidden_states, self.gate_up_proj)
            + self.gate_up_proj_bias[..., None, :]
        )
        gate, up = gate_up[..., ::2], gate_up[..., 1::2]
        gate = gate.clamp(min=None, max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        glu = gate * torch.sigmoid(gate * self.alpha)
        next_states = torch.bmm(((up + 1) * glu), self.down_proj)
        next_states = next_states + self.down_proj_bias[..., None, :]
        next_states = next_states.view(num_experts, batch_size, -1, self.hidden_size)
        next_states = (
            next_states
            * routing_weights.transpose(0, 1).view(num_experts, batch_size, -1)[
                ..., None
            ]
        )
        next_states = next_states.sum(dim=0)
    return next_states


def _sparse_mlp_forward(self, hidden_states):
    """Monkey-patched GptOssMLP.forward matching 4.57.1 behavior."""
    _, router_scores, router_indices = self.router(hidden_states)
    routed_out = self.experts(
        hidden_states, router_indices=router_indices, routing_weights=router_scores
    )
    return routed_out, router_scores


# Monkey patch to restore 4.57.1 interfaces:
# - Router returns full [T, E] sparse routing weights
# - Experts has CPU per-expert loop + device dense bmm
# - Bypasses @use_experts_implementation decorator dispatch
try:
    from transformers.models.gpt_oss.modeling_gpt_oss import (
        GptOssExperts,
        GptOssMLP,
        GptOssTopKRouter,
    )

    GptOssTopKRouter.forward = _router_forward
    GptOssExperts.forward = _experts_forward
    GptOssMLP.forward = _sparse_mlp_forward
except ImportError:
    pass


def _lfm2_short_conv_slow_forward(self, x, past_key_values=None, cache_position=None, attention_mask=None):
    """Monkey-patched Lfm2ShortConv.slow_forward that avoids cache_position[0] > 0,
    which causes aten.item() inside the compiled XLA graph (INTERNAL Error code: 13).
    Uses seqlen == 1 as a compile-friendly proxy: decode is always single-token."""
    import torch.nn.functional as _F
    from transformers.models.lfm2.modeling_lfm2 import apply_mask_to_padding_states

    seqlen = x.shape[1]
    x = apply_mask_to_padding_states(x, attention_mask)
    BCx = self.in_proj(x).transpose(-1, -2)
    B, C, x = BCx.chunk(3, dim=-2)
    Bx = B * x

    if past_key_values is not None and seqlen == 1:
        conv_state = past_key_values.conv_cache[self.layer_idx]
        cache_position = cache_position.clamp(0, self.L_cache - 1)
        conv_state = conv_state.roll(shifts=-1, dims=-1)
        conv_state[:, :, cache_position] = Bx.to(device=conv_state.device, dtype=conv_state.dtype)
        past_key_values.conv_cache[self.layer_idx].copy_(conv_state)
        conv_out = torch.sum(conv_state.to(Bx.device) * self.conv.weight[:, 0, :], dim=-1)
        if self.bias:
            conv_out += self.conv.bias
        conv_out = conv_out.unsqueeze(-1)
    else:
        if past_key_values is not None:
            conv_state = _F.pad(Bx, (self.L_cache - Bx.shape[-1], 0))
            past_key_values.conv_cache[self.layer_idx].copy_(conv_state)
        conv_out = self.conv(Bx)[..., :seqlen]

    y = C * conv_out
    y = y.transpose(-1, -2).contiguous()
    y = self.out_proj(y)
    return y


try:
    from transformers.models.lfm2.modeling_lfm2 import Lfm2ShortConv

    Lfm2ShortConv.slow_forward = _lfm2_short_conv_slow_forward
except ImportError:
    pass


def _lfm2_vl_model_get_image_features(self, pixel_values, spatial_shapes, pixel_attention_mask=None, **kwargs):
    """Monkey-patched Lfm2VlModel.get_image_features.

    Avoids pjrt-device-to-host (INTERNAL Error 13): computes feature lengths from
    spatial_shapes (numpy array, always CPU-accessible) instead of
    pixel_attention_mask.sum(dim=1) (TT int32 tensor used as a Python slice bound
    inside the compiled XLA graph, which requires a device-to-host transfer).

    By NaFlex patchification design, pixel_attention_mask[i].sum() == h * w from
    spatial_shapes[i], so this is numerically equivalent.
    """
    kwargs.pop("return_dict", None)  # consumed by the @can_return_tuple decorator in the original

    image_outputs = self.vision_tower(
        pixel_values=pixel_values,
        spatial_shapes=spatial_shapes,
        pixel_attention_mask=pixel_attention_mask,
        return_dict=True,
        **kwargs,
    )
    last_hidden_state = image_outputs.last_hidden_state

    image_features = []
    for img_idx in range(last_hidden_state.size(0)):
        feature = last_hidden_state[img_idx]
        feature_org_h = int(spatial_shapes[img_idx, 0])
        feature_org_w = int(spatial_shapes[img_idx, 1])
        feature = feature[: feature_org_h * feature_org_w, :].unsqueeze(0)
        feature = feature.reshape(1, feature_org_h, feature_org_w, -1)
        img_embedding = self.multi_modal_projector(feature)
        img_embedding = img_embedding.reshape(-1, img_embedding.size(-1))
        image_features.append(img_embedding)

    image_outputs.pooler_output = image_features
    return image_outputs


try:
    from transformers.models.lfm2_vl.modeling_lfm2_vl import Lfm2VlModel

    Lfm2VlModel.get_image_features = _lfm2_vl_model_get_image_features
except ImportError:
    pass
