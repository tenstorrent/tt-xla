# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Union

import torch
from torch import Tensor
from torch_xla.experimental.mark_pattern_utils import StableHLOCompositeBuilder

"""
From XLA documentation: '(Composite operation) Encapsulates an operation made up (composed) of
other StableHLO operations, taking inputs and composite_attributes and producing results.
The semantics of the op are implemented by the decomposition attribute. The composite op can be
replaced with its decomposition without changing program semantics.'

So, composite op is made because of high-level ops that are decomposed into dozens stable HLO
operations. It enables custom backends to handle op however they want. Since we have a native
support for, let's say, gelu operation in ttir, we will wrap torch gelu op into gelu composite
op and handle it in XLA without decompostions, enabling our device to execute custom gelu implementation.

Since we want to run torch models wihout modifying them, we will substitute torch operations
(that we have a direct support for) with composite ops. This way, user will not have to change
anything in model in order to get performance improvement.
"""

################# function replacements #################


def composite_gelu(input: Tensor, approximate: str = "none") -> Tensor:
    """
    Creates composite gelu operation for torch xla using StableHLOCompositeBuilder.
    Note that operation name must be tenstorrent.gelu[_tanh] for MLIR to handle it.

    Returns a tensor.
    """
    tanh = approximate == "tanh"
    name = "tenstorrent.gelu" + ("_tanh" if tanh else "")
    attr = {"approximate": "tanh"} if tanh else None

    builder = StableHLOCompositeBuilder(name=name, attr=attr)

    input = builder.mark_inputs(input)
    input = torch.nn.functional.gelu(input, approximate=approximate)
    input = builder.mark_outputs(input)

    return input


def composite_rms_norm(
    input: Tensor, normalized_shape, weight=None, eps=None
) -> Tensor:
    """
    Creates composite RMS norm operation for torch xla using StableHLOCompositeBuilder.
    Note that operation name must be tenstorrent.rms_norm for MLIR to handle it.

    Args:
        input: Input tensor
        normalized_shape: Shape over which to normalize (tuple of ints)
        weight: Optional learnable weight parameter
        eps: Epsilon for numerical stability (default: None)

    Returns a tensor.
    """
    attr = {"normalized_shape": normalized_shape}
    if eps is not None:
        attr["epsilon"] = eps

    builder = StableHLOCompositeBuilder(name="tenstorrent.rms_norm", attr=attr)

    if weight is not None:
        input, weight = builder.mark_inputs(input, weight)
    else:
        input = builder.mark_inputs(input)

    output = torch.nn.functional.rms_norm(input, normalized_shape, weight, eps)
    output = builder.mark_outputs(output)

    return output


def composite_layer_norm(
    input: Tensor,
    normalized_shape: Union[int, List[int], torch.Size],
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
) -> Tensor:
    """
    Creates composite layer_norm operation for torch xla using StableHLOCompositeBuilder.
    Operation name is tenstorrent.layer_norm for MLIR to handle it.

    Args:
        input: Input tensor to normalize
        normalized_shape: Shape over which to normalize (int, list, tuple, or torch.Size)
        weight: Optional learnable weight parameter for affine transformation
        bias: Optional learnable bias parameter for affine transformation
        eps: Small constant for numerical stability (default: 1e-5)

    Returns:
        Normalized tensor with same shape as input
    """
    if isinstance(normalized_shape, int):
        normalized_shape_list = [normalized_shape]
    else:
        normalized_shape_list = list(normalized_shape)

    attr = {"normalized_shape": normalized_shape_list, "epsilon": eps}

    builder = StableHLOCompositeBuilder(name="tenstorrent.layer_norm", attr=attr)

    if weight is not None and bias is not None:
        input, weight, bias = builder.mark_inputs(input, weight, bias)
    elif weight is not None:
        input, weight = builder.mark_inputs(input, weight)
    elif bias is not None:
        input, bias = builder.mark_inputs(input, bias)
    else:
        input = builder.mark_inputs(input)

    output = torch.nn.functional.layer_norm(
        input, normalized_shape_list, weight, bias, eps
    )

    output = builder.mark_outputs(output)

    return output

def composite_sdpa(query: Tensor, key: Tensor, value: Tensor, attn_mask: Tensor | None = None, dropout_p: float = 0.0, is_causal: bool = False, scale = None, enable_gqa: bool = False) -> Tensor:
    """
    Creates composite scaled dot product attention operation for torch xla using StableHLOCompositeBuilder.
    Note that operation name must be tenstorrent.scaled_dot_product_attention for MLIR to handle it.

    Args:
        query: Query tensor [batch, num_heads, query_seq_len, head_size]
        key: Key tensor [batch, num_kv_heads, kv_seq_len, head_size]
        value: Value tensor [batch, num_kv_heads, kv_seq_len, head_size]
        attn_mask: Optional attention mask tensor
        dropout_p: Dropout probability (default: 0.0)
        is_causal: Whether to apply causal mask (default: False)
        scale: Optional scale factor for attention scores
        enable_gqa: Whether to enable group query attention optimization (TT-specific, default: False)

    Returns a tensor with shape [batch, num_heads, query_seq_len, head_size].
    
    NOTE: The decomposition uses a an equivalent implementation of SDPA
    because XLA's standard SDPA decomposition fails to legalize the operation.
    Since tt-mlir will replace this composite with ttir.scaled_dot_product_attention,
    the decomposition body is never actually executed on TT hardware, it just serves as golden.
    """

    def __scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
        """
        An equivalent implementation of SDPA that can successfully legalize on XLA.
        Source: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
        """
        import math
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias = attn_mask + attn_bias

        if enable_gqa:
            key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
            value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        return attn_weight @ value
    
    attr = {
        "is_causal": is_causal,
    }
    if scale is not None:
        attr["scale"] = float(scale)  # Ensure it's a Python float

    builder = StableHLOCompositeBuilder(name="tenstorrent.scaled_dot_product_attention", attr=attr)

    if attn_mask is not None:
        query, key, value, attn_mask = builder.mark_inputs(query, key, value, attn_mask)
    else:
        query, key, value = builder.mark_inputs(query, key, value)

    
    output = __scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)
    
    output = builder.mark_outputs(output)
    return output


################# module replacements #################


def replace_layer_norm_module(
    gm: torch.fx.GraphModule,
    node: torch.fx.Node,
    module: torch.nn.LayerNorm,
) -> None:
    """
    Replace nn.LayerNorm call_module node with composite_layer_norm call_function.

    Transformation:
        BEFORE: %out = call_module[target=layer_norm](args=(%x,))
        AFTER:  %weight = get_attr[target=layer_norm.weight]
                %bias = get_attr[target=layer_norm.bias]
                %out = call_function[target=composite_layer_norm](
                    args=(%x,),
                    kwargs={normalized_shape: (768,), weight: %weight,
                            bias: %bias, eps: 1e-5}
                )

    Args:
        gm: GraphModule containing the node
        node: call_module node to replace
        module: nn.LayerNorm instance
    """
    normalized_shape = module.normalized_shape
    eps = module.eps
    has_weight = module.weight is not None and module.elementwise_affine
    has_bias = module.bias is not None and module.elementwise_affine

    input_tensor = node.args[0]

    kwargs = {"normalized_shape": normalized_shape, "eps": eps}

    with gm.graph.inserting_before(node):
        if has_weight:
            weight_node = gm.graph.get_attr(f"{node.target}.weight")
            kwargs["weight"] = weight_node
        else:
            kwargs["weight"] = None

        if has_bias:
            bias_node = gm.graph.get_attr(f"{node.target}.bias")
            kwargs["bias"] = bias_node
        else:
            kwargs["bias"] = None

        new_node = gm.graph.call_function(
            composite_layer_norm, args=(input_tensor,), kwargs=kwargs
        )

    node.replace_all_uses_with(new_node)
    gm.graph.erase_node(node)


"""
Dictionary holding replacement composite functions for torch functions and modules.
Maps torch API calls and module types to composite implementations.
Used for call_function and call_module nodes where node.target is a function reference or module type.
"""
replacements = {
    # function replacements
    torch.nn.functional.gelu: composite_gelu,
    torch.rms_norm: composite_rms_norm,
    torch.nn.functional.layer_norm: composite_layer_norm,
    torch.nn.functional.scaled_dot_product_attention: composite_sdpa,
    # module replacements
    torch.nn.LayerNorm: replace_layer_norm_module,
}
