# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, List, Optional, Union

import torch
from torch import Tensor
from torch_xla.experimental.mark_pattern_utils import StableHLOCompositeBuilder
from ttxla_tools.logging import logger

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
    input: Tensor, normalized_shape, weight=None, eps=None, residual=None
) -> Tensor:
    """
    Creates composite RMS norm operation for torch xla using StableHLOCompositeBuilder.
    Note that operation name must be tenstorrent.rms_norm for MLIR to handle it.

    Args:
        input: Input tensor
        normalized_shape: Shape over which to normalize (tuple of ints)
        weight: Optional learnable weight parameter
        eps: Epsilon for numerical stability (default: None)
        residual: Optional residual to sum into input before normalization.
            When provided, the composite carries a `has_residual=True`
            attribute and the residual is the LAST marked input. tt-mlir's
            StableHLOLegalizeCompositePass uses that attribute to route the
            operand into ttir.rms_norm's residual slot, which the runtime
            forwards to ttnn::rms_norm's residual_input_tensor parameter.

    Returns a tensor.
    """
    attr = {"normalized_shape": normalized_shape}
    if eps is not None:
        attr["epsilon"] = eps
    if residual is not None:
        attr["has_residual"] = True

    builder = StableHLOCompositeBuilder(name="tenstorrent.rms_norm", attr=attr)

    # Operand order: (input, [weight], [residual]). Residual is always the
    # trailing marked input when present so the legalize pass can pop it
    # off based on `has_residual` regardless of whether weight is present.
    if weight is not None and residual is not None:
        input, weight, residual = builder.mark_inputs(input, weight, residual)
    elif weight is not None:
        input, weight = builder.mark_inputs(input, weight)
    elif residual is not None:
        input, residual = builder.mark_inputs(input, residual)
    else:
        input = builder.mark_inputs(input)

    summed = input + residual if residual is not None else input
    output = torch.nn.functional.rms_norm(summed, normalized_shape, weight, eps)
    output = builder.mark_outputs(output)

    return output


def _gated_activation_composite(
    name: str,
    activation_fn: Callable[[Tensor], Tensor],
    input: Tensor,
    dim: int,
) -> Tensor:
    """Shared body for the 4 gated-activation composites (SwiGLU/GLU/GeGLU/ReGLU).

    All 4 variants share the structure ``activation(input[:H]) * input[H:]``
    along ``dim``, differing only in the activation function. We emit one
    composite per variant with a stable Tenstorrent name so tt-mlir's
    StableHLOLegalizeCompositePass can collapse it back into a single
    ttir.gated_activation op (with the activation kind stored as an attr).

    The ``dim`` is carried as a composite attribute so the legalize pass can
    materialize the slice axis without re-deriving from the StableHLO
    ``slice`` operands.
    """
    builder = StableHLOCompositeBuilder(name=name, attr={"dim": dim})

    input = builder.mark_inputs(input)
    half = input.shape[dim] // 2
    out = activation_fn(input.narrow(dim, 0, half)) * input.narrow(dim, half, half)
    return builder.mark_outputs(out)


def composite_swiglu(input: Tensor, dim: int = -1) -> Tensor:
    """SwiGLU gated activation: ``SiLU(input[:H]) * input[H:]`` along ``dim``.

    Wraps the decomposed PyTorch impl with StableHLOCompositeBuilder so
    tt-mlir's StableHLOLegalizeCompositePass can collapse it back into
    a single ttir.gated_activation op with activation kind ``swiglu``.
    """
    return _gated_activation_composite(
        "tenstorrent.swiglu", torch.nn.functional.silu, input, dim
    )


def composite_glu(input: Tensor, dim: int = -1) -> Tensor:
    """GLU gated activation: ``Sigmoid(input[:H]) * input[H:]`` along ``dim``."""
    return _gated_activation_composite(
        "tenstorrent.glu", torch.sigmoid, input, dim
    )


def composite_geglu(input: Tensor, dim: int = -1) -> Tensor:
    """GeGLU gated activation: ``GELU(input[:H]) * input[H:]`` along ``dim``."""
    return _gated_activation_composite(
        "tenstorrent.geglu", torch.nn.functional.gelu, input, dim
    )


def composite_reglu(input: Tensor, dim: int = -1) -> Tensor:
    """ReGLU gated activation: ``ReLU(input[:H]) * input[H:]`` along ``dim``."""
    return _gated_activation_composite(
        "tenstorrent.reglu", torch.nn.functional.relu, input, dim
    )


def _normalize_slice_spec(rank: int, narrow_specs):
    """Materialize ``begins``, ``ends``, ``step`` arrays of length ``rank``
    from a list of (dim, begin, length) triplets covering only the
    actually-sliced dims. Unsliced dims default to ``[0:input_size]`` —
    callers must supply ``input_shape`` (a list of ints) to fill those.

    Returns the populated (begins, ends, step) lists. Raises ValueError if
    a dim repeats or is out of range.
    """
    begins = [0] * rank
    ends_list = [None] * rank
    step = [1] * rank
    for dim, b, length in narrow_specs:
        if dim < 0:
            dim = dim + rank
        if dim < 0 or dim >= rank:
            raise ValueError(f"dim {dim} out of range for rank {rank}")
        if ends_list[dim] is not None:
            raise ValueError(f"duplicate narrow on dim {dim}")
        begins[dim] = b
        ends_list[dim] = b + length
    return begins, ends_list, step


def composite_slice_reshape(
    input: Tensor,
    narrow_specs: List,
    shape: List[int],
) -> Tensor:
    """Fused slice + reshape composite emitter.

    Mirrors quetzal's fuse_slice_reshape pass: a same-rank
    ``slice_static(input, begins, ends, step)`` feeding a ``reshape(shape)``,
    bundled as a single ``tenstorrent.slice_reshape`` StableHLO composite so
    tt-mlir's StableHLOLegalizeCompositePass can collapse the two into a
    single ttir.slice_reshape op (and one program-executor case at runtime,
    saving one host-dispatch round-trip).

    Parameters:
    - ``input``: The tensor to slice and reshape.
    - ``narrow_specs``: A list of ``(dim, begin, length)`` triplets — one
      per sliced dim. Unsliced dims are inferred from ``input.shape``.
      Carrying the per-dim spec instead of fully-materialized
      ``begins``/``ends`` lets the FX-level matcher operate without
      knowing the source shape (symbolic_trace doesn't propagate shapes
      into ``meta``); the composite emitter sees the concrete tensor and
      can fill in the rest.
    - ``shape``: The final reshape target shape.

    The slice / reshape parameters travel as composite_attributes so
    tt-mlir's TenstorrentSliceReshapeConversionPattern can emit them as
    I32ArrayAttrs on the TTIR op without re-deriving anything from the
    StableHLO body.
    """
    rank = input.dim()
    input_shape = list(input.shape)
    begins, ends_list, step = _normalize_slice_spec(rank, narrow_specs)
    for d in range(rank):
        if ends_list[d] is None:
            ends_list[d] = input_shape[d]
    ends = list(ends_list)

    # Composite attributes are carried as plain Python lists; the
    # StableHLOCompositeBuilder serializes them as DenseI64ArrayAttr by
    # default, and tt-mlir's TenstorrentSliceReshapeConversionPattern
    # accepts both that and ArrayAttr variants.
    attr = {
        "begins": begins,
        "ends": ends,
        "step": step,
        "shape": list(shape),
    }
    builder = StableHLOCompositeBuilder(name="tenstorrent.slice_reshape", attr=attr)

    input = builder.mark_inputs(input)
    # Body: slice (same-rank) followed by reshape, mirroring the runtime
    # composition. Use torch.narrow per dim to keep the body decomposable
    # by torch_xla into stablehlo.slice ops; reshape via .reshape().
    sliced = input
    for dim, (b, e, s) in enumerate(zip(begins, ends, step)):
        if b == 0 and e == input_shape[dim] and s == 1:
            continue  # full-slice — no-op narrow.
        if s == 1:
            sliced = sliced.narrow(dim, b, e - b)
        else:
            idx = torch.arange(b, e, s, device=sliced.device)
            sliced = torch.index_select(sliced, dim, idx)
    out = sliced.reshape(shape)
    return builder.mark_outputs(out)


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


def composite_group_norm(
    input: Tensor,
    num_groups: int,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
) -> Tensor:
    """
    Creates composite group_norm operation for torch xla using StableHLOCompositeBuilder.
    Operation name is tenstorrent.group_norm for MLIR to handle it.

    Args:
        input: Input tensor to normalize
        num_groups: Number of groups to divide the channels into
        weight: Optional learnable weight parameter for affine transformation
        bias: Optional learnable bias parameter for affine transformation
        eps: Small constant for numerical stability (default: 1e-5)

    Returns:
        Normalized tensor with same shape as input
    """
    attr = {"num_groups": num_groups, "epsilon": eps, "channel_dim": 1}

    builder = StableHLOCompositeBuilder(name="tenstorrent.group_norm", attr=attr)

    if weight is not None and bias is not None:
        input, weight, bias = builder.mark_inputs(input, weight, bias)
    elif weight is not None:
        input, weight = builder.mark_inputs(input, weight)
    elif bias is not None:
        input, bias = builder.mark_inputs(input, bias)
    else:
        input = builder.mark_inputs(input)

    output = torch.nn.functional.group_norm(input, num_groups, weight, bias, eps)

    output = builder.mark_outputs(output)

    return output


def composite_topk(
    input: Tensor,
    k: int,
    dim: int = -1,
    largest: bool = True,
    sorted: bool = True,
    *,
    out: tuple[Tensor, ...] | list[Tensor] | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Creates composite topk operation for torch-xla using StableHLOCompositeBuilder.
    Returns a (values, indices) tuple.
    """
    attrs = {"k": k, "dim": dim, "largest": largest, "sorted": sorted}

    builder = StableHLOCompositeBuilder(name="tenstorrent.topk", attr=attrs)

    input = builder.mark_inputs(input)
    values, indices = torch.topk(input, k, dim, largest, sorted, out=out)
    values, indices = builder.mark_outputs(values, indices)
    return (values, indices)


def composite_topk_values(
    input: Tensor,
    k: int,
    dim: int = -1,
    largest: bool = True,
    sorted: bool = True,
    *,
    out: tuple[Tensor, ...] | list[Tensor] | None = None,
) -> Tensor:
    """Composite topk returning only values. Marks a single output at pos=0."""
    attrs = {"k": k, "dim": dim, "largest": largest, "sorted": sorted}
    builder = StableHLOCompositeBuilder(name="tenstorrent.topk_values", attr=attrs)
    input = builder.mark_inputs(input)
    values, _ = torch.topk(input, k, dim, largest, sorted)
    values = builder.mark_outputs(values)
    return values


def composite_topk_indices(
    input: Tensor,
    k: int,
    dim: int = -1,
    largest: bool = True,
    sorted: bool = True,
    *,
    out: tuple[Tensor, ...] | list[Tensor] | None = None,
) -> Tensor:
    """Composite topk returning only indices. Marks a single output at pos=0."""
    attrs = {"k": k, "dim": dim, "largest": largest, "sorted": sorted}
    builder = StableHLOCompositeBuilder(name="tenstorrent.topk_indices", attr=attrs)
    input = builder.mark_inputs(input)
    _, indices = torch.topk(input, k, dim, largest, sorted)
    indices = builder.mark_outputs(indices)
    return indices


def composite_scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
) -> Tensor:
    """
    Creates composite scaled_dot_product_attention operation for torch xla
    """

    attr = {"is_causal": is_causal}
    if scale is not None:
        attr["scale"] = scale

    builder = StableHLOCompositeBuilder(
        name="tenstorrent.scaled_dot_product_attention", attr=attr
    )

    if attn_mask is not None:
        query, key, value, attn_mask = builder.mark_inputs(query, key, value, attn_mask)
    else:
        query, key, value = builder.mark_inputs(query, key, value)

    output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
    )
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
        # Copy metadata from original node to preserve stack_trace and nn_module_stack
        # This ensures layer_norm maintains proper source location in MLIR output
        new_node.meta = node.meta.copy()

    node.replace_all_uses_with(new_node)
    gm.graph.erase_node(node)


def replace_group_norm_module(
    gm: torch.fx.GraphModule,
    node: torch.fx.Node,
    module: torch.nn.GroupNorm,
) -> None:
    """
    Replace nn.GroupNorm call_module node with composite_group_norm call_function.

    Transformation:
        BEFORE: %out = call_module[target=group_norm](args=(%x,))
        AFTER:  %weight = get_attr[target=group_norm.weight]
                %bias = get_attr[target=group_norm.bias]
                %out = call_function[target=composite_group_norm](
                    args=(%x,),
                    kwargs={num_groups: 8, weight: %weight,
                            bias: %bias, eps: 1e-5}
                )

    Args:
        gm: GraphModule containing the node
        node: call_module node to replace
        module: nn.GroupNorm instance
    """
    num_groups = module.num_groups
    eps = module.eps
    has_weight = module.weight is not None and module.affine
    has_bias = module.bias is not None and module.affine

    input_tensor = node.args[0]

    kwargs = {"num_groups": num_groups, "eps": eps}

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
            composite_group_norm, args=(input_tensor,), kwargs=kwargs
        )

    node.replace_all_uses_with(new_node)
    gm.graph.erase_node(node)


################# composite constraint checks #################


def _check_sdpa_constraints(node: torch.fx.Node) -> bool:
    """
    Check that SDPA inputs are bfloat16, the only dtype our composite supports.
    Also, check for dropout_p > 0, which is not supported in composite SDPA.
    If either of these conditions are met, skip the composite and use the native implementation.
    """
    # Dropout is not supported in composite SDPA
    dropout_p = node.kwargs.get("dropout_p", 0.0)
    if dropout_p is not None and dropout_p > 0:
        logger.debug(
            "composite scaled_dot_product_attention does not support dropout = {dropout_p}, "
            "skipping composite and using native implementation."
        )
        return False

    # Check all inputs are bfloat16
    tensor_args = list(node.args) + [
        v for v in node.kwargs.values() if isinstance(v, torch.fx.Node)
    ]
    for arg in tensor_args:
        if hasattr(arg, "meta"):
            val = arg.meta.get("example_value", None)
            if val is not None and val.dtype != torch.bfloat16:
                logger.debug(
                    "composite scaled_dot_product_attention only supports bfloat16 inputs, "
                    "skipping composite and using native implementation."
                )
                return False

    return True


@torch.fx.wrap
def _torch_residual_rms_norm(
    input: Tensor,
    residual: Tensor,
    normalized_shape,
    weight=None,
    eps=None,
) -> Tensor:
    """Marker function emitted by ResidualRMSNormFusionProvider.

    @torch.fx.wrap keeps this as an opaque call_function node in the FX
    graph so handle_composite_ops can pivot it over to the composite-emitting
    wrapper below. Without the wrap, FX would inline the body and we'd lose
    the fusion site.
    """
    return torch.nn.functional.rms_norm(input + residual, normalized_shape, weight, eps)


def _composite_residual_rms_norm(
    input: Tensor,
    residual: Tensor,
    normalized_shape,
    weight=None,
    eps=None,
) -> Tensor:
    """Composite-emitting form of `_torch_residual_rms_norm`.

    Delegates to composite_rms_norm with `residual=` so the resulting
    StableHLO carries a `tenstorrent.rms_norm` composite with
    `has_residual=True` — the form tt-mlir's StableHLOLegalizeCompositePass
    knows how to route into the new ttir.rms_norm residual operand.
    """
    return composite_rms_norm(
        input, normalized_shape, weight=weight, eps=eps, residual=residual
    )


@torch.fx.wrap
def _torch_swiglu(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Marker function emitted by SwiGLUFusionProvider.

    Like _torch_residual_rms_norm, this is `@torch.fx.wrap`-decorated so FX
    keeps it as an opaque call_function node. handle_composite_ops then maps
    it to the composite-emitting wrapper which produces the
    `tenstorrent.swiglu` StableHLO composite.
    """
    half = input.shape[dim] // 2
    return torch.nn.functional.silu(input.narrow(dim, 0, half)) * input.narrow(
        dim, half, half
    )


@torch.fx.wrap
def _torch_glu(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Marker for GLU (sigmoid gated)."""
    half = input.shape[dim] // 2
    return torch.sigmoid(input.narrow(dim, 0, half)) * input.narrow(dim, half, half)


@torch.fx.wrap
def _torch_geglu(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Marker for GeGLU (GELU gated)."""
    half = input.shape[dim] // 2
    return torch.nn.functional.gelu(input.narrow(dim, 0, half)) * input.narrow(
        dim, half, half
    )


@torch.fx.wrap
def _torch_reglu(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Marker for ReGLU (ReLU gated)."""
    half = input.shape[dim] // 2
    return torch.nn.functional.relu(input.narrow(dim, 0, half)) * input.narrow(
        dim, half, half
    )


def _composite_swiglu(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Composite-emitting form of `_torch_swiglu`. Delegates to composite_swiglu
    so the resulting StableHLO carries a `tenstorrent.swiglu` composite that
    tt-mlir's StableHLOLegalizeCompositePass routes to ttir.gated_activation."""
    return composite_swiglu(input, dim=dim)


def _composite_glu(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return composite_glu(input, dim=dim)


def _composite_geglu(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return composite_geglu(input, dim=dim)


def _composite_reglu(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return composite_reglu(input, dim=dim)


@torch.fx.wrap
def _torch_slice_reshape(
    input: torch.Tensor,
    narrow_specs: List,
    shape: List[int],
) -> torch.Tensor:
    """Marker function emitted by SliceReshapeFusionProvider.

    @torch.fx.wrap keeps this as an opaque call_function node so
    handle_composite_ops can pivot it to the composite-emitting wrapper
    below. The body matches the unfused decomposition (per-dim narrows
    followed by a reshape), so this marker is also a valid functional
    fallback if the composite swap is disabled.

    ``narrow_specs`` is a list of ``(dim, begin, length)`` triplets — one
    per sliced dim. Unsliced dims pass through unchanged. Carrying the
    per-dim spec (instead of fully-materialized ``begins``/``ends``)
    lets the FX-level matcher operate without knowing the source shape.
    """
    sliced = input
    for dim, b, length in narrow_specs:
        sliced = sliced.narrow(dim, b, length)
    return sliced.reshape(shape)


def _composite_slice_reshape(
    input: torch.Tensor,
    narrow_specs: List,
    shape: List[int],
) -> torch.Tensor:
    """Composite-emitting form of `_torch_slice_reshape`.

    Delegates to composite_slice_reshape so the resulting StableHLO carries
    a `tenstorrent.slice_reshape` composite that tt-mlir's
    StableHLOLegalizeCompositePass routes to ttir.slice_reshape.
    """
    return composite_slice_reshape(input, narrow_specs, shape)


def can_apply_composite(node: torch.fx.Node) -> bool:
    """
    Check whether a composite replacement should be applied for the given node.
    Returns True if the replacement is valid, False if it should be skipped.
    """
    return constraints.get(node.target, lambda _: True)(node)


"""
Dictionary holding constraints for composite replacements.
Maps torch API calls to constraint functions.
"""
constraints = {
    torch.nn.functional.scaled_dot_product_attention: _check_sdpa_constraints,
}


"""
Dictionary holding replacement composite functions for torch functions and modules.
Maps torch API calls and module types to composite implementations.
Used for call_function and call_module nodes where node.target is a function reference or module type.

When the node.target is a torch function returning just a single output, we use the
composite function directly.

When the node.target is a torch function returning multiple outputs, we use a dictionary
of composite functions, keyed by the frozenset of output indices used.
"""
replacements = {
    # function replacements
    torch.nn.functional.gelu: composite_gelu,
    torch.rms_norm: composite_rms_norm,
    torch.nn.functional.rms_norm: composite_rms_norm,
    _torch_residual_rms_norm: _composite_residual_rms_norm,
    _torch_swiglu: _composite_swiglu,
    _torch_glu: _composite_glu,
    _torch_geglu: _composite_geglu,
    _torch_reglu: _composite_reglu,
    _torch_slice_reshape: _composite_slice_reshape,
    torch.nn.functional.layer_norm: composite_layer_norm,
    torch.nn.functional.scaled_dot_product_attention: composite_scaled_dot_product_attention,
    # TODO: uncomment once https://github.com/tenstorrent/tt-metal/issues/40916 is fixed
    # torch.nn.functional.group_norm: composite_group_norm,
    torch.topk: {
        frozenset({0, 1}): composite_topk,
        frozenset({0}): composite_topk_values,
        frozenset({1}): composite_topk_indices,
    },
    # module replacements
    torch.nn.LayerNorm: replace_layer_norm_module,
    # TODO: uncomment once https://github.com/tenstorrent/tt-metal/issues/40916 is fixed
    # torch.nn.GroupNorm: replace_group_norm_module,
}

"""
Maps tensor method name strings to their torch function equivalents.
Used to resolve call_method FX nodes (e.g. x.topk(k)) where dynamo sets
node.target to the method name string "topk".
The mapped function must be a key in `replacements` for the rewrite to apply.
"""
method_name_to_function = {
    "topk": torch.topk,
}
