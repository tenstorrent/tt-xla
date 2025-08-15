# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Callable, Dict, List, Optional, Sequence, Union

import contextlib
import threading

import torch
from torch._decomp import get_decompositions, remove_decompositions

DecompositionTable = Dict[torch._ops.OperatorBase, Callable]
DecompositionOpsList = Sequence[
    Union[torch._ops.OperatorBase, torch._ops.OpOverloadPacket]
]

# Manages "scopes" for decompositions used. Each unique scope is an attribute on
# the _decomp_local. If the attribute is missing, then the default
# decompositions are used. The scope "aot" is used for all AOT cases.
_decomp_local = threading.local()


@contextlib.contextmanager
def _extend_context_manager(
    scope: str,
    *,
    from_current: bool = True,
    add_ops: Optional[DecompositionOpsList] = None,
    remove_ops: Optional[DecompositionOpsList] = None,
):
    table: DecompositionTable
    if from_current:
        table = dict(_current(scope))
    else:
        table = {}
    if add_ops:
        table.update(get_decompositions(add_ops))
    if remove_ops:
        remove_decompositions(table, remove_ops)  # type: ignore
    stack = _get_decomp_stack(scope)
    stack.append(table)
    try:
        yield table
    finally:
        popped = stack.pop()
        assert (
            popped is table
        ), "contextmanager unbalanced: popped different that pushed"


def _current(scope: str) -> DecompositionTable:
    """Gets the current decomposition table (which may be the default)."""
    stack = _get_decomp_stack(scope)
    if stack:
        return dict(stack[-1])
    else:
        return dict(CUSTOM_DECOMPOSITION_TABLE)

def _get_decomp_stack(scope: str) -> List[DecompositionTable]:
    try:
        return getattr(_decomp_local, scope)
    except AttributeError:
        stack: List[DecompositionTable] = []
        setattr(_decomp_local, scope, stack)
        return stack


# This method is derived from the implementation of jax.image.resize in JAX:
#     https://github.com/jax-ml/jax/blob/354bd5271077654af983965c8e01ee462ce4ce91/jax/_src/image/scale.py#L52
#
# I've modified it to use numpy rather than JAX. I've also added the ability
# to generate a weight matrix that allows the matmul to be identical to to
# torch's upsample_bilinear2d when align_corners=True.
# This logic was derived from @brentyi's implementation in:
#    https://github.com/jax-ml/jax/issues/11206#issuecomment-1423140760
def compute_linear_weight(input_size, output_size, scale, align_corners, dtype):
    if input_size == 1:
        return torch.ones(1, output_size, dtype=dtype)
    translation = 0
    if align_corners:
        scale = (output_size - 1) / (input_size - 1)
        translation = 0.5 - (scale / 2)

    inv_scale = 1 / scale
    sample_f = (
        (torch.arange(output_size, dtype=torch.float64) + 0.5) * inv_scale
        - translation * inv_scale
        - 0.5
    )
    x = torch.abs(sample_f - torch.arange(input_size, dtype=torch.float64).unsqueeze(1))

    weights = torch.relu(1 - torch.abs(x))

    total_weight_sum = torch.sum(weights, axis=0, keepdims=True)
    weights = torch.divide(
        weights,
        torch.where(total_weight_sum != 0, total_weight_sum, 1),
    )

    weights = torch.where(
        torch.logical_and(sample_f >= -0.5, sample_f <= input_size - 0.5),
        weights,
        0,
    )
    weights = weights.squeeze()
    return weights.to(dtype)


def upsample_linear(
    input: torch.Tensor,
    output_size: List[int],
    align_corners: bool,
    scales: List[Optional[float]],
) -> torch.Tensor:
    input_size = input.shape[-len(scales) :]

    for i in range(len(scales)):
        scales[i] = float(output_size[i]) / float(input_size[i])

    res = input
    for i in range(len(scales)):
        weight = compute_linear_weight(
            input_size[i], output_size[i], scales[i], align_corners, input.dtype
        ).to(input.device)
        res = (res.transpose(i - len(scales), -1) @ weight).transpose(
            i - len(scales), -1
        )
    return res


def upsample_nearest(
    input: torch.Tensor,
    output_size: List[int],
    scales: List[Optional[float]],
    exact: bool = False,
):
    input_size = input.shape[-len(scales) :]

    # Find the indices which we should gather from the input tensor
    # but use them to construct weight matrices to perform the interpolation
    # rather than simply gather.
    indices = []
    for (scale, in_size, out_size) in zip(scales, input_size, output_size):
        # To map from output indices to input indices we need to multiply
        # the output index by the reciprocal of the scale
        scale = 1 / scale if scale is not None else in_size / out_size

        all_output_indices = torch.arange(out_size)
        input_indices = (
            torch.floor(all_output_indices * scale)
            .to(torch.int32)
            .unsqueeze(0)
            .transpose(-2, -1)
        )
        # input_indices currently contains which indices to gather from the
        # input tensor for each output index we are going to concatenate the
        # output indices to this tensor so that we can use it to map from
        # output indices to input indices.
        input_indices = torch.cat(
            [
                input_indices,
                torch.arange(out_size, dtype=torch.int32)
                .unsqueeze(0)
                .transpose(-2, -1),
            ],
            dim=-1,
        )

        # input_indices is in the form [input_index, output_index]. That is to say
        # that for the nth output index, input_index[n, 0] is the index to gather
        # from the input tensor, and input_index[n, 1] is the output index (n).
        indices.append(input_indices)

    # Must use torch.ones so this input is consteval-able
    one = torch.ones(1, dtype=input.dtype)
    res = input
    for dim, indices_dim in enumerate(indices):
        weight_ = torch.zeros(input_size[dim], output_size[dim], dtype=input.dtype)
        weight = weight_.index_put((indices_dim[:, 0], indices_dim[:, 1]), one).to(
            input.device
        )  # use out-of-place index_put so graph remains consteval-able

        res = (res.transpose(dim - len(indices), -1) @ weight).transpose(
            dim - len(indices), -1
        )

    return res


def upsample_linear_vec(
    input: torch.Tensor,
    output_size: Optional[List[int]],
    align_corners: bool,
    scale_factors: Optional[List[float]],
) -> torch.Tensor:
    scale_factors = scale_factors if output_size is None else None
    osize = torch._decomp.decompositions.upsample_compute_output_size(
        input.size(), output_size, scale_factors
    )
    scales = scale_factors if scale_factors else [None] * len(osize)
    return upsample_linear(input, osize, align_corners, scales)


def upsample_linear_default(
    input: torch.Tensor,
    output_size: list[int],
    align_corners: bool,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
    scales_d: Optional[float] = None,
) -> torch.Tensor:
    scale_factors = [scales_h, scales_w, scales_d] if output_size is None else None
    osize = torch._decomp.decompositions.upsample_compute_output_size(
        input.size(), output_size, scale_factors
    )
    scales = scale_factors if scale_factors else [None] * len(osize)
    return upsample_linear(input, osize, align_corners, scales)


def upsample_nearest_vec(
    input: torch.Tensor,
    output_size: Optional[List[int]],
    scale_factors: Optional[List[float]],
) -> torch.Tensor:
    scale_factors = scale_factors if output_size is None else None
    osize = torch._decomp.decompositions.upsample_compute_output_size(
        input.size(), output_size, scale_factors
    )
    scales = (
        scale_factors if scale_factors else [None] * len(osize)  # type: ignore[list-item]
    )
    return upsample_nearest(input, osize, scales, scales)


def upsample_nearest_default(
    input: torch.Tensor,
    output_size: list[int],
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
    scales_d: Optional[float] = None,
) -> torch.Tensor:
    scale_factors = [scales_h, scales_w, scales_d] if output_size is None else None
    osize = torch._decomp.decompositions.upsample_compute_output_size(
        input.size(), output_size, scale_factors
    )
    scales = (
        scale_factors if scale_factors else [None] * len(osize)  # type: ignore[list-item]
    )
    return upsample_nearest(input, osize, scales, scales)


# TODO: Remove this decomposition when we can lower a stablehlo.reduce_window which is equivalent to a sum-pool
# to ttir
def avg_pool2d(
    input: torch.Tensor,
    kernel_size: Union[int, List[int]],
    stride: Optional[Union[int, List[int]]] = None,
    padding: Union[int, List[int]] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: Optional[int] = None,
) -> torch.Tensor:

    if isinstance(kernel_size, int):
        kernel_size = [kernel_size, kernel_size]
    if stride is None:
        stride = kernel_size
    if isinstance(stride, int):
        stride = [stride, stride]
    if isinstance(padding, int):
        padding = [padding, padding, padding, padding]

    input_size = list(input.shape[-len(stride) :])
    if stride == kernel_size == input_size and padding == [0, 0, 0, 0]:
        return input.mean(dim=[-2, -1], keepdim=True)

    # If we call the regular torch.nn.functional.avg_pool2d, it will infinitely recurse into this function.
    # Returning NotImplemented allows the tracer to use the default implementation.
    return NotImplemented


# TODO: Test if this is still necesarry when compiling via torch-xla
def split_with_sizes(
    self: torch.Tensor, split_sizes: List[int], dim: int = 0
) -> List[torch.Tensor]:
    # NB: Perform the check_is_size tests first so that the
    # sum test does not try to do a replacement
    for i in range(len(split_sizes)):
        torch._check_is_size(
            split_sizes[i],
            lambda: "split_with_sizes expects split_sizes have only non-negative entries",
        )
    torch._check_with(
        ValueError,
        sum(split_sizes) == self.shape[dim],
        lambda: f"Split sizes add up to {sum(split_sizes)} but got the tensor's size of {self.shape[dim]}",
    )
    num_splits = len(split_sizes)
    splits = []
    start_idx = 0

    for i in range(num_splits):
        length = split_sizes[i]
        splits.append(self.narrow(dim, start_idx, length))
        start_idx += length
    return splits

# TODO: Test if this is still necesarry when compiling via torch-xla
def masked_fill_tensor(input, mask, value):
    if value.device != input.device:
        value = value.to(input.device)
        return torch.masked_fill(input, mask, value)
    return NotImplemented


# Squeeze is defined as an aten::prim op in some circumstances. This
# causes issues during the passes we run on the GraphModule in `torch_pass_pipeline`.
# This decomposition converts the squeeze to a reshape.
def squeeze(input, dims):
    shape = input.shape
    newshape = [s for i, s in enumerate(shape) if i not in dims]
    return input.reshape(newshape)


# TODO: DO we ever need this?
def _get_default_decomposition_ops() -> DecompositionOpsList:
    aten = torch.ops.aten
    # default decompositions pulled from SHARK / torch._decomp
    return [
        aten.embedding_dense_backward,
        aten.native_layer_norm_backward,
        aten.slice_backward,
        aten.select_backward,
        aten.norm.ScalarOpt_dim,
        aten.native_group_norm,
        aten.split.Tensor,
        # aten.split_with_sizes,
        aten.native_layer_norm,
        aten.masked_fill.Tensor,
        aten.masked_fill.Scalar,
        aten.t,
        aten.addmm,
        # decompositions that aid us in handling nn.BatchNorm2d
        aten._native_batch_norm_legit_functional,
        aten._native_batch_norm_legit,
        aten._native_batch_norm_legit.no_stats,
        aten.squeeze.dims,
        # decompositions for miscellaneous ops that are not handled in torch-mlir but have available decompositions
        aten.soft_margin_loss,
        aten.im2col,
        aten._euclidean_dist,
        aten.index_copy,
        aten.index_copy_,
        aten.grid_sampler_2d,
        aten.log_sigmoid_forward,
        aten.unsafe_split.Tensor,
        aten.binary_cross_entropy,
        aten.dot,
        aten._adaptive_avg_pool2d,
        aten._prelu_kernel,
        aten.full,
        aten._log_softmax,
        aten.nll_loss_forward,
        aten.nll_loss_backward,
        aten._to_copy,
        aten._log_softmax_backward_data,
        aten.lift_fresh_copy.default,
        aten._unsafe_index.Tensor,
        aten.unbind.int,
        aten.linspace.Tensor_Tensor,
        aten._scaled_dot_product_flash_attention_for_cpu.default,
        aten.slice_scatter,
    ]


def _get_custom_decopositions() -> DecompositionTable:
    aten = torch.ops.aten
    return {
        # Interpolation decompositions here perform interpolation
        # using a series of matmuls agains constant tensors.
        # They are necesarry as the default aten decompositions
        # use gather, which we cannot lower from ttir-to ttnn 
        # in the form presented by this decomposition.
        # The better (and more performant) solution to this is 
        # to fuse the gather-based pattern in tt-mlir to the correct
        # interpolation op.
        aten.upsample_nearest1d.vec: upsample_nearest_vec,
        aten.upsample_nearest2d.vec: upsample_nearest_vec,
        aten.upsample_nearest3d.vec: upsample_nearest_vec,
        aten.upsample_linear1d.vec: upsample_linear_vec,
        aten.upsample_bilinear2d.vec: upsample_linear_vec,
        aten.upsample_trilinear3d.vec: upsample_linear_vec,
        aten.upsample_nearest1d.default: upsample_nearest_default,
        aten.upsample_nearest2d.default: upsample_nearest_default,
        aten.upsample_nearest3d.default: upsample_nearest_default,
        aten.upsample_linear1d.default: upsample_linear_default,
        aten.upsample_bilinear2d.default: upsample_linear_default,
        aten.upsample_trilinear3d.default: upsample_linear_default,

        # TODO: Test if this is still necesarry when compiling via torch-xla
        aten.adaptive_avg_pool2d.default: aten._adaptive_avg_pool2d,

        # TODO: Test if this is still necesarry when compiling via torch-xla
        aten.avg_pool2d.default: avg_pool2d,
        aten.split_with_sizes.default: split_with_sizes,
        aten.masked_fill.Tensor: masked_fill_tensor,
        torch.ops.prims.squeeze.default: squeeze,
    }


CUSTOM_DECOMPOSITION_TABLE = get_decompositions(_get_default_decomposition_ops())
CUSTOM_DECOMPOSITION_TABLE.update(_get_custom_decopositions())
