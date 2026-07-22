# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import operator
import re

import torch
from torch.export.graph_signature import InputKind, OutputKind
from tt_torch import composite_ops
from tt_torch.fusion_providers import FusionProvider
from tt_torch.slice_bounds import clamp_neg_slice_key
from ttxla_tools.logging import logger


def rewrite_interpolate_to_matmul(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    Rewrite F.interpolate(mode='bilinear'/'nearest') to the matmul-based implementation.

    AOTAutograd's functionalization trace decomposes F.interpolate via the C++
    CompositeImplicitAutograd kernel into gather/index ops before the Python
    decomposition table is consulted.  This pass rewrites the calls at the FX
    graph level so they use our efficient matmul-based interpolation instead.
    """
    from tt_torch.backend.decompositions import (
        upsample_linear_vec,
        upsample_nearest_vec,
    )

    graph = gm.graph
    modified = False

    for node in list(graph.nodes):
        if (
            node.op != "call_function"
            or node.target is not torch.nn.functional.interpolate
        ):
            continue

        # Extract arguments — F.interpolate(input, size=, scale_factor=, mode=, align_corners=, ...)
        input_node = node.args[0] if node.args else node.kwargs.get("input")
        size = node.kwargs.get("size", node.args[1] if len(node.args) > 1 else None)
        scale_factor = node.kwargs.get("scale_factor", None)
        mode = node.kwargs.get("mode", "nearest")
        align_corners = node.kwargs.get("align_corners", False)

        # Normalize size to a list or None
        if size is not None:
            if isinstance(size, int):
                output_size = [size, size]
            else:
                output_size = list(size)
            scale_factors = None
        else:
            output_size = None
            if isinstance(scale_factor, (int, float)):
                scale_factors = [float(scale_factor), float(scale_factor)]
            elif scale_factor is not None:
                scale_factors = [float(s) for s in scale_factor]
            else:
                continue

        # Use the vec wrappers which handle output_size/scale_factor normalization
        if mode == "bilinear":
            replacement_fn = upsample_linear_vec
            new_args = (input_node, output_size, align_corners, scale_factors)
        elif mode == "nearest":
            replacement_fn = upsample_nearest_vec
            new_args = (input_node, output_size, scale_factors)
        else:
            continue

        with graph.inserting_after(node):
            new_node = graph.call_function(
                replacement_fn,
                args=new_args,
            )
            node.replace_all_uses_with(new_node)
            graph.erase_node(node)
            modified = True

    if modified:
        gm.recompile()

    return gm


def rewrite_adaptive_avgpool_to_mean(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    Rewrite call_module nodes targeting AdaptiveAvgPool1d/2d with output_size=1/(1,1)
    to use torch.mean instead.

    This works around an XLA + FunctionalTensorMode incompatibility where inplace_view
    ops (aten.as_strided_ inside adaptive pooling) are re-executed under no_dispatch()
    for metadata fixup, causing dispatch to XLA's kernel on wrapper subclass tensors
    that XLA can't handle.
    """
    graph = gm.graph
    modified = False

    for node in list(graph.nodes):
        if node.op == "call_module" and isinstance(node.target, str):
            target_module = gm.get_submodule(node.target)
            if isinstance(target_module, torch.nn.AdaptiveAvgPool1d):
                output_size = target_module.output_size
                if output_size == 1 or output_size == (1,) or output_size == [1]:
                    with graph.inserting_after(node):
                        mean_node = graph.call_function(
                            torch.mean,
                            args=(node.args[0],),
                            kwargs={"dim": [-1], "keepdim": True},
                        )
                        node.replace_all_uses_with(mean_node)
                        graph.erase_node(node)
                        modified = True
            elif isinstance(target_module, torch.nn.AdaptiveAvgPool2d):
                output_size = target_module.output_size
                if output_size == 1 or output_size == (1, 1) or output_size == [1, 1]:
                    with graph.inserting_after(node):
                        mean_node = graph.call_function(
                            torch.mean,
                            args=(node.args[0],),
                            kwargs={"dim": [-2, -1], "keepdim": True},
                        )
                        node.replace_all_uses_with(mean_node)
                        graph.erase_node(node)
                        modified = True

    if modified:
        gm.recompile()

    return gm


_VIEW_OPS = (
    torch.ops.aten.view.default,
    torch.ops.aten._unsafe_view.default,
    torch.ops.aten.reshape.default,
)
_TRANSPOSE_OPS = (
    torch.ops.aten.transpose.int,
    torch.ops.aten.permute.default,
)


def _fx_node_shape(n):
    """Static shape from an FX node's val meta, or None if unavailable."""
    if not isinstance(n, torch.fx.Node):
        return None
    val = n.meta.get("val")
    return tuple(int(d) for d in val.shape) if val is not None else None


def _swaps_last_two_only(node):
    """True iff `node` is a transpose/permute that swaps exactly the last two dims."""
    src_shape = _fx_node_shape(node.args[0])
    if src_shape is None or len(src_shape) < 2:
        return False
    rank = len(src_shape)
    if node.target is torch.ops.aten.transpose.int:
        d0 = node.args[1] % rank
        d1 = node.args[2] % rank
        return {d0, d1} == {rank - 2, rank - 1}
    # permute
    perm = list(node.args[1])
    expected = list(range(rank))
    expected[-2], expected[-1] = expected[-1], expected[-2]
    return perm == expected


def _trace_to_rank_n_source(operand, target_batch_prefix):
    """Walk back from a bmm operand through view/reshape and last-two-dim
    transpose nodes until we land on a tensor whose shape's batch prefix
    matches `target_batch_prefix`.  Returns `(node, transposed_last_two)`
    on success, or `None` on failure."""
    target_rank = len(target_batch_prefix) + 2
    cur = operand
    transposed = False

    while isinstance(cur, torch.fx.Node):
        shp = _fx_node_shape(cur)
        if shp is None:
            return None
        if len(shp) == target_rank and shp[:-2] == target_batch_prefix:
            return cur, transposed
        if cur.target in _VIEW_OPS:
            cur = cur.args[0]
            continue
        if cur.target in _TRANSPOSE_OPS and _swaps_last_two_only(cur):
            transposed = not transposed
            cur = cur.args[0]
            continue
        return None
    return None


def fold_view_bmm_view_to_einsum(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """Recover rank-preserving matmul from the post-AOTA view->bmm->view sandwich.

    AOTAutograd's SDPA decomp (and PyTorch's N-D matmul -> bmm folding,
    LinearAlgebra.cpp#L1999) reshapes (..., M, K) operands to (B*H, M, K)
    before bmm, then reshapes the output back.  Under tensor parallelism with
    the head dim sharded, that destroys head sharding and forces SPMD to
    insert an f32 all_gather on the 4-D Q/K/V, which OOMs L1.

    Detect the sandwich and rewrite it to a rank-preserving einsum, which
    torch-xla lowers to a 4-D stablehlo.dot_general with head sharding intact.
    aten.einsum.default is opaque (we pop its decomp in populate_decompositions),
    so it survives to torch-xla unchanged.

    Walks each bmm operand backwards through any chain of view/reshape and
    last-two-dim transpose/permute ops (the latter absorbed into the einsum
    equation, which handles the typical K^T in Q*K^T).  Anything else
    (expand, repeat_interleave, convert_element_type, clone, ...) stops the
    walk and the bmm is left alone.
    """
    graph = gm.graph
    rewritten = 0

    for bmm in list(graph.nodes):
        if bmm.op != "call_function" or bmm.target is not torch.ops.aten.bmm.default:
            continue
        if _try_rewrite_bmm_to_einsum(graph, bmm):
            rewritten += 1

    if rewritten:
        graph.eliminate_dead_code()
        graph.lint()
        gm.recompile()

    return gm


def _try_rewrite_bmm_to_einsum(graph, bmm):
    """Try to rewrite one bmm site.  Returns True iff rewritten."""
    if len(bmm.users) != 1:
        return False
    out_v = next(iter(bmm.users))
    if not (isinstance(out_v, torch.fx.Node) and out_v.target in _VIEW_OPS):
        return False
    out_shape = _fx_node_shape(out_v)
    if out_shape is None or len(out_shape) < 4:
        return False

    batch_prefix = out_shape[:-2]
    M_out, N_out = out_shape[-2], out_shape[-1]

    lhs_res = _trace_to_rank_n_source(bmm.args[0], batch_prefix)
    if lhs_res is None:
        return False
    rhs_res = _trace_to_rank_n_source(bmm.args[1], batch_prefix)
    if rhs_res is None:
        return False

    lhs_4d, lhs_T = lhs_res
    rhs_4d, rhs_T = rhs_res
    lshp = _fx_node_shape(lhs_4d)
    rshp = _fx_node_shape(rhs_4d)

    # Resolve logical (M, K) / (K, N) given the recorded transpose flags.
    M_lhs, K_lhs = (lshp[-1], lshp[-2]) if lhs_T else (lshp[-2], lshp[-1])
    K_rhs, N_rhs = (rshp[-1], rshp[-2]) if rhs_T else (rshp[-2], rshp[-1])

    if (M_lhs, N_rhs) != (M_out, N_out) or K_lhs != K_rhs:
        return False

    lhs_eq = "...km" if lhs_T else "...mk"
    rhs_eq = "...nk" if rhs_T else "...kn"
    equation = f"{lhs_eq},{rhs_eq}->...mn"

    with graph.inserting_before(bmm):
        einsum = graph.call_function(
            torch.ops.aten.einsum.default,
            args=(equation, [lhs_4d, rhs_4d]),
        )
    einsum.meta.update(out_v.meta)
    out_v.replace_all_uses_with(einsum)
    return True


def run_fusion_passes(gm: torch.fx.GraphModule) -> None:
    """
    Run all registered fusion passes on a GraphModule.

    Args:
        gm: The GraphModule to transform
    """
    total_replacements = 0

    for provider_cls in FusionProvider.get_registered_providers():
        provider = provider_cls()
        num_replaced = provider.replace_pattern(gm)
        if num_replaced > 0:
            logger.debug(f"[Fusion] {provider.name}: {num_replaced} match(es)")
            total_replacements += num_replaced

    if total_replacements > 0:
        gm.graph.lint()
        gm.recompile()


def _get_used_output_indices(node: torch.fx.Node) -> frozenset:
    """Return frozenset of output indices consumed by live getitem users of node."""
    # For example: when running torch.topk(x, k=5) the following Torch FX IR gets generated:
    # %topk = call_function[target=torch.topk](x, k=5)
    # %getitem_0 = call_function[target=operator.getitem](%topk, 0) // Values result
    # %getitem_1 = call_function[target=operator.getitem](%topk, 1) // Indices result

    # Where:
    # - %topk is the node provided as input
    # - [%getitem_0, %getitem_1] is the node.users list
    # - (%topk, 0) is an example of user.args:
    #     - user.args[0] = %topk = source node
    #     - user.args[1] = 0 = output index of the result of the node
    used = set()
    for user in node.users:
        if (
            user.op == "call_function"
            and user.target is operator.getitem
            and isinstance(user.args[1], int)
            and len(user.users) > 0  # Ensure that the result is actually being used
        ):
            used.add(user.args[1])
    return frozenset(used)


def _replace_multi_output_op(gm, node, output_variants):
    """Select the correct composite variant for a multi-output op and rewire the graph."""
    used_indices = _get_used_output_indices(node)
    replacement_fn = output_variants.get(used_indices)
    if replacement_fn is None:  # fallback to most-outputs variant
        fallback_key = max(output_variants.keys(), key=len)
        replacement_fn = output_variants[fallback_key]

    node.target = replacement_fn

    # Collect all getitem children of this node
    getitem_nodes = [
        u
        for u in list(node.users.keys())
        if u.op == "call_function" and u.target is operator.getitem
    ]

    # For single-output replacement, redirect the live getitem's users to point
    # directly at the composite node, then erase all getitems
    if len(used_indices) == 1:
        used_idx = next(iter(used_indices))
        for gi in getitem_nodes:
            if gi.args[1] == used_idx:
                gi.replace_all_uses_with(node)
        for gi in getitem_nodes:
            gm.graph.erase_node(gi)


# SDPA input tensors are laid out [..., num_heads, seq_len, head_dim], so the head
# axis is the third-from-last dimension.
_SDPA_HEAD_AXIS_FROM_END = 3


def _fx_node_shape(node) -> tuple | None:
    """Return the static shape of an fx node from its fake-tensor meta, or None."""
    if not isinstance(node, torch.fx.Node):
        return None
    val = node.meta.get("example_value", node.meta.get("val", None))
    if val is None or not hasattr(val, "shape"):
        return None
    try:
        return tuple(int(d) for d in val.shape)
    except (TypeError, ValueError):
        # Symbolic (dynamic) dims are not integers; we can't verify the pattern.
        return None


def _normalize_dim(dim: int, rank: int) -> int:
    return dim + rank if dim < 0 else dim


def _match_unsqueeze_after(node: torch.fx.Node, head_axis: int):
    """Return src if `node` inserts a single unit dim immediately after head_axis."""
    # Form 1: advanced-index unsqueeze, e.g. x[:, :, None, :, :]
    if node.op == "call_function" and node.target is operator.getitem:
        index = node.args[1]
        if not isinstance(index, tuple):
            return None
        none_positions = [i for i, e in enumerate(index) if e is None]
        # Exactly one inserted axis, and every other entry must be a full slice
        # (reject integer/advanced indexing that would drop or reorder dims).
        if len(none_positions) != 1:
            return None
        if any(e is not None and e != slice(None) for e in index):
            return None
        if none_positions[0] != head_axis + 1:
            return None
        return node.args[0]
    # Form 2: explicit unsqueeze
    if node.op == "call_method" and node.target == "unsqueeze":
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
        src_shape = _fx_node_shape(node.args[0])
        if dim is None or src_shape is None:
            return None
        if _normalize_dim(dim, len(src_shape) + 1) == head_axis + 1:
            return node.args[0]
    return None


def _match_repeat_interleave(node: torch.fx.Node, head_axis: int):
    """Return src if `node` is a repeat_interleave along the head axis."""
    is_ri = (node.op == "call_function" and node.target is torch.repeat_interleave) or (
        node.op == "call_method" and node.target == "repeat_interleave"
    )
    if not is_ri:
        return None
    dim = node.kwargs.get("dim", node.args[2] if len(node.args) > 2 else None)
    out_shape = _fx_node_shape(node)
    # A missing dim means the tensor is flattened first -> not a head repeat.
    if dim is None or out_shape is None:
        return None
    if _normalize_dim(dim, len(out_shape)) != head_axis:
        return None
    # `repeats` must be a uniform scalar to match enable_gqa's interleave ordering.
    # A per-head tensor of repeats can satisfy the output-shape gate while producing
    # a different head ordering, which would make the enable_gqa substitution wrong.
    repeats = node.args[1] if len(node.args) > 1 else node.kwargs.get("repeats")
    if not isinstance(repeats, int):
        return None
    return node.args[0]


def _match_expand_reshape(node: torch.fx.Node, head_axis: int):
    """Return src if `node` is an HF-style reshape(expand(unsqueeze(src)))."""
    if not (node.op == "call_method" and node.target in ("reshape", "view")):
        return None
    expand = node.args[0]
    if not (
        isinstance(expand, torch.fx.Node)
        and expand.op == "call_method"
        and expand.target == "expand"
    ):
        return None
    src = _match_unsqueeze_after(expand.args[0], head_axis)
    if src is None:
        return None
    src_shape = _fx_node_shape(src)
    exp_shape = _fx_node_shape(expand)
    unsq_shape = _fx_node_shape(expand.args[0])
    out_shape = _fx_node_shape(node)
    if None in (src_shape, exp_shape, unsq_shape, out_shape):
        return None
    rank = len(out_shape)
    # unsqueeze/expand insert exactly one axis right after the head axis.
    if len(unsq_shape) != rank + 1 or len(exp_shape) != rank + 1:
        return None
    if unsq_shape[head_axis + 1] != 1:
        return None
    if exp_shape[head_axis] != src_shape[head_axis]:
        return None
    n_rep = exp_shape[head_axis + 1]
    if n_rep <= 1:
        return None
    # reshape merges (head, rep) back into head*rep, leaving other dims intact.
    if out_shape[head_axis] != src_shape[head_axis] * n_rep:
        return None
    return src


def _uninflated_kv_source(kv_node, query_heads):
    """
    If `kv_node` is the output of a grouped-query `repeat_kv` head-expansion, return
    the pre-expansion source node; otherwise None.

    Only the two expansion forms that reproduce torch's `enable_gqa=True` ordering
    (interleaved repetition along the head axis) are matched, so that replacing the
    expansion with `enable_gqa=True` is numerically identical. Tile/`repeat`-style
    expansion produces a different head ordering and is intentionally not matched.

    The match is gated on head-axis-only integer expansion whose result equals the
    query head count and whose source head count divides it -- a valid GQA grouping
    with no other dimension disturbed.
    """
    out_shape = _fx_node_shape(kv_node)
    if out_shape is None or len(out_shape) < _SDPA_HEAD_AXIS_FROM_END:
        return None
    rank = len(out_shape)
    head_axis = rank - _SDPA_HEAD_AXIS_FROM_END

    src = _match_repeat_interleave(kv_node, head_axis) or _match_expand_reshape(
        kv_node, head_axis
    )
    if src is None:
        return None

    src_shape = _fx_node_shape(src)
    if src_shape is None or len(src_shape) != rank:
        return None

    src_heads = src_shape[head_axis]
    out_heads = out_shape[head_axis]
    # Head axis must grow by an integer factor; every other dim must be unchanged.
    if src_heads == 0 or out_heads == src_heads or out_heads % src_heads != 0:
        return None
    for i in range(rank):
        if i != head_axis and src_shape[i] != out_shape[i]:
            return None
    # Result must be a valid GQA grouping for this query's head count.
    if query_heads is None or query_heads % src_heads != 0 or out_heads != query_heads:
        return None
    return src


def _sdpa_operand(node: torch.fx.Node, pos: int, name: str):
    """Fetch an SDPA operand that may be passed positionally or by keyword."""
    if pos < len(node.args):
        return node.args[pos]
    return node.kwargs.get(name)


def erase_repeat_kv(gm: torch.fx.GraphModule) -> None:
    """
    Erase grouped-query `repeat_kv` head-expansion off the key/value operands of
    scaled_dot_product_attention and set `enable_gqa=True` instead.

    This lets the downstream (composite / ttnn) SDPA consume the non-inflated K/V
    and broadcast the heads on device, avoiding materialization of the expanded
    K/V. Only expansions whose head ordering matches torch's `enable_gqa` semantics
    are removed, so the result is numerically identical. Non-GQA attention with
    n_rep == 1 never matches (no expansion node), so this is safe to run
    unconditionally.
    """
    sdpa = torch.nn.functional.scaled_dot_product_attention
    changed = False
    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target is not sdpa:
            continue

        query = _sdpa_operand(node, 0, "query")
        q_shape = _fx_node_shape(query)
        q_heads = (
            q_shape[-_SDPA_HEAD_AXIS_FROM_END]
            if q_shape is not None and len(q_shape) >= _SDPA_HEAD_AXIS_FROM_END
            else None
        )

        # Match key and value independently, but only rewrite if BOTH deflate.
        # enable_gqa requires key and value to share the same head count, so a
        # one-sided rewrite (e.g. deflated key + still-expanded value) would
        # leave SDPA with mismatched K/V heads and produce a wrong/failed result.
        matched = {}
        for pos, kw in ((1, "key"), (2, "value")):
            src = _uninflated_kv_source(_sdpa_operand(node, pos, kw), q_heads)
            if src is not None:
                matched[(pos, kw)] = src

        if len(matched) != 2:
            continue

        src_heads = {
            _fx_node_shape(src)[-_SDPA_HEAD_AXIS_FROM_END] for src in matched.values()
        }
        if len(src_heads) != 1:
            # Deflated key and value disagree on head count -> not a valid GQA
            # grouping for enable_gqa; leave the node untouched.
            continue

        new_args = list(node.args)
        new_kwargs = dict(node.kwargs)
        for (pos, kw), src in matched.items():
            if pos < len(new_args):
                new_args[pos] = src
            else:
                new_kwargs[kw] = src

        new_kwargs["enable_gqa"] = True
        node.args = tuple(new_args)
        node.kwargs = new_kwargs
        changed = True
        logger.debug(
            "erase_repeat_kv: erased repeat_kv for SDPA node %s",
            node.name,
        )

    if changed:
        # Do NOT call gm.graph.eliminate_dead_code() here (pre-export). KV-cache
        # fills are in-place ops (kcache.copy_(k), kcache[:, :, pos] = k) whose
        # return is unused -- SDPA depends on them through the buffer, not a data
        # edge. Pre-export fx cannot see this:
        #   fill = call_method (copy_ / __setitem__)  ->  is_impure() = False
        #   users = 0  ->  DCE deletes the fill  ->  SDPA reads an empty cache (PCC ~0.3)
        # No fix in this pass makes DCE safe here; the rule is "nobody sweeps
        # pre-export". After torch.export / run_decompositions these become real
        # mutation edges (is_impure() = True), so DCE there cleans up the orphaned
        # repeat_kv nodes safely.
        gm.graph.lint()
        gm.recompile()


def handle_composite_ops(gm: torch.fx.GraphModule) -> None:
    """
    Replaces torch ops with composite ops if we have a proper replacement.

    Handles three types of nodes:
    1. call_function nodes: torch and torch.nn.functional ops
       - node.target is a function reference
       - Replaced by changing node.target to composite function

    2. call_module nodes: nn.Module instances
       - node.target is a string like "some_module"
       - Replaced by creating new call_function node (composite function) with get_attr for parameters

    3. call_method nodes: tensor method calls (e.g. x.topk(k) instead of torch.topk(x, k))
       - node.target is a method name string like "topk"
       - Resolved via composite_ops.method_name_to_function, then promoted to call_function
    """
    for node in list(gm.graph.nodes):  # snapshot to allow mid-loop erasure
        if node.op == "call_function":
            if (
                node.target in composite_ops.replacements
                and composite_ops.can_apply_composite(node)
            ):
                replacement = composite_ops.replacements[node.target]
                if isinstance(replacement, dict):
                    _replace_multi_output_op(gm, node, replacement)
                else:
                    node.target = replacement

        elif node.op == "call_module":
            module = gm.get_submodule(node.target)
            module_type = type(module)
            if module_type in composite_ops.replacements:
                composite_ops.replacements[module_type](gm, node, module)

        elif node.op == "call_method":
            # This happens when the method is called as `input.function(args)` instead of
            # `function(input, args)`.
            torch_fn = composite_ops.method_name_to_function.get(node.target)
            if torch_fn is not None and torch_fn in composite_ops.replacements:
                replacement = composite_ops.replacements[torch_fn]
                # Promote call_method to call_function (args layout is identical)
                node.op = "call_function"
                if isinstance(replacement, dict):
                    _replace_multi_output_op(gm, node, replacement)
                else:
                    node.target = replacement

    gm.graph.lint()


def insert_argument_type_markers(
    gm: torch.fx.GraphModule,
    graph_signature,
    flat_name_to_original_fqn: dict = None,
) -> torch.fx.GraphModule:

    if flat_name_to_original_fqn is None:
        flat_name_to_original_fqn = {}

    normalized_fqn_lookup = _build_normalized_fqn_lookup(flat_name_to_original_fqn)

    input_nodes = gm.graph.find_nodes(op="get_attr") + gm.graph.find_nodes(
        op="placeholder"
    )
    input_signature = graph_signature.input_specs
    output_signature = graph_signature.output_specs

    # Keep track of buffers which are mutated as we do not want these arguments to be hoisted into a consteval graph.
    mutated_buffer_targets = set()
    for out_spec in output_signature:
        if out_spec.kind == OutputKind.BUFFER_MUTATION:
            mutated_buffer_targets.add(out_spec.target)

    input_type_dict = {}
    for in_spec in input_signature:
        type_str = None
        if in_spec.kind == InputKind.USER_INPUT:
            type_str = "input"
        # We do not model these argument types in tt-mlir. To avoid graph transformations that would
        # impact how these inputs are handled (i.e. consteval), we will mark them as "input".
        elif in_spec.kind in [InputKind.TOKEN, InputKind.CUSTOM_OBJ]:
            type_str = "input"
        elif in_spec.kind == InputKind.PARAMETER:
            type_str = "parameter"
        elif in_spec.kind == InputKind.CONSTANT_TENSOR:
            type_str = "constant"
        # If a buffer is mutated, we do not want to hoist the argument into a consteval graph.
        # This is because the argument will be mutated in place, and we do not want to used the cached
        # version of the input from the first iteration of the graph. If it is not mutated then we can
        # mark it as a constant.
        elif in_spec.kind == InputKind.BUFFER:
            if in_spec.target in mutated_buffer_targets:
                type_str = "input"
            else:
                type_str = "constant"
        else:
            assert False, f"Unexpected input kind: {in_spec.kind}"

        # Index by target (for get_attr nodes) and by arg.name (for placeholder
        # nodes in the AOTAutograd path).
        if in_spec.target is not None:
            input_type_dict[in_spec.target] = type_str
        input_type_dict[in_spec.arg.name] = type_str

    for input_node in input_nodes:
        users = list(input_node.users.keys())
        if len(users) == 0:
            continue

        argument_type = input_type_dict.get(input_node.target) or input_type_dict.get(
            input_node.name
        )
        if argument_type is None:
            continue

        mangled_name = input_node.target if input_node.target else input_node.name
        clean_name = _demangle_name(mangled_name, normalized_fqn_lookup)

        with gm.graph.inserting_after(input_node):
            new_input = gm.graph.create_node(
                "call_function",
                torch.ops.tt.mark_argument_attributes,
                args=(input_node,),
                kwargs={"argument_type": argument_type, "name": clean_name},
            )

        for user in users:
            # Replacing the input to an in-place copy_ op with a `tt.mark_argument_attributes` result
            # causes XLA to handle the copying into an input tensor incorrectly. So, we do not
            # replace the destination tensor with the `tt.mark_argument_attributes` result.
            if (
                user.target == torch.ops.aten.copy_.default
                and user.args[0] == input_node
            ):
                continue
            user.replace_input_with(input_node, new_input)

    _validate_demangling(gm, flat_name_to_original_fqn)

    return gm


def bypass_assert_tensor_metadata(gm):
    """
    Bypass assert_tensor_metadata nodes.
    This is a noop node that is used to assert the tensor metadata.
    This is used to remove these assertion ops as we may be preparing a GraphModule
    which was originally compiled on CPU. Thus, these assertions will assert
    that its input tensor is on CPU. When running the graph on an XLA device later,
    it would fail the assertion.
    """
    for node in gm.graph.nodes:
        if (
            node.op == "call_function"
            and node.target == torch.ops.aten._assert_tensor_metadata.default
        ):
            gm.graph.erase_node(node)
    return gm


def bypass_redundant_getitem(gm):
    """
    Replaces `getitem` calls with a direct reference to the tensor being retrieved.
    """
    for node in gm.graph.nodes:
        if node.op == "call_function" and "getitem" in node.name:
            if isinstance(node.args[0], tuple):
                idx = node.args[1]
                if isinstance(idx, int):
                    node.replace_all_uses_with(node.args[0][idx])
    return gm


def run_shape_prop(gm, example_inputs):
    """
    Propagates shape and dtype information through the graph.
    """
    shape_prop = torch.fx.passes.shape_prop.ShapeProp(gm)
    if shape_prop.fake_mode is not None:
        fake_args = [
            (
                shape_prop.fake_mode.from_tensor(act, static_shapes=True)
                if isinstance(act, torch.Tensor)
                else act
            )
            for act in example_inputs
        ]
    else:
        fake_args = example_inputs
    shape_prop.run(*fake_args)


def clamp_neg_slice_bounds(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """Clamp out-of-range negative ``aten.slice.Tensor`` start/end bounds to ``-dim_size``.

    CPU's ``Tensor.__getitem__`` clamps a bound ``< -dim_size`` to ``-dim_size``;
    XLA's strict ``aten.slice.Tensor`` instead raises "Value out of range". Only a
    constant negative start/end on a statically known dim is rewritten, so valid
    slices are untouched. Refs: tt-xla #5199.
    """
    changed = False
    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target is not torch.ops.aten.slice.Tensor:
            continue
        # aten.slice.Tensor(input, dim, start, end=None, step=1)
        if len(node.args) < 3:
            continue
        inp, dim = node.args[0], node.args[1]
        if not isinstance(inp, torch.fx.Node) or not isinstance(dim, int):
            continue
        shape = getattr(inp.meta.get("val"), "shape", None)
        if shape is None:
            shape = getattr(inp.meta.get("tensor_meta"), "shape", None)
        if shape is None:
            continue
        ndim = len(shape)
        d = dim + ndim if dim < 0 else dim
        if not (0 <= d < ndim):
            continue
        dim_size = shape[d]
        if not isinstance(dim_size, int):  # skip symbolic / dynamic dims
            continue
        # Clamp the start (args[2]) and end (args[3]) bounds independently.
        new_args = list(node.args)
        node_changed = False
        for i in (2, 3):
            bound = new_args[i] if i < len(new_args) else None
            if isinstance(bound, int) and bound < -dim_size:
                new_args[i] = -dim_size
                node_changed = True
        if node_changed:
            logger.debug(
                f"clamp_neg_slice_bounds: clamped {node.name} start/end "
                f"{tuple(node.args[2:4])} -> {tuple(new_args[2:4])} "
                f"(dim {dim}, size {dim_size})"
            )
            node.args = tuple(new_args)
            changed = True
    if changed:
        gm.graph.lint()
        gm.recompile()
    return gm


def clamp_neg_getitem_bounds(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """Pre-AOTAutograd clamp for OOB negative slices as ``operator.getitem`` nodes.

    Must run before AOTAutograd so functionalization records the clamped view-meta
    sequence; otherwise the runtime view-alias replay uses the original bound and
    raises "Value out of range". Companion to clamp_neg_slice_bounds. Refs: #4465.
    """
    changed = False
    for node in gm.graph.nodes:
        if (
            node.op != "call_function"
            or node.target is not operator.getitem
            or len(node.args) != 2
        ):
            continue
        inp, key = node.args
        if not isinstance(inp, torch.fx.Node):
            continue
        key_tuple = key if isinstance(key, tuple) else (key,)
        if not any(isinstance(k, slice) for k in key_tuple):
            continue
        shape = getattr(inp.meta.get("example_value"), "shape", None)
        if shape is None:
            shape = getattr(inp.meta.get("val"), "shape", None)
        if shape is None:
            continue
        new_key, node_changed = clamp_neg_slice_key(shape, key)
        if node_changed:
            logger.debug(
                f"clamp_neg_getitem_bounds: clamped {node.name} key {key} -> {new_key}"
            )
            node.args = (inp, new_key)
            changed = True
    if changed:
        gm.graph.lint()
        gm.recompile()
    return gm


def bypass_dtype_promotion_and_redundant_cast(gm, example_inputs):
    """
    Removes casting of nodes to float32 unless they were explicitly set by the user.
    Pytorch insists on casting nodes to float32 during decompositions, even though the
    user may have specified a different dtype.
    Also removes redundant casts.
    """
    removed_non_redundant_casts = False
    for node in gm.graph.nodes:
        if (
            node.op == "call_function"
            and hasattr(node.target, "name")
            and "prims::convert_element_type" in node.target.name()
        ):
            is_unwanted_dtype_promotion = (
                node.meta["original_aten"]._name != "aten::_to_copy"
                and node.args[1] == torch.float32
            )
            node_meta = node.args[0].meta
            meta_val = node_meta.get("tensor_meta") or node_meta.get("val")
            is_redundant_cast = meta_val is not None and meta_val.dtype == node.args[1]

            if is_unwanted_dtype_promotion or is_redundant_cast:
                node.replace_all_uses_with(node.args[0])
                removed_non_redundant_casts |= is_unwanted_dtype_promotion

    gm.graph.eliminate_dead_code()
    gm.graph.lint()

    if removed_non_redundant_casts:
        # if non redundant nodes were removed, re-propagate shape and dtype and re-run pass to remove redundant casts
        run_shape_prop(gm, example_inputs)
        gm = bypass_dtype_promotion_and_redundant_cast(gm, example_inputs)

    return gm


def _demangle_name(mangled_name, normalized_fqn_lookup):
    """Look up the original clean name (e.g., "classifier.6.weight") for a mangled
    FX node name by normalizing it and matching against the lookup table."""
    return normalized_fqn_lookup.get(_normalize_fx_name(mangled_name), mangled_name)


def _normalize_fx_name(name):
    """Normalize a mangled FX name to a canonical form for dictionary matching.

    FX mangles module attribute paths into flat names. The mangling rules and how
    this function handles each one:

    Example mangled key (from flat_name_to_original_fqn):
      "getattr_getattr_L__self___resampler_layers___0___ff___1___net___2___weight"

    Demangled value (original state_dict key):
      "resampler.layers.0.ff.1.net.2.weight"

    Rule 1: "getattr_" prefixes (one per __getattr__ call for nn.Sequential/ModuleList
            indexing). There can be multiple: ff[1].net[2] needs two __getattr__ calls,
            so the key gets "getattr_getattr_" prepended.
            -> strip ALL leading "getattr_" prefixes with a while loop.

    Rule 2: "L__self___" prefix means "self." (the root module reference).
            -> strip the "L__self___" prefix.

    Rule 3: "___N___" (triple underscores around digits) marks integer-indexed module
            access like .layers[0]. -> "___0___". This is unambiguous and structurally
            parseable, unlike single underscores.
            -> collapse all runs of underscores to single "_" via regex.

    Rule 4: Remaining single "_" replaces "." (dot separator between named attributes).
            This is LOSSY: "layer_norm1" in the mangled name could mean either
            "layer_norm1" (literal underscore) or "layer.norm1" (dot separator).
            -> replace "." with "_" so both sides use the same separator.
            -> The flat_name_to_original_fqn dictionary resolves this ambiguity.
    """
    s = str(name)
    # Strip all getattr_ prefixes (one per level of __getattr__ indexing)
    while s.startswith("getattr_"):
        s = s[len("getattr_") :]
    # Strip the self reference prefix
    if s.startswith("L__self___"):
        s = s[len("L__self___") :]
    # Replace dots with underscores (node targets use dots, dict keys use underscores)
    s = s.replace(".", "_")
    # Collapse runs of underscores (triple underscores around indices -> single)
    s = re.sub(r"_+", "_", s)
    return s


def _build_normalized_fqn_lookup(flat_name_to_original_fqn):
    """Build a lookup table mapping normalized FX names to original clean FQNs."""
    return {_normalize_fx_name(k): v for k, v in flat_name_to_original_fqn.items()}


def _validate_demangling(gm, flat_name_to_original_fqn):
    """Validate demangling results using structural knowledge of FX mangling.

    Checks that:
    - All parameter/constant names were successfully demangled (contain dots).
    - Integer indices marked by ___N___ in the mangled name appear as .N. in the
      demangled result.
    """

    if not flat_name_to_original_fqn:
        return

    unresolved = []
    mismatched = []
    for node in gm.graph.nodes:
        if (
            node.op != "call_function"
            or node.target != torch.ops.tt.mark_argument_attributes
        ):
            continue
        clean = str(node.kwargs.get("name", ""))
        arg_type = node.kwargs.get("argument_type", "")
        if arg_type not in ("parameter", "constant"):
            continue

        # A successfully demangled name always contains dots (e.g., "layers.0.weight").
        if "." not in clean:
            unresolved.append(clean)
            continue

        # Cross-check: integer indices marked by ___N___ in the mangled name
        # must appear as .N. (or .N at end) in the demangled name.
        mangled = str(node.args[0].target) if node.args else ""
        for match in re.finditer(r"___(\d+)___", mangled):
            idx = match.group(1)
            if f".{idx}." not in clean and not clean.endswith(f".{idx}"):
                mismatched.append((clean, mangled, idx))

    if unresolved:
        logger.debug(
            f"Failed to demangle {len(unresolved)} argument name(s): {unresolved}"
        )
    if mismatched:
        logger.debug(
            f"Demangled names inconsistent with mangled structure for "
            f"{len(mismatched)} argument(s): "
            + ", ".join(
                f"'{c}' (from '{m}', missing index {i})" for c, m, i in mismatched
            )
        )
