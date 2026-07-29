# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Builder for Shardy ``sdy.op_sharding_rule`` attributes.

This module is intentionally framework-neutral (it does not import torch),
so it can be used from either the JAX or the PyTorch frontend. The rule is
constructed and printed by Shardy itself via its MLIR Python bindings, which
ship inside jaxlib at ``jaxlib.mlir.dialects.sdy`` and are a guaranteed
dependency of the ``pjrt_plugin_tt`` wheel (see ``requirements.txt``).

Example:
    >>> from tt_torch import make_sharding_rule
    >>> make_sharding_rule(
    ...     operand_mappings=[("B", "M"), ("M", "N")],
    ...     result_mappings=[("B", "N")],
    ...     factor_sizes={"B": 8, "M": 16, "N": 32},
    ...     reduction_factors=["M"],
    ... )
    '#sdy.op_sharding_rule<([i, j], [j, k])->([i, k]) {i=8, j=16, k=32} reduction={j}, custom>'
"""

from typing import Iterable, Mapping, Sequence


def _import_sdy_bindings():
    """Import Shardy's MLIR Python bindings from jaxlib.

    These ship inside jaxlib at ``jaxlib.mlir.dialects.sdy`` (JAX's packaging
    of the upstream Shardy/StableHLO MLIR bindings). jaxlib is a hard
    dependency of the ``pjrt_plugin_tt`` wheel, so this import is expected to
    succeed in any environment where the plugin is installed; the import is
    done lazily only to keep ``import tt_torch`` from eagerly pulling in jax.
    """
    from jaxlib.mlir import ir
    from jaxlib.mlir.dialects import sdy

    return ir, sdy


def _factor_index_map(
    operand_mappings: Sequence[Sequence[str]],
    result_mappings: Sequence[Sequence[str]],
    factor_sizes: Mapping[str, int],
) -> "dict[str, int]":
    """Assign each factor name a canonical shardy index by first-appearance order.

    Factor names not seen in any mapping are appended at the end so that
    ``factor_sizes`` may declare a size-1 factor that is never used in a
    tensor mapping (e.g. a broadcast axis kept for documentation).
    """
    order: list[str] = []
    seen: set[str] = set()

    def visit(name: str) -> None:
        if name in seen:
            return
        if name not in factor_sizes:
            raise ValueError(
                f"factor {name!r} used in a tensor mapping but not present "
                f"in factor_sizes {sorted(factor_sizes)!r}"
            )
        seen.add(name)
        order.append(name)

    for mapping in operand_mappings:
        for name in mapping:
            visit(name)
    for mapping in result_mappings:
        for name in mapping:
            visit(name)
    for name in factor_sizes:
        visit(name)

    return {name: index for index, name in enumerate(order)}


def _factor_indices(
    keyword: str,
    factors: Iterable[str],
    name_to_index: Mapping[str, int],
) -> "list[int]":
    """Validate a factor-name list and map it to canonical shardy indices."""
    indices: list[int] = []
    seen: set[str] = set()
    for name in factors:
        if name in seen:
            raise ValueError(f"duplicate factor {name!r} in {keyword} list")
        if name not in name_to_index:
            raise ValueError(
                f"factor {name!r} in {keyword} list is not declared in " "factor_sizes"
            )
        seen.add(name)
        indices.append(name_to_index[name])
    return indices


def make_sharding_rule(
    operand_mappings: Sequence[Sequence[str]],
    result_mappings: Sequence[Sequence[str]],
    factor_sizes: Mapping[str, int],
    *,
    reduction_factors: Iterable[str] = (),
    need_replication_factors: Iterable[str] = (),
    permutation_factors: Iterable[str] = (),
    blocked_propagation_factors: Iterable[str] = (),
) -> str:
    """Build the MLIR text of an ``sdy.op_sharding_rule`` attribute.

    The rule is constructed and printed by Shardy's own MLIR bindings
    (``jaxlib.mlir.dialects.sdy.OpShardingRuleAttr``), so the emitted text
    matches Shardy's canonical printer exactly and structural errors (e.g. a
    factor index out of range) surface at construction time. The returned
    string can be attached to a ``stablehlo.custom_call`` under the
    ``xla.sdy.custom_sharding_rule`` frontend attribute; tt-mlir's
    ``register-user-sharding-rule`` pass recognizes it and parses it back
    into a real ``sdy.op_sharding_rule`` handed to Shardy propagation.

    Every rule is emitted with the ``custom`` marker so Shardy preserves it
    through propagation; that is required for user-defined rules on
    ``stablehlo.custom_call`` ops.

    Args:
        operand_mappings: One sequence per operand, each listing the factor
            names covering the operand's dims in order. Use ``()`` for a
            scalar tensor.
        result_mappings: Same, but for results.
        factor_sizes: Maps each factor name to its size. All names used in
            mappings must be present. Names may be any strings; they are
            renamed to the shardy canonical ``i, j, k, ...`` sequence in
            the emitted text based on first-appearance order.
        reduction_factors: Factor names requiring reduction (e.g. matmul
            contracting dims).
        need_replication_factors: Factor names requiring full replication.
        permutation_factors: Factor names requiring a collective-permute
            when sharded.
        blocked_propagation_factors: Factor names along which shardings
            must not be propagated.

    Returns:
        A string of the form
        ``"#sdy.op_sharding_rule<([i, j], [j, k])->([i, k]) {i=8, j=16, k=32}"``
        ``" reduction={j}, custom>"`` (with fields omitted when empty).

    Example:
        >>> make_sharding_rule(
        ...     operand_mappings=[("B", "M"), ("M", "N")],
        ...     result_mappings=[("B", "N")],
        ...     factor_sizes={"B": 8, "M": 16, "N": 32},
        ...     reduction_factors=["M"],
        ... )
        '#sdy.op_sharding_rule<([i, j], [j, k])->([i, k]) {i=8, j=16, k=32} reduction={j}, custom>'
    """
    for name, size in factor_sizes.items():
        if not isinstance(size, int) or size <= 0:
            raise ValueError(
                f"factor_sizes[{name!r}] must be a positive int, got {size!r}"
            )

    ir, sdy = _import_sdy_bindings()

    name_to_index = _factor_index_map(operand_mappings, result_mappings, factor_sizes)
    ordered_names = sorted(name_to_index, key=name_to_index.get)
    factor_sizes_list = [factor_sizes[name] for name in ordered_names]

    with ir.Context() as ctx, ir.Location.unknown():
        sdy.register_dialect(ctx)

        def tensor_mapping(mapping: Sequence[str]):
            # Each entry names the single factor covering one tensor dim, so
            # every dim maps to exactly one factor index (one-factor-per-dim).
            dim_mappings = [
                sdy.DimMappingAttr.get([name_to_index[name]]) for name in mapping
            ]
            return sdy.TensorMappingAttr.get(dim_mappings)

        rule = sdy.OpShardingRuleAttr.get(
            factor_sizes=factor_sizes_list,
            operand_mappings=[tensor_mapping(m) for m in operand_mappings],
            result_mappings=[tensor_mapping(m) for m in result_mappings],
            reduction_factors=_factor_indices(
                "reduction", reduction_factors, name_to_index
            ),
            need_replication_factors=_factor_indices(
                "need_replication", need_replication_factors, name_to_index
            ),
            permutation_factors=_factor_indices(
                "permutation", permutation_factors, name_to_index
            ),
            blocked_propagation_factors=_factor_indices(
                "blocked_propagation", blocked_propagation_factors, name_to_index
            ),
            is_custom=True,
        )
        return str(rule)


def make_fully_replicated_sharding_rule(
    operand_shapes: Sequence[Sequence[int]],
    *,
    out_indices: Sequence[int],
) -> str:
    """Build an explicit full-replication ``sdy.op_sharding_rule`` for an op.

    Every operand/result dimension is mapped to a factor listed in
    ``need_replication_factors``, so Shardy must keep all tensors fully
    replicated. Operand shapes that are identical reuse the same factor
    names (pointwise-friendly).

    The SHLO ``tt.tt_lang_op`` custom_call is functional (``in`` tensors
    only); DPS ``out`` buffers are reintroduced later in StableHLO→TTIR.
    This rule therefore maps only the ``in``-tagged shapes as operands and
    the ``out``-tagged shapes as results.

    This is the default rule emitted for tt-lang ops when the caller does
    not supply a ``sharding_rule``, so tt-mlir always sees an explicit
    rule rather than having to special-case a missing attribute.
    """
    if not operand_shapes:
        raise ValueError("operand_shapes must contain at least one shape")
    if not out_indices:
        raise ValueError("out_indices must contain at least one index")
    out_set = set(out_indices)
    for idx in out_indices:
        if not 0 <= idx < len(operand_shapes):
            raise ValueError(
                f"out index {idx} out of range for {len(operand_shapes)} operands"
            )

    shape_to_mapping: dict[tuple[int, ...], tuple[str, ...]] = {}
    factor_sizes: dict[str, int] = {}
    all_factors: list[str] = []
    counter = 0

    def mapping_for(shape: Sequence[int]) -> tuple[str, ...]:
        nonlocal counter
        key = tuple(int(dim) for dim in shape)
        cached = shape_to_mapping.get(key)
        if cached is not None:
            return cached
        names: list[str] = []
        for dim in key:
            name = f"f{counter}"
            counter += 1
            factor_sizes[name] = dim
            all_factors.append(name)
            names.append(name)
        mapping = tuple(names)
        shape_to_mapping[key] = mapping
        return mapping

    in_mappings = [
        mapping_for(shape) for i, shape in enumerate(operand_shapes) if i not in out_set
    ]
    if not in_mappings:
        raise ValueError(
            "make_fully_replicated_sharding_rule requires at least one "
            "'in' operand shape; pure-out ops are not supported on the "
            "functional SHLO path."
        )
    result_mappings = [mapping_for(operand_shapes[i]) for i in out_indices]
    return make_sharding_rule(
        operand_mappings=in_mappings,
        result_mappings=result_mappings,
        factor_sizes=factor_sizes,
        need_replication_factors=all_factors,
    )
