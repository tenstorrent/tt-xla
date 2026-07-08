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
                f"factor {name!r} in {keyword} list is not declared in "
                "factor_sizes"
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
    is_custom: bool = True,
) -> str:
    """Build the MLIR text of an ``sdy.op_sharding_rule`` attribute.

    The rule is constructed and printed by Shardy's own MLIR bindings
    (``jaxlib.mlir.dialects.sdy.OpShardingRuleAttr``), so the emitted text
    matches Shardy's canonical printer exactly and structural errors (e.g. a
    factor index out of range) surface at construction time. The returned
    string can be attached to a ``stablehlo.custom_call`` under the
    ``xla.sdy.sharding_rule`` frontend attribute; tt-mlir's
    ``register-custom-sharding-rule`` pass recognizes it and parses it back
    into a real ``sdy.op_sharding_rule`` handed to Shardy propagation.

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
        is_custom: Emit the ``custom`` marker. A custom rule is preserved
            forever by Shardy propagation and is required for user-defined
            rules on ``stablehlo.custom_call`` ops.

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
            is_custom=is_custom,
        )
        return str(rule)
