# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Central registry of all fusion pattern providers.

All fusion provider classes are defined in this file.
"""

from abc import ABC, abstractmethod
from typing import Callable, List, Type

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.fx.subgraph_rewriter import replace_pattern_with_filters
from ttxla_tools.logging import logger


class FusionProvider(ABC):
    """Base class for all fusion pattern providers.

    Subclasses are automatically registered via __init_subclass__.

    To create a new fusion provider:
    1. Inherit from FusionProvider
    2. Implement the `name` property
    3. Implement the `pattern` static method (the pattern to match)
    4. Implement the `replacement` static method (the replacement function)

    For providers with multiple pattern variants, override `get_patterns`
    to return a list of (pattern, replacement) tuples instead.

    Optional:
    5. Implement the `match_filter` method (single match filter to apply to the pattern)
    or alternatively,
    Implement the `get_match_filters` method (list of match filters to apply to the pattern)
    """

    _registered_providers: List[Type["FusionProvider"]] = []
    default_enabled: bool = True

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        FusionProvider._registered_providers.append(cls)

    @classmethod
    def get_registered_providers(
        cls,
        provider_names: List[str] | None = None,
        include_default_disabled: bool = False,
    ) -> List[Type["FusionProvider"]]:
        """Return registered providers filtered by name and default-enabled state."""
        selected_names = set(provider_names) if provider_names is not None else None
        selected: List[Type["FusionProvider"]] = []

        for provider_cls in cls._registered_providers:
            provider = provider_cls()
            if not include_default_disabled and not provider_cls.default_enabled:
                continue
            if selected_names is not None and provider.name not in selected_names:
                continue
            selected.append(provider_cls)

        return selected

    @classmethod
    def get_registered_provider_names(
        cls, include_default_disabled: bool = False
    ) -> List[str]:
        return [
            provider_cls().name
            for provider_cls in cls.get_registered_providers(
                include_default_disabled=include_default_disabled
            )
        ]

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this provider."""
        pass

    @staticmethod
    @abstractmethod
    def pattern(*args, **kwargs) -> Tensor:
        """The pattern to match in the graph."""
        pass

    @staticmethod
    @abstractmethod
    def replacement(*args, **kwargs) -> Tensor:
        """The replacement function."""
        pass

    @staticmethod
    def match_filter(*args, **kwargs) -> bool:
        """The match filter to apply to the pattern."""
        return True

    def get_match_filters(self) -> List[Callable]:
        """Return the match filters for the provider."""
        return [self.match_filter]

    def get_patterns(self) -> List[tuple]:
        """Return (pattern, replacement) pairs. Override for multi-pattern providers."""
        return [(self.pattern, self.replacement)]

    def replace_pattern(self, gm: torch.fx.GraphModule) -> int:
        """
        Replace patterns in the graph.

        Iterates over all (pattern, replacement) pairs from get_patterns().

        Args:
            gm: The GraphModule to transform

        Returns:
            Number of replacements made
        """
        total = 0
        for pattern, replacement in self.get_patterns():
            replaced = replace_pattern_with_filters(
                gm,
                pattern,
                replacement,
                match_filters=self.get_match_filters(),
            )
            total += len(replaced)
        return total


# ================================ Fusion Providers ================================


class RMSNormFusionProvider(FusionProvider):
    """
    Provides fusion patterns for RMS Normalization operations.

    Matches patterns like LlamaRMSNorm (common case):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + epsilon)
        return weight * hidden_states.to(input_dtype)

    There is also a GPT-OSS variant where the cast happens after multiply with weight:
        return (weight * hidden_states).to(input_dtype)

    Both are replaced with torch.nn.functional.rms_norm.
    """

    @property
    def name(self) -> str:
        return "rms_norm_fusion"

    @staticmethod
    def pattern(hidden_states: Tensor, weight: Tensor, eps: float, dtype) -> Tensor:
        """
        Llama variant: cast happens before multiply with weight.

        Matches: weight * hidden_states.to(input_dtype)

        Note:
            Uses method calls (.add(), .mul()) instead of operators (+, *)
            because dynamo traces tensor operations as call_method, not call_function.

            The dtype parameter allows matching any dtype variant, it becomes a
            wildcard in the pattern graph that matches any value.
        """
        hidden_fp32 = hidden_states.to(torch.float32)
        variance = hidden_fp32.pow(2).mean(-1, keepdim=True)
        variance_eps = variance.add(eps)
        rsqrt_var = torch.rsqrt(variance_eps)
        hidden_normalized = hidden_fp32.mul(rsqrt_var)
        hidden_cast = hidden_normalized.to(dtype)
        return weight.mul(hidden_cast)

    @staticmethod
    def pattern_cast_after_mul(
        hidden_states: Tensor, weight: Tensor, eps: float, dtype
    ) -> Tensor:
        """
        GPT-OSS variant: cast happens after multiply with weight.

        Matches: (weight * hidden_states).to(input_dtype)
        """
        hidden_fp32 = hidden_states.to(torch.float32)
        variance = hidden_fp32.pow(2).mean(-1, keepdim=True)
        variance_eps = variance.add(eps)
        rsqrt_var = torch.rsqrt(variance_eps)
        hidden_normalized = hidden_fp32.mul(rsqrt_var)
        result = weight.mul(hidden_normalized)
        return result.to(dtype)

    @staticmethod
    def replacement(hidden_states: Tensor, weight: Tensor, eps: float, dtype) -> Tensor:
        """Shared replacement for both RMS norm pattern variants."""
        return torch.nn.functional.rms_norm(
            hidden_states, normalized_shape=weight.shape, weight=weight, eps=eps
        )

    def get_patterns(self) -> List[tuple]:
        return [
            (self.pattern, self.replacement),
            (self.pattern_cast_after_mul, self.replacement),
        ]


class ResidualRMSNormFusionProvider(FusionProvider):
    """Fold `add(x, y) → rms_norm(...)` into a single residual-aware rms_norm.

    Direct evidence in Llama-3.2-1B prefill MLIR shows 32 unfused
    `ttnn.add → ttnn.rms_norm` pairs per 16-layer model. Each is one
    extra kernel launch + DRAM round-trip per layer that this fusion
    eliminates by routing the residual through ttnn::rms_norm's existing
    `residual_input_tensor` parameter.

    **Why we override `replace_pattern` instead of using subgraph_rewriter.**
    In Llama the residual add `hidden_states = residual + attn_output` has
    TWO downstream users: the rms_norm we want to fuse into AND the *next*
    residual add at end-of-MLP. FX's `replace_pattern_with_filters` uses
    `SubgraphMatcher`, which rejects matches where any internal node has
    users outside the matched subgraph. Result: zero matches in real
    transformer code. So we walk the graph manually — find rms_norm nodes
    whose first arg is an add, rewrite the rms_norm in place, and leave
    the add alone (its other users keep working).

    Must run AFTER RMSNormFusionProvider (which collapses HF's manual
    rms_norm decomposition into a `torch.nn.functional.rms_norm` call).
    Default-disabled; invoked via the `late` rewrite hook in backend.py.

    The rewritten call site uses _torch_residual_rms_norm (a `@torch.fx.wrap`'d
    marker) so handle_composite_ops can swap it to _composite_residual_rms_norm
    which emits the `tenstorrent.rms_norm` composite with `has_residual=True`.
    """

    default_enabled = False

    @property
    def name(self) -> str:
        return "fuse_residual_rms_norm"

    @staticmethod
    def pattern(*args, **kwargs):
        # Unused — base class abstract method shim. We override replace_pattern.
        raise NotImplementedError

    @staticmethod
    def replacement(*args, **kwargs):
        raise NotImplementedError

    def replace_pattern(self, gm: torch.fx.GraphModule) -> int:
        """Walk the FX graph; for each `torch.nn.functional.rms_norm` whose
        first arg is an `add` (call_method 'add' OR call_function operator.add),
        rewrite the rms_norm node to call `_torch_residual_rms_norm(x, r, ...)`
        in place. Leave the `add` node alone — it may have other users."""
        from tt_torch.composite_ops import _torch_residual_rms_norm

        replaced = 0
        for node in list(gm.graph.nodes):
            if node.op != "call_function":
                continue
            if node.target is not torch.nn.functional.rms_norm:
                continue
            # First positional arg is the activation tensor.
            if not node.args:
                continue
            input_node = node.args[0]
            if not isinstance(input_node, torch.fx.Node):
                continue
            # Detect both call_method (dynamo) and call_function (symbolic_trace) add forms.
            is_method_add = input_node.op == "call_method" and input_node.target == "add"
            import operator

            is_func_add = (
                input_node.op == "call_function" and input_node.target is operator.add
            )
            if not (is_method_add or is_func_add):
                continue
            if len(input_node.args) != 2:
                continue
            x, residual = input_node.args
            # Build call args for the marker:
            # _torch_residual_rms_norm(input, residual, normalized_shape, weight=, eps=)
            normalized_shape = node.args[1] if len(node.args) > 1 else None
            weight = node.kwargs.get("weight")
            eps = node.kwargs.get("eps")
            with gm.graph.inserting_before(node):
                new_node = gm.graph.call_function(
                    _torch_residual_rms_norm,
                    args=(x, residual, normalized_shape, weight, eps),
                )
            node.replace_all_uses_with(new_node)
            gm.graph.erase_node(node)
            replaced += 1

        if replaced:
            gm.graph.lint()
            gm.recompile()
        return replaced


class SwiGLUFusionProvider(FusionProvider):
    """Fold the slice→silu→slice→mul SwiGLU pattern into a single
    ``_torch_swiglu(x, dim)`` marker call.

    Pattern (along ``dim`` = ``-1``):

        half = x.shape[-1] // 2
        gate = x[..., :half]
        up   = x[..., half:]
        return F.silu(gate) * up

    handle_composite_ops then swaps ``_torch_swiglu`` for the
    composite-emitting wrapper, producing a ``tenstorrent.swiglu`` StableHLO
    composite carrying ``dim`` as an attribute. tt-mlir's
    StableHLOLegalizeCompositePass collapses that into a single
    ttir.gated_activation op (activation = ``swiglu``).

    **Why we walk the graph manually instead of using
    `subgraph_rewriter.replace_pattern_with_filters`.** Two reasons:

    1. Dynamo and symbolic_trace disagree on the trace of the slice. Dynamo
       lowers ``x[..., :half]`` to ``aten.slice``-style call_functions;
       symbolic_trace produces ``operator.getitem`` with a tuple
       ``(Ellipsis, slice(...))`` argument. ``narrow`` is yet a third form
       (``call_method[narrow]``). A single template-based pattern can't
       cover all three; per LANDING_A_FUSION.md we'd need at least one
       pattern per surface form, and SubgraphMatcher is sensitive to
       positional-vs-kwargs and node-kind structural differences.

    2. The mul operand order isn't fixed. Both ``silu(gate) * up`` and
       ``up * silu(gate)`` are valid SwiGLU spellings; we want to match
       both. Walking manually lets us check both operand positions in one
       place.

    Default-disabled; gated through the early ``QuetzalRewrite`` hook (it
    runs BEFORE ``run_fusion_passes``, but for SwiGLU there's no upstream
    provider whose output we depend on, so either hook works in principle —
    we use the early hook since it pairs naturally with ``fuse_gelu`` and
    ``reconstruct_sdpa`` which are also slice/elementwise rewrites).

    For now this provider scopes to SiLU (SwiGLU) only. The sibling
    variants (GLU/GeGLU/ReGLU) reuse the same composite plumbing —
    ``_torch_glu`` / ``_torch_geglu`` / ``_torch_reglu`` markers are
    registered in composite_ops.replacements — but discovery providers are
    a TODO unless model evidence motivates them.
    """

    default_enabled = False

    @property
    def name(self) -> str:
        return "fuse_swiglu"

    @staticmethod
    def pattern(*args, **kwargs):
        # Unused — base class abstract method shim. We override replace_pattern.
        raise NotImplementedError

    @staticmethod
    def replacement(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _is_mul(node: torch.fx.Node) -> bool:
        """True if node is an elementwise mul (function or method form)."""
        import operator as _op

        if node.op == "call_function" and node.target is _op.mul:
            return True
        if node.op == "call_method" and node.target == "mul":
            return True
        return False

    @staticmethod
    def _is_silu(node: torch.fx.Node) -> bool:
        """True if node is an F.silu call (function form). silu has no
        common method spelling on Tensor, so we only check call_function."""
        if node.op == "call_function" and node.target is torch.nn.functional.silu:
            return True
        return False

    @staticmethod
    def _slice_descriptor(node: torch.fx.Node):
        """If ``node`` is a half-slice of a tensor along some dim, return
        ``(source, start, length, dim, kind)``. Otherwise return ``None``.

        ``start`` and ``length`` may be Python ints OR FX Nodes — symbolic
        shapes (``half = x.shape[-1] // 2``) trace through as call_function
        ``operator.floordiv`` nodes that get plumbed into ``narrow`` /
        ``slice`` arguments. We deliberately preserve those node references
        so the caller can compare the gate-half and up-half by identity
        without trying to constant-fold.

        ``kind`` is ``"first"`` (start is a literal 0; length is the
        half-symbol) or ``"second"`` (start is the half-symbol; length is
        the half-symbol or absent).

        Recognized surface forms:

        * ``x.narrow(dim, start, length)`` — call_method[narrow] (the form
          ``torch.fx.symbolic_trace`` produces for ``Tensor.narrow``).
          We tolerate length being either an int or an FX node.
        * ``x[..., start:stop]`` — call_function[operator.getitem] with a
          tuple key whose last entry is a ``slice``. The slice's
          ``start``/``stop`` may be FX nodes (the symbolic half).
        """
        import operator as _op

        # narrow form: x.narrow(dim, start, length).
        is_method_narrow = node.op == "call_method" and node.target == "narrow"
        is_func_narrow = (
            node.op == "call_function"
            and getattr(node.target, "__name__", None) == "narrow"
        )
        if is_method_narrow or is_func_narrow:
            if len(node.args) != 4:
                return None
            source, dim_arg, start_arg, length_arg = node.args
            if not isinstance(source, torch.fx.Node):
                return None
            if not isinstance(dim_arg, int):
                return None
            # gate: start == int 0, length is the half symbol (int OR Node).
            # up: start is the half symbol, length is also the half symbol.
            if isinstance(start_arg, int) and start_arg == 0:
                # First-half candidate. length_arg is the half symbol.
                return (source, 0, length_arg, dim_arg, "first")
            if isinstance(start_arg, torch.fx.Node) or isinstance(start_arg, int):
                # Second-half candidate. start_arg is the half symbol; we
                # carry length_arg (also the half symbol) so the caller can
                # confirm gate.length == up.length.
                return (source, start_arg, length_arg, dim_arg, "second")
            return None

        # getitem form: x[..., :half] or x[..., half:].
        if node.op == "call_function" and node.target is _op.getitem:
            if len(node.args) != 2:
                return None
            source, key = node.args
            if not isinstance(source, torch.fx.Node):
                return None
            if not isinstance(key, tuple) or len(key) == 0:
                return None
            last = key[-1]
            if not isinstance(last, slice):
                return None
            if last.step not in (None, 1):
                return None
            # Confirm leading entries are Ellipsis or full slice(None) — we
            # only support slicing the trailing dim.
            for entry in key[:-1]:
                if entry is Ellipsis:
                    continue
                if isinstance(entry, slice) and entry == slice(None):
                    continue
                return None
            # First-half: start is None (interpreted as 0); stop is the half symbol.
            # Second-half: start is the half symbol; stop is None.
            if last.start is None and last.stop is not None:
                return (source, 0, last.stop, -1, "first")
            if last.start is not None and last.stop is None:
                # length is implicit (until end of dim) — we don't have a
                # symbol for it. Mark it absent (None) so the matcher can
                # match against the gate's length symbol but not require it.
                return (source, last.start, None, -1, "second")
            return None

        return None

    def replace_pattern(self, gm: torch.fx.GraphModule) -> int:
        """Walk the FX graph; for each elementwise ``mul`` whose two inputs
        are ``silu(slice_a(x))`` and ``slice_b(x)`` from the SAME source
        tensor, with ``slice_a`` covering the first half and ``slice_b``
        the second half along the same dim, rewrite to a single
        ``_torch_swiglu(x, dim=...)`` marker call.

        Both operand orders (``silu(gate) * up`` and ``up * silu(gate)``)
        are accepted. The leftover slice/silu nodes are erased only if
        they have no other users — be conservative, since SwiGLU's halves
        are produced by the same upstream tensor and we don't want to
        accidentally orphan a downstream use.
        """
        from tt_torch.composite_ops import _torch_swiglu

        replaced = 0
        for node in list(gm.graph.nodes):
            if not self._is_mul(node):
                continue
            # Pull the two operands. call_method mul has args = (self, other);
            # call_function operator.mul has args = (lhs, rhs). Either way
            # the first two args are the operands.
            if len(node.args) < 2:
                continue
            lhs, rhs = node.args[0], node.args[1]
            if not (isinstance(lhs, torch.fx.Node) and isinstance(rhs, torch.fx.Node)):
                continue

            # Try both orderings: (silu_side, up_side) ∈ {(lhs,rhs), (rhs,lhs)}.
            for silu_side, up_side in ((lhs, rhs), (rhs, lhs)):
                if not self._is_silu(silu_side):
                    continue
                if not silu_side.args:
                    continue
                gate_slice_node = silu_side.args[0]
                if not isinstance(gate_slice_node, torch.fx.Node):
                    continue
                gate_desc = self._slice_descriptor(gate_slice_node)
                up_desc = self._slice_descriptor(up_side)
                if gate_desc is None or up_desc is None:
                    continue
                gate_src, gate_start, gate_length, gate_dim, gate_kind = gate_desc
                up_src, up_start, up_length, up_dim, up_kind = up_desc
                if gate_src is not up_src:
                    continue
                if gate_dim != up_dim:
                    continue
                # gate must be the FIRST half (start == 0), up must be the
                # SECOND half (start == half-symbol). Both halves must
                # reference the SAME half-symbol — comparing by Python `is`
                # / `==` on FX Nodes (identity) catches the case where
                # `narrow(x, dim, 0, H)` and `narrow(x, dim, H, H)` use the
                # same `floordiv` node for ``H``.
                if gate_kind != "first" or up_kind != "second":
                    continue
                if gate_start != 0:
                    continue
                # gate_length is the half symbol (int or FX Node).
                # up_start must equal it.
                if gate_length is None:
                    continue
                if up_start is not gate_length and up_start != gate_length:
                    continue
                # up_length, if present, must also equal the half symbol
                # (the narrow form provides it; the getitem ``x[..., H:]``
                # form leaves it as None and that's fine — slicing to
                # end-of-dim is consistent with SwiGLU).
                if up_length is not None and up_length is not gate_length and (
                    up_length != gate_length
                ):
                    continue

                # Rewrite the mul node into a _torch_swiglu(source, dim) call.
                with gm.graph.inserting_before(node):
                    new_node = gm.graph.call_function(
                        _torch_swiglu,
                        args=(gate_src,),
                        kwargs={"dim": gate_dim},
                    )
                node.replace_all_uses_with(new_node)
                gm.graph.erase_node(node)
                # Try to clean up the silu + slice nodes if they're now
                # orphaned. Don't force-erase; another consumer may exist.
                for orphan in (silu_side, gate_slice_node, up_side):
                    if isinstance(orphan, torch.fx.Node) and len(orphan.users) == 0:
                        try:
                            gm.graph.erase_node(orphan)
                        except RuntimeError:
                            # Node still has users despite the count check —
                            # fine, leave it for DCE.
                            pass
                replaced += 1
                break  # done with this mul node

        if replaced:
            gm.graph.lint()
            gm.recompile()
        return replaced


class QuetzalFuseGELUProvider(FusionProvider):
    """Collapse tanh-GELU decompositions back to torch.nn.functional.gelu."""

    default_enabled = False

    @property
    def name(self) -> str:
        return "fuse_gelu"

    @staticmethod
    def pattern(x: Tensor) -> Tensor:
        return QuetzalFuseGELUProvider.pattern_operator(x)

    @staticmethod
    def pattern_method(x: Tensor) -> Tensor:
        half_x = x.mul(0.5)
        x_pow_3 = x.pow(3.0)
        cubic_term = x_pow_3.mul(0.044715)
        inner = x.add(cubic_term)
        tanh_input = inner.mul(0.7978845608028654)
        tanh_output = torch.tanh(tanh_input)
        return half_x.mul(tanh_output.add(1.0))

    @staticmethod
    def pattern_method_function_pow(x: Tensor) -> Tensor:
        # Dynamo traces binary operators (* +) as call_method on a Tensor Proxy,
        # but leaves named torch.* calls (torch.pow, torch.tanh) as call_function.
        # HF transformers' NewGELUActivation (GPT-2 et al) uses `0.5 * x` and
        # `torch.pow(x, 3.0)` — so after dynamo the graph has call_method mul/add
        # plus a call_function torch.pow. Symbolic_trace of `x.pow(3.0)` would
        # instead produce a call_method pow, which is why the other method-form
        # pattern does NOT match a dynamo-captured GELU subgraph.
        half_x = x.mul(0.5)
        x_pow_3 = torch.pow(x, 3.0)
        cubic_term = x_pow_3.mul(0.044715)
        inner = x.add(cubic_term)
        tanh_input = inner.mul(0.7978845608028654)
        tanh_output = torch.tanh(tanh_input)
        return half_x.mul(tanh_output.add(1.0))

    @staticmethod
    def pattern_operator(x: Tensor) -> Tensor:
        return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * (x + 0.044715 * x**3)))

    @staticmethod
    def pattern_operator_method_pow(x: Tensor) -> Tensor:
        return 0.5 * x * (
            1.0 + torch.tanh(0.7978845608028654 * (x + 0.044715 * x.pow(3.0)))
        )

    @staticmethod
    def replacement(x: Tensor) -> Tensor:
        return F.gelu(x, approximate="tanh")

    def get_patterns(self) -> List[tuple]:
        return [
            (self.pattern_method, self.replacement),
            (self.pattern_method_function_pow, self.replacement),
            (self.pattern_operator, self.replacement),
            (self.pattern_operator_method_pow, self.replacement),
        ]


class QuetzalReconstructSDPAProvider(FusionProvider):
    """Reconstruct SDPA from manual matmul/softmax/matmul attention."""

    default_enabled = False

    @property
    def name(self) -> str:
        return "reconstruct_sdpa"

    @staticmethod
    def pattern(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        return QuetzalReconstructSDPAProvider.pattern_unscaled_operator(
            query, key, value
        )

    @staticmethod
    def pattern_scaled_method(
        query: Tensor, key: Tensor, value: Tensor, scale: float
    ) -> Tensor:
        key_t = key.transpose(-2, -1)
        scores = torch.matmul(query, key_t)
        scaled_scores = scores.mul(scale)
        weights = torch.softmax(scaled_scores, dim=-1)
        return torch.matmul(weights, value)

    @staticmethod
    def pattern_scaled_operator(
        query: Tensor, key: Tensor, value: Tensor, scale: float
    ) -> Tensor:
        return torch.softmax((query @ key.transpose(-2, -1)) * scale, dim=-1) @ value

    @staticmethod
    def pattern_unscaled_method(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        key_t = key.transpose(-2, -1)
        scores = torch.matmul(query, key_t)
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, value)

    @staticmethod
    def pattern_unscaled_operator(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        return torch.softmax(query @ key.transpose(-2, -1), dim=-1) @ value

    @staticmethod
    def replacement_scaled(
        query: Tensor, key: Tensor, value: Tensor, scale: float
    ) -> Tensor:
        return F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False, scale=scale
        )

    @staticmethod
    def replacement_unscaled(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        return F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False
        )

    @staticmethod
    def replacement(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        return QuetzalReconstructSDPAProvider.replacement_unscaled(query, key, value)

    def get_patterns(self) -> List[tuple]:
        return [
            (self.pattern_scaled_method, self.replacement_scaled),
            (self.pattern_scaled_operator, self.replacement_scaled),
            (self.pattern_unscaled_method, self.replacement_unscaled),
            (self.pattern_unscaled_operator, self.replacement_unscaled),
        ]
