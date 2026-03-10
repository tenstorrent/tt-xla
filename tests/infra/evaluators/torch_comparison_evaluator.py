# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
from infra.runners import run_on_cpu
from infra.utilities import Framework, PyTree
from torch.utils._pytree import tree_flatten, tree_map
from transformers import Cache, DynamicCache, EncoderDecoderCache

from .comparison_evaluator import ComparisonEvaluator
from .evaluation_config import AllcloseConfig, AtolConfig, PccConfig


class TorchComparisonEvaluator(ComparisonEvaluator):
    """ComparisonEvaluator for Torch tensors/pytrees."""

    # @override
    def _is_single_element(self, tensor: PyTree) -> bool:
        """Returns True if the tensor has only a single element."""
        if isinstance(tensor, torch.Tensor):
            return tensor.numel() == 1
        # For pytrees, check if all leaves are single-element
        leaves, _ = tree_flatten(tensor)
        return all(
            leaf.numel() == 1 for leaf in leaves if isinstance(leaf, torch.Tensor)
        )

    @staticmethod
    def _cache_to_legacy(cache: Cache) -> tuple:
        """
        Convert a Cache (DynamicCache, StaticCache, etc.) to legacy tuple of
        (key, value) tensors per layer for comparison.
        """
        if hasattr(cache, "to_legacy_cache"):
            return cache.to_legacy_cache()
        # Fallback for any Cache with .layers (DynamicCache, StaticCache in transformers 5.x,
        # and any other Cache subclass). Each layer exposes .keys and .values via CacheLayerMixin.
        if hasattr(cache, "layers"):
            return tuple((layer.keys, layer.values) for layer in cache.layers)
        # Handle EncoderDecoderCache (transformers 5.x): contains self_attention_cache and
        # cross_attention_cache as separate DynamicCache objects instead of .layers directly.
        if hasattr(cache, "self_attention_cache") and hasattr(
            cache, "cross_attention_cache"
        ):
            return (
                TorchComparisonEvaluator._cache_to_legacy(cache.self_attention_cache),
                TorchComparisonEvaluator._cache_to_legacy(cache.cross_attention_cache),
            )
        return cache

    # @override
    @staticmethod
    @run_on_cpu(Framework.TORCH)
    def _match_data_types(tensors: PyTree) -> PyTree:
        def match(tensor):
            if isinstance(tensor, torch.Tensor) and tensor.dtype != torch.float64:
                tensor = tensor.to(torch.float64)
            return tensor

        def convert_and_match(tensor):
            if isinstance(tensor, Cache):
                # New transformers library uses Cache classes (DynamicCache, StaticCache)
                # with CacheLayers/StaticLayers instead of raw tensors. Convert to legacy
                # (keys, values) tuple per layer so the comparator can compare tensors.
                tensor = TorchComparisonEvaluator._cache_to_legacy(tensor)
            if isinstance(tensor, torch.Tensor) and tensor.dtype != torch.float64:
                tensor = tensor.to(torch.float64)
            return tree_map(match, tensor)

        return tree_map(convert_and_match, tensors)

    # @override
    @staticmethod
    @run_on_cpu(Framework.TORCH)
    def _compare_equal(device_output: PyTree, golden_output: PyTree) -> bool:
        def _equal_leaf(x, y):
            if x is None and y is None:
                return True
            return torch.equal(x, y)

        passed = tree_map(_equal_leaf, device_output, golden_output)
        flat_passed, _ = tree_flatten(passed)
        return all(flat_passed)

    # @override
    @staticmethod
    @run_on_cpu(Framework.TORCH)
    def _compare_atol(
        device_output: PyTree, golden_output: PyTree, atol_config: AtolConfig
    ) -> float:

        def _atol_leaf(x, y):
            if x is None and y is None:
                return torch.tensor(0.0)
            return torch.max(torch.abs(x - y))

        leaf_atols = tree_map(_atol_leaf, device_output, golden_output)
        flat_atols, _ = tree_flatten(leaf_atols)
        filtered_atols = [atol for atol in flat_atols if atol is not None]
        atol = max(filtered_atols)
        return float(atol)

    # @override
    @run_on_cpu(Framework.TORCH)
    def _compare_pcc(
        self,
        device_output: PyTree,
        golden_output: PyTree,
        pcc_config: PccConfig,
        pcc_mask: torch.Tensor | None = None,
    ) -> float:
        def apply_real_token_mask(x: torch.Tensor, y: torch.Tensor):
            if pcc_mask is None:
                return x, y

            assert (
                pcc_mask.ndim == 2
            ), f"Expected pcc_mask to have shape [batch, seq], got {tuple(pcc_mask.shape)}"
            batch_size, seq_len = pcc_mask.shape

            if x.dim() < 2 or y.dim() < 2:
                return x, y
            if x.shape[0] != batch_size or y.shape[0] != batch_size:
                return x, y

            # Known LLM outputs:
            # - rank-3 tensors (e.g. logits [B, S, V]) -> sequence dim is -2
            # - rank-4 tensors (e.g. KV [B, H, S, D]) -> sequence dim is -2
            assert (
                x.shape == y.shape
            ), f"pcc_mask requires matching shapes, got {x.shape} vs {y.shape}"
            assert x.dim() in (
                3,
                4,
            ), f"pcc_mask only supports rank-3 (logits) or rank-4 (KV) tensors, got rank {x.dim()} with shape {x.shape}"
            assert (
                x.shape[-2] == seq_len
            ), f"dim -2 should be seq_len ({seq_len}), got {x.shape[-2]} in shape {x.shape}"

            x_seq_first = x.movedim(-2, 1)
            y_seq_first = y.movedim(-2, 1)

            x, y = x_seq_first[pcc_mask], y_seq_first[pcc_mask]
            return x, y

        def compute_pcc(x: torch.Tensor, y: torch.Tensor):
            if x is None and y is None:
                return None
            x, y = apply_real_token_mask(x, y)
            # PCC formula can be ill conditioned. If inputs are allclose, fudge the result to 1.0.
            # Done per tensor to avoid cases where some pairs in a pytree are not allclose and others enter the ill-conditioned region.
            if TorchComparisonEvaluator._compare_allclose(x, y, pcc_config.allclose):
                return 1.0

            # PCC is undefined for single-element tensors (no variance), but we want to fail if we came to this.
            if self._is_single_element(x):
                return 0.0

            x_flat, y_flat = x.flatten(), y.flatten()
            vx, vy = x_flat - x_flat.mean(), y_flat - y_flat.mean()
            denom = vx.norm() * vy.norm()

            return torch.tensor(float("nan")) if denom == 0 else (vx @ vy) / denom

        leaf_pccs = tree_map(compute_pcc, device_output, golden_output)
        flat_pccs, _ = tree_flatten(leaf_pccs)
        filtered_pccs = [pcc for pcc in flat_pccs if pcc is not None]
        pcc = min(filtered_pccs)
        return float(pcc)

    # @override
    @staticmethod
    @run_on_cpu(Framework.TORCH)
    def _compare_allclose(
        device_output: PyTree,
        golden_output: PyTree,
        allclose_config: AllcloseConfig,
    ) -> bool:
        def _allclose_leaf(x, y):
            if x is None and y is None:
                return True
            return torch.allclose(
                x, y, rtol=allclose_config.rtol, atol=allclose_config.atol
            )

        all_close = tree_map(_allclose_leaf, device_output, golden_output)
        flat_close, _ = tree_flatten(all_close)
        return all(flat_close)
