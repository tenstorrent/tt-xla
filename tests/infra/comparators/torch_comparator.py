# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
from infra.runners import run_on_cpu
from infra.utilities import Framework, PyTree
from torch.utils._pytree import tree_flatten, tree_map
from transformers import DynamicCache, EncoderDecoderCache

from .comparator import Comparator
from .comparison_config import AllcloseConfig, AtolConfig, ComparisonConfig, PccConfig
from loguru import logger

class TorchComparator(Comparator):
    """Comparator for Torch tensors/pytrees."""

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

    # LLM decode tests may include transformers StaticCache objects in outputs (e.g., past_key_values).
    # These are not torch.Tensors, so we detect them and treat matching StaticCache leaves as equal.
    # TODO https://github.com/tenstorrent/tt-xla/issues/2743: Enable checking for allclose, pcc, atol, equal.
    @staticmethod
    def _is_static_cache(x: object) -> bool:
        # Avoid importing transformers at module import time.
        try:
            from transformers.cache_utils import StaticCache  # type: ignore

            return isinstance(x, StaticCache)
        except Exception:
            return False

    @staticmethod
    def _both_static_cache(x: object, y: object) -> bool:
        is_sc = TorchComparator._is_static_cache
        return is_sc(x) and is_sc(y)

    # @override
    @staticmethod
    @run_on_cpu(Framework.TORCH)
    def _match_data_types(tensors: PyTree) -> PyTree:
        def match(tensor):
            if isinstance(tensor, torch.Tensor) and tensor.dtype != torch.float64:
                tensor = tensor.to(torch.float64)
            return tensor

        def convert_and_match(tensor):
            if isinstance(tensor, DynamicCache) or isinstance(
                tensor, EncoderDecoderCache
            ):
                # New transformers library has changed the Cache classes
                # to contain and arrays of CacheLayers instead of an array of
                # torch.tensors, we need to extract the torch tensors from CacheLayers
                # before comparing values in the comparator.
                tensor = tensor.to_legacy_cache()
            if isinstance(tensor, torch.Tensor) and tensor.dtype != torch.float64:
                tensor = tensor.to(torch.float64)
            return tree_map(match, tensor)

        return tree_map(convert_and_match, tensors)

    # @override
    @staticmethod
    @run_on_cpu(Framework.TORCH)
    def _compare_equal(device_output: PyTree, golden_output: PyTree) -> bool:
        def _equal_leaf(x, y):
            if TorchComparator._both_static_cache(x, y) or (x is None and y is None):
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
            if TorchComparator._both_static_cache(x, y) or (x is None and y is None):
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
        self, device_output: PyTree, golden_output: PyTree, pcc_config: PccConfig
    ) -> float:
        def compute_pcc(x: torch.Tensor, y: torch.Tensor):
            if TorchComparator._both_static_cache(x, y):
                return torch.tensor(1.0)
            if x is None and y is None:
                return None
            # PCC formula can be ill conditioned. If inputs are allclose, fudge the result to 1.0.
            # Done per tensor to avoid cases where some pairs in a pytree are not allclose and others enter the ill-conditioned region.
            if TorchComparator._compare_allclose(x, y, pcc_config.allclose):
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
        logger.info("flat_pccs={}",flat_pccs)
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
            if TorchComparator._both_static_cache(x, y) or (x is None and y is None):
                return True
            return torch.allclose(
                x, y, rtol=allclose_config.rtol, atol=allclose_config.atol
            )

        all_close = tree_map(_allclose_leaf, device_output, golden_output)
        flat_close, _ = tree_flatten(all_close)
        return all(flat_close)
