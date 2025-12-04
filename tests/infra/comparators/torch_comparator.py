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
    @staticmethod
    @run_on_cpu(Framework.TORCH)
    def _match_data_types(tensors: PyTree) -> PyTree:
        def match(tensor):
            if isinstance(tensor, torch.Tensor) and tensor.dtype != torch.float32:
                tensor = tensor.to(torch.float32)
            return tensor

        def convert_and_match(tensor):
            if isinstance(tensor, DynamicCache) or isinstance(
                tensor, EncoderDecoderCache
            ):
                tensor = tensor.to_legacy_cache()
            if isinstance(tensor, torch.Tensor) and tensor.dtype != torch.float32:
                tensor = tensor.to(torch.float32)
            return tree_map(match, tensor)

        return tree_map(convert_and_match, tensors)

    # @override
    @staticmethod
    @run_on_cpu(Framework.TORCH)
    def _compare_equal(device_output: PyTree, golden_output: PyTree) -> bool:
        passed = tree_map(
            lambda x, y: True if x is None and y is None else torch.equal(x, y),
            device_output,
            golden_output,
        )
        flat_passed, _ = tree_flatten(passed)
        return all(flat_passed)

    # @override
    @staticmethod
    @run_on_cpu(Framework.TORCH)
    def _compare_atol(
        device_output: PyTree, golden_output: PyTree, atol_config: AtolConfig
    ) -> float:
        leaf_atols = tree_map(
            lambda x, y: (
                None if x is None and y is None else torch.max(torch.abs(x - y))
            ),
            device_output,
            golden_output,
        )
        flat_atols, _ = tree_flatten(leaf_atols)
        filtered_atols = [atol for atol in flat_atols if atol is not None]
        atol = max(filtered_atols)
        return float(atol)

    # @override
    @staticmethod
    @run_on_cpu(Framework.TORCH)
    def _compare_pcc(
        device_output: PyTree, golden_output: PyTree, pcc_config: PccConfig
    ) -> float:
        def compute_pcc(x: torch.Tensor, y: torch.Tensor):
            if x is None and y is None:
                return None
            x_flat, y_flat = x.flatten(), y.flatten()
            vx, vy = x_flat - x_flat.mean(), y_flat - y_flat.mean()
            denom = vx.norm() * vy.norm()

            return torch.tensor(float("nan")) if denom == 0 else (vx @ vy) / denom

        # If tensors are really close, pcc will be nan. Handle that before calculating
        # pcc by checking allclose first.
        if TorchComparator._compare_allclose(
            device_output, golden_output, pcc_config.allclose
        ):
            return 1.0  # Perfect correlation when values are essentially identical

        # Calculate PCC for non-identical values
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
        all_close = tree_map(
            lambda x, y: (
                True
                if x is None and y is None
                else torch.allclose(
                    x, y, rtol=allclose_config.rtol, atol=allclose_config.atol
                )
            ),
            device_output,
            golden_output,
        )
        flat_close, _ = tree_flatten(all_close)
        return all(flat_close)
