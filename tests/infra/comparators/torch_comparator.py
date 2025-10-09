# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np
from infra.runners import run_on_cpu
from infra.utilities import Framework, PyTree
from torch.utils._pytree import tree_flatten, tree_map

from .comparator import Comparator
from .comparison_config import AllcloseConfig, AtolConfig, ComparisonConfig, PccConfig


class TorchComparator(Comparator):
    """Comparator for Torch tensors/pytrees."""

    @staticmethod
    def _convert_to_torch(obj):
        """Convert numpy arrays and scalars to PyTorch tensors."""
        if isinstance(obj, np.ndarray):
            return torch.from_numpy(obj)
        elif isinstance(obj, (np.floating, np.integer, np.complexfloating)):
            return torch.tensor(obj.item())
        else:
            return obj

    @staticmethod
    def _tree_map_to_torch(tensors: PyTree) -> PyTree:
        """Convert any numpy arrays and scalars in the PyTree to PyTorch tensors."""
        return tree_map(TorchComparator._convert_to_torch, tensors)

    # @override
    @staticmethod
    @run_on_cpu(Framework.TORCH)
    def _match_data_types(tensors: PyTree) -> PyTree:
        tensors = TorchComparator._tree_map_to_torch(tensors)
        return tree_map(
            lambda tensor: (
                tensor.to(torch.float32)
                if isinstance(tensor, torch.Tensor) and tensor.dtype != torch.float32
                else tensor
            ),
            tensors,
        )

    # @override
    @staticmethod
    @run_on_cpu(Framework.TORCH)
    def _compare_equal(device_output: PyTree, golden_output: PyTree) -> None:
        device_output = TorchComparator._tree_map_to_torch(device_output)
        golden_output = TorchComparator._tree_map_to_torch(golden_output)
        passed = tree_map(lambda x, y: torch.equal(x, y), device_output, golden_output)
        flat_passed, _ = tree_flatten(passed)
        assert all(flat_passed), "Equal comparison failed."

    # @override
    @staticmethod
    @run_on_cpu(Framework.TORCH)
    def _compare_atol(
        device_output: PyTree, golden_output: PyTree, atol_config: AtolConfig
    ) -> None:
        device_output = TorchComparator._tree_map_to_torch(device_output)
        golden_output = TorchComparator._tree_map_to_torch(golden_output)
        leaf_atols = tree_map(
            lambda x, y: torch.max(torch.abs(x - y)), device_output, golden_output
        )
        flat_atols, _ = tree_flatten(leaf_atols)
        atol = max(flat_atols)
        assert atol <= atol_config.required_atol, (
            f"Atol comparison failed. "
            f"Calculated: atol={atol}. Required: atol={atol_config.required_atol}."
        )

    # @override
    @staticmethod
    @run_on_cpu(Framework.TORCH)
    def _compare_pcc(
        device_output: PyTree, golden_output: PyTree, pcc_config: PccConfig
    ) -> None:
        device_output = TorchComparator._tree_map_to_torch(device_output)
        golden_output = TorchComparator._tree_map_to_torch(golden_output)

        def compute_pcc(x: torch.Tensor, y: torch.Tensor):
            x_flat, y_flat = x.flatten(), y.flatten()
            vx, vy = x_flat - x_flat.mean(), y_flat - y_flat.mean()
            denom = vx.norm() * vy.norm()

            return torch.tensor(float("nan")) if denom == 0 else (vx @ vy) / denom

        # If tensors are really close, pcc will be nan. Handle that before calculating
        # pcc.
        try:
            TorchComparator._compare_allclose(
                device_output, golden_output, pcc_config.allclose
            )
        except AssertionError:
            leaf_pccs = tree_map(compute_pcc, device_output, golden_output)
            flat_pccs, _ = tree_flatten(leaf_pccs)
            pcc = min(flat_pccs)
            assert pcc >= pcc_config.required_pcc, (
                f"PCC comparison failed. "
                f"Calculated: pcc={pcc}. Required: pcc={pcc_config.required_pcc}."
            )

    # @override
    @staticmethod
    @run_on_cpu(Framework.TORCH)
    def _compare_allclose(
        device_output: PyTree,
        golden_output: PyTree,
        allclose_config: AllcloseConfig,
    ) -> None:
        device_output = TorchComparator._tree_map_to_torch(device_output)
        golden_output = TorchComparator._tree_map_to_torch(golden_output)
        all_close = tree_map(
            lambda x, y: torch.allclose(
                x, y, rtol=allclose_config.rtol, atol=allclose_config.atol
            ),
            device_output,
            golden_output,
        )
        flat_close, _ = tree_flatten(all_close)
        assert all(flat_close), (
            f"Allclose comparison failed. "
            f"Required: atol={allclose_config.atol}, rtol={allclose_config.rtol}."
        )
