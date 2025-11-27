# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
from infra.runners import run_on_cpu
from infra.utilities import Framework, PyTree
from torch.utils._pytree import tree_flatten, tree_map

from .comparator import Comparator
from .comparison_config import AllcloseConfig, AtolConfig, ComparisonConfig, PccConfig


class TorchComparator(Comparator):
    """Comparator for Torch tensors/pytrees."""

    # @override
    @staticmethod
    @run_on_cpu(Framework.TORCH)
    def _match_data_types(tensors: PyTree) -> PyTree:
        return tree_map(
            lambda tensor: (
                tensor.to(torch.float64)
                if isinstance(tensor, torch.Tensor) and tensor.dtype != torch.float64
                else tensor
            ),
            tensors,
        )

    # @override
    @staticmethod
    @run_on_cpu(Framework.TORCH)
    def _compare_equal(device_output: PyTree, golden_output: PyTree) -> bool:
        passed = tree_map(lambda x, y: torch.equal(x, y), device_output, golden_output)
        flat_passed, _ = tree_flatten(passed)
        return all(flat_passed)

    # @override
    @staticmethod
    @run_on_cpu(Framework.TORCH)
    def _compare_atol(
        device_output: PyTree, golden_output: PyTree, atol_config: AtolConfig
    ) -> float:
        leaf_atols = tree_map(
            lambda x, y: torch.max(torch.abs(x - y)), device_output, golden_output
        )
        flat_atols, _ = tree_flatten(leaf_atols)
        atol = max(flat_atols)
        return float(atol)

    # @override
    @staticmethod
    @run_on_cpu(Framework.TORCH)
    def _compare_pcc(
        device_output: PyTree, golden_output: PyTree, pcc_config: PccConfig
    ) -> float:
        def compute_pcc(x: torch.Tensor, y: torch.Tensor):
            # PCC formula can be ill conditioned. If inputs are allclose, fudge the result to 1.0.
            # Done per tensor to avoid cases where some pairs in a pytree are not allclose and others enter the ill-conditioned region.
            if TorchComparator._compare_allclose(
                device_output, golden_output, pcc_config.allclose
            ):
                return 1.0

            x_flat, y_flat = x.flatten(), y.flatten()
            vx, vy = x_flat - x_flat.mean(), y_flat - y_flat.mean()
            denom = vx.norm() * vy.norm()

            return torch.tensor(float("nan")) if denom == 0 else (vx @ vy) / denom

        leaf_pccs = tree_map(compute_pcc, device_output, golden_output)
        flat_pccs, _ = tree_flatten(leaf_pccs)
        pcc = min(flat_pccs)
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
            lambda x, y: torch.allclose(
                x, y, rtol=allclose_config.rtol, atol=allclose_config.atol
            ),
            device_output,
            golden_output,
        )
        flat_close, _ = tree_flatten(all_close)
        return all(flat_close)
