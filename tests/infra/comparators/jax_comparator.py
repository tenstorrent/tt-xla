# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
from infra.runners import run_on_cpu
from infra.utilities import Framework, PyTree
from jax.tree import map as tree_map

from .comparator import Comparator
from .comparison_config import AllcloseConfig, AtolConfig, ComparisonConfig, PccConfig


class JaxComparator(Comparator):
    """Comparator for JAX tensors/pytrees."""

    # @override
    @staticmethod
    @run_on_cpu(Framework.JAX)
    def _match_data_types(tensors: PyTree) -> PyTree:
        return tree_map(
            lambda tensor: (
                tensor.astype("float64")
                if isinstance(tensor, jax.Array) and tensor.dtype.str != "float64"
                else tensor
            ),
            tensors,
        )

    # @override
    @staticmethod
    @run_on_cpu(Framework.JAX)
    def _compare_equal(device_output: PyTree, golden_output: PyTree) -> bool:
        passed = tree_map(lambda x, y: (x == y).all(), device_output, golden_output)
        return jax.tree.all(passed)

    # @override
    @staticmethod
    @run_on_cpu(Framework.JAX)
    def _compare_atol(
        device_output: PyTree, golden_output: PyTree, atol_config: AtolConfig
    ) -> float:
        leaf_atols = tree_map(
            lambda x, y: jnp.max(jnp.abs(x - y)),
            device_output,
            golden_output,
        )
        atol = jax.tree.reduce(lambda x, y: jnp.maximum(x, y), leaf_atols)
        return float(atol)

    # @override
    @staticmethod
    @run_on_cpu(Framework.JAX)
    def _compare_pcc(
        device_output: PyTree, golden_output: PyTree, pcc_config: PccConfig
    ) -> float:
        def compute_pcc(x: jax.Array, y: jax.Array):
            # PCC formula can be ill conditioned. If inputs are allclose, fudge the result to 1.0.
            # Done per tensor to avoid cases where some pairs in a pytree are not allclose and others enter the ill-conditioned region.
            if JaxComparator._compare_allclose(
                device_output, golden_output, pcc_config.allclose
            ):
                return 1.0

            x_flat, y_flat = x.flatten(), y.flatten()
            vx, vy = x_flat - jnp.mean(x_flat), y_flat - jnp.mean(y_flat)
            denom = jnp.linalg.norm(vx) * jnp.linalg.norm(vy)

            return jnp.nan if denom == 0 else jnp.dot(vx, vy) / denom

        leaf_pccs = jax.tree.map(compute_pcc, device_output, golden_output)
        flat_pccs, _ = jax.tree_util.tree_flatten(leaf_pccs)
        pcc = min(flat_pccs)
        return float(pcc)

    # @override
    @staticmethod
    @run_on_cpu(Framework.JAX)
    def _compare_allclose(
        device_output: PyTree,
        golden_output: PyTree,
        allclose_config: AllcloseConfig,
    ) -> bool:
        all_close = tree_map(
            lambda x, y: jnp.allclose(
                x, y, rtol=allclose_config.rtol, atol=allclose_config.atol
            ),
            device_output,
            golden_output,
        )
        passed = jax.tree.reduce(lambda x, y: x and y, all_close)
        return bool(passed)
