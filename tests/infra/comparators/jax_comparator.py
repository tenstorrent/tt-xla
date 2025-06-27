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

    # -------------------- Private methods --------------------

    # --- Overrides ---

    # @override
    @staticmethod
    @run_on_cpu(Framework.JAX)
    def _compare_equal(device_output: PyTree, golden_output: PyTree) -> None:
        passed = tree_map(lambda x, y: (x == y).all(), device_output, golden_output)
        assert jax.tree.all(passed), f"Equal comparison failed."

    # @override
    @staticmethod
    @run_on_cpu(Framework.JAX)
    def _compare_atol(
        device_output: PyTree, golden_output: PyTree, atol_config: AtolConfig
    ) -> None:
        leaf_atols = tree_map(
            lambda x, y: jnp.max(jnp.abs(x - y)),
            device_output,
            golden_output,
        )
        atol = jax.tree.reduce(lambda x, y: jnp.maximum(x, y), leaf_atols)
        assert atol <= atol_config.required_atol, (
            f"Atol comparison failed. "
            f"Calculated: atol={atol}. Required: atol={atol_config.required_atol}."
        )

    # @override
    @staticmethod
    @run_on_cpu(Framework.JAX)
    def _compare_allclose(
        device_output: PyTree,
        golden_output: PyTree,
        allclose_config: AllcloseConfig,
    ) -> None:
        all_close = tree_map(
            lambda x, y: jnp.allclose(
                x, y, rtol=allclose_config.rtol, atol=allclose_config.atol
            ),
            device_output,
            golden_output,
        )
        passed = jax.tree.reduce(lambda x, y: x and y, all_close)
        assert passed, (
            f"Allclose comparison failed. "
            f"Required: atol={allclose_config.atol}, rtol={allclose_config.rtol}."
        )

    # @override
    @staticmethod
    @run_on_cpu(Framework.JAX)
    def _compare_pcc(
        device_output: PyTree, golden_output: PyTree, pcc_config: PccConfig
    ) -> None:
        def compute_pcc(x: jax.Array, y: jax.Array):
            x_flat, y_flat = x.flatten(), y.flatten()
            vx, vy = x_flat - jnp.mean(x_flat), y_flat - jnp.mean(y_flat)
            denom = jnp.linalg.norm(vx) * jnp.linalg.norm(vy)

            return jnp.nan if denom == 0 else jnp.dot(vx, vy) / denom

        # If tensors are really close, pcc will be nan. Handle that before calculating
        # pcc.
        try:
            JaxComparator._compare_allclose(
                device_output, golden_output, pcc_config.allclose
            )
        except AssertionError:
            leaf_pccs = jax.tree.map(compute_pcc, device_output, golden_output)
            flat_pccs, _ = jax.tree_util.tree_flatten(leaf_pccs)
            pcc = min(flat_pccs)
            assert pcc >= pcc_config.required_pcc, (
                f"PCC comparison failed. "
                f"Calculated: pcc={pcc}. Required: pcc={pcc_config.required_pcc}."
            )

    # @override
    @staticmethod
    @run_on_cpu(Framework.JAX)
    def _match_data_types(tensors: PyTree) -> PyTree:
        return tree_map(
            lambda tensor: (
                tensor.astype("float32")
                if isinstance(tensor, jax.Array) and tensor.dtype.str != "float32"
                else tensor
            ),
            tensors,
        )
