# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import PyTree

from .device_runner import run_on_cpu


@dataclass
class ConfigBase:
    enabled: bool = True

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False


@dataclass
class EqualConfig(ConfigBase):
    pass


@dataclass
class AtolConfig(ConfigBase):
    required_atol: float = 1.6e-1


@dataclass
class AllcloseConfig(ConfigBase):
    rtol: float = 1e-2
    atol: float = 1e-2


# When tensors are too close, pcc will output NaN values.
# Therefore, for each test it should be possible to separately tune the threshold of allclose.rtol and allclose.atol
# below which pcc won't be calculated and therefore test will be able to pass without pcc comparison.
@dataclass
class PccConfig(ConfigBase):
    required_pcc: float = 0.99
    allclose: AllcloseConfig = AllcloseConfig()


@dataclass
class ComparisonConfig:
    equal: EqualConfig = EqualConfig(False)
    atol: AtolConfig = AtolConfig(False)
    pcc: PccConfig = PccConfig()
    allclose: AllcloseConfig = AllcloseConfig(False)

    def enable_all(self) -> None:
        self.equal.enable()
        self.atol.enable()
        self.allclose.enable()
        self.pcc.enable()

    def disable_all(self) -> None:
        self.equal.disable()
        self.atol.disable()
        self.allclose.disable()
        self.pcc.disable()


# TODO functions below rely on jax functions, should be generalized for all supported
# frameworks in the future.


@run_on_cpu
def compare_equal(device_output: PyTree, golden_output: PyTree) -> None:
    passed = jax.tree.map(lambda x, y: (x == y).all(), device_output, golden_output)
    assert jax.tree.all(passed), f"Equal comparison failed."


@run_on_cpu
def compare_atol(
    device_output: PyTree, golden_output: PyTree, atol_config: AtolConfig
) -> None:
    leaf_atols = jax.tree.map(
        lambda x, y: jnp.max(jnp.abs(x - y)),
        device_output,
        golden_output,
    )
    atol = jax.tree.reduce(lambda x, y: jnp.maximum(x, y), leaf_atols)
    assert atol <= atol_config.required_atol, (
        f"Atol comparison failed: calculated={atol}, required={atol_config.required_atol}. "
        f"{print_max_diff_elements(device_output, golden_output)}"
    )


@run_on_cpu
def compare_pcc(
    device_output: PyTree, golden_output: PyTree, pcc_config: PccConfig
) -> None:
    # Note, minmimum of pccs is not the same as pcc across all elements.
    # If the user wants to compare pcc across all elements, they should concatenate the tensors themselves
    # This should be fine, as it's effectively what would be done if n comparisons are done independently.
    try:  # If tensors are really close, pcc will be nan. Handle that before calculating pcc.
        compare_allclose(device_output, golden_output, pcc_config.allclose)
    except AssertionError:
        leaf_pccs = jax.tree.map(
            lambda x, y: jnp.min(jnp.corrcoef(x.flatten(), y.flatten())),
            device_output,
            golden_output,
        )
        pcc = jax.tree.reduce(lambda x, y: jnp.minimum(x, y), leaf_pccs)
        assert pcc >= pcc_config.required_pcc, (
            f"PCC comparison failed. "
            f"Calculated: pcc={pcc}. Required: pcc={pcc_config.required_pcc}."
        )


@run_on_cpu
def compare_allclose(
    device_output: PyTree, golden_output: PyTree, allclose_config: AllcloseConfig
) -> None:
    all_close = jax.tree.map(
        lambda x, y: jnp.allclose(
            x, y, rtol=allclose_config.rtol, atol=allclose_config.atol
        ),
        device_output,
        golden_output,
    )
    passed = jax.tree.reduce(lambda x, y: x and y, all_close)
    assert passed, (
        f"Allclose comparison failed, required: atol={allclose_config.atol}, rtol={allclose_config.rtol}. "
        f"{print_max_diff_elements(device_output, golden_output)}"
    )


def print_max_diff_elements(device_output: PyTree, golden_output: PyTree) -> str:
    device_output_arr, _ = jax.flatten_util.ravel_pytree(device_output)
    golden_output_arr, _ = jax.flatten_util.ravel_pytree(golden_output)

    max_diff_idx = jnp.argmax(jnp.abs(device_output_arr - golden_output_arr))
    device_el = device_output_arr[max_diff_idx]
    golden_el = golden_output_arr[max_diff_idx]

    return f"Max diff elements: device={device_el}, golden={golden_el}."
