# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .device_runner import run_on_cpu
from .types import Tensor


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
    required_atol: float = 1e-1


@dataclass
class PccConfig(ConfigBase):
    required_pcc: float = 0.99


@dataclass
class AllcloseConfig(ConfigBase):
    rtol: float = 1e-2
    atol: float = 1e-2


@dataclass
class ComparisonConfig:
    equal: EqualConfig = EqualConfig(False)
    atol: AtolConfig = AtolConfig()
    pcc: PccConfig = PccConfig()
    allclose: AllcloseConfig = AllcloseConfig()

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
def compare_equal(device_output: Tensor, golden_output: Tensor) -> None:
    assert isinstance(device_output, jax.Array) and isinstance(
        golden_output, jax.Array
    ), f"Currently only jax.Array is supported"

    eq = (device_output == golden_output).all()

    assert eq, f"Equal comparison failed"


@run_on_cpu
def compare_atol(
    device_output: Tensor, golden_output: Tensor, atol_config: AtolConfig
) -> None:
    assert isinstance(device_output, jax.Array) and isinstance(
        golden_output, jax.Array
    ), f"Currently only jax.Array is supported {type(device_output)}, {type(golden_output)}"

    atol = jnp.max(jnp.abs(device_output - golden_output))

    assert (
        atol <= atol_config.required_atol
    ), f"Atol comparison failed. Calculated atol={atol}"


@run_on_cpu
def compare_pcc(
    device_output: Tensor, golden_output: Tensor, pcc_config: PccConfig
) -> None:
    assert isinstance(device_output, jax.Array) and isinstance(
        golden_output, jax.Array
    ), f"Currently only jax.Array is supported"

    # If tensors are really close, pcc will be nan. Handle that before calculating pcc.
    try:
        compare_allclose(
            device_output, golden_output, AllcloseConfig(rtol=1e-2, atol=1e-2)
        )
    except AssertionError:
        pcc = jnp.corrcoef(device_output.flatten(), golden_output.flatten())
        pcc = jnp.min(pcc)

        assert (
            pcc >= pcc_config.required_pcc
        ), f"PCC comparison failed. Calculated pcc={pcc}"


@run_on_cpu
def compare_allclose(
    device_output: Tensor, golden_output: Tensor, allclose_config: AllcloseConfig
) -> None:
    assert isinstance(device_output, jax.Array) and isinstance(
        golden_output, jax.Array
    ), f"Currently only jax.Array is supported"

    allclose = jnp.allclose(
        device_output,
        golden_output,
        rtol=allclose_config.rtol,
        atol=allclose_config.atol,
    )

    assert allclose, f"Allclose comparison failed."
