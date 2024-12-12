# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


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
    required_atol: float = 1e-2


@dataclass
class PccConfig(ConfigBase):
    required_pcc: float = 0.99


@dataclass
class AllcloseConfig(ConfigBase):
    rtol: float = 1e-2
    atol: float = 1e-2


@dataclass
class ComparisonConfig:
    equal: EqualConfig = EqualConfig()
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


def compare_equal(device_output: jax.Array, golden_output: jax.Array) -> bool:
    return (device_output == golden_output).all()


def compare_atol(
    device_output: jax.Array, golden_output: jax.Array, atol_config: AtolConfig
) -> bool:
    atol = jnp.max(jnp.abs(device_output - golden_output))
    return atol <= atol_config.required_atol


def compare_pcc(
    device_output: jax.Array, golden_output: jax.Array, pcc_config: PccConfig
) -> bool:
    # If tensors are really close, pcc will be nan. Handle that before calculating pcc.
    if compare_allclose(
        device_output, golden_output, AllcloseConfig(rtol=1e-3, atol=1e-3)
    ):
        return True

    pcc = jnp.corrcoef(device_output.flatten(), golden_output.flatten())
    return jnp.min(pcc) >= pcc_config.required_pcc


def compare_allclose(
    device_output: jax.Array, golden_output: jax.Array, allclose_config: AllcloseConfig
) -> bool:
    allclose = jnp.allclose(
        device_output,
        golden_output,
        rtol=allclose_config.rtol,
        atol=allclose_config.atol,
    )
    return allclose
