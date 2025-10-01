# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field


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


@dataclass
class PccConfig(ConfigBase):
    required_pcc: float = 0.99
    # When tensors are too close, pcc will output NaN values. To prevent that, we do
    # allclose comparison in that case. For each test it should be possible to
    # separately tune the allclose config for which pcc won't be calculated and
    # therefore test will be able to pass without pcc comparison.
    allclose: AllcloseConfig = field(default_factory=AllcloseConfig)


@dataclass
class ComparisonConfig:
    equal: EqualConfig = field(default_factory=lambda: EqualConfig(False))
    atol: AtolConfig = field(default_factory=lambda: AtolConfig(False))
    pcc: PccConfig = field(default_factory=PccConfig)
    allclose: AllcloseConfig = field(default_factory=lambda: AllcloseConfig(False))
    assert_on_failure: bool = True  # Default to True for backwards compatibility

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
