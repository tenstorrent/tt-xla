# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from utilities.types import Framework

from .comparator import Comparator
from .comparison_config import ComparisonConfig
from .jax_comparator import JaxComparator
from .torch_comparator import TorchComparator


class ComparatorFactory:
    """Factory creating Comparators based on provided framework."""

    # -------------------- Public methods --------------------

    def __init__(self, framework: Framework) -> None:
        self._framework = framework

    def create_comparator(self, comparison_config: ComparisonConfig) -> Comparator:
        if self._framework == Framework.JAX:
            return JaxComparator(comparison_config)
        elif self._framework == Framework.TORCH:
            return TorchComparator(comparison_config)
        else:
            raise ValueError(f"Unsupported framework {self._framework}")
