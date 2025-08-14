# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from infra.utilities import Framework

from .comparator import Comparator
from .comparison_config import ComparisonConfig
from .jax_comparator import JaxComparator
from .torch_comparator import TorchComparator


class ComparatorFactory:
    """Factory creating Comparators based on provided framework."""

    @staticmethod
    def create_comparator(
        framework: Framework, comparison_config: ComparisonConfig
    ) -> Comparator:
        if framework == Framework.JAX:
            return JaxComparator(comparison_config)
        elif framework == Framework.TORCH:
            return TorchComparator(comparison_config)
        else:
            raise ValueError(f"Unsupported framework {framework}")
