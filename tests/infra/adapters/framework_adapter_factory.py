# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from utilities.types import Framework

from .framework_adapter import FrameworkAdapter
from .jax_adapter import JaxAdapter
from .torch_adapter import TorchAdapter


class FrameworkAdapterFactory:
    """Factory creating FrameworkAdapters based on provided framework."""

    # -------------------- Public methods --------------------

    def __init__(self, framework: Framework) -> None:
        self._framework = framework

    def create_adapter(self) -> FrameworkAdapter:
        if self._framework == Framework.JAX:
            return JaxAdapter()
        elif self._framework == Framework.TORCH:
            return TorchAdapter()
        else:
            raise ValueError(f"Unsupported framework {self._framework}")
