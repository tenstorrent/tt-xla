# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC
from typing import Callable, Sequence

import jax

from .comparison import (
    ComparisonConfig,
    compare_allclose,
    compare_atol,
    compare_equal,
    compare_pcc,
)
from .device_runner import DeviceRunner
from .types import Tensor


class BaseTester(ABC):
    """
    Abstract base class all testers must inherit.

    Provides just a couple of common methods.
    """

    def __init__(
        self, comparison_config: ComparisonConfig = ComparisonConfig()
    ) -> None:
        self._comparison_config = comparison_config

    @staticmethod
    def _compile(executable: Callable) -> Callable:
        """Sets up `executable` for just-in-time compile."""
        return jax.jit(executable)

    def _compare(
        self,
        device_out: Tensor,
        golden_out: Tensor,
    ) -> None:
        device_output, golden_output = DeviceRunner.put_tensors_on_cpu(
            device_out, golden_out
        )
        device_output, golden_output = self._match_data_types(
            device_output, golden_output
        )

        if self._comparison_config.equal.enabled:
            compare_equal(device_output, golden_output)
        if self._comparison_config.atol.enabled:
            compare_atol(device_output, golden_output, self._comparison_config.atol)
        if self._comparison_config.pcc.enabled:
            compare_pcc(device_output, golden_output, self._comparison_config.pcc)
        if self._comparison_config.allclose.enabled:
            compare_allclose(
                device_output, golden_output, self._comparison_config.allclose
            )

    def _match_data_types(self, *tensors: Tensor) -> Sequence[Tensor]:
        """
        Casts all tensors to float32 if not already in that format.

        Tensors need to be in same data format in order to compare them.
        """
        return [
            tensor.astype("float32") if tensor.dtype.str != "float32" else tensor
            for tensor in tensors
        ]
