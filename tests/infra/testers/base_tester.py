# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from adapters import FrameworkAdapter, FrameworkAdapterFactory
from comparators import Comparator, ComparatorFactory, ComparisonConfig
from runners import DeviceRunner, DeviceRunnerFactory
from utilities.types import Framework, Tensor
from utilities.workloads import Workload


class BaseTester(ABC):
    """Abstract base class all testers must inherit."""

    # -------------------- Protected methods --------------------

    def __init__(
        self,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        framework: Optional[Framework] = None,
    ) -> None:
        """Protected constructor for subclasses to use."""
        self._comparison_config = comparison_config
        self._framework = framework
        # Placeholders for objects that will be set in `_initialize_all_components`.
        # Easier to spot if located in constructor instead of dynamically creating them
        # somewhere in methods.
        self._device_runner: DeviceRunner = None
        self._adapter: FrameworkAdapter = None
        self._comparator: Comparator = None

        # Initialize rest of the class.
        self._initialize_all_components()

    @abstractmethod
    def _initialize_all_components(self) -> None:
        """
        Helper initialization method allowing subclasses to define a piece by piece
        object construction.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def _initialize_framework_specific_helpers(self) -> None:
        """
        Initializes `self._device_runner`, `self._adapter` and `self._comparator`.

        This function triggers connection to device.
        """
        # Creating runner will register plugin and connect the device properly.
        self._device_runner = DeviceRunnerFactory(self._framework).create_runner()
        self._adapter = FrameworkAdapterFactory(self._framework).create_adapter()
        self._comparator = ComparatorFactory(self._framework).create_comparator(
            self._comparison_config
        )

    # --- Convenience wrappers ---

    def _compile(self, workload: Workload) -> Workload:
        """
        Compiles workload into optimized kernels.

        Returns new "compiled" Workload.
        """
        return self._adapter.compile(workload)

    def _run_on_tt_device(self, compiled_workload: Workload) -> Tensor:
        """Runs workload on TT device."""
        return self._device_runner.run_on_tt_device(compiled_workload)

    def _run_on_cpu(self, compiled_workload: Workload) -> Tensor:
        """Runs workload on CPU."""
        return self._device_runner.run_on_cpu(compiled_workload)

    def _compare(self, device_out: Tensor, golden_out: Tensor) -> None:
        """Compares device with golden output."""
        self._comparator.compare(device_out, golden_out)
