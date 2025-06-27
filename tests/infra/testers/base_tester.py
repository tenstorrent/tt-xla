# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from infra.comparators import Comparator, ComparatorFactory, ComparisonConfig
from infra.runners import DeviceRunner, DeviceRunnerFactory
from infra.utilities import Framework, Tensor
from infra.workloads import Workload


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
        # Placeholders for objects that will be set during
        # `_initialize_framework_specific_helpers`. Easier to spot if located in
        # constructor instead of dynamically creating them somewhere in methods.
        self._device_runner: DeviceRunner = None
        self._comparator: Comparator = None

        # Automatically initialize framework-specific helpers. Leave to subclasses to
        # override `_initialize_all_components` if there are more initialization steps.
        self._initialize_framework_specific_helpers()
        # Initialize rest of the class.
        self._initialize_all_components()

    def _initialize_framework_specific_helpers(self) -> None:
        """
        Initializes `self._device_runner` and `self._comparator`.

        Based on the framework instantiates a DeviceRunner (which internally
        instantiates a DeviceConnector singleton, ensuring plugin registration and
        connection to the device) and a Comparator, instantiates and stores the concrete
        model instance and finally packs model or its forward method and its arguments
        in a Workload.

        This function triggers connection to device.
        """
        assert self._framework is not None
        # Creating runner will register plugin and connect the device properly.
        self._device_runner = DeviceRunnerFactory.create_runner(self._framework)
        self._comparator = ComparatorFactory.create_comparator(
            self._framework, self._comparison_config
        )

    def _initialize_all_components(self) -> None:
        """
        Helper initialization method allowing subclasses to define a piece by piece
        object construction.

        Subclasses should override this method if there are more initialization steps
        to do upon object creation.
        """
        pass

    @abstractmethod
    def _compile(self, workload: Workload) -> Workload:
        """Compiles `workload`."""
        raise NotImplementedError("Subclasses must implement this method.")

    # --- Convenience wrappers ---

    def _run_on_tt_device(self, compiled_workload: Workload) -> Tensor:
        """Runs workload on TT device."""
        return self._device_runner.run_on_tt_device(compiled_workload)

    def _run_on_cpu(self, compiled_workload: Workload) -> Tensor:
        """Runs workload on CPU."""
        return self._device_runner.run_on_cpu(compiled_workload)

    def _compare(self, device_out: Tensor, golden_out: Tensor) -> None:
        """Compares device with golden output."""
        self._comparator.compare(device_out, golden_out)
