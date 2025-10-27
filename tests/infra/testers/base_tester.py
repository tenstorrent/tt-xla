# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Optional

from infra.comparators import Comparator, ComparatorFactory, ComparisonConfig
from infra.runners import DeviceRunner, DeviceRunnerFactory
from infra.utilities import Framework
from infra.workloads import Workload


class BaseTester(ABC):
    """Abstract base class all testers must inherit."""

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

        # Automatically initialize framework-specific helpers
        self._initialize_framework_specific_helpers()

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

    @abstractmethod
    def _compile_for_cpu(self, workload: Workload) -> None:
        """Compiles `workload` for CPU."""
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def _compile_for_tt_device(self, workload: Workload) -> None:
        """Compiles `workload` for TT device."""
        raise NotImplementedError("Subclasses must implement this method.")

    def serialize_compilation_artifacts(self, test_name: str) -> None:
        """Serialize the model with the appropriate output prefix.

        Args:
            test_name: Test name to generate output prefix from.
        """

        # Keep the test name but replace special chars with underscores
        # Example: test_mnist_mlp_inference[256-128-64] -> output/test_mnist_mlp_inference_256_128_64
        clean_name = re.sub(r"[\[\](),\-\s]+", "_", test_name)
        # Remove trailing underscores
        clean_name = clean_name.rstrip("_")

        output_prefix = f"output_artifact/{clean_name}"

        self.serialize_on_device(output_prefix)

    @abstractmethod
    def serialize_on_device(self, output_prefix: str) -> None:
        """
        Serializes the model workload on TT device with proper compiler configuration.

        Args:
            output_prefix: Base path and filename prefix for output files
        """
        raise NotImplementedError("Subclasses must implement this method.")
