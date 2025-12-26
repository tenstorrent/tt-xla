# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from infra.evaluators import (
    ComparisonConfig,
    Evaluator,
    EvaluatorFactory,
    EvaluatorType,
)
from infra.runners import DeviceRunner, DeviceRunnerFactory
from infra.utilities import Framework, sanitize_test_name


class BaseTester(ABC):
    """Abstract base class all testers must inherit."""

    def __init__(
        self,
        framework: Optional[Framework] = None,
        evaluator_type: Optional[EvaluatorType] = None,
        comparison_config: Optional[ComparisonConfig] = None,
    ) -> None:
        """Protected constructor for subclasses to use."""
        self._framework = framework
        self._evaluator_type = evaluator_type
        self._comparison_config = comparison_config
        # Placeholders for objects that will be set during
        # `_initialize_framework_specific_helpers`. Easier to spot if located in
        # constructor instead of dynamically creating them somewhere in methods.
        self._device_runner: DeviceRunner | None = None
        self._evaluator: Evaluator | None = None

        # Automatically initialize framework-specific helpers
        self._initialize_framework_specific_helpers()

    def _initialize_framework_specific_helpers(self) -> None:
        """
        Initializes `self._device_runner` and optionally `self._evaluator`.

        Based on the framework instantiates a DeviceRunner (which internally
        instantiates a DeviceConnector singleton, ensuring plugin registration and
        connection to the device). If evaluator_type is provided, also creates an
        Evaluator. Subclasses can manage their own evaluator by passing
        evaluator_type=None.

        This function triggers connection to device.
        """
        assert self._framework is not None
        # Creating runner will register plugin and connect the device properly.
        self._device_runner = DeviceRunnerFactory.create_runner(self._framework)

        # Only create evaluator if evaluator_type is specified
        if self._evaluator_type is not None:
            self._evaluator = EvaluatorFactory.create_evaluator(
                self._evaluator_type,
                framework=self._framework,
                comparison_config=self._comparison_config,
            )

    def serialize_compilation_artifacts(self, test_name: str) -> None:
        """Serialize the model with the appropriate output prefix.

        Args:
            test_name: Test name to generate output prefix from.
        """
        clean_name = sanitize_test_name(test_name)
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
