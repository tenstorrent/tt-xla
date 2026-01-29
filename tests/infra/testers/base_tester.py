# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from infra.evaluators import (
    ComparisonConfig,
    Evaluator,
    EvaluatorFactory,
    QualityConfig,
)
from infra.runners import DeviceRunner, DeviceRunnerFactory
from infra.utilities import Framework, sanitize_test_name


class BaseTester(ABC):
    """Abstract base class all testers must inherit."""

    def __init__(
        self,
        evaluator_type: str,
        comparison_config: Optional[ComparisonConfig] = None,
        framework: Optional[Framework] = None,
        quality_config: Optional[QualityConfig] = None,
        metric_names: Optional[List[str]] = None,
        metric_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        """Protected constructor for subclasses to use."""
        self._evaluator_type = evaluator_type
        self._comparison_config = (
            comparison_config if comparison_config is not None else ComparisonConfig()
        )
        self._framework = framework
        self._quality_config = (
            quality_config if quality_config is not None else QualityConfig()
        )
        self._metric_names = metric_names
        self._metric_kwargs = metric_kwargs

        # Placeholders for objects that will be set during
        # `_initialize_framework_specific_helpers`. Easier to spot if located in
        # constructor instead of dynamically creating them somewhere in methods.
        self._device_runner: DeviceRunner = None
        self._evaluator: Optional[Evaluator] = None

        # Automatically initialize framework-specific helpers
        self._initialize_framework_specific_helpers()

    def _initialize_framework_specific_helpers(self) -> None:
        """
        Initializes `self._device_runner` and `self._evaluator`.

        Based on the framework instantiates a DeviceRunner (which internally
        instantiates a DeviceConnector singleton, ensuring plugin registration and
        connection to the device) and a Comparator, instantiates and stores the concrete
        model instance and finally packs model or its forward method and its arguments
        in a Workload.

        This function triggers connection to device.
        """
        if self._framework is not None:
            # Creating runner will register plugin and connect the device properly.
            self._device_runner = DeviceRunnerFactory.create_runner(self._framework)
        self._initialize_evaluator()

    def _initialize_evaluator(self) -> None:
        """Initialize evaluator using factory with stored params."""
        # Skip if quality evaluator needs lazy init (no metric_names yet)
        if self._evaluator_type == "quality" and not self._metric_names:
            return

        self._evaluator = EvaluatorFactory.create_evaluator(
            evaluation_type=self._evaluator_type,
            framework=self._framework,
            comparison_config=self._comparison_config,
            quality_config=self._quality_config,
            metric_names=self._metric_names,
            metric_kwargs=self._metric_kwargs,
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
