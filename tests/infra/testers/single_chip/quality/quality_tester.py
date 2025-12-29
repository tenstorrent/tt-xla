# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Optional

from infra.evaluators import EvaluatorFactory, QualityEvaluator, QualityResult
from infra.evaluators.quality_config import QualityConfig
from infra.runners import DeviceRunnerFactory
from infra.utilities import Framework


class QualityTester:
    """
    Abstract base class for quality metric-based testing.

    Unlike OpTester/GraphTester which compare CPU vs TT device outputs using PCC,
    QualityTester runs workloads on the target device and evaluates output quality
    using application-specific metrics.
    """

    def __init__(
        self,
        quality_config: Optional[QualityConfig] = None,
        metric_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        """
        Initialize the quality tester.

        Args:
            quality_config: Configuration for quality thresholds
            metric_kwargs: Optional dict mapping metric_name -> kwargs for metric creation
        """
        self._quality_config = (
            quality_config if quality_config is not None else QualityConfig()
        )
        self._metric_kwargs = metric_kwargs or {}
        self._quality_evaluator: Optional[QualityEvaluator] = None
        self._last_result: Optional[QualityResult] = None

        # Create device runner for torch framework (quality tests use torch)
        self._device_runner = DeviceRunnerFactory.create_runner(Framework.TORCH)

    def _initialize_quality_evaluator(self) -> None:
        """
        Initialize the QualityEvaluator with configured metrics.

        Called lazily when metrics are first needed to allow subclasses
        to set up metric_kwargs in their constructors.
        """
        metric_names = self._get_metric_names()
        evaluator = EvaluatorFactory.create_evaluator(
            evaluation_type="quality",
            quality_config=self._quality_config,
            metric_names=metric_names,
            metric_kwargs=self._metric_kwargs,
        )
        # Factory returns Evaluator base type, but we know it's QualityEvaluator
        assert isinstance(evaluator, QualityEvaluator)
        self._quality_evaluator = evaluator

    @abstractmethod
    def _get_metric_names(self) -> List[str]:
        """
        Return list of metric names to evaluate.

        Override in subclasses to specify which metrics to use.
        E.g., ["clip"] or ["clip", "fid"]
        """
        raise NotImplementedError("Subclasses must implement _get_metric_names()")

    @abstractmethod
    def _generate_outputs(self) -> Any:
        """
        Generate outputs to evaluate (e.g., images).

        Returns framework-specific output tensor.
        """
        raise NotImplementedError("Subclasses must implement _generate_outputs()")

    def _get_prompts(self) -> Optional[List[str]]:
        """
        Return prompts if needed for metrics like CLIP.

        Override in subclasses that use prompt-based metrics.
        Returns None by default.
        """
        return None

    def test(self) -> QualityResult:
        """
        Main test entry point.

        Generates outputs, evaluates quality, and optionally asserts on failure.

        Returns:
            QualityResult with computed metrics and pass/fail status
        """
        # lazy init
        if self._quality_evaluator is None:
            self._initialize_quality_evaluator()

        outputs = self._generate_outputs()
        prompts = self._get_prompts()

        # At this point, evaluator is guaranteed to be initialized
        assert self._quality_evaluator is not None
        self._last_result = self._quality_evaluator.evaluate(outputs, prompts)

        if self._quality_config.assert_on_failure and not self._last_result.passed:
            QualityEvaluator._assert_on_results(self._last_result)

        return self._last_result

    @property
    def metrics(self) -> Dict[str, Any]:
        """Returns the computed metrics after test() has been called."""
        if self._last_result is None:
            return {}
        return self._last_result.metrics or {}

    @property
    def result(self) -> Optional[QualityResult]:
        """Returns the last evaluation result."""
        return self._last_result

    @property
    def quality_config(self) -> QualityConfig:
        """Returns the quality configuration."""
        return self._quality_config

    @abstractmethod
    def serialize_on_device(self, output_prefix: str) -> None:
        """
        Serializes the model workload on TT device with proper compiler configuration.

        Args:
            output_prefix: Base path and filename prefix for output files
        """
        raise NotImplementedError("Subclasses must implement this method.")
