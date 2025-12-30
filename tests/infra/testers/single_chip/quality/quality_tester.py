# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Optional

from infra.evaluators import EvaluatorFactory, QualityEvaluator, QualityResult
from infra.evaluators.quality_config import QualityConfig


class QualityTester:
    """
    Abstract base class for quality metric-based testing.
    """

    def __init__(
        self,
        quality_config: Optional[QualityConfig] = None,
        metric_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
        metric_names: Optional[List[str]] = None,
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
        self._metric_names: List[str] = metric_names or []

    def _initialize_quality_evaluator(self) -> None:
        assert (
            self._metric_names is not None and len(self._metric_names) > 0
        ), "Metric names are required for quality evaluators"
        evaluator = EvaluatorFactory.create_evaluator(
            evaluation_type="quality",
            quality_config=self._quality_config,
            metric_names=self._metric_names,
            metric_kwargs=self._metric_kwargs,
        )

        assert isinstance(evaluator, QualityEvaluator)
        self._quality_evaluator = evaluator

    @abstractmethod
    def test(self) -> QualityResult:
        raise NotImplementedError("Subclasses must implement test()")

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
