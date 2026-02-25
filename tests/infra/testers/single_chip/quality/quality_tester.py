# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Optional

from infra.evaluators import QualityConfig, QualityResult
from infra.testers.base_tester import BaseTester


class QualityTester(BaseTester):
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
            metric_names: List of metric names to evaluate
        """
        self._last_result: Optional[QualityResult] = None

        super().__init__(
            evaluator_type="quality",
            quality_config=(
                quality_config if quality_config is not None else QualityConfig()
            ),
            metric_names=metric_names or [],
            metric_kwargs=metric_kwargs or {},
        )

    @abstractmethod
    def test(self, request=None) -> QualityResult:
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
    def serialize_on_device(self, workload=None, output_prefix: str = None) -> None:
        """
        Serializes the model workload on TT device with proper compiler configuration.

        Args:
            output_prefix: Base path and filename prefix for output files
        """
        raise NotImplementedError("Subclasses must implement this method.")
