# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Dict, List, Optional

from infra.utilities import Framework

from .comparison_evaluator import ComparisonEvaluator
from .evaluation_config import ComparisonConfig
from .evaluator import Evaluator
from .jax_comparison_evaluator import JaxComparisonEvaluator
from .quality_config import QualityConfig
from .quality_evaluator import QualityEvaluator
from .torch_comparison_evaluator import TorchComparisonEvaluator


class EvaluatorFactory:
    """
    Factory for creating evaluators based on evaluation type and framework.
    """

    @staticmethod
    def create_evaluator(
        evaluation_type: str,
        framework: Optional[Framework] = None,
        comparison_config: Optional[ComparisonConfig] = None,
        quality_config: Optional[QualityConfig] = None,
        metric_names: Optional[List[str]] = None,
        metric_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Evaluator:
        """
        Create an evaluator based on the specified type.

        Args:
            evaluation_type: Type of evaluator to create
            framework: Required for comparison evaluators (JAX or TORCH)
            comparison_config: Config for comparison evaluator
            quality_config: Config for quality evaluator
            metric_names: Metric names for quality evaluator
            metric_kwargs: Additional kwargs per metric for quality evaluator

        Returns:
            Appropriate Evaluator subclass instance

        Raises:
            ValueError: If evaluation_type is not recognized or required params missing
        """
        if evaluation_type == "comparison":
            return EvaluatorFactory._create_comparison_evaluator(
                framework, comparison_config
            )
        elif evaluation_type == "quality":
            assert (
                framework is None or framework == Framework.TORCH
            ), "Quality evaluators currently only support Pytorch framework"
            return EvaluatorFactory._create_quality_evaluator(
                quality_config, metric_names, metric_kwargs
            )
        else:
            raise ValueError(
                f"Unsupported evaluation type: {evaluation_type}. "
                f"Use 'comparison' or 'quality'."
            )

    @staticmethod
    def _create_comparison_evaluator(
        framework: Optional[Framework],
        comparison_config: Optional[ComparisonConfig],
    ) -> ComparisonEvaluator:
        """
        Create framework-specific ComparisonEvaluator.

        Args:
            framework: Which framework comparison evaluator to create
            comparison_config: Comparison configuration
        """
        if framework is None:
            raise ValueError("framework is required for comparison evaluators")
        if comparison_config is None:
            comparison_config = ComparisonConfig()

        if framework == Framework.JAX:
            return JaxComparisonEvaluator(comparison_config)
        elif framework == Framework.TORCH:
            return TorchComparisonEvaluator(comparison_config)
        else:
            raise ValueError(f"Unsupported framework: {framework}")

    @staticmethod
    def _create_quality_evaluator(
        quality_config: Optional[QualityConfig],
        metric_names: Optional[List[str]],
        metric_kwargs: Optional[Dict[str, Dict[str, Any]]],
    ) -> QualityEvaluator:
        """
        Create QualityEvaluator with specified metrics.

        Args:
            quality_config: Quality configuration with thresholds
            metric_names: List of metric names to compute
            metric_kwargs: Additional kwargs per metric

        Returns:
            QualityEvaluator instance
        """
        if metric_names is None or len(metric_names) == 0:
            raise ValueError("metric_names is required for quality evaluators")
        if quality_config is None:
            quality_config = QualityConfig()

        return QualityEvaluator(
            metric_names=metric_names,
            quality_config=quality_config,
            metric_kwargs=metric_kwargs,
        )
