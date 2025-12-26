# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Optional

from infra.utilities import Framework

from .evaluation_config import ComparisonConfig, QualityConfig
from .evaluator import Evaluator, EvaluatorType
from .jax_comparison_evaluator import JaxComparisonEvaluator
from .quality_evaluator import QualityEvaluator
from .torch_comparison_evaluator import TorchComparisonEvaluator


class EvaluatorFactory:
    """Factory for creating evaluator instances."""

    @staticmethod
    def create_evaluator(
        evaluator_type: EvaluatorType,
        *,
        framework: Optional[Framework] = None,
        comparison_config: Optional[ComparisonConfig] = None,
        quality_config: Optional[QualityConfig] = None,
        metric: Optional[str] = None,
        metric_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Evaluator:
        """
        Create an evaluator of the specified type.

        Args:
            evaluator_type: Type of evaluator to create (EvaluatorType.COMPARISON or EvaluatorType.QUALITY)
            framework: The framework (JAX or TORCH). Required for comparison evaluators.
            comparison_config: Configuration for comparison thresholds. Used for comparison evaluators.
            quality_config: Configuration for quality thresholds. Used for quality evaluators.
            metric: Metric name string ('clip', 'fid', etc.). Required for quality evaluators.
            metric_kwargs: Additional arguments for metric creation (e.g., FID statistics).

        Returns:
            Evaluator instance of the requested type

        Raises:
            ValueError: If required arguments are missing or invalid
        """
        if evaluator_type == EvaluatorType.COMPARISON:
            if framework is None:
                raise ValueError("framework is required for comparison evaluators")
            config = comparison_config or ComparisonConfig()

            if framework == Framework.JAX:
                return JaxComparisonEvaluator(config)
            elif framework == Framework.TORCH:
                return TorchComparisonEvaluator(config)
            else:
                raise ValueError(f"Unsupported framework: {framework}")

        elif evaluator_type == EvaluatorType.QUALITY:
            if metric is None:
                raise ValueError("metric is required for quality evaluators")
            config = quality_config or QualityConfig()
            return QualityEvaluator(
                config=config, metric=metric, **(metric_kwargs or {})
            )

        else:
            raise ValueError(f"Unsupported evaluator type: {evaluator_type}")
