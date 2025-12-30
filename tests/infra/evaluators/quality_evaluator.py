# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from infra.metrics import QualityMetric, get_metric

from .evaluator import Evaluator, QualityResult
from .quality_config import QualityConfig


class QualityEvaluator(Evaluator):
    """
    Evaluator that assesses output quality using application-specific metrics.
    """

    def __init__(
        self,
        metric_names: List[str],
        quality_config: QualityConfig,
        metric_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        """
        Initialize the quality evaluator.

        Args:
            metric_names: List of metric names to compute (e.g., ["clip", "fid"])
            quality_config: Configuration with thresholds for each metric
            metric_kwargs: Optional dict mapping metric_name -> kwargs for metric creation
                          e.g., {"fid": {"statistics_mean": ..., "statistics_cov": ...}}
        """
        self._quality_config = quality_config
        self._metric_kwargs = metric_kwargs or {}

        # Initialize metrics using string-based selection
        self._metrics: Dict[str, QualityMetric] = {}
        for name in metric_names:
            kwargs = self._metric_kwargs.get(name, {})
            self._metrics[name] = get_metric(name, **kwargs)

    def evaluate(
        self,
        images: torch.Tensor,
        prompts: Optional[List[str]] = None,
        **kwargs,
    ) -> QualityResult:
        """
        Evaluate output quality using configured metrics.

        Args:
            images: Generated images tensor (N, C, H, W)
            prompts: Optional text prompts used for generation (required for CLIP)
            **kwargs: Additional metric-specific parameters

        Returns:
            QualityResult with computed metrics and pass/fail status
        """
        computed_metrics = self._compute_all_metrics(images, prompts, **kwargs)
        passed, error_message = self._evaluate_thresholds(computed_metrics)

        return QualityResult(
            passed=passed,
            error_message=error_message,
            metrics=computed_metrics,
        )

    def _compute_all_metrics(
        self,
        images: torch.Tensor,
        prompts: Optional[List[str]],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Compute all configured metrics.

        Args:
            images: Generated images tensor
            prompts: Optional text prompts
            **kwargs: Additional metric-specific parameters

        Returns:
            Dictionary of computed metric values
        """
        results = {}
        for name, metric in self._metrics.items():
            result = metric.compute(images, prompts, **kwargs)
            # Handle tuple returns (like CLIP which returns mean, min)
            if isinstance(result, tuple):
                results[f"{name}_mean"] = result[0]
                results[f"{name}_min"] = result[1]
            else:
                results[name] = result
        return results

    def _evaluate_thresholds(
        self,
        metrics: Dict[str, Any],
    ) -> Tuple[bool, str | None]:
        """
        Evaluate computed metrics against configured thresholds.

        Args:
            metrics: Dictionary of computed metric values

        Returns:
            Tuple of (passed, error_message)
        """
        passed = True
        error_messages = []

        # Check CLIP threshold (higher is better, use clip_min for worst-case)
        if "clip_min" in metrics:
            min_clip = self._quality_config.min_clip_threshold
            if metrics["clip_min"] < min_clip:
                passed = False
                error_messages.append(
                    f"CLIP quality check failed. "
                    f"Calculated: clip_min={metrics['clip_min']:.2f}. "
                    f"Required: min_clip_threshold={min_clip:.2f}."
                )

        # Check FID threshold (lower is better)
        if "fid" in metrics:
            max_fid = self._quality_config.max_fid_threshold
            if metrics["fid"] > max_fid:
                passed = False
                error_messages.append(
                    f"FID quality check failed. "
                    f"Calculated: fid={metrics['fid']:.2f}. "
                    f"Required: max_fid_threshold={max_fid:.2f}."
                )

        combined_error = "\n".join(error_messages) if error_messages else None
        return passed, combined_error

    @property
    def metrics(self) -> Dict[str, QualityMetric]:
        """Return the configured metrics."""
        return self._metrics

    @property
    def quality_config(self) -> QualityConfig:
        """Return the quality configuration."""
        return self._quality_config

    @staticmethod
    def _assert_on_results(result: QualityResult) -> None:
        """Raise AssertionError if quality check failed."""
        if not result.passed:
            raise AssertionError(result.error_message)
