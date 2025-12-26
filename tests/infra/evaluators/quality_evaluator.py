# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any, List, Optional

import torch
from infra.metrics import QualityMetric, get_metric

from .evaluation_config import QualityConfig
from .evaluator import Evaluator


@dataclass
class QualityResult:
    """
    Result from quality evaluation using metrics like CLIP or FID.

    Attributes:
        passed: Whether the quality check passed (None if not evaluated)
        clip_mean: Mean CLIP score across samples (if CLIP metric used)
        clip_min: Minimum CLIP score (worst case, if CLIP metric used)
        fid_score: FID score (if FID metric used)
        num_samples: Number of samples evaluated
        error_message: Error message if evaluation failed
    """

    passed: bool | None
    clip_mean: float | None = None
    clip_min: float | None = None
    fid_score: float | None = None
    num_samples: int | None = None
    error_message: str | None = None


class QualityEvaluator(Evaluator):
    """
    Evaluator that assesses output quality using metrics like CLIP and FID.

    Unlike ComparisonEvaluator which compares device output to golden reference,
    QualityEvaluator computes quality metrics on the device output alone.
    This is useful for generative models where there's no single correct output.

    Example usage:
        evaluator = QualityEvaluator(config=quality_config, metric="clip")
        result = evaluator.evaluate(images, prompts=captions)
        if not result.passed:
            raise AssertionError(result.error_message)
    """

    def __init__(
        self, config: QualityConfig, metric: str, **metric_kwargs: Any
    ) -> None:
        """
        Initialize the quality evaluator.

        Args:
            config: Quality configuration with thresholds
            metric: Metric name string ('clip', 'fid', etc.)
            **metric_kwargs: Additional arguments for metric creation (e.g., FID statistics)
        """
        self._config = config
        self._metric = get_metric(metric, **metric_kwargs)

    def evaluate(
        self,
        device_output: torch.Tensor,
        *,
        prompts: Optional[List[str]] = None,
        **kwargs,
    ) -> QualityResult:
        """
        Evaluate quality of generated images using the configured metric.

        Args:
            device_output: Generated images tensor of shape (N, C, H, W)
            prompts: Text prompts used for generation (required for CLIP)
            **kwargs: Additional metric-specific arguments

        Returns:
            QualityResult with computed metrics and pass/fail status
        """
        metric_name = self._metric.name.lower()

        if metric_name == "clip":
            return self._evaluate_clip(device_output, prompts)
        elif metric_name == "fid":
            return self._evaluate_fid(device_output)
        else:
            raise ValueError(f"Unsupported metric: {metric_name}")

    def _evaluate_clip(
        self, images: torch.Tensor, prompts: Optional[List[str]]
    ) -> QualityResult:
        """Evaluate using CLIP metric."""
        if prompts is None:
            return QualityResult(
                passed=False,
                error_message="CLIP metric requires prompts but none were provided.",
            )

        clip_mean, clip_min = self._metric.compute(images, prompts)
        num_samples = images.shape[0]

        # Evaluate against threshold
        passed, error_message = self._check_clip_threshold(clip_min)

        return QualityResult(
            passed=passed,
            clip_mean=float(clip_mean),
            clip_min=float(clip_min),
            num_samples=num_samples,
            error_message=error_message,
        )

    def _evaluate_fid(self, images: torch.Tensor) -> QualityResult:
        """Evaluate using FID metric."""
        fid_score = self._metric.compute(images)
        num_samples = images.shape[0]

        # Evaluate against threshold
        passed, error_message = self._check_fid_threshold(fid_score)

        return QualityResult(
            passed=passed,
            fid_score=float(fid_score),
            num_samples=num_samples,
            error_message=error_message,
        )

    def _check_clip_threshold(self, clip_min: float) -> tuple[bool, str | None]:
        """Check if CLIP score meets the minimum threshold."""
        min_threshold = self._config.min_clip_threshold

        if clip_min < min_threshold:
            return False, (
                f"CLIP score regression detected: "
                f"clip_min={clip_min:.2f} < threshold={min_threshold:.2f}"
            )
        return True, None

    def _check_fid_threshold(self, fid_score: float) -> tuple[bool, str | None]:
        """Check if FID score meets the maximum threshold."""
        max_threshold = self._config.max_fid_threshold

        if fid_score > max_threshold:
            return False, (
                f"FID score regression detected: "
                f"fid_score={fid_score:.2f} > threshold={max_threshold:.2f}"
            )
        return True, None

    @property
    def metric(self) -> QualityMetric:
        """Return the underlying metric."""
        return self._metric

    @property
    def config(self) -> QualityConfig:
        """Return the quality configuration."""
        return self._config
