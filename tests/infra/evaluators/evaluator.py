# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Tuple


@dataclass
class EvaluationResult:
    """Base result type for all evaluators."""

    passed: bool
    error_message: str | None = None


@dataclass
class ComparisonResult(EvaluationResult):
    """Result from comparison-based evaluation (device vs golden)."""

    pcc: float | None = None
    atol: float | None = None
    allclose: bool | None = None
    equal: bool | None = None


@dataclass
class QualityResult(EvaluationResult):
    """Result from quality metric-based evaluation."""

    metrics: dict[str, Any] | None = None


class Evaluator(ABC):
    """
    Abstract base class for all evaluators.
    """

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> EvaluationResult:
        """
        Perform evaluation and return result.
        This is a standard API for evaluators to implement.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @staticmethod
    def _assert_on_results(
        result: EvaluationResult | Tuple[EvaluationResult, ...],
    ) -> None:
        """
        Assert based on evaluation results if any checks failed.

        Args:
            result: Either a single EvaluationResult or a tuple of EvaluationResults.
            There may be multiple results for each test, eg. forward and backward pass
            results for training.
        """
        if isinstance(result, EvaluationResult):
            results = (result,)
        else:
            results = result

        error_messages = []
        for i, res in enumerate(results):
            if not res.passed:
                error_messages.append(
                    f"Evaluation result {i} failed: {res.error_message}"
                )
        if error_messages:
            assert False, "\n".join(error_messages)
