# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any


class EvaluatorType(Enum):
    """Enum representing the type of evaluator."""

    COMPARISON = "comparison"
    QUALITY = "quality"


class Evaluator(ABC):
    """
    Abstract base class for all evaluators.

    Evaluators assess the quality of device outputs. This includes:
    - Comparison-based evaluation: comparing device output to golden reference (CPU)
    - Quality-based evaluation: assessing output quality using metrics (CLIP, FID, etc.)

    Subclasses must implement the evaluate() method with their specific signature.
    """

    @abstractmethod
    def evaluate(self, device_output: Any, **kwargs) -> Any:
        """
        Evaluate the device output.

        Args:
            device_output: The output from the device to evaluate
            **kwargs: Additional arguments specific to the evaluator type

        Returns:
            An evaluation result object (ComparisonResult, QualityResult, etc.)
        """
        raise NotImplementedError("Subclasses must implement this method")
