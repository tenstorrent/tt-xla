# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from .comparison_evaluator import ComparisonEvaluator, ComparisonResult
from .evaluation_config import (
    AllcloseConfig,
    AtolConfig,
    ComparisonConfig,
    ConfigBase,
    EqualConfig,
    PccConfig,
    QualityConfig,
)
from .evaluator import Evaluator, EvaluatorType
from .evaluator_factory import EvaluatorFactory
from .jax_comparison_evaluator import JaxComparisonEvaluator
from .quality_evaluator import QualityEvaluator, QualityResult
from .torch_comparison_evaluator import TorchComparisonEvaluator

# Backward compatibility aliases - deprecated, use new names instead
Comparator = ComparisonEvaluator
JaxComparator = JaxComparisonEvaluator
TorchComparator = TorchComparisonEvaluator
ComparatorFactory = EvaluatorFactory
