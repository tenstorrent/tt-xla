# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Quality metrics for evaluating model output quality.

This module provides a factory-based system for creating quality metrics
using string-based identifiers, simplifying test authoring.

Example usage:
    >>> from tests.infra.metrics import get_metric
    >>> clip = get_metric('clip')
    >>> fid = get_metric('fid', statistics_mean=mean, statistics_cov=cov)
"""

from .metrics import QualityMetric, get_metric

__all__ = [
    "QualityMetric",
    "get_metric",
]
