# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class QualityConfig:
    """Configuration for quality-based evaluation thresholds."""

    # CLIP score thresholds (higher is better)
    min_clip_threshold: float = 25.0

    # FID score threshold (lower is better)
    max_fid_threshold: float = float("inf")

    # Generic threshold map for extensibility
    # Allows adding new metrics without code changes
    # Format: {"metric_name": {"threshold": value, "comparison": "gt" | "lt"}}
    # "gt" means metric must be greater than threshold (higher is better)
    # "lt" means metric must be less than threshold (lower is better)
    custom_thresholds: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Whether to assert on threshold failure (consistent with ComparisonConfig)
    assert_on_failure: bool = True
