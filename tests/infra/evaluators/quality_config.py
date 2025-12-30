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
    max_fid_threshold: float = 350.0

    assert_on_failure: bool = True
