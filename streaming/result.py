# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Return value of `streaming.core.run_streaming`."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class StreamingResult:
    """Outcome of a streaming inference run."""

    # generated_ids[i] = full token sequence for batch row i (prefill + decode);
    # EOS trimming happens at print time.
    generated_ids: List[List[int]] = field(default_factory=list)
    prompts_used: List[str] = field(default_factory=list)

    # Wall-clock seconds per phase.
    timing: Dict[str, float] = field(default_factory=dict)
