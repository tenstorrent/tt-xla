#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Run ``tests/benchmark/scripts/debug_gpt_oss_input_sharding.py`` from repo root.

    python scripts/debug_gpt_oss_input_sharding.py --mode compare
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_TARGET = _REPO / "tests/benchmark/scripts/debug_gpt_oss_input_sharding.py"
_BENCH = _REPO / "tests/benchmark"


def main() -> None:
    if not _TARGET.is_file():
        print(f"Missing {_TARGET}", file=sys.stderr)
        raise SystemExit(1)
    raise SystemExit(
        subprocess.call(
            [sys.executable, str(_TARGET), *sys.argv[1:]],
            cwd=_BENCH,
        )
    )


if __name__ == "__main__":
    main()
