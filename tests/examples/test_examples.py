# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import os
import sys
import subprocess
from pathlib import Path

import pytest

EXAMPLES_DIR = Path("examples")
PATTERN = "**/*.py"
TIMEOUT_SECS = 5 * 60


def _discover_examples() -> list[Path]:
    files = [
        p for p in EXAMPLES_DIR.glob(PATTERN) if p.is_file() and p.name != "__init__.py"
    ]
    return sorted(files, key=lambda p: str(p))


@pytest.mark.push
@pytest.mark.parametrize("script", _discover_examples(), ids=lambda p: str(p))
def test_examples(script: Path):
    cmd = [sys.executable, str(script)]
    env = os.environ.copy()

    proc = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_SECS,
    )

    if proc.returncode != 0:
        out_tail = "\n".join(proc.stdout.splitlines()[-100:])
        err_tail = "\n".join(proc.stderr.splitlines()[-100:])
        pytest.fail(
            f"Example failed: {script}\n"
            f"returncode: {proc.returncode}\n"
            f"--- stdout (tail) ---\n{out_tail}\n"
            f"--- stderr (tail) ---\n{err_tail}\n"
        )
