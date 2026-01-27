# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent.parent / "examples"
PATTERN = "**/*.py"
TIMEOUT_SECS = 20 * 60

# Directories with known issues - tests will be marked as xfail with the given reason
XFAIL_DIRS: dict[str, str] = {
    "vllm": "vLLM examples require server setup and are not suitable for automated testing",
}


def _get_xfail_reason(filepath: Path) -> str | None:
    """Return xfail reason if filepath is in an xfail directory, None otherwise."""
    for xfail_dir, reason in XFAIL_DIRS.items():
        if xfail_dir in filepath.parts:
            return reason
    return None


def _has_pytest_tests(filepath: Path) -> bool:
    """Check if a file contains pytest test functions."""
    try:
        source = filepath.read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                return True
    except (SyntaxError, UnicodeDecodeError):
        pass
    return False


def _has_main_block(filepath: Path) -> bool:
    r"""Check if a file has an `if __name__ == '__main__':` block."""
    try:
        source = filepath.read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                # Check for: if __name__ == "__main__"
                test = node.test
                if isinstance(test, ast.Compare):
                    if (
                        isinstance(test.left, ast.Name)
                        and test.left.id == "__name__"
                        and len(test.ops) == 1
                        and isinstance(test.ops[0], ast.Eq)
                        and len(test.comparators) == 1
                        and isinstance(test.comparators[0], ast.Constant)
                        and test.comparators[0].value == "__main__"
                    ):
                        return True
    except (SyntaxError, UnicodeDecodeError):
        pass
    return False


def _discover_pytest_examples() -> list[Path]:
    """Discover example files that contain pytest tests."""
    files = [
        p
        for p in EXAMPLES_DIR.glob(PATTERN)
        if p.is_file() and p.name != "__init__.py" and _has_pytest_tests(p)
    ]
    return sorted(files, key=lambda p: str(p))


def _discover_script_examples() -> list[Path]:
    """Discover example files that have a main block but no pytest tests."""
    files = [
        p
        for p in EXAMPLES_DIR.glob(PATTERN)
        if p.is_file()
        and p.name != "__init__.py"
        and _has_main_block(p)
        and not _has_pytest_tests(p)
    ]
    return sorted(files, key=lambda p: str(p))


@pytest.mark.push
@pytest.mark.parametrize("script", _discover_pytest_examples(), ids=lambda p: str(p))
def test_pytest_examples(script: Path):
    """Run example files that contain pytest tests using pytest."""
    xfail_reason = _get_xfail_reason(script)
    if xfail_reason:
        pytest.xfail(xfail_reason)

    cmd = [sys.executable, "-m", "pytest", "-v", str(script)]
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
            f"Pytest example failed: {script}\n"
            f"returncode: {proc.returncode}\n"
            f"--- stdout (tail) ---\n{out_tail}\n"
            f"--- stderr (tail) ---\n{err_tail}\n"
        )


@pytest.mark.push
@pytest.mark.parametrize("script", _discover_script_examples(), ids=lambda p: str(p))
def test_script_examples(script: Path):
    """Run example files that have a main block as standalone scripts."""
    xfail_reason = _get_xfail_reason(script)
    if xfail_reason:
        pytest.xfail(xfail_reason)

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
            f"Script example failed: {script}\n"
            f"returncode: {proc.returncode}\n"
            f"--- stdout (tail) ---\n{out_tail}\n"
            f"--- stderr (tail) ---\n{err_tail}\n"
        )
