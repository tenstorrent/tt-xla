# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Emit/load tests for tensor-parallel (SPMD) graphs across the full mesh.
Emit is a dry run that only generates code (the device returns zero buffers, so
correctness is not checked in emit mode); the load run executes the edited
generated Python and is checked against the CPU golden. Emit and load run in
separate processes since torch_xla caches compiled graphs in-process."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

# TODO(#5481): Skip in CI until EmitPy execution is available in the CI build.
pytestmark = pytest.mark.skipif(
    os.environ.get("GITHUB_ACTIONS") == "true",
    reason="codegen load needs EmitPy execution (PythonModelRunner), which the "
    "manylinux CI wheel omits; see tenstorrent/tt-xla#5481",
)

TP_HELPER = Path(__file__).parent / "codegen_emit_load_multichip_helper.py"


def run_tp(env_extra):
    env = {**os.environ, **env_extra}
    return subprocess.run(
        [sys.executable, str(TP_HELPER)],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )


def insert_sentinel(main_py: Path, sentinel: Path):
    lines = main_py.read_text().splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.startswith("def forward("):
            lines.insert(i + 1, f'    open(r"{sentinel}", "w").write("hit")\n')
            break
    else:
        raise AssertionError(f"no forward() in {main_py}")
    main_py.write_text("".join(lines))


@pytest.mark.nightly
@pytest.mark.llmbox
def test_tp_emit_then_load_with_edit(tmp_path):
    export_dir = tmp_path / "emitted"

    result = run_tp({"TTXLA_CODEGEN_EXPORT_DIR": str(export_dir)})
    assert result.returncode == 0, result.stderr[-2000:]
    assert (export_dir / "manifest.json").exists(), result.stderr[-2000:]
    dirs = [d for d in export_dir.iterdir() if (d / "module_key").exists()]
    assert len(dirs) == 1, f"expected 1 graph dir, got {dirs}"
    key_lines = (dirs[0] / "module_key").read_text().split()
    assert len(key_lines) == 3 and "x" in key_lines[1], key_lines

    sentinel = tmp_path / "sentinel"
    insert_sentinel(dirs[0] / "main.py", sentinel)

    result = run_tp({"TTXLA_CODEGEN_LOAD_DIR": str(export_dir)})
    assert "Codegen load: graph" in result.stderr + result.stdout
    assert sentinel.exists(), "edited main.py was not executed in load mode"
    assert "match: True" in result.stdout, "loaded TP output does not match CPU golden"
