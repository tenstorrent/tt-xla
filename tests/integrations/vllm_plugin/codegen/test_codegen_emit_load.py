# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Emit TTNN Python codegen during a vLLM run (TTXLA_CODEGEN_EXPORT_DIR), edit
the emitted code, then reload it instead of compiling (TTXLA_CODEGEN_LOAD_DIR).
Each phase runs in its own process since torch_xla caches graphs in-process."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

# Standalone vLLM generation, run in its own process per phase so torch_xla's
# in-process graph cache doesn't leak between emit and load.
VLLM_SCRIPT = Path(__file__).parent / "vllm_serve_helper.py"


def run_vllm(tmp_path, env_extra, name):
    out_file = tmp_path / f"{name}.txt"
    env = {**os.environ, **env_extra}
    result = subprocess.run(
        [sys.executable, str(VLLM_SCRIPT), str(out_file)],
        env=env,
        capture_output=True,
        text=True,
        timeout=3000,
    )
    return result, out_file


def insert_sentinel(main_py: Path, sentinel: Path):
    lines = main_py.read_text().splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.startswith("def forward("):
            lines.insert(i + 1, f'    open(r"{sentinel}", "a").write("hit")\n')
            break
    else:
        raise AssertionError(f"no forward() in {main_py}")
    main_py.write_text("".join(lines))


@pytest.mark.skip(
    reason=(
        "Pre-existing compiler failure, unrelated to buffer keying: the emit "
        "subprocess trips the known torch-xla 'Bad StatusOr access: INTERNAL: "
        "Error code: 13' at torch_xla.sync during graph extraction "
        "(https://github.com/tenstorrent/tt-xla/issues/5338). It also fails on "
        "main. On device this can manifest as a hang rather than a fast crash; "
        "the subprocess then runs out its 3000s timeout and leaves the device "
        "wedged, cascading a 240-min job timeout into the next test "
        "(test_b1_prefill_ttft). Skip until #5338 is fixed so it cannot wedge "
        "the shared device for neighbouring tests."
    )
)
@pytest.mark.nightly
@pytest.mark.single_device
def test_vllm_codegen_emit_then_load(tmp_path):
    export_dir = tmp_path / "emitted"

    result, emit_out = run_vllm(
        tmp_path, {"TTXLA_CODEGEN_EXPORT_DIR": str(export_dir)}, "emit"
    )
    assert result.returncode == 0, result.stderr[-4000:]
    assert emit_out.exists()

    graph_dirs = sorted(d for d in export_dir.iterdir() if (d / "module_key").exists())
    assert len(graph_dirs) >= 3, f"expected several graphs, got {graph_dirs}"
    assert (export_dir / "manifest.json").exists()

    sentinels = []
    for d in graph_dirs:
        sentinel = tmp_path / f"sentinel_{d.name}"
        insert_sentinel(d / "main.py", sentinel)
        sentinels.append(sentinel)

    result, load_out = run_vllm(
        tmp_path, {"TTXLA_CODEGEN_LOAD_DIR": str(export_dir)}, "load"
    )
    assert result.returncode == 0, result.stderr[-4000:]

    executed = [s.name for s in sentinels if s.exists()]
    # Check that something was executed and produced
    assert executed, "no edited graph was executed in load mode"
    assert load_out.read_text().strip(), "load mode produced no output"
