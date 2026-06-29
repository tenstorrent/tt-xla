# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for codegen emit/load: TTXLA_CODEGEN_EXPORT_DIR emits and executes
generated Python per graph; TTXLA_CODEGEN_LOAD_DIR runs the saved (possibly
user-edited) code instead of compiling. Emit and load run in separate
processes since torch_xla caches compiled graphs in-process."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

# Runs num_graphs forwards with distinct batch sizes (so each compiles a
# separate graph) and verifies outputs against CPU.
MLP_SCRIPT = textwrap.dedent(
    """
    import sys
    import torch
    import torch.nn as nn
    import torch_xla
    import torch_xla.runtime as xr

    num_graphs = int(sys.argv[1])

    xr.set_device_type("TT")
    device = torch_xla.device()

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10)).to(
        torch.bfloat16
    )
    inputs = [
        torch.randn(4 * 2**i, 64, dtype=torch.bfloat16) for i in range(num_graphs)
    ]
    goldens = [model(x) for x in inputs]

    xla_model = model.to(device)
    for x, golden in zip(inputs, goldens):
        out = xla_model(x.to(device))
        torch_xla.sync()
        ok = torch.allclose(out.cpu().float(), golden.float(), atol=0.1)
        print(f"batch {x.shape[0]} match: {ok}")
        assert ok
    """
)


def run_mlp(env_extra, num_graphs=1):
    env = {**os.environ, **env_extra}
    return subprocess.run(
        [sys.executable, "-c", MLP_SCRIPT, str(num_graphs)],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )


def graph_dirs(export_dir: Path):
    return sorted(d for d in export_dir.iterdir() if (d / "module_key").exists())


def insert_sentinel(main_py: Path, sentinel: Path):
    lines = main_py.read_text().splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.startswith("def forward("):
            lines.insert(i + 1, f'    open(r"{sentinel}", "w").write("hit")\n')
            break
    else:
        raise AssertionError(f"no forward() in {main_py}")
    main_py.write_text("".join(lines))


@pytest.mark.push
@pytest.mark.single_device
def test_emit_then_load_with_edit(tmp_path):
    export_dir = tmp_path / "emitted"

    result = run_mlp({"TTXLA_CODEGEN_EXPORT_DIR": str(export_dir)})
    assert result.returncode == 0, result.stderr[-2000:]

    dirs = graph_dirs(export_dir)
    assert len(dirs) == 1, f"expected 1 graph dir, got {dirs}"
    assert (dirs[0] / "main.py").exists()
    assert (export_dir / "manifest.json").exists()

    sentinel = tmp_path / "sentinel"
    insert_sentinel(dirs[0] / "main.py", sentinel)

    result = run_mlp({"TTXLA_CODEGEN_LOAD_DIR": str(export_dir)})
    assert result.returncode == 0, result.stderr[-2000:]
    assert sentinel.exists(), "edited main.py was not executed in load mode"


@pytest.mark.push
@pytest.mark.single_device
def test_load_missing_graph_fails(tmp_path):
    export_dir = tmp_path / "emitted"

    result = run_mlp({"TTXLA_CODEGEN_EXPORT_DIR": str(export_dir)})
    assert result.returncode == 0, result.stderr[-2000:]

    result = run_mlp({"TTXLA_CODEGEN_LOAD_DIR": str(export_dir)}, num_graphs=2)
    assert result.returncode != 0
    assert "no saved graph with hash" in (result.stderr + result.stdout)


@pytest.mark.push
@pytest.mark.single_device
def test_emit_then_load_multi_graph(tmp_path):
    export_dir = tmp_path / "emitted"

    result = run_mlp({"TTXLA_CODEGEN_EXPORT_DIR": str(export_dir)}, num_graphs=2)
    assert result.returncode == 0, result.stderr[-2000:]
    dirs = graph_dirs(export_dir)
    assert len(dirs) == 2, f"expected 2 graph dirs, got {dirs}"

    sentinels = []
    for d in dirs:
        sentinel = tmp_path / f"sentinel_{d.name}"
        insert_sentinel(d / "main.py", sentinel)
        sentinels.append(sentinel)

    result = run_mlp({"TTXLA_CODEGEN_LOAD_DIR": str(export_dir)}, num_graphs=2)
    assert result.returncode == 0, result.stderr[-2000:]
    for sentinel in sentinels:
        assert sentinel.exists(), f"{sentinel.name} not executed in load mode"
