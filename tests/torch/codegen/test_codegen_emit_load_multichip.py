# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Emit/load tests for tensor-parallel (SPMD) graphs across the full mesh.
The load run executes the edited generated Python; emit and load run in
separate processes since torch_xla caches compiled graphs in-process."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

TP_SCRIPT = textwrap.dedent(
    """
    import os
    import sys
    os.environ['CONVERT_SHLO_TO_SHARDY'] = '1'
    import numpy as np
    import torch
    import torch.nn as nn
    import torch_xla
    import torch_xla.runtime as xr
    import torch_xla.distributed.spmd as xs
    from torch_xla.distributed.spmd import Mesh

    out_file = sys.argv[1]

    xr.set_device_type("TT")
    xr.use_spmd()
    device = torch_xla.device()

    num_devices = xr.global_runtime_device_count()
    mesh = Mesh(np.array(range(num_devices)), (1, num_devices), ("batch", "model"))

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64)).to(
        torch.bfloat16
    )
    x = torch.randn(8, 64, dtype=torch.bfloat16)
    golden = model(x)

    model = model.to(device)
    xs.mark_sharding(model[0].weight, mesh, ("model", None))
    xs.mark_sharding(model[0].bias, mesh, ("model",))
    xs.mark_sharding(model[2].weight, mesh, (None, "model"))
    xs.mark_sharding(model[2].bias, mesh, (None,))

    x_dev = x.to(device)
    xs.mark_sharding(x_dev, mesh, (None, None))

    out = model(x_dev)
    torch_xla.sync()
    out_cpu = out.detach().cpu().float()
    np.save(out_file, out_cpu.numpy())
    print(f"match: {torch.allclose(out_cpu, golden.float(), atol=0.1)}")
    """
)


def run_tp(out_file, env_extra):
    env = {**os.environ, **env_extra}
    return subprocess.run(
        [sys.executable, "-c", TP_SCRIPT, str(out_file)],
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

    result = run_tp(
        tmp_path / "emit.npy", {"TTXLA_CODEGEN_EXPORT_DIR": str(export_dir)}
    )
    assert (export_dir / "manifest.json").exists(), result.stderr[-2000:]
    assert "match: True" in result.stdout, "emitted TP output does not match CPU golden"
    dirs = [d for d in export_dir.iterdir() if (d / "module_key").exists()]
    assert len(dirs) == 1, f"expected 1 graph dir, got {dirs}"
    key_lines = (dirs[0] / "module_key").read_text().split()
    assert len(key_lines) == 3 and "x" in key_lines[1], key_lines

    sentinel = tmp_path / "sentinel"
    insert_sentinel(dirs[0] / "main.py", sentinel)

    result = run_tp(tmp_path / "load.npy", {"TTXLA_CODEGEN_LOAD_DIR": str(export_dir)})
    assert "Codegen load: graph" in result.stderr + result.stdout
    assert sentinel.exists(), "edited main.py was not executed in load mode"
    assert np.array_equal(
        np.load(tmp_path / "emit.npy"), np.load(tmp_path / "load.npy")
    )
    assert "match: True" in result.stdout, "loaded TP output does not match CPU golden"
