# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Regression test for codegen ``export_tensors=True`` with multi-device sharded tensors.

See https://github.com/tenstorrent/tt-xla/issues/5177.

When codegen runs with ``export_tensors=True`` on a multi-device mesh, the
input and weight tensors that are sharded across the mesh must be serialized to
``tensors/arg*.tensorbin`` as the *full distributed* tensors (all shards), not as
a single device shard.

The original bug serialized single shard, so reloading via
``ttnn.load_tensor`` + ``ttnn.to_device(mesh)`` silently replicated that one
shard across every device. A TP/EP-sharded weight then ran with device-0's
slice on every device.

This test shards a one-op ``x + w`` model across the ``model`` mesh axis, runs
codegen, and asserts that every serialized tensorbin contains the full tensor
(correct volume) split into one shard per device.
"""

import glob
import os

import numpy as np
import pytest
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from infra.utilities.torch_multichip_utils import enable_spmd
from torch_xla.distributed.spmd import Mesh

# Global tensor shape.
ROWS = 256
COLS = 128


class Model(torch.nn.Module):
    """Single-op model: x + w, with w a sharded parameter."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(ROWS, COLS, dtype=torch.bfloat16))

    def forward(self, x):
        return x + self.w


@pytest.mark.push
@pytest.mark.dual_chip
def test_codegen_export_tensors_preserves_sharding():
    """Codegen export_tensors must serialize full sharded tensors, not one shard."""
    import ttnn

    enable_spmd()
    xr.set_device_type("TT")
    device = torch_xla.device()

    num_devices = xr.global_runtime_device_count()
    assert num_devices > 1, "This regression test requires a multi-device mesh."
    assert (
        ROWS % num_devices == 0
    ), f"Global row dim {ROWS} must be divisible by device count {num_devices}."

    mesh = Mesh(np.arange(num_devices), (1, num_devices), ("batch", "model"))

    model = Model().to(device, torch.bfloat16)
    xs.mark_sharding(model.w, mesh, ("model", None))

    export_path = "codegen_export"
    torch_xla.set_custom_compile_options(
        {
            "backend": "codegen_py",
            "export_path": export_path,
            "export_tensors": True,
        }
    )
    model.compile(backend="tt", options={"tt_legacy_compile": True})

    x = torch.randn(ROWS, COLS, dtype=torch.bfloat16).to(device)
    xs.mark_sharding(x, mesh, ("model", None))

    with torch.no_grad():
        model(x)

    # We need to sync here - in order for everything to be exported
    # before we proceed with verifying the exported artifacts.
    torch_xla.sync(wait=True)

    tensorbins = sorted(
        glob.glob(
            os.path.join(export_path, "**", "tensors", "arg*.tensorbin"),
            recursive=True,
        )
    )
    assert tensorbins, (
        f"No arg*.tensorbin files were exported under {export_path}; "
        "codegen export_tensors produced nothing."
    )

    expected_nshards = num_devices
    expected_per_shard_volume = (ROWS // num_devices) * COLS

    for path in tensorbins:
        tensor = ttnn.load_tensor(path)
        nshards = len(ttnn.get_device_tensors(tensor))
        per_shard_volume = tensor.volume()
        assert (
            nshards == expected_nshards
            and per_shard_volume == expected_per_shard_volume
        ), (
            f"{path}: expected full sharded tensor"
            f"(per_shard_volume={expected_per_shard_volume}, "
            f"nshards={expected_nshards}) but got shape={list(tensor.shape)}, "
            f"per_shard_volume={per_shard_volume}, nshards={nshards}. "
        )
