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


_XFAIL_REASON = (
    "codegen with export_tensors=True does not preserve TP/EP sharding: "
    "single shard is serialized to tensors/arg*.tensorbin instead of the "
    "full distributed tensor, so a sharded tensor is silently replicated across the "
    "mesh - https://github.com/tenstorrent/tt-xla/issues/5177"
)


@pytest.mark.push
@pytest.mark.dual_chip
@pytest.mark.xfail(reason=_XFAIL_REASON)
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

    # Both inputs (x and w) are sharded on dim 0 over the "model" axis, so
    # every exported tensorbin must reassemble to the full global volume and
    # carry one shard per device.
    expected_volume = ROWS * COLS
    expected_nshards = num_devices

    for path in tensorbins:
        tensor = ttnn.load_tensor(path)
        nshards = len(ttnn.get_device_tensors(tensor))
        volume = tensor.volume()
        assert nshards == expected_nshards and volume == expected_volume, (
            f"{path}: expected full sharded tensor "
            f"(volume={expected_volume}, nshards={expected_nshards}) but got "
            f"shape={list(tensor.shape)}, volume={volume}, nshards={nshards}. "
            "Only a single device shard was serialized "
            "(see tenstorrent/tt-xla#5177)."
        )
