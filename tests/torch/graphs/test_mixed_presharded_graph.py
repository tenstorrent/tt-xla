# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Demonstrates the mixed presharded/unsharded result issue tracked in:
#   tt-xla: https://github.com/tenstorrent/tt-xla/issues/3386
#   tt-mlir: https://github.com/tenstorrent/tt-mlir/issues/7002
#
# In this graph:
#   - result `y`    has a sdy.sharding annotation  (sharded across devices)
#   - result `loss` has NO sdy.sharding annotation (replicated scalar)
#
# torch-xla always expects ALL results as a per-device list of tensors regardless
# of annotation.  The current compiler infers result shard status from annotations,
# so it marks `loss` as unsharded and sets up the runtime to receive a single
# tensor — but torch-xla hands back a list.  That mismatch causes a runtime error.
#
# Fix: tt-xla must explicitly pass result-presharded pipeline options once
# tt-mlir PR #7120 lands.

import numpy as np
import pytest
import torch
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from torch_xla.distributed.spmd import Mesh

from tests.utils import Category, parametrize_arch


@pytest.mark.nightly
@pytest.mark.known_failure_xfail(
    reason="Mixed presharded/unsharded results not yet supported (tt-xla#3386, tt-mlir#7002)"
)
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@parametrize_arch(["llmbox"])
def test_mixed_presharded_graph(arch):
    """Graph that returns one sharded tensor and one unsharded scalar.

    The column-parallel linear produces a sharded output `y` (has sdy.sharding).
    The subsequent .mean() reduction produces a scalar `loss` (no sdy.sharding).
    torch-xla expects both as per-device lists; compiler only sets that up for `y`.
    """
    xr.set_device_type("TT")

    class MixedOutputModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(8, 8, bias=False, dtype=torch.bfloat16)

        def forward(self, x):
            y = self.linear(x)  # sharded — weight is column-parallel, output annotated
            loss = y.mean()  # scalar — no sdy.sharding annotation on this result
            return y, loss

    num_devices = xr.global_runtime_device_count()
    mesh = Mesh(np.arange(num_devices), (1, num_devices), ("batch", "model"))

    def get_shard_spec(model, args, kwargs):
        # Column-parallel: split weight along output dim → y is sharded across devices
        return {model.linear.weight: ("model", None)}

    run_graph_test(
        MixedOutputModule(),
        [torch.randn(4, 8, dtype=torch.bfloat16)],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )
