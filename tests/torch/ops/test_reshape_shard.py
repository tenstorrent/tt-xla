# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
import torch_xla.runtime as xr
from infra import Framework, run_op_test
from torch_xla.distributed.spmd import Mesh


def test_reshape_split_sharded_dim():
    class Reshape(torch.nn.Module):
        def forward(self, x):
            # https://huggingface.co/krea/krea-realtime-video/blob/main/transformer/causal_model.py#L1085
            return x.unflatten(1, (6, 5120))

    num_devices = xr.global_runtime_device_count()
    mesh = Mesh(np.array(range(num_devices)), (1, num_devices), ("batch", "model"))

    x = torch.randn(3, 30720, dtype=torch.bfloat16)

    run_op_test(
        Reshape(),
        [x],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=lambda m, args, kwargs: {args[0]: (None, "model")},
    )
