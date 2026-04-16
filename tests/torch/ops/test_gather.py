# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
import torch
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.testers.compiler_config import CompilerConfig
from torch import nn
from torch_xla.distributed.spmd import Mesh


@pytest.mark.parametrize("batch_size", [1, 4, 32, 64])
@pytest.mark.parametrize("seq_len", [32, 128, 512, 1024, 2048])
def test_gather_indices(batch_size, seq_len):
    xr.set_device_type("TT")

    class GatherIndices(nn.Module):
        def __init__(self, topk_indices: torch.Tensor):
            super().__init__()
            self.topk_indices = topk_indices

        def forward(self, x):
            gather_idx = self.topk_indices.squeeze(1)
            index = gather_idx.to(x.device).unsqueeze(-1).expand(-1, -1, x.size(-1))
            gathered_x = torch.gather(x, 1, index)
            return gathered_x

    # Setup mesh
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (2, 4)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

    def get_shard_spec(gather_module, args, kwargs):
        shard_specs = {}
        shard_specs[args[0]] = ("batch", None, None)
        return shard_specs

    kv_lora_rank = 512
    index_topk = 16
    x = torch.randn(batch_size, seq_len, kv_lora_rank, dtype=torch.bfloat16)
    topk_indices = torch.stack(
        [torch.randperm(seq_len)[:index_topk] for _ in range(batch_size)]
    ).unsqueeze(
        1
    )  # (batch_size, 1, index_topk)

    gather_indices = GatherIndices(topk_indices)

    run_graph_test(
        gather_indices,
        [x],
        framework=Framework.TORCH,
        shard_spec_fn=get_shard_spec,
        mesh=mesh,
        # compiler_config=CompilerConfig(enable_const_eval_on_cpu=False),
    )
