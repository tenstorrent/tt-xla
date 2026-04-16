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

        def forward(self, x, topk_indices):
            gather_idx = topk_indices.squeeze(1)  # (bsz, topk)
            index = gather_idx.unsqueeze(-1).expand(
                -1, -1, x.size(-1)
            )  # (bsz, topk, kv_lora_rank)
            gathered_x = torch.gather(x, 1, index)  # (bsz, topk, kv_lora_rank)
            return gathered_x

    class GatherIndicesFancy(nn.Module):
        def __init__(self, topk_indices: torch.Tensor):
            super().__init__()

        def forward(self, x, topk_indices):
            gather_idx = topk_indices.squeeze(1)  # (bsz, topk)
            batch_idx = (
                torch.arange(gather_idx.size(0)).view(-1, 1).to(x.device)
            )  # (bsz, 1)
            gathered_x = x[batch_idx, gather_idx]  # (bsz, topk, kv_lora_rank)
            return gathered_x

    # Setup mesh
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (2, 4)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

    def get_shard_spec(gather_module, args, kwargs):
        mesh_batch_axis_size = mesh.shape()["batch"]
        batch_size = args[0].shape[0]
        # Conditionally shard weights that involve batch axis
        batch_axis = "batch" if batch_size >= mesh_batch_axis_size else None
        shard_specs = {}
        shard_specs[args[0]] = (batch_axis, None, None)  # x
        shard_specs[args[1]] = (batch_axis, None, None)  # topk_indices
        # shard_specs[gather_module.topk_indices] = (batch_axis, None, None)
        return shard_specs

    kv_lora_rank = 512
    index_topk = 16
    x = torch.randn(batch_size, seq_len, kv_lora_rank, dtype=torch.bfloat16)
    topk_indices = torch.stack(
        [torch.randperm(seq_len)[:index_topk] for _ in range(batch_size)]
    ).unsqueeze(1)  # (batch_size, 1, index_topk)

    gather_indices = GatherIndices(topk_indices)
    gather_indices_fancy = GatherIndicesFancy(topk_indices)

    use_fancy = False

    run_graph_test(
        gather_indices_fancy if use_fancy else gather_indices,
        [x, topk_indices],
        framework=Framework.TORCH,
        shard_spec_fn=get_shard_spec,
        mesh=mesh,
        # compiler_config=CompilerConfig(enable_const_eval_on_cpu=False),
    )
