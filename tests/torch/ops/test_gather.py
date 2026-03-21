import pytest
import torch
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from torch import nn


# Erroneous combinations: [128-32]
@pytest.mark.parametrize("batch_size", [1, 4, 32, 64])
@pytest.mark.parametrize("seq_len", [32, 128, 512, 2048])
def test_gather_indices(batch_size, seq_len):
    xr.set_device_type("TT")

    class GatherIndices(nn.Module):
        def __init__(self, topk_indices: torch.Tensor):
            super().__init__()
            self.topk_indices = topk_indices

        def forward(self, x):
            gather_idx = self.topk_indices.squeeze(1)
            batch_idx = torch.arange(gather_idx.size(0)).view(-1, 1)

            gathered_x = x[batch_idx, gather_idx]
            return gathered_x

    kv_lora_rank = 512
    index_topk = 16
    x = torch.randn(batch_size, seq_len, kv_lora_rank, dtype=torch.bfloat16)
    topk_indices = torch.stack(
        [torch.randperm(seq_len)[:index_topk] for _ in range(batch_size)]
    ).unsqueeze(1)  # (batch_size, 1, index_topk)

    gather_indices = GatherIndices(topk_indices)

    run_graph_test(
        gather_indices,
        [x],
        framework=Framework.TORCH,
    )
