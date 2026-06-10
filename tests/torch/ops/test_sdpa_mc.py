# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Multi-chip (tensor-parallel) SDPA sanity repro for the HunyuanImage-2.1 fusion gap.

Single-chip (test_sdpa.py) lowers the composite to ttnn.scaled_dot_product_attention.
This shards Q/K/V across the head dim (28 -> heads/chip) like the model, so the SDPA
lands inside an `sdy.manual_computation` region. Use it to check whether the composite
still lowers to ttnn.scaled_dot_product_attention or falls back to the f32 decomposition
(the suspected sharded-path fusion gap):

    grep -c 'ttnn.scaled_dot_product_attention' <log>   # expect >0 if it fuses
    grep -c '5224x5224xf32'                     <log>   # >0 => decomposed score matrix

Config mirrors the model joint-attention block:
    query/key/value : [1, 28, 5224, 128] bf16, sharded on heads (dim 1)
    attn_mask       : [1, 1, 1, 5224] bool (broadcast over heads -> replicated)
"""

import numpy as np
import pytest
import torch
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from infra import ComparisonConfig, Framework, run_graph_test
from torch.nn import functional as F

BATCH_SIZE = 1
NUM_HEADS = 28
SEQ_LEN = 5224
HEAD_DIM = 128
DTYPE = torch.bfloat16


class SDPAModel(torch.nn.Module):
    def forward(self, query, key, value, attn_mask):
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
        )


@pytest.mark.nightly
@pytest.mark.dual_chip
def test_sdpa_multichip():
    num_devices = xr.global_runtime_device_count()
    if NUM_HEADS % num_devices != 0:
        pytest.skip(f"num_heads={NUM_HEADS} not divisible by num_devices={num_devices}")

    model = SDPAModel().to(DTYPE)
    model.eval()

    query = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, dtype=DTYPE)
    key = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, dtype=DTYPE)
    value = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, dtype=DTYPE)
    attn_mask = torch.ones((BATCH_SIZE, 1, 1, SEQ_LEN), dtype=torch.bool)
    attn_mask[..., -224:] = False  # padding -> masked

    # 1 x num_devices mesh; shard the head dim (dim 1) over the "model" axis.
    mesh_shape = (1, num_devices)
    device_ids = np.array(range(num_devices))
    mesh = xs.Mesh(device_ids, mesh_shape, ("batch", "model"))

    # Tensor-parallel: shard Q/K/V on heads; mask is head-broadcast -> replicated.
    def get_shard_spec(args, kwargs):
        head_sharded = (None, "model", None, None)
        return {args[0]: head_sharded, args[1]: head_sharded, args[2]: head_sharded}

    run_graph_test(
        model,
        [query, key, value, attn_mask],
        comparison_config=ComparisonConfig(),
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
        torch_options={"tt_enable_composite_ops": True},
    )
