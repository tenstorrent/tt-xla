# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.utilities.torch_multichip_utils import get_mesh
from loguru import logger


def _mesh():
    """(1×4) mesh matching the real model's sharded run."""
    n = xr.global_runtime_device_count()
    from third_party.tt_forge_models.hidream_i1.pytorch.src.model_utils import (
        MESH_SHAPES,
        MESH_NAMES,
    )
    shape = MESH_SHAPES.get(n, (1, 4))
    return get_mesh(shape, MESH_NAMES)


@pytest.mark.nightly
@pytest.mark.llmbox
def test_scatter_reduce_mc():
    """Sharded repro: pre-reshaped index, mirrors the exact presharded context of
    the crashing program 762. expert_cache/expert_out/index all sharded (None,'model')
    → 640 per device on the model axis, same as the real MoE scatter-back graph."""
    xr.set_device_type("TT")

    class ScatterReducePrereshaped(torch.nn.Module):
        def forward(self, expert_cache, exp_token_idx_after_reshape, expert_out):
            return expert_cache.scatter_reduce_(
                0,
                exp_token_idx_after_reshape,
                expert_out,
                reduce="sum",
            )

    expert_cache = torch.load("expert_cache.pt", map_location="cpu")
    exp_token_idx_after_reshape = torch.load(
        "exp_token_idx_after_reshape.pt", map_location="cpu"
    )
    expert_out = torch.load("expert_out.pt", map_location="cpu")

    logger.info("expert_cache shape={} dtype={}", expert_cache.shape, expert_cache.dtype)
    logger.info("expert_out shape={} dtype={}", expert_out.shape, expert_out.dtype)
    logger.info(
        "exp_token_idx_after_reshape shape={} dtype={}",
        exp_token_idx_after_reshape.shape,
        exp_token_idx_after_reshape.dtype,
    )

    mesh = _mesh()

    def shard_spec_fn(model, args, kwargs):
        cache, idx, out = args
        # Hidden dim (2560) split across model axis → 640 per device.
        # Matches the presharded context seen in the crashing IR.
        return {
            cache: (None, "model"),
            out: (None, "model"),
            idx: (None, "model"),  # index already reshaped to (N, hidden)
        }

    run_graph_test(
        ScatterReducePrereshaped().to(torch.bfloat16),
        [expert_cache, exp_token_idx_after_reshape, expert_out],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=shard_spec_fn,
    )


