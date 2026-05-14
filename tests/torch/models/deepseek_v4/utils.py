# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import pytest
import torch
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.evaluators import ComparisonConfig, PccConfig
from torch import nn
from torch_xla.distributed.spmd import Mesh
from tt_torch.sparse_mlp import enable_sparse_mlp

PCC_99 = ComparisonConfig(pcc=PccConfig(enabled=True, required_pcc=0.99))


def setup_spmd():
    pass


def make_2d_mesh() -> Mesh:
    num_devices = xr.global_runtime_device_count()
    assert num_devices in (
        8,
        32,
    ), "Need to run on either an LLMBox or Galaxy in order to make a 2D Mesh"
    if num_devices == 32:
        mesh_shape = (4, 8)
    else:
        mesh_shape = (2, 4)
    device_ids = np.array(range(num_devices))
    return Mesh(device_ids, mesh_shape, ("batch", "model"))


def real_args(**overrides) -> ModelArgs:
    """Attention-focused args copied from the real config.json with bounded test sizes."""
    defaults = dict(
        max_batch_size=1,
        max_seq_len=256,
        vocab_size=129280,
        dim=4096,
        moe_inter_dim=2048,
        n_layers=43,
        n_hash_layers=3,
        n_mtp_layers=0,
        n_heads=64,
        n_routed_experts=256,
        n_shared_experts=1,
        n_activated_experts=6,
        score_func="sqrtsoftplus",
        route_scale=1.5,
        swiglu_limit=10.0,
        q_lora_rank=1024,
        head_dim=512,
        rope_head_dim=64,
        o_groups=8,
        o_lora_rank=1024,
        window_size=128,
        original_seq_len=65536,
        rope_theta=10000,
        rope_factor=16,
        beta_fast=32,
        beta_slow=1,
        index_n_heads=64,
        index_head_dim=128,
        index_topk=512,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        dtype="bf16",
        scale_fmt=None,
        expert_dtype=None,
        compress_rope_theta=160000,
        compress_ratios=(
            0,
            0,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            0,
        ),
    )
    defaults.update(overrides)
    return ModelArgs(**defaults)
