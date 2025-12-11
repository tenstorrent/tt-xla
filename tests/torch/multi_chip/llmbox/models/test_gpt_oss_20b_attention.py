# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
from typing import Callable

import numpy as np
import pytest
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.comparators.comparison_config import ComparisonConfig, PccConfig
from torch_xla.distributed.spmd import Mesh
from transformers.models.gpt_oss.modeling_gpt_oss import (
    ALL_ATTENTION_FUNCTIONS,
    eager_attention_forward,
)

from third_party.tt_forge_models.gpt_oss.pytorch.loader import ModelLoader, ModelVariant


def sdpa(
    attention_module,
    query_states,
    key_states,
    value_states,
    attention_mask,
    dropout,
    scaling,
):
    attention_interface: Callable = eager_attention_forward
    if attention_module.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[
            attention_module.config._attn_implementation
        ]

    attn_output, attn_weights = attention_interface(
        attention_module,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=dropout,
        scaling=scaling,
    )
    return attn_output, attn_weights


def gpt_oss():
    loader = ModelLoader(variant=ModelVariant.GPT_OSS_20B, num_layers=1)
    model = loader.load_model()
    config = loader.load_config()
    inputs = loader.load_inputs()
    batch_size = inputs["input_ids"].shape[0]  # 1
    seq_len = inputs["input_ids"].shape[1]  # 71

    num_heads = config.num_attention_heads
    num_key_value_heads = getattr(config, "num_key_value_heads", num_heads)
    head_dim = config.head_dim

    attention = model.model.layers[0].self_attn

    dropout = 0.0
    scaling = attention.scaling

    query_states = torch.randn(
        (batch_size, num_heads, seq_len, head_dim), dtype=torch.bfloat16
    )
    key_states = torch.randn(
        (batch_size, num_key_value_heads, seq_len, head_dim), dtype=torch.bfloat16
    )
    value_states = torch.randn(
        (batch_size, num_key_value_heads, seq_len, head_dim), dtype=torch.bfloat16
    )
    attention_mask = torch.rand(batch_size, 1, seq_len, seq_len, dtype=torch.bfloat16)

    comparison_config = ComparisonConfig(pcc=PccConfig(required_pcc=0.98))

    # Create mesh for multi-device
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (1, num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

    def get_shard_spec(sdpa_fn, args, kwargs):
        # Extract attention module from args (it's the first argument to sdpa)
        attention = args[0]
        shard_specs = {}
        shard_specs[attention.q_proj.weight] = ("model", None)
        shard_specs[attention.k_proj.weight] = ("model", None)
        shard_specs[attention.v_proj.weight] = ("model", None)
        shard_specs[attention.o_proj.weight] = (None, "model")
        shard_specs[attention.sinks] = ("model",)
        return shard_specs

    run_graph_test(
        sdpa,
        [
            attention,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout,
            scaling,
        ],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
        comparison_config=comparison_config,
    )


if __name__ == "__main__":
    torch._dynamo.reset()
    gpt_oss()
