# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from infra import ComparisonConfig, Framework, run_graph_test
from loguru import logger

from third_party.tt_forge_models.hidream_i1.pytorch.src.model_utils import (
    HIDREAM_REPO_ID,
    MESH_NAMES,
    MESH_SHAPES,
    load_transformer,
    shard_hidream_transformer_specs,
)

DTYPE = torch.bfloat16

HIDDEN_STATES_PT = "hidden_states.pt"
TEMB_PT = "temb.pt"


class DoubleStreamBlock0Sanity(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.adaLN_modulation = block.adaLN_modulation
        self.norm1_i = block.norm1_i

    def forward(self, hidden_states, temb):
        # Mirrors HiDreamImageTransformerBlock.forward up to the modulation.
        wtype = hidden_states.dtype

        (
            shift_msa_i,
            scale_msa_i,
            gate_msa_i,
            shift_mlp_i,
            scale_mlp_i,
            gate_mlp_i,
            shift_msa_t,
            scale_msa_t,
            gate_msa_t,
            shift_mlp_t,
            scale_mlp_t,
            gate_mlp_t,
        ) = self.adaLN_modulation(temb)[:, None].chunk(12, dim=-1)

        norm_hidden_states = self.norm1_i(hidden_states).to(dtype=wtype)

        return (
            shift_msa_i,
            (1 + scale_msa_i) + shift_msa_i,
            norm_hidden_states * (1 + scale_msa_i) + shift_msa_i,
        )


@pytest.mark.nightly
def test_double_stream_block0_adaln_chunk():
    xr.set_device_type("TT")
    torch.manual_seed(42)

    num_devices = xr.global_runtime_device_count()

    hidden_states = torch.load(HIDDEN_STATES_PT, map_location="cpu").to(DTYPE)
    temb = torch.load(TEMB_PT, map_location="cpu").to(DTYPE)

    logger.info(
        "hidden_states shape={} dtype={}",
        tuple(hidden_states.shape),
        hidden_states.dtype,
    )
    logger.info("hidden_states: {}", hidden_states)
    logger.info("temb shape={} dtype={}", tuple(temb.shape), temb.dtype)
    logger.info("temb: {}", temb)

    transformer = load_transformer(HIDREAM_REPO_ID, DTYPE)
    block = transformer.double_stream_blocks[0].block
    model = DoubleStreamBlock0Sanity(block).eval()

    mesh = xs.Mesh(np.array(range(num_devices)), MESH_SHAPES[num_devices], MESH_NAMES)

    def get_shard_spec(model, args, kwargs):
        model_specs = shard_hidream_transformer_specs(transformer)
        param_ids = {id(p) for p in model.parameters()}
        specs = {
            tensor: spec
            for tensor, spec in model_specs.items()
            if id(tensor) in param_ids
        }
        linear = model.adaLN_modulation[1]
        assert linear.weight in specs and linear.bias in specs, (
            "adaLN specs missing — model.to(xla) replaced the Parameter objects "
            "before shard_hidream_transformer_specs re-read them"
        )
        logger.info(
            "adaLN weight spec={} bias spec={}",
            specs[linear.weight],
            specs[linear.bias],
        )
        specs[args[0]] = (None, None, "model")
        return specs

    run_graph_test(
        model,
        [hidden_states, temb],
        comparison_config=ComparisonConfig(),
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )
