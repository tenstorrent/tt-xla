# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Krea Realtime — CausalWanModel (14B DiT) component test."""

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.utilities.torch_multichip_utils import get_mesh

from third_party.tt_forge_models.krea_realtime_video.pytorch import (
    ModelLoader,
    ModelVariant,
)
from loguru import logger

@pytest.mark.skip(
    reason="OOM on single device — CausalWanModel exceeds single-chip memory; sharded variant runs"
)
def test_transformer():
    _run(sharded=False)


# @pytest.mark.xfail(
#     reason="error: failed to legalize unresolved materialization from ('tensor<0x2xf64>') to ('tensor<0xcomplex<f64>>') that remained live after conversion — https://github.com/tenstorrent/tt-mlir/issues/8291"
# )
def test_transformer_sharded():
    _run(sharded=True)


def _run(sharded: bool):
    xr.set_device_type("TT")
    torch.manual_seed(42)

    loader = ModelLoader(ModelVariant.TRANSFORMER)
    model = loader.load_model(dtype_override=torch.bfloat16)
    
    logger.info("model={}", model)
    
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)

    mesh = None
    shard_spec_fn = None
    if sharded:
        mesh_shape, mesh_names = loader.get_mesh_config(
            xr.global_runtime_device_count()
        )
        mesh = get_mesh(mesh_shape, mesh_names)
        shard_spec_fn = loader.load_shard_spec

    run_graph_test(
        model,
        inputs,
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=shard_spec_fn,
    )
