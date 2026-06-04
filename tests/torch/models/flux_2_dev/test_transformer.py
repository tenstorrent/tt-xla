# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""FLUX.2-dev — Flux2Transformer2DModel (DiT) component test (~32.2B).

tensor_parallel / PROMOTION-ONLY: at 32.2B bf16 (~64 GB) the transformer is
weight-bound on every single chip (n150=12 GiB, p150=32 GiB), so it only runs
on a multichip TP mesh. The Megatron-1D shard spec in the loader is a best-effort
scaffold validated/refined by /model-bringup-multichip before this passes.

Captured I/O (64x64, seq=512):
  hidden_states          [1, 16, 128]      bf16
  encoder_hidden_states  [1, 512, 15360]   bf16
  timestep               [1]               bf16
  img_ids                [1, 16, 4]        int64
  txt_ids                [1, 512, 4]       int64
  guidance               [1]               float32
OUT: [1, 16, 128] bf16
"""

import os

import pytest
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from infra.utilities.torch_multichip_utils import get_mesh

from third_party.tt_forge_models.flux_2_dev.pytorch import ModelLoader, ModelVariant


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.tensor_parallel
@pytest.mark.large
def test_transformer_sharded():
    # tensor_parallel needs >= 2 chips; skip on single-chip hosts rather than
    # hard-failing (host_probe selects the full mesh at run time).
    if xr.global_runtime_device_count() < 2:
        pytest.skip("Flux2 transformer is tensor_parallel; needs a multichip mesh.")

    # Not using run_graph_test: it runs a CPU golden first, and a 32B-param
    # transformer CPU pass is impractical (> 10 min). TT-only until the sharded
    # TT path is green, then wire run_graph_test with required_pcc=0.99.
    # TODO(model-bringup-multichip): re-enable CPU golden + PCC 0.99.
    torch_xla.set_custom_compile_options(
        {"experimental-enable-dram-space-saving-optimization": "true"}
    )
    xr.set_device_type("TT")
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()
    torch.manual_seed(42)

    device = xm.xla_device()

    loader = ModelLoader(ModelVariant.FLUX2_DEV_TRANSFORMER)
    model = loader.load_model(dtype_override=torch.bfloat16).eval().to(device)

    compiled = torch.compile(model, backend="tt")

    mesh_shape, mesh_names = loader.get_mesh_config(xr.global_runtime_device_count())
    mesh = get_mesh(mesh_shape, mesh_names)
    shard_spec = loader.load_shard_spec(model)
    for tensor, partition_spec in shard_spec.items():
        xs.mark_sharding(tensor, mesh, partition_spec)

    inputs = loader.load_inputs(dtype_override=torch.bfloat16)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        compiled(**inputs)
