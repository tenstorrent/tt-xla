# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Krea Realtime — CausalWanModel (14B DiT) standalone component test.

Temporary diagnostic: recovered from the version #5838 deleted, markers stripped.
A single sharded forward on a fresh device — no pipeline, no cache evolution, no
accumulated programs. Used to tell apart:
  * the RoPE subtract's 383 MB buffer being too big to place at all (per-op), vs
  * the buffer fitting fine on a clean pool and only failing in the pipeline
    because the resident transformer has fragmented DRAM (#6047).
"""

import torch
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.utilities.torch_multichip_utils import get_mesh

from third_party.tt_forge_models.krea_realtime_video.pytorch import (
    ModelLoader,
    ModelVariant,
)


def test_transformer_sharded():
    xr.set_device_type("TT")
    torch.manual_seed(42)

    loader = ModelLoader(ModelVariant.TRANSFORMER)
    model = loader.load_model(dtype_override=torch.bfloat16)
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)

    mesh_shape, mesh_names = loader.get_mesh_config(xr.global_runtime_device_count())
    mesh = get_mesh(mesh_shape, mesh_names)

    run_graph_test(
        model,
        inputs,
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=loader.load_shard_spec,
    )
