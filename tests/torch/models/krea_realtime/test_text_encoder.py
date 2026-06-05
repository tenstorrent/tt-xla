# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Krea Realtime — UMT5EncoderModel (UMT5-XXL) text encoder component test (~5.68B).

tensor_parallel: at 5.68B bf16 (~10.6 GiB) the encoder exceeds the n150/n300
weight budget (10.2 GiB) and fits p150 only by weight; it is sharded onto the
uniform krea pipeline mesh alongside the TP transformer. The 2D FSDP-style shard
spec in the loader (per-layer q/k/v/o + wi_0/wi_1/wo on the UMT5 encoder, on a
("batch", "model") mesh) is validated/refined by /model-bringup-multichip.

Reuses the text_encoder from Wan-AI/Wan2.1-T2V-14B-Diffusers.

Verified PASSED on n300-llmbox (mesh (2,4) = 8 chips, TT_VISIBLE_DEVICES=0,1,2,3):
run_graph_test runs a CPU golden + the sharded TT path and asserts PCC >= 0.99.

Captured I/O (seq=512):
  input_ids       [1, 512]   int64
  attention_mask  [1, 512]   int64
OUT: last_hidden_state [1, 512, 4096] bf16
"""

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.evaluators import ComparisonConfig, PccConfig
from infra.utilities.torch_multichip_utils import get_mesh

from third_party.tt_forge_models.krea_realtime_video.pytorch import (
    ModelLoader,
    ModelVariant,
)


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.tensor_parallel
@pytest.mark.large
def test_text_encoder_sharded():
    # tensor_parallel needs >= 2 chips; skip on single-chip hosts rather than
    # hard-failing (host_probe selects the full mesh at run time).
    if xr.global_runtime_device_count() < 2:
        pytest.skip("Krea text_encoder is tensor_parallel; needs a multichip mesh.")

    xr.set_device_type("TT")
    torch.manual_seed(42)

    loader = ModelLoader(ModelVariant.TEXT_ENCODER)
    encoder = loader.load_model(dtype_override=torch.bfloat16)
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)

    mesh_shape, mesh_names = loader.get_mesh_config(xr.global_runtime_device_count())
    mesh = get_mesh(mesh_shape, mesh_names)

    run_graph_test(
        encoder,
        inputs,
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=loader.load_shard_spec,
        comparison_config=ComparisonConfig(pcc=PccConfig(required_pcc=0.99)),
    )
