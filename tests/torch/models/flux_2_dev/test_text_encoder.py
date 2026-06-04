# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""FLUX.2-dev — Mistral3ForConditionalGeneration text encoder (~24.0B).

tensor_parallel / PROMOTION-ONLY: at 24.0B bf16 (~48 GB) the text encoder is
weight-bound on every single chip (n150=12 GiB, p150=32 GiB), so it only runs
on a multichip TP mesh. The Megatron-1D shard spec in the loader (per-layer
q/k/v/o + gate/up/down on the Mistral language model) is validated/refined by
/model-bringup-multichip before this passes.

Captured I/O (seq=512):
  input_ids       [1, 512]   int64
  attention_mask  [1, 512]   int64
  output_hidden_states=True
OUT: hidden_states = tuple[41] of [1, 512, 5120] bf16
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
def test_text_encoder_sharded():
    # tensor_parallel needs >= 2 chips; skip on single-chip hosts rather than
    # hard-failing (host_probe selects the full mesh at run time).
    if xr.global_runtime_device_count() < 2:
        pytest.skip("Flux2 text_encoder is tensor_parallel; needs a multichip mesh.")

    # Not using run_graph_test: a 24B-param CPU golden is impractical. TT-only
    # until the sharded TT path is green, then wire run_graph_test PCC 0.99.
    # TODO(model-bringup-multichip): re-enable CPU golden + PCC 0.99.
    torch_xla.set_custom_compile_options(
        {"experimental-enable-dram-space-saving-optimization": "true"}
    )
    xr.set_device_type("TT")
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()
    torch.manual_seed(42)

    device = xm.xla_device()

    loader = ModelLoader(ModelVariant.FLUX2_DEV_TEXT_ENCODER)
    model = loader.load_model(dtype_override=torch.bfloat16).eval().to(device)

    compiled = torch.compile(model, backend="tt")

    mesh_shape, mesh_names = loader.get_mesh_config(xr.global_runtime_device_count())
    mesh = get_mesh(mesh_shape, mesh_names)
    shard_spec = loader.load_shard_spec(model)
    for tensor, partition_spec in shard_spec.items():
        xs.mark_sharding(tensor, mesh, partition_spec)

    inputs = loader.load_inputs(dtype_override=torch.bfloat16)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        # logits_to_keep=1: lm_head on the last token only -> logits [1,1,131072]
        # instead of [1,512,131072] (~134MB replicated). hidden_states (what the
        # FLUX.2 pipeline actually uses) are still produced for every layer.
        compiled(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            logits_to_keep=1,
        )
