# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""LongCat-Image — Qwen2_5_VLForConditionalGeneration (text_encoder) component test.

The ~7.7B Qwen2.5-VL text encoder does not fit a single n150. It is brought up
tensor-parallel on a multichip n300 llmbox with an FSDP-style
("batch", "model") mesh; the GQA (4 KV heads) caps the model axis at 4, so 8
chips map to (2, 4). See https://github.com/tenstorrent/tt-xla/issues/5168
"""

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.testers.single_chip.model.torch_model_tester import _mask_jax_accelerator
from infra.utilities.torch_multichip_utils import get_mesh

from third_party.tt_forge_models.longcat_image.pytorch import ModelLoader, ModelVariant


@pytest.mark.nightly
@pytest.mark.llmbox
@pytest.mark.model_test
@pytest.mark.tensor_parallel
def test_text_encoder_sharded():
    xr.set_device_type("TT")
    torch.manual_seed(42)

    loader = ModelLoader(ModelVariant.TEXT_ENCODER)
    encoder = loader.load_model(dtype_override=torch.bfloat16)
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)

    mesh_shape, mesh_names = loader.get_mesh_config(xr.global_runtime_device_count())

    with _mask_jax_accelerator():
        run_graph_test(
            encoder,
            inputs,
            framework=Framework.TORCH,
            mesh=get_mesh(mesh_shape, mesh_names),
            shard_spec_fn=loader.load_shard_spec,
        )
