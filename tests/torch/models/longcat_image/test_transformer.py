# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""LongCat-Image — LongCatImageTransformer2DModel (transformer) component test.

The ~6B Flux-style MMDiT does not fit a single n150: the compiler const-evals
the 41 AdaLayerNorm modulation linears (they all consume the same timestep
embedding) into one fused ~6.87 GB f32 weight buffer. It is brought up
tensor-parallel on a multichip n300 llmbox with an FSDP-style
("batch", "model") mesh (model axis = 4, capped by the shared text-encoder
GQA), so 8 chips map to (2, 4); row-parallel modulation weights shard that
fused buffer. See https://github.com/tenstorrent/tt-xla/issues/5169
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
def test_transformer_sharded():
    xr.set_device_type("TT")
    torch.manual_seed(42)

    loader = ModelLoader(ModelVariant.TRANSFORMER)
    model = loader.load_model(dtype_override=torch.bfloat16)
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)

    mesh_shape, mesh_names = loader.get_mesh_config(xr.global_runtime_device_count())

    with _mask_jax_accelerator():
        run_graph_test(
            model,
            inputs,
            framework=Framework.TORCH,
            mesh=get_mesh(mesh_shape, mesh_names),
            shard_spec_fn=loader.load_shard_spec,
        )
