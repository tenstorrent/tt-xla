# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
import torch
import torch_xla.runtime as xr

from infra import Framework, run_graph_test
from infra.evaluators import ComparisonConfig, PccConfig
from torch_xla.distributed.spmd import Mesh

from tests.utils import parametrize_arch
from third_party.tt_forge_models.gpt_oss.pytorch.s_loader import (
    ModelLoader as GPTOSSModelLoader,
    ModelVariant as GPTOSSModelVariant,
)
from third_party.tt_forge_models.gpt_oss.pytorch.loader import (
    ModelLoader as GPTOSSCurrentModelLoader,
    ModelVariant as GPTOSSCurrentModelVariant,
)


@pytest.mark.nightly
@parametrize_arch(["llmbox"])
@pytest.mark.parametrize(
    "variant",
    [GPTOSSModelVariant.GPT_OSS_20B],
    ids=["20B"],
)
def test_gpt_oss_replay_model_layer(variant, arch):
    """Run a full sparse decoder layer (attention + MoE) using _ReplayModel."""
    xr.set_device_type("TT")

    # Load the model with capture_after_layer=0 so that the output of layer 0
    # is saved as frozen_input for the replay of layer 1.
    loader = GPTOSSModelLoader(
        variant=variant,
        num_layers=4,
        capture_after_layer=0,
        mlp_type="a2a_sparse",
    )
    model = loader.load_model()
    inputs = loader.load_inputs()

    # CPU forward pass to trigger the capture hook on layer 0.
    with torch.no_grad():
        model(**inputs)

    assert loader.captured_output is not None, (
        "Forward hook did not fire — captured_output is None"
    )

    # Build the single-layer replay model (layer 1, mode="full" by default).
    replay_model = loader.create_replay_model()

    num_devices = xr.global_runtime_device_count()
    mesh_shape = (2, 4)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

    def get_shard_spec(model, args, kwargs):
        return loader.load_replay_shard_spec(model)

    run_graph_test(
        replay_model,
        [inputs["input_ids"], inputs["attention_mask"]],
        comparison_config=ComparisonConfig(pcc=PccConfig(required_pcc=0.85)),
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )

@pytest.mark.nightly
@parametrize_arch(["llmbox"])
@pytest.mark.parametrize(
    "variant",
    [GPTOSSCurrentModelVariant.GPT_OSS_20B],
    ids=["20B"],
)
def test_gpt_oss_decoder_layer(variant, arch):
    """Full GPT-OSS forward with A2a sparse MoE."""
    xr.set_device_type("TT")

    loader = GPTOSSCurrentModelLoader(
        variant=variant,
        num_layers=None,
    )
    model = loader.load_model()
    inputs = loader.load_inputs()

    num_devices = xr.global_runtime_device_count()
    mesh_shape, axis_names = loader.get_mesh_config(num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, axis_names)

    def get_shard_spec(model, args, kwargs):
        return loader.load_shard_spec(model)

    run_graph_test(
        model,
        [inputs["input_ids"], inputs["attention_mask"]],
        comparison_config=ComparisonConfig(pcc=PccConfig(required_pcc=0.99)),
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )
