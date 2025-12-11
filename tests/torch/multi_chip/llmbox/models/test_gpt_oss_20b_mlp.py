# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
import torch
import torch_xla.runtime as xr
from infra import Framework, run_graph_test  # Add this import
from torch_xla.distributed.spmd import Mesh
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssMLP

from third_party.tt_forge_models.gpt_oss.pytorch.loader import ModelLoader, ModelVariant


def test_gpt_oss_mlp():
    # Set up environment
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()
    xr.set_device_type("TT")

    # Load model and config
    loader = ModelLoader(variant=ModelVariant.GPT_OSS_20B, num_layers=1)
    model = loader.load_model()
    config = loader.load_config()
    inputs = loader.load_inputs()

    batch_size = inputs["input_ids"].shape[0]  # 1
    seq_len = inputs["input_ids"].shape[1]  # 128 with padding

    # Get MLP module
    mlp: GptOssMLP = model.model.layers[0].mlp

    # Create input
    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )

    # Create mesh for multi-device
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (1, num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

    # Define shard spec function
    def get_shard_spec(mlp, args, kwargs):
        shard_specs = {}
        # Router weights (not sharded)
        shard_specs[mlp.router.weight] = (None, None)
        shard_specs[mlp.router.bias] = (None,)

        # Shard experts across devices: 32 / 8 -> 4 experts per device
        shard_specs[mlp.experts.gate_up_proj] = ("model", None, None)
        shard_specs[mlp.experts.gate_up_proj_bias] = ("model", None)
        shard_specs[mlp.experts.down_proj] = ("model", None, None)
        shard_specs[mlp.experts.down_proj_bias] = ("model", None)

        return shard_specs

    # Run the test using run_graph_test
    run_graph_test(
        mlp,
        [hidden_states],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


if __name__ == "__main__":
    test_gpt_oss_mlp()
