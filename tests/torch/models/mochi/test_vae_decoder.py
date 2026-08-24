# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Mochi — VAE decoder component test at original resolution (0.36B)."""

import pytest
import torch
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.utilities.torch_multichip_utils import get_mesh

from tests.infra.testers.compiler_config import CompilerConfig
from third_party.tt_forge_models.mochi.pytorch import ModelLoader, ModelVariant
from third_party.tt_forge_models.mochi.pytorch.src.utils import (
    load_vae_decoder_inputs_full_res,
)


def test_vae_decoder():
    xr.set_device_type("TT")
    torch.manual_seed(42)

    loader = ModelLoader(ModelVariant.MOCHI, subfolder="vae")
    vae = loader.load_model(dtype_override=torch.bfloat16)
    decoder = vae.decoder
    inputs = load_vae_decoder_inputs_full_res(dtype=torch.bfloat16)

    # Megatron-style sharding. run_graph_test moves the model to the device and
    # applies mark_sharding; TorchWorkload enables SPMD when a multichip mesh
    # and a shard spec are both present.
    mesh_shape, mesh_names = loader.get_mesh_config(xr.global_runtime_device_count())
    mesh = get_mesh(mesh_shape, mesh_names)

    run_graph_test(
        decoder,
        inputs,
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=loader.load_shard_spec,
    )


def test_vae_decoder_opt1():
    xr.set_device_type("TT")
    torch.manual_seed(42)

    loader = ModelLoader(ModelVariant.MOCHI, subfolder="vae")
    vae = loader.load_model(dtype_override=torch.bfloat16)
    decoder = vae.decoder
    inputs = load_vae_decoder_inputs_full_res(dtype=torch.bfloat16)

    # Megatron-style sharding. run_graph_test moves the model to the device and
    # applies mark_sharding; TorchWorkload enables SPMD when a multichip mesh
    # and a shard spec are both present.
    mesh_shape, mesh_names = loader.get_mesh_config(xr.global_runtime_device_count())
    mesh = get_mesh(mesh_shape, mesh_names)

    run_graph_test(
        decoder,
        inputs,
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=loader.load_shard_spec,
        compiler_config=CompilerConfig(optimization_level=1),
    )


def test_vae_decoder_opt2():
    xr.set_device_type("TT")
    torch.manual_seed(42)

    loader = ModelLoader(ModelVariant.MOCHI, subfolder="vae")
    vae = loader.load_model(dtype_override=torch.bfloat16)
    decoder = vae.decoder
    inputs = load_vae_decoder_inputs_full_res(dtype=torch.bfloat16)

    # Megatron-style sharding. run_graph_test moves the model to the device and
    # applies mark_sharding; TorchWorkload enables SPMD when a multichip mesh
    # and a shard spec are both present.
    mesh_shape, mesh_names = loader.get_mesh_config(xr.global_runtime_device_count())
    mesh = get_mesh(mesh_shape, mesh_names)

    run_graph_test(
        decoder,
        inputs,
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=loader.load_shard_spec,
        compiler_config=CompilerConfig(optimization_level=2),
    )