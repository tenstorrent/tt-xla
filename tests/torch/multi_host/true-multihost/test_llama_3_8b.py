# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Multi-host tensor parallel test for Llama 3.1 8B across different topologies.

Loads model and inputs via the tt_forge_models loader, applies tensor parallel
sharding, compiles with backend 'tt', and runs inference.

Example:
    # Run on quad_galaxy
    pytest -svv tests/torch/multi_host/true-multihost/test_llama_31_8b.py -k "quad_galaxy"
"""

import pytest
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from infra.utilities.torch_multichip_utils import enable_spmd, get_mesh
from torch_xla.distributed.spmd import Mesh

from third_party.tt_forge_models.llama.causal_lm.pytorch.loader import (
    ModelLoader,
    ModelVariant,
)


def setup_spmd():
    """Enable SPMD mode with Shardy conversion (required for tensor parallel)."""
    enable_spmd()


def create_device_mesh(mesh_shape: tuple[int, int]) -> Mesh:
    """Create device mesh with specified shape and names ('batch', 'model')."""
    return get_mesh(mesh_shape, ("batch", "model"))


@pytest.mark.tensor_parallel
@pytest.mark.parametrize("topology", ["dual_bh_quietbox", "dual_galaxy", "quad_galaxy"])
def test_llama_31_8b_tensor_parallel(topology, configure_topology, mesh_shape):
    """
    Run Llama 3.1 8B inference with tensor parallelism.

    Loads model and inputs via the tt_forge_models loader, moves to device,
    applies loader's shard spec (model weights), compiles with backend 'tt',
    runs forward, and validates output shape.
    """
    xr.set_device_type("TT")
    setup_spmd()
    device = torch_xla.device()
    mesh = create_device_mesh(mesh_shape)

    # Load model and inputs (same pattern as test_models.py / DynamicTorchModelTester)
    loader = ModelLoader(variant=ModelVariant.LLAMA_3_1_8B)
    model = loader.load_model(dtype_override=torch.bfloat16)
    model.eval()
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)

    # Move model and inputs to device
    model = model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Apply tensor parallel sharding from loader (model weights only)
    shard_specs = loader.load_shard_spec(model)
    assert (
        shard_specs is not None
    ), "Llama 3.1 8B loader must provide shard specs for tensor parallel"
    for tensor, shard_spec in shard_specs.items():
        xs.mark_sharding(tensor, mesh, shard_spec)

    # Compile and run
    compiled_model = torch.compile(model, backend="tt")
    compiled_model = compiled_model.to(device)
    with torch.no_grad():
        output = compiled_model(**inputs)

    # Unpack logits (CausalLMOutputWithPast -> logits)
    logits = loader.unpack_forward_output(output)
    logits_cpu = logits.cpu()

    # Basic sanity: logits shape (batch, seq_len, vocab_size)
    assert logits_cpu.dim() == 3, f"Expected 3D logits, got {logits_cpu.dim()}"
    batch, seq_len, vocab_size = logits_cpu.shape
    assert batch == inputs["input_ids"].shape[0]
    assert seq_len == inputs["input_ids"].shape[1]
    assert vocab_size > 0

    # Optional: compare against CPU reference (expensive; can be gated by a marker)
    # For a quick run we only assert shape and that the run completed.
    print(
        f"Llama 3.1 8B tensor parallel ({mesh_shape[0]}x{mesh_shape[1]} mesh): "
        f"logits shape {tuple(logits_cpu.shape)}"
    )
