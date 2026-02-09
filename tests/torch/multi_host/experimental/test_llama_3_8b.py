# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Multi-host tensor parallel test for Llama 3.1 8B across different topologies.

Loads model and inputs via the tt_forge_models loader, applies tensor parallel
sharding, compiles with backend 'tt', and runs inference.

Example:
    # Run on quad_galaxy
    pytest -svv tests/torch/multi_host/experimental/test_llama_3_8b.py -k "quad_galaxy"
"""

import pytest
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from infra.evaluators import ComparisonConfig, PccConfig, TorchComparisonEvaluator
from infra.utilities.torch_multichip_utils import enable_spmd, get_mesh
from torch_xla.distributed.spmd import Mesh
from transformers import AutoTokenizer

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
def test_llama_3_8b_tensor_parallel(topology, configure_topology, mesh_shape):
    """
    Run Llama 3.1 8B inference with tensor parallelism.

    Loads model and inputs via the tt_forge_models loader, runs CPU reference,
    then runs distributed inference on TT devices with tensor parallel sharding,
    compares results with PCC, and decodes the output tokens.
    """
    # Load model and inputs
    loader = ModelLoader(variant=ModelVariant.LLAMA_3_1_8B)
    model = loader.load_model(dtype_override=torch.bfloat16)
    model.eval()
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)

    # Run CPU reference
    print("Running CPU reference inference...")
    with torch.no_grad():
        cpu_output = model(**inputs)
    cpu_logits = cpu_output.logits
    print(f"CPU logits shape: {tuple(cpu_logits.shape)}")

    # Now run on TT devices with tensor parallelism
    xr.set_device_type("TT")
    setup_spmd()
    device = torch_xla.device()
    mesh = create_device_mesh(mesh_shape)

    # Move model and inputs to device
    model_tt = model.to(device)
    inputs_tt = {k: v.to(device) for k, v in inputs.items()}

    # Apply tensor parallel sharding from loader (model weights only)
    shard_specs = loader.load_shard_spec(model_tt)
    assert (
        shard_specs is not None
    ), "Llama 3.1 8B loader must provide shard specs for tensor parallel"
    for tensor, shard_spec in shard_specs.items():
        xs.mark_sharding(tensor, mesh, shard_spec)

    # Compile and run
    print(f"Running TT inference on {mesh_shape[0]}x{mesh_shape[1]} mesh...")
    compiled_model = torch.compile(model_tt, backend="tt")
    with torch.no_grad():
        tt_output = compiled_model(**inputs_tt)

    # Extract logits from CausalLMOutputWithPast
    tt_logits = tt_output.logits
    tt_logits_cpu = tt_logits.cpu()
    print(f"TT logits shape: {tuple(tt_logits_cpu.shape)}")

    # Compare with CPU reference using PCC
    comparison_config = ComparisonConfig(
        pcc=PccConfig(required_pcc=0.95), assert_on_failure=True
    )
    comparator = TorchComparisonEvaluator(comparison_config)
    comparator.evaluate(tt_logits_cpu, cpu_logits)
    print(f"PCC validation passed for topology {topology}")

    # Decode and print output tokens
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

    # Get the predicted token IDs (argmax over vocab dimension)
    predicted_token_ids = tt_logits_cpu.argmax(dim=-1)

    # Decode for each batch item
    batch_size = predicted_token_ids.shape[0]
    for batch_idx in range(batch_size):
        input_ids = inputs["input_ids"][batch_idx].tolist()
        input_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        print(f"\nBatch {batch_idx} input: {input_text}")

        tokens = predicted_token_ids[batch_idx].tolist()
        decoded_text = tokenizer.decode(tokens, skip_special_tokens=True)
        print(f"Batch {batch_idx} decoded output: {decoded_text}")

    print(
        f"Llama 3.1 8B tensor parallel ({mesh_shape[0]}x{mesh_shape[1]} mesh): "
        f"Test completed successfully"
    )
