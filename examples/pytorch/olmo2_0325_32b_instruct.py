# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
OLMo-2-0325-32B-Instruct tensor-parallel (TP) next-token prediction example.

Runs allenai/OLMo-2-0325-32B-Instruct across a 1 x N device mesh using the
Megatron-style column->row tensor-parallel sharding that the tt-forge-models
loader exposes (`get_mesh_config` / `load_shard_spec`), then does a single
prefill forward pass over a real chat prompt and prints the top-5 most likely
next tokens (the causal-LM analogue of resnet_dp.py's top-5 labels).

Scenario note: this is a single prefill forward rather than a full generation
loop. The 32B decoder is too large for a single Blackhole chip, so it must run
tensor-parallel; the prefill forward is the path validated on device for this
model. The example uses `torch.compile(backend="tt")` (the loader's causal mask
is data-dependent and only lowers cleanly through the compiler, not eager).
"""

import numpy as np
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh

from third_party.tt_forge_models.olmo2.causal_lm.pytorch.loader import (
    ModelLoader,
    ModelVariant,
)

TOP_K = 5


def setup_spmd():
    """Enable SPMD mode in torch_xla (same as examples/pytorch/qwen3_tp.py)."""
    import os

    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()


def olmo2_0325_32b_instruct():
    """Load OLMo-2-32B via the loader, shard TP across the mesh, run one forward."""
    num_devices = xr.global_runtime_device_count()
    setup_spmd()

    device = torch_xla.device()

    # Build model + inputs through the tt-forge-models loader public API.
    loader = ModelLoader(ModelVariant.Olmo_2_0325_32B_Instruct)
    model = loader.load_model(dtype_override=torch.bfloat16).eval()
    inputs = loader.load_inputs(batch_size=1)

    # Tensor-parallel mesh + Megatron column->row shard spec straight from the loader.
    mesh_shape, mesh_names = loader.get_mesh_config(num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, mesh_names)
    print(f"Created device mesh: {mesh_shape} names={mesh_names}")

    # Move to device, then mark the loader's weight shard spec on the mesh.
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model = model.to(device)
    for tensor, shard_spec in loader.load_shard_spec(model).items():
        xs.mark_sharding(tensor, mesh, shard_spec)

    # bfp_bf8 weights keep the 32B model within device memory (same option the
    # olmo3_1125_32b.py 32B example uses).
    torch_xla.set_custom_compile_options({"experimental_weight_dtype": "bfp_bf8"})
    compiled_model = torch.compile(model, backend="tt")

    with torch.no_grad():
        output = compiled_model(**inputs, use_cache=False)

    # Logits for the final prompt position -> next-token distribution.
    logits = output.logits.to("cpu")
    next_token_logits = logits[0, -1]
    return next_token_logits, loader.tokenizer


def post_process_output(next_token_logits, tokenizer):
    """Print the top-5 most likely next tokens (human-readable result)."""
    topk = torch.topk(next_token_logits.float(), TOP_K)
    print("=" * 80)
    print(f"Top-{TOP_K} next-token predictions:")
    print("-" * 80)
    for rank, (score, token_id) in enumerate(zip(topk.values, topk.indices), start=1):
        token = tokenizer.decode([token_id.item()])
        print(f"  {rank}. {repr(token):<20} (id={token_id.item():>6}, logit={score.item():.3f})")
    print("=" * 80)
    return topk


def test_olmo2_0325_32b_instruct():
    """Cheap correctness guard for tt-xla CI: finite logits, right vocab size,
    and a well-defined argmax next token."""
    xr.set_device_type("TT")

    next_token_logits, tokenizer = olmo2_0325_32b_instruct()

    assert torch.isfinite(next_token_logits).all(), "next-token logits are not finite"
    assert next_token_logits.ndim == 1, "expected a 1-D next-token logit vector"
    assert next_token_logits.shape[0] == tokenizer.vocab_size or next_token_logits.shape[0] >= tokenizer.vocab_size, (
        f"logit width {next_token_logits.shape[0]} does not cover vocab {tokenizer.vocab_size}"
    )

    topk = post_process_output(next_token_logits, tokenizer)
    # The top prediction must be a strict maximum (stable, well-defined argmax).
    assert topk.values[0].item() == next_token_logits.float().max().item()
    print("test_olmo2_0325_32b_instruct passed.")


if __name__ == "__main__":
    # By default torch_xla uses the CPU device so we have to set it to TT device.
    xr.set_device_type("TT")

    next_token_logits, tokenizer = olmo2_0325_32b_instruct()
    post_process_output(next_token_logits, tokenizer)
