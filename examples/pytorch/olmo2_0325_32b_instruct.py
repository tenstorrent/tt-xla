# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
OLMo-2-0325-32B-Instruct tensor-parallel next-token prediction example.

allenai/OLMo-2-0325-32B-Instruct is a 64.5 GB (bf16) dense decoder that does not
fit on a single 32 GB Blackhole chip, so it is run with 4-way Megatron-style
tensor parallelism (column-parallel q/k/v + gate/up, row-parallel o_proj +
down_proj) on a (1, N) device mesh. The sharding topology and the model/tokenizer
all come from the tt-forge-models ``ModelLoader`` public API
(``get_mesh_config`` / ``load_shard_spec``), mirroring examples/pytorch/qwen3_tp.py.

This runs a single prefill forward over the prompt and reports the most likely
continuation tokens. OLMo-2's autoregressive *decode* graph is a known gap on TT
hardware (the perf bringup found first-decode PCC ~0), so this example deliberately
demonstrates the validated prefill path rather than a multi-token generation loop.
"""

import os

import numpy as np
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh

from third_party.tt_forge_models.olmo2.causal_lm.pytorch import ModelLoader
from third_party.tt_forge_models.olmo2.causal_lm.pytorch.loader import ModelVariant


def setup_spmd():
    """Enable SPMD mode and the Shardy lowering used for tensor parallelism."""
    print("Setting up XLA environment...")
    # Converts the StableHLO emitted by torch-xla to the Shardy dialect.
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()
    print("XLA environment configured.")


# --------------------------------
# OLMo-2-0325-32B-Instruct tensor-parallel prefill example
# --------------------------------
def olmo2_0325_32b_instruct():
    num_devices = xr.global_runtime_device_count()
    if num_devices < 2:
        raise RuntimeError(
            "OLMo-2-0325-32B-Instruct is 64.5 GB (bf16) and does not fit on a single "
            f"32 GB chip; this example needs a multi-chip mesh (found {num_devices})."
        )

    setup_spmd()

    # Build the model, tokenizer and inputs from the tt-forge-models loader.
    loader = ModelLoader(ModelVariant.Olmo_2_0325_32B_Instruct)
    model = loader.load_model(dtype_override=torch.bfloat16).eval()
    tokenizer = loader.tokenizer
    inputs = loader.load_inputs(batch_size=1)

    # Build the tensor-parallel mesh from the loader's public mesh config.
    mesh_shape, mesh_names = loader.get_mesh_config(num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, mesh_names)
    print(f"Created device mesh: {mesh_shape} {mesh_names} with {num_devices} devices")

    # Move model + inputs to device, then apply the loader's Megatron shard spec.
    device = torch_xla.device()
    model = model.to(device)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    for tensor, shard_spec in loader.load_shard_spec(model).items():
        xs.mark_sharding(tensor, mesh, shard_spec)

    compiled_model = torch.compile(model, backend="tt")

    with torch.no_grad():
        output = compiled_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )

    return output.logits.to("cpu"), tokenizer, loader.sample_text


def post_process_output(logits, tokenizer, prompt, top_k=5):
    """Print the prompt and the model's most likely next tokens (greedy + top-k)."""
    next_token_logits = logits[0, -1]
    greedy_id = int(next_token_logits.argmax(dim=-1))
    topk = torch.topk(next_token_logits.float(), k=top_k)

    print("=" * 80)
    print("PROMPT:")
    print(prompt)
    print("-" * 80)
    print(f"Greedy next token: {tokenizer.decode([greedy_id])!r} (id={greedy_id})")
    print(f"Top-{top_k} next-token candidates:")
    for rank, (score, tok_id) in enumerate(zip(topk.values, topk.indices), start=1):
        tok = tokenizer.decode([int(tok_id)])
        print(f"  {rank}. {tok!r:<20} (id={int(tok_id):>6}, logit={float(score):.3f})")
    print("=" * 80)
    return greedy_id


def test_olmo2_0325_32b_instruct():
    """Prefill forward over the loader's sample prompt must produce finite logits
    with the expected vocabulary width, and a stable greedy next token."""
    xr.set_device_type("TT")

    logits, tokenizer, prompt = olmo2_0325_32b_instruct()

    assert torch.isfinite(logits).all(), "logits contain non-finite values"
    assert logits.shape[-1] == tokenizer.vocab_size or logits.shape[-1] >= len(
        tokenizer
    ), f"unexpected vocab width {logits.shape[-1]}"

    greedy_id = post_process_output(logits, tokenizer, prompt)
    # Greedy token must be a real, decodable vocabulary id.
    assert 0 <= greedy_id < logits.shape[-1]


if __name__ == "__main__":
    # By default torch_xla uses the CPU device so we have to set it to TT device.
    xr.set_device_type("TT")

    logits, tokenizer, prompt = olmo2_0325_32b_instruct()
    post_process_output(logits, tokenizer, prompt)
