# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Gemma-4 26B-A4B-it sparse-MoE decoder — tensor-parallel prefill on Tenstorrent.

google/gemma-4-26B-A4B-it is a Gemma4ForConditionalGeneration VLM whose text
decoder is a 128-expert / top-8 sparse MoE (~45.7 GB of expert weights) that
does not fit on a single chip. This example brings up its text prefill graph
tensor-parallel across the mesh: one forward pass over a prompt, returning the
next-token logits (``use_cache=False`` — prefill only, no decode loop).

Everything specific to the model comes from the tt-forge-models loader:

* ``load_model`` swaps the data-dependent MoE expert dispatch for a static
  dense form (so the prefill graph is fully traceable),
* ``get_mesh_config`` gives the Megatron-1D ``(1, num_devices)`` /
  ``("batch", "model")`` mesh,
* ``load_shard_spec`` gives the tensor-parallel weight map — query/output and
  MLP projections column/row-parallel on the sliding layers, and the 128
  experts sharded expert-parallel on their leading dim (a cross-device reduce
  sums the per-expert contributions),
* ``load_inputs`` returns the tokenized prompt plus a precomputed additive
  causal-mask dict (the model's own mask construction lowers to a scatter the
  SPMD pipeline cannot shard), and
* ``unpack_forward_output`` pulls the logits out of the unified model output.

Modelled after ``qwen3_tp.py`` (the loader-agnostic TP example) but sourcing
the mesh and shard spec from the loader, as the MoE topology is model-specific.
"""

import os

import numpy as np
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh

from third_party.tt_forge_models.gemma4.pytorch import ModelLoader, ModelVariant


def _to_device(obj, device):
    """Move a tensor, or a (possibly nested) dict of tensors, to ``device``.

    ``load_inputs`` returns ``attention_mask`` as a dict of per-mask-type
    additive tensors (see the loader), so a plain ``.to(device)`` is not enough.
    """
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    return obj


def gemma_4_26b_a4b_it_tp():
    """Run one tensor-parallel prefill forward and return (loader, ids, logits)."""
    # Gemma4's masks/MoE routing lean on the Shardy SPMD path.
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()

    num_devices = xr.global_runtime_device_count()

    # Everything model-specific comes from the loader.
    loader = ModelLoader(ModelVariant.GEMMA_4_26B_A4B_IT)
    model = loader.load_model(dtype_override=torch.bfloat16).eval()
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)
    # Keep a host copy of the prompt ids for human-readable post-processing.
    input_ids_cpu = inputs["input_ids"].clone()

    # Megatron-1D mesh from the loader: (1, num_devices) over ("batch", "model").
    mesh_shape, mesh_names = loader.get_mesh_config(num_devices)
    mesh = Mesh(np.array(range(num_devices)), mesh_shape, mesh_names)

    device = torch_xla.device()
    model = model.to(device)
    inputs = _to_device(inputs, device)

    # Tensor-parallel weight sharding (expert-parallel MoE + Megatron attention/MLP).
    for tensor, shard_spec in loader.load_shard_spec(model).items():
        xs.mark_sharding(tensor, mesh, shard_spec)

    compiled_model = torch.compile(model, backend="tt")
    with torch.no_grad():
        output = compiled_model(**inputs)

    logits = loader.unpack_forward_output(output).cpu()
    return loader, input_ids_cpu, logits


def post_process_output(loader, input_ids, logits):
    """Print the prompt and the greedily-predicted next token from the logits."""
    tokenizer = loader.tokenizer
    prompt = tokenizer.decode(input_ids[0], skip_special_tokens=True)

    # Prefill has no padding on the single prompt, so the last position holds
    # the next-token distribution.
    next_id = int(logits[0, -1].argmax())
    next_token = tokenizer.decode([next_id], skip_special_tokens=False)

    print(f"Prompt:             {prompt!r}")
    print(f"Logits shape:       {tuple(logits.shape)}")
    print(f"Predicted next id:  {next_id}")
    print(f"Predicted next tok: {next_token!r}")
    print(f"Continuation:       {prompt}{next_token}")
    return next_id, next_token


def test_gemma_4_26b_a4b_it_tp():
    """Guard: TP prefill produces finite, correctly-shaped next-token logits.

    Checks the vocab dimension matches the text config and the greedy next
    token is a valid id — a cheap, stable correctness signal for the example.
    """
    xr.set_device_type("TT")

    loader, input_ids, logits = gemma_4_26b_a4b_it_tp()

    assert torch.isfinite(logits).all(), "logits contain non-finite values"

    text_cfg = getattr(loader.config, "text_config", loader.config)
    vocab_size = text_cfg.vocab_size
    assert logits.shape[0] == input_ids.shape[0]
    assert logits.shape[1] == input_ids.shape[1]
    assert logits.shape[-1] == vocab_size, (
        f"logits vocab dim {logits.shape[-1]} != config vocab_size {vocab_size}"
    )

    next_id, _ = post_process_output(loader, input_ids, logits)
    assert 0 <= next_id < vocab_size, f"predicted id {next_id} out of vocab range"


if __name__ == "__main__":
    # By default torch_xla uses the CPU device so we have to set it to TT device.
    xr.set_device_type("TT")

    loader, input_ids, logits = gemma_4_26b_a4b_it_tp()
    post_process_output(loader, input_ids, logits)
