# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tensor-parallel image-text-to-text example for google/gemma-4-31B-it.

google/gemma-4-31B-it is a ``Gemma4ForConditionalGeneration`` vision-language
model: a SigLIP-style vision tower feeding a 60-layer Gemma4 text decoder
(GQA 32/16 heads, head_dim 256, interleaved sliding-window / full attention).
The full model is far too large for a single Blackhole chip, so this example
runs it under Megatron-style tensor parallelism across the whole board
(a 1 x N device mesh), exactly the topology the bring-up validated on
qb2-blackhole.

The scenario is a single VLM forward: a candy photo plus the prompt
"What do you see in this image?" are pushed through the sharded model, and the
last-position logits are decoded to show which token(s) the model would begin
its answer with. All weights, inputs, the TP mesh and the shard map come from
the tt-forge-models ``ModelLoader`` public API.
"""

import os

import numpy as np
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh
from tt_torch.transformers_overrides import override_model_sliding_window_causal_mask

from third_party.tt_forge_models.gemma4.multimodal.pytorch import (
    ModelLoader,
    ModelVariant,
)

# opt-level 2 fails with an L1 out-of-memory TT_FATAL on this model (see the
# gemma4-31b-it vLLM TP config, tt-xla #5440); opt-level 1 is the known-good
# level the TP bring-up compiled with.
OPTIMIZATION_LEVEL = 1


def gemma_4_31b_it():
    # By default torch_xla uses the CPU device, so select the TT device and
    # enable SPMD for the multi-chip tensor-parallel run.
    xr.set_device_type("TT")
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()

    num_devices = xr.global_runtime_device_count()

    # Build the model and image+text inputs from the loader's public API.
    loader = ModelLoader(ModelVariant.GEMMA_4_31B_IT)
    model = loader.load_model(dtype_override=torch.bfloat16).eval()
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)

    # Gemma4's interleaved sliding-window / full attention needs the TT-friendly
    # causal-mask rewrite before compiling (loader.requires_model_rewrites).
    if loader.requires_model_rewrites:
        override_model_sliding_window_causal_mask(model)

    # Build the Megatron TP mesh from the loader (1 x num_devices, axes
    # ("batch", "model")).
    mesh_shape, mesh_names = loader.get_mesh_config(num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, mesh_names)

    # Move the model and inputs onto the TT device.
    device = torch_xla.device()
    model = model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Apply the loader's Megatron shard map. Weights not in the map (vision
    # tower, norms, embeddings) and the inputs stay replicated.
    for tensor, shard_spec in loader.load_shard_spec(model).items():
        xs.mark_sharding(tensor, mesh, shard_spec)

    torch_xla.set_custom_compile_options({"optimization_level": OPTIMIZATION_LEVEL})
    compiled_model = torch.compile(model, backend="tt")

    with torch.no_grad():
        output = compiled_model(**inputs)

    logits = loader.unpack_forward_output(output).to("cpu")
    return loader, inputs, logits


def post_process_output(loader, inputs, logits, top_k=5):
    """Print the prompt and the token(s) the model would answer with."""
    tokenizer = getattr(loader.processor, "tokenizer", loader.processor)

    # The next token the model predicts for the final prompt position — the
    # first token of its answer to "What do you see in this image?".
    next_token_logits = logits[0, -1]
    top_scores, top_ids = next_token_logits.topk(top_k)

    print("=" * 80)
    print(f"PROMPT: {loader.sample_text}  (+ candy image)")
    print("-" * 80)
    print(f"Model answer begins with: {tokenizer.decode(top_ids[0].item())!r}")
    print(f"Top-{top_k} next-token candidates:")
    for score, tok_id in zip(top_scores.tolist(), top_ids.tolist()):
        print(f"  {tokenizer.decode(tok_id)!r:>16}  (logit {score:.3f})")
    print("=" * 80)

    return top_ids


def test_gemma_4_31b_it():
    """Guard the TP VLM example: a single forward yields finite, correctly
    shaped logits and a decodable next token."""
    loader, inputs, logits = gemma_4_31b_it()

    # Logits are (batch, seq_len, vocab_size) and must be finite.
    assert logits.ndim == 3, f"expected 3D logits, got shape {tuple(logits.shape)}"
    assert torch.isfinite(logits).all(), "logits contain NaN/Inf"

    text_config = loader.config.get_text_config()
    assert logits.shape[-1] == text_config.vocab_size, (
        f"logits vocab dim {logits.shape[-1]} != config vocab "
        f"{text_config.vocab_size}"
    )

    top_ids = post_process_output(loader, inputs, logits)
    tokenizer = getattr(loader.processor, "tokenizer", loader.processor)
    assert tokenizer.decode(top_ids[0].item()) != "", "top token decoded to empty string"


if __name__ == "__main__":
    loader, inputs, logits = gemma_4_31b_it()
    post_process_output(loader, inputs, logits)
