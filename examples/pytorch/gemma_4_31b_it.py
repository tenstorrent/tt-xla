# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Tensor-parallel greedy text generation with google/gemma-4-31B-it.

google/gemma-4-31B-it is a ~31B Gemma4 image-text-to-text model; this example
drives its *text-only* causal-LM path (the tractable single-/multi-device
bringup target) through the tt-forge-models ``gemma4`` loader. The decoder is
standard GQA (32 query heads), so it shards cleanly with Megatron-style 1D
tensor parallelism across the mesh — q/MLP-gate/up are column-parallel, o_proj
and down_proj are row-parallel, and KV is replicated (see the loader's
``load_shard_spec``).

The loader pins ``use_cache=False`` (the Gemma4 unified head does not accept a
``use_cache`` kwarg), so there is no KV cache to step. To keep the compiled
graph shape-stable — and therefore compile only *once* — generation runs over a
fixed-length buffer: the prompt sits at the front, the tail is padding masked
out by ``attention_mask``, and each new token is written into the next slot.
Every forward sees the same ``[1, T]`` shape, so torch.compile does not recompile
per step.

Modelled after ``examples/pytorch/olmo3_1125_32b.py`` (multi-chip TP generation
loop) and ``examples/pytorch/qwen3_tp.py`` (Megatron sharding), but sourcing the
model, inputs, mesh and shard spec from the loader's public API.
"""

import numpy as np
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh

from third_party.tt_forge_models.gemma4.pytorch import ModelLoader, ModelVariant

MAX_NEW_TOKENS = 20


def setup_spmd():
    """Enable torch_xla SPMD mode for tensor-parallel execution.

    Mirrors examples/pytorch/olmo3_1125_32b.py / qwen3_tp.py.
    """
    import os

    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()


def gemma_4_31b_it():
    num_devices = xr.global_runtime_device_count()
    setup_spmd()

    device = torch_xla.device()

    # --- Model + inputs from the loader's public API -------------------------
    loader = ModelLoader(ModelVariant.GEMMA_4_31B_IT)
    model = loader.load_model(dtype_override=torch.bfloat16).eval()
    inputs = loader.load_inputs(batch_size=1)
    tokenizer = loader.tokenizer

    input_ids = inputs["input_ids"]  # [1, L]
    prompt_len = input_ids.shape[1]
    prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

    # --- Megatron-style tensor-parallel mesh + sharding from the loader ------
    mesh_shape, mesh_names = loader.get_mesh_config(num_devices)
    mesh = Mesh(np.array(range(num_devices)), mesh_shape, mesh_names)
    xs.set_global_mesh(mesh)

    model = model.to(device)
    for weight, shard_spec in loader.load_shard_spec(model).items():
        xs.mark_sharding(weight, mesh, shard_spec)

    # --- Compile (match the bringup baseline for this 31B model on Blackhole):
    #   bfp8 weights to fit the decoder across the mesh; opt level 0 because
    #   higher levels abort compile on the Blackhole OpModel grid.
    torch_xla.set_custom_compile_options(
        {"optimization_level": 0, "experimental_weight_dtype": "bfp_bf8"}
    )
    compiled_model = torch.compile(model, backend="tt")

    # --- Fixed-length greedy decode (no KV cache; shape-stable) --------------
    # buffer_ids holds prompt at the front and pad tokens in the tail; the
    # attention mask hides the not-yet-generated tail so each [1, T] forward is
    # causally identical to a growing-sequence decode but never recompiles.
    total_len = prompt_len + MAX_NEW_TOKENS
    pad_id = tokenizer.pad_token_id
    buffer_ids = torch.full((1, total_len), pad_id, dtype=input_ids.dtype)
    buffer_ids[0, :prompt_len] = input_ids[0]
    buffer_mask = torch.zeros((1, total_len), dtype=inputs["attention_mask"].dtype)
    buffer_mask[0, :prompt_len] = 1

    generated_ids = []
    with torch.no_grad():
        for step in range(MAX_NEW_TOKENS):
            cur_len = prompt_len + step
            output = compiled_model(
                input_ids=buffer_ids.to(device),
                attention_mask=buffer_mask.to(device),
            )
            logits = loader.unpack_forward_output(output).to("cpu")
            next_token_id = logits[0, cur_len - 1].argmax(dim=-1).item()
            print(f"[Step {step}] {'Prefill' if step == 0 else 'Decode'} ...", flush=True)

            if next_token_id == tokenizer.eos_token_id:
                break

            generated_ids.append(next_token_id)
            buffer_ids[0, cur_len] = next_token_id
            buffer_mask[0, cur_len] = 1

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return prompt_text, generated_text


def post_process_output(prompt_text, generated_text):
    """Print the prompt and the model's greedily decoded continuation."""
    print("=" * 80)
    print("PROMPT:")
    print(prompt_text)
    print("-" * 80)
    print("GENERATED:")
    print(generated_text)
    print("=" * 80)


def test_gemma_4_31b_it():
    """Smoke test: tensor-parallel Gemma 4 31B-it produces a non-empty decode.

    Asserts the greedy loop emitted at least one real token (the model ran and
    argmax'd a sensible continuation, not an immediate EOS / empty string).
    """
    xr.set_device_type("TT")

    prompt_text, generated_text = gemma_4_31b_it()
    post_process_output(prompt_text, generated_text)

    assert generated_text.strip(), "Gemma 4 31B-it generated no tokens"
    print("Gemma 4 31B-it tensor-parallel generation produced a non-empty result.")


if __name__ == "__main__":
    # By default torch_xla uses the CPU device, so set it to the TT device.
    xr.set_device_type("TT")

    prompt_text, generated_text = gemma_4_31b_it()
    post_process_output(prompt_text, generated_text)
