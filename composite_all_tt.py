# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Composite FLUX.2-dev with ALL THREE components on Tenstorrent (TT).

Unlike test_multichip.py (text encoder + VAE on CPU, only denoiser on TT), this
runs every compute component on device:

  * text encoder (Mistral3, ~24B)  -> TT, tensor-parallel (sharded), mesh (1, N)
  * transformer / denoiser (~32B)  -> TT, tensor-parallel (sharded), mesh (1, N)
  * VAE decoder (~84M)             -> TT, replicated across the mesh

Memory strategy (avoids holding 24B + 32B resident at once): run the components
SEQUENTIALLY in the order the pipeline uses them.
  Stage 1: place the text encoder on device, encode the prompt -> prompt_embeds,
           then move it back to CPU and free its device buffers.
  Stage 2: place the denoiser (sharded) + VAE (replicated) on device and run
           pipe(prompt_embeds=...) so encode_prompt skips the (freed) encoder.

Peak device memory is therefore max(text-encoder, denoiser) rather than the sum.

Resolution env-overridable: FLUX2_HEIGHT / FLUX2_WIDTH / FLUX2_STEPS (default 1024).

Run:
  source venv/activate
  HF_TOKEN=... timeout 3000 python composite_all_tt.py
"""

import gc
import os
import sys

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
THIRD_PARTY = os.path.join(THIS_DIR, "third_party")
sys.path.insert(0, THIRD_PARTY)
sys.path.insert(0, os.path.join(THIS_DIR, "tests"))

import torch_xla  # noqa: E402
import torch_xla.distributed.spmd as xs  # noqa: E402
import torch_xla.runtime as xr  # noqa: E402
from infra.utilities.torch_multichip_utils import enable_spmd, get_mesh  # noqa: E402

from tt_forge_models.flux2.pytorch.src.model_utils import (  # noqa: E402
    DTYPE,
    GUIDANCE_SCALE,
    MESH_NAMES,
    PROMPT,
    REPO_ID,
    SEED,
    Mistral3TextEncoderWrapper,
    shard_text_encoder_specs,
    shard_transformer_specs,
    tokenize_prompt,
)

HEIGHT = int(os.environ.get("FLUX2_HEIGHT", "1024"))
WIDTH = int(os.environ.get("FLUX2_WIDTH", "1024"))
NUM_INFERENCE_STEPS = int(os.environ.get("FLUX2_STEPS", "4"))
OUT_DIR = os.environ.get("FLUX2_OUT_DIR", THIS_DIR)


def _free_device():
    gc.collect()
    torch_xla.sync()


class DeviceDenoiser:
    """Routes Flux2Pipeline's transformer calls to the TP-sharded model on device."""

    def __init__(self, transformer, mesh):
        self._dev = torch_xla.device()
        self.config = transformer.config
        self.dtype = next(transformer.parameters()).dtype

        transformer = transformer.to(self._dev)
        if hasattr(transformer, "tie_weights"):
            transformer.tie_weights()
        for tensor, spec in shard_transformer_specs(transformer).items():
            xs.mark_sharding(tensor, mesh, spec)
        self._compiled = torch.compile(transformer, backend="tt")

    def __call__(self, **kwargs):
        moved = {
            k: (v.to(self._dev) if torch.is_tensor(v) else v) for k, v in kwargs.items()
        }
        out = self._compiled(**moved)
        torch_xla.sync()
        if isinstance(out, (tuple, list)):
            return type(out)(o.cpu() if torch.is_tensor(o) else o for o in out)
        return out.cpu()


class DeviceVAEDecoder:
    """Routes Flux2Pipeline's vae.decode() to device (replicated across the mesh).

    The pipeline reads ``vae.bn`` / ``vae.config`` / ``vae.dtype`` (for the host-side
    batch-norm denorm) and then calls ``vae.decode(latents, return_dict=False)[0]``.
    We move the VAE to device (no shard spec -> replicated) and run decode there,
    returning a CPU tensor; the proxied attributes keep the host denorm working.
    """

    def __init__(self, vae, mesh):
        self._dev = torch_xla.device()
        self.config = vae.config
        self.dtype = next(vae.parameters()).dtype
        self.bn = vae.bn  # stays on CPU; pipeline reads it host-side for denorm
        self._vae = vae  # moved to device lazily on first decode (after denoise loop)
        self._compiled = None

    def decode(self, latents, return_dict=False):
        # Lazy device placement: keep VAE off-device during the denoise loop so it
        # does not inflate the denoiser's peak DRAM; place it only now, when the
        # loop's activations have been freed.
        if self._compiled is None:
            vae = self._vae.to(self._dev)
            self._compiled = torch.compile(
                lambda z: vae.decode(z, return_dict=False)[0], backend="tt"
            )
        out = self._compiled(latents.to(self._dev))
        torch_xla.sync()
        image = out.cpu() if torch.is_tensor(out) else out
        return (image,)


def main():
    from diffusers import Flux2Pipeline

    enable_spmd()
    num_devices = xr.global_runtime_device_count()
    print(f"Visible TT devices: {num_devices}", flush=True)
    mesh = get_mesh((1, num_devices), MESH_NAMES)
    dev = torch_xla.device()

    print(f"Loading Flux2Pipeline ({REPO_ID}) on CPU ...", flush=True)
    pipe = Flux2Pipeline.from_pretrained(REPO_ID, torch_dtype=DTYPE)

    # ---- Stage 1: TEXT ENCODER on TT (sharded) -> prompt_embeds, then free ----
    print("Stage 1: placing text encoder on device (tensor-parallel) ...", flush=True)
    text_encoder = pipe.text_encoder
    encoder_wrapper = Mistral3TextEncoderWrapper(text_encoder).eval()
    input_ids, attention_mask = tokenize_prompt(PROMPT)

    text_encoder = text_encoder.to(dev)
    if hasattr(text_encoder, "tie_weights"):
        text_encoder.tie_weights()
    te_specs = shard_text_encoder_specs(text_encoder)
    print(f"  text-encoder sharded tensors: {len(te_specs)}", flush=True)
    if len(te_specs) == 0:
        raise RuntimeError(
            "text-encoder shard spec is empty — descent failed, weights would "
            "replicate and OOM (encoder-OOM fix not effective)."
        )
    for tensor, spec in te_specs.items():
        xs.mark_sharding(tensor, mesh, spec)
    te_compiled = torch.compile(encoder_wrapper, backend="tt")

    print("  encoding prompt on device ...", flush=True)
    with torch.no_grad():
        prompt_embeds = te_compiled(input_ids.to(dev), attention_mask.to(dev))
    torch_xla.sync()
    prompt_embeds = prompt_embeds.cpu()
    print(f"  prompt_embeds: {tuple(prompt_embeds.shape)} {prompt_embeds.dtype}", flush=True)

    # Free the 24B encoder from device before loading the 32B denoiser.
    text_encoder = text_encoder.to("cpu")
    pipe.text_encoder = text_encoder
    del te_compiled, encoder_wrapper
    _free_device()

    # ---- Stage 2: DENOISER (sharded) + VAE (replicated) on TT ----
    print("Stage 2: placing denoiser (sharded) + VAE (replicated) on device ...", flush=True)
    pipe.transformer = DeviceDenoiser(pipe.transformer, mesh)
    pipe.vae = DeviceVAEDecoder(pipe.vae, mesh)

    print(
        f"Generating {HEIGHT}x{WIDTH}, {NUM_INFERENCE_STEPS} steps, "
        f"guidance={GUIDANCE_SCALE} (text-enc + denoiser + vae ALL on TT) ...",
        flush=True,
    )
    generator = torch.Generator().manual_seed(SEED)
    result = pipe(
        prompt=None,
        prompt_embeds=prompt_embeds,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator,
    )
    image = result.images[0]

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "composite_all_tt.png")
    image.save(out_path)
    print(f"COMPOSITE (ALL-TT) SUCCESS -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
