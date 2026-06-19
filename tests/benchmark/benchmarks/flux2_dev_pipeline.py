# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""FLUX.2-dev — benchmark-side composite pipeline for the imagegen harness.

Mirrors the validated component plan in
``third_party/tt_forge_models/flux2/pytorch/test_multichip.py``:

  * Mistral3 text encoder (~24B)      -> CPU
  * Flux2 transformer / denoiser (~32B) -> DEVICE, tensor-parallel across the
                                           full mesh (1 x num_devices), resident
                                           through every scheduler step
  * AutoencoderKLFlux2 decoder (~84M) -> CPU

The transformer is the compute-dominant, device-resident component and the
sharding target, so it runs on device. The text encoder and VAE stay on CPU to
keep the single-process composite within device memory (each is validated on
device separately by the loader's component tests). Diffusers' ``Flux2Pipeline``
owns the scheduler loop and latent glue, so we only swap its ``.transformer`` for
the TP-sharded on-device denoiser and time each component into ``self._perf`` for
the imagegen harness.
"""

import os
import time
from typing import Optional

import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from infra.utilities.torch_multichip_utils import enable_spmd, get_mesh

from third_party.tt_forge_models.flux2.pytorch.src.model_utils import (
    DTYPE,
    GUIDANCE_SCALE,
    HEIGHT,
    MESH_NAMES,
    REPO_ID,
    SEED,
    WIDTH,
    shard_transformer_specs,
)


class Flux2DevConfig:
    def __init__(self, compile_options: Optional[dict] = None):
        self.repo_id = REPO_ID
        self.height = HEIGHT
        self.width = WIDTH
        self.guidance_scale = GUIDANCE_SCALE
        self.dtype = DTYPE
        # Harness-set compile options (forwarded; the denoiser compiles with
        # whatever is active globally via torch_xla.set_custom_compile_options).
        self.compile_options = compile_options or {}


class _DeviceDenoiser:
    """Route Flux2Pipeline's transformer calls to the TP-sharded model on device.

    Mirrors ``test_multichip.py::DeviceDenoiser`` plus per-call timing into the
    shared ``_perf["unet_steps"]`` list (one entry per scheduler step).
    """

    def __init__(self, transformer, mesh, perf):
        self._dev = torch_xla.device()
        # Flux2Pipeline reads transformer.config / transformer.dtype directly.
        self.config = transformer.config
        self.dtype = next(transformer.parameters()).dtype
        self._perf = perf

        transformer = transformer.to(self._dev)
        if hasattr(transformer, "tie_weights"):
            transformer.tie_weights()
        for tensor, spec in shard_transformer_specs(transformer).items():
            xs.mark_sharding(tensor, mesh, spec)
        # Optional perf knob (model-perf-tuning): per-weight dtype override on the
        # TP-sharded denoiser. Applied AFTER mark_sharding so the sharding
        # annotations sit on the leaf weights; the override op then runs per-shard
        # inside the compiled graph. Off by default — opt in via FLUX2_WEIGHT_DTYPE.
        weight_dtype = os.environ.get("FLUX2_WEIGHT_DTYPE")
        if weight_dtype:
            from tt_torch.weight_dtype import apply_weight_dtype_overrides

            applied = apply_weight_dtype_overrides(transformer, weight_dtype)
            print(f"Applied weight_dtype_override={weight_dtype} to {len(applied)} tensors")
        self._compiled = torch.compile(transformer, backend="tt")

    def __call__(self, **kwargs):
        t0 = time.perf_counter()
        moved = {
            k: (v.to(self._dev) if torch.is_tensor(v) else v)
            for k, v in kwargs.items()
        }
        out = self._compiled(**moved)
        # cpu() forces a sync, so the timer below covers the full device step.
        torch_xla.sync()
        if isinstance(out, (tuple, list)):
            out = type(out)(o.cpu() if torch.is_tensor(o) else o for o in out)
        else:
            out = out.cpu()
        self._perf["unet_steps"].append(time.perf_counter() - t0)
        return out


class Flux2DevPipeline:
    """FLUX.2-dev composite pipeline with the denoiser tensor-parallel on TT."""

    def __init__(self, config: Flux2DevConfig):
        self.config = config
        self._perf = None

    def setup(self):
        from diffusers import Flux2Pipeline

        enable_spmd()
        num_devices = xr.global_runtime_device_count()
        # Full-mesh tensor parallel: (batch=1, model=num_devices). The transformer
        # shard specs reference the "model" axis; the residual stream stays
        # replicated. Any visible device count maps to a single (1, N) mesh.
        self.mesh = get_mesh((1, num_devices), MESH_NAMES)

        self.pipe = Flux2Pipeline.from_pretrained(
            self.config.repo_id, torch_dtype=self.config.dtype
        )
        # Swap the denoiser for the TP-sharded on-device component; the text
        # encoder and VAE remain on CPU inside the diffusers pipeline.
        self.pipe.transformer = _DeviceDenoiser(
            self.pipe.transformer, self.mesh, perf={"unet_steps": []}
        )

    def generate(self, prompt: str, num_inference_steps: int) -> torch.Tensor:
        # Per-component forward+sync times (reset every generate() call).
        self._perf = {
            "te1": 0.0,
            "te2": 0.0,  # FLUX.2 has a single text encoder; te2 stays 0.
            "unet_steps": [],
            "vae": 0.0,
            "total": None,
        }
        # Point the denoiser at this run's step list.
        self.pipe.transformer._perf = self._perf

        t_total = time.perf_counter()
        with torch.no_grad():
            # ── Text encoder (Mistral3, ~24B) on CPU ──────────────────────
            t0 = time.perf_counter()
            prompt_embeds, _ = self.pipe.encode_prompt(prompt, device="cpu")
            self._perf["te1"] = time.perf_counter() - t0

            # ── Denoising loop (transformer on device, TP) + VAE on CPU ───
            generator = torch.Generator().manual_seed(SEED)
            result = self.pipe(
                prompt_embeds=prompt_embeds,
                height=self.config.height,
                width=self.config.width,
                num_inference_steps=num_inference_steps,
                guidance_scale=self.config.guidance_scale,
                generator=generator,
                output_type="pt",
            )

        self._perf["total"] = time.perf_counter() - t_total
        # The VAE decode runs on CPU inside Flux2Pipeline.__call__ and is not
        # separately isolated here; its wall time is surfaced by the harness as
        # CPU overhead (total - text_encoder - denoiser steps).

        # output_type="pt" yields image(s) in [0, 1]; the harness's save_image
        # expects [-1, 1] (it maps x/2 + 0.5 -> [0, 1]). Take the first image,
        # restore the batch dim, and rescale.
        images = result.images
        image = images[0]
        if image.dim() == 3:
            image = image.unsqueeze(0)
        return image * 2.0 - 1.0
