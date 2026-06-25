# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""FIBO (briaai/FIBO) — benchmark-side pipeline for the imagegen harness.

FIBO is BRIA AI's 8.3B DiT, flow-matching text-to-image model (SmolLM3-3B text
encoder, ``AutoencoderKLWan`` VAE). Unlike the SDXL / Playground pipelines we
do *not* re-implement the denoising loop: ``BriaFiboPipeline.__call__`` already
drives a Flux-style flow-matching loop, so we reuse it verbatim and only
relocate the heavy 8.3B transformer (the denoiser) onto Tenstorrent. The text
encoder, scheduler and VAE stay on CPU — only the DiT denoiser was confirmed on
device during loader bringup, and it is the component whose throughput this
benchmark reports.

Two interpositions make that work:

  - The transformer is ``compile(backend="tt")``-d and moved to the XLA device,
    then wrapped so each forward bridges its inputs CPU->TT, runs the compiled
    graph, and brings the output TT->CPU (the cast forces a sync, ending the
    per-step timer). Attribute reads the pipeline makes on the transformer
    (``config`` / ``dtype`` / ``*_blocks``) are delegated to the wrapped module.
  - The pipeline derives its execution device from the first registered module
    (now the transformer, on TT); we force ``_execution_device`` back to CPU so
    latents, embeds and scheduler tensors are built on CPU and only the
    transformer forward crosses to TT.

Per-step transformer forward+sync times are collected into ``self._perf`` for
the harness to read after each ``generate()`` call.
"""

import time
from typing import Optional

import torch
import torch_xla.core.xla_model as xm

from third_party.tt_forge_models.fibo.pytorch.src.model_utils import load_pipe

MODEL_ID = "briaai/FIBO"


def _to_device(value, device):
    """Recursively move tensors in args/kwargs/containers to ``device``.

    Non-tensor leaves (bools, ints, ``None``, scalars) are passed through
    unchanged so the transformer's mixed positional/keyword signature survives
    the CPU<->TT bridge.
    """
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, list):
        return [_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_device(v, device) for v in value)
    if isinstance(value, dict):
        return {k: _to_device(v, device) for k, v in value.items()}
    return value


class FiboConfig:
    def __init__(
        self,
        transformer_on_tt: bool = True,
        height: int = 512,
        width: int = 512,
        guidance_scale: float = 1.0,
        compile_options: Optional[dict] = None,
    ):
        self.model_id = MODEL_ID
        self.transformer_on_tt = transformer_on_tt
        self.height = height
        self.width = width
        # guidance_scale <= 1 disables classifier-free guidance, keeping the
        # denoiser at batch 1 (halves activation memory on a single chip).
        self.guidance_scale = guidance_scale
        # Harness-set compile options, retained for symmetry with the other
        # imagegen pipelines (FIBO does not need an inline opt-level switch
        # because its VAE stays on CPU).
        self.compile_options = compile_options or {}


class FiboPipeline:
    """FIBO pipeline with the 8.3B DiT denoiser on TT, everything else on CPU."""

    def __init__(self, config: FiboConfig):
        self.config = config
        self._perf = None

    def setup(self):
        # Load the full FIBO pipeline on CPU in bf16 (model card reference dtype).
        self.pipe = load_pipe(self.config.model_id, dtype_override=torch.bfloat16)
        if self.config.transformer_on_tt:
            self._move_transformer_to_tt()

    def _move_transformer_to_tt(self):
        device = xm.xla_device()
        transformer = self.pipe.transformer
        transformer.compile(backend="tt")
        transformer = transformer.to(device)

        owner = self

        class _TTTransformer(torch.nn.Module):
            """Device-bridge + per-step timer around the compiled FIBO DiT."""

            def __init__(self, inner: torch.nn.Module):
                super().__init__()
                self.inner = inner

            def forward(self, *args, **kwargs):
                args = _to_device(args, device)
                kwargs = _to_device(kwargs, device)
                t0 = time.perf_counter()
                out = self.inner(*args, **kwargs)
                # TT -> CPU cast forces a sync; the timer ends after it.
                out = _to_device(out, "cpu")
                if owner._perf is not None:
                    owner._perf["unet_steps"].append(time.perf_counter() - t0)
                return out

            # Delegate the attribute reads BriaFiboPipeline makes on its
            # transformer to the wrapped module.
            @property
            def config(self):
                return self.inner.config

            @property
            def dtype(self):
                return self.inner.dtype

            @property
            def transformer_blocks(self):
                return self.inner.transformer_blocks

            @property
            def single_transformer_blocks(self):
                return self.inner.single_transformer_blocks

        self.pipe.transformer = _TTTransformer(transformer)

        # _execution_device is inferred from the first registered module (now the
        # transformer, on TT). Pin it to CPU so the pipeline builds latents /
        # embeds / scheduler tensors on CPU and only the transformer forward
        # crosses to TT (bridged above).
        pipe_cls = type(self.pipe)

        class _CpuExecPipeline(pipe_cls):
            @property
            def _execution_device(self):
                return torch.device("cpu")

        self.pipe.__class__ = _CpuExecPipeline

    def generate(
        self,
        prompt: str,
        num_inference_steps: int,
        seed: Optional[int] = 42,
    ) -> torch.Tensor:
        # Per-component forward+sync times (reset every generate() call). Only the
        # denoiser runs on TT; te1/te2/vae stay 0.0 (text encoder + VAE on CPU).
        self._perf = {
            "te1": 0.0,
            "te2": 0.0,
            "unet_steps": [],
            "vae": 0.0,
            "total": None,
        }
        t_total_start = time.perf_counter()

        generator = torch.Generator(device="cpu")
        if seed is not None:
            generator.manual_seed(seed)

        with torch.no_grad():
            out = self.pipe(
                prompt=prompt,
                height=self.config.height,
                width=self.config.width,
                num_inference_steps=num_inference_steps,
                guidance_scale=self.config.guidance_scale,
                generator=generator,
                # Stop at the denoised latent. This benchmark measures the TT
                # denoiser; the Wan VAE decode runs on CPU only (not on TT) and
                # is impractically slow at native resolution, so decoding it
                # would dominate the wall clock without exercising the device.
                # This mirrors the loader, which also drives the pipeline with
                # output_type="latent".
                output_type="latent",
                return_dict=False,
            )

        image = out[0]
        if isinstance(image, list):
            image = image[0]

        self._perf["total"] = time.perf_counter() - t_total_start
        return image
