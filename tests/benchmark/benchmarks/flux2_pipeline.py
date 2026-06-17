# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""FLUX.2-dev — benchmark-side pipeline for the imagegen harness.

Mirrors the validated composite in
``third_party/tt_forge_models/flux2/pytorch/test_multichip.py``: the
compute-dominant denoiser (Flux2Transformer2DModel, ~32B) runs on device
tensor-parallel across the full mesh and stays resident through every
scheduler step; the 24B Mistral3 text encoder and the VAE decoder stay on CPU
to keep the single-process composite within device memory (each passes its own
on-device component test separately).

The full ``diffusers`` ``Flux2Pipeline`` owns the latent packing / scheduler /
guidance glue that cannot be traced as a single graph, so we reuse it verbatim
and only (a) swap ``pipe.transformer`` for an on-device, TP-sharded denoiser and
(b) install lightweight timers on the text encoder and VAE so the harness can
read a per-component breakdown from ``self._perf`` after each ``generate()``.

``_perf`` matches the contract the imagegen harness expects (see
``benchmarks/imagegen_benchmark.py``): ``te1`` is the single text encoder,
``te2`` is unused (0.0), ``unet_steps`` collects per-step denoiser times, ``vae``
is the decode time and ``total`` is the wall clock for the whole generation.
"""

import time
from typing import Optional

import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from infra.utilities.torch_multichip_utils import enable_spmd, get_mesh
from loguru import logger

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


def _mesh_shape(num_devices: int):
    # ("batch", "model") mesh; the transformer shard specs only use "model", so
    # batch stays 1 and every chip holds a 1/num_devices slice of the denoiser.
    return (1, num_devices)


class _DeviceDenoiser:
    """Routes Flux2Pipeline's transformer calls to the TP-sharded model on device.

    Same shape as ``test_multichip.DeviceDenoiser`` (move → shard → compile,
    then move inputs on / results off each step) with per-step timing folded in
    so the harness sees one entry per scheduler step in ``_perf['unet_steps']``.
    """

    def __init__(self, transformer, mesh, perf_owner):
        self._dev = torch_xla.device()
        self._owner = perf_owner
        # Flux2Pipeline reads .config / .dtype off the transformer it calls.
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
            k: (v.to(self._dev) if torch.is_tensor(v) else v)
            for k, v in kwargs.items()
        }
        t0 = time.perf_counter()
        out = self._compiled(**moved)
        # Cast back to CPU forces the XLA sync, so the timer captures the full
        # on-device step (compile is a no-op after the warmup pass).
        if isinstance(out, (tuple, list)):
            out = type(out)(o.cpu() if torch.is_tensor(o) else o for o in out)
        else:
            out = out.cpu()
        self._owner._perf["unet_steps"].append(time.perf_counter() - t0)
        return out


class Flux2BenchmarkPipeline:
    """FLUX.2-dev composite: denoiser on device (TP), text encoder + VAE on CPU."""

    def __init__(self, compile_options: Optional[dict] = None):
        # Stored for parity with the imagegen harness contract; the denoiser
        # compiles with the harness-set global options, so nothing to merge here.
        self.compile_options = compile_options or {}
        self._perf = {}

    def setup(self):
        from diffusers import Flux2Pipeline

        enable_spmd()
        num_devices = xr.global_runtime_device_count()
        mesh = get_mesh(_mesh_shape(num_devices), MESH_NAMES)

        logger.info(f"Loading Flux2Pipeline ({REPO_ID}) on CPU ...")
        self.pipe = Flux2Pipeline.from_pretrained(REPO_ID, torch_dtype=DTYPE)

        logger.info("Placing denoiser on device (tensor-parallel) ...")
        self.pipe.transformer = _DeviceDenoiser(self.pipe.transformer, mesh, self)

        self._install_timers()

    def _install_timers(self):
        """Wrap the CPU text encoder + VAE so each generate() records its time."""
        text_encoder = self.pipe.text_encoder
        orig_te_forward = text_encoder.forward

        def timed_text_encoder(*args, **kwargs):
            t0 = time.perf_counter()
            out = orig_te_forward(*args, **kwargs)
            self._perf["te1"] = time.perf_counter() - t0
            return out

        text_encoder.forward = timed_text_encoder

        vae = self.pipe.vae
        orig_decode = vae.decode

        def timed_decode(*args, **kwargs):
            t0 = time.perf_counter()
            out = orig_decode(*args, **kwargs)
            self._perf["vae"] = time.perf_counter() - t0
            # Capture the raw decode sample (range [-1, 1], BCHW) so generate()
            # can hand it to the harness's save_image (which expects [-1, 1]).
            if isinstance(out, (tuple, list)):
                self._last_decode = out[0]
            else:
                self._last_decode = getattr(out, "sample", out)
            return out

        vae.decode = timed_decode

    def generate(self, prompt: str, num_inference_steps: int) -> torch.Tensor:
        self._perf = {
            "te1": 0.0,
            "te2": 0.0,  # FLUX.2 has a single text encoder.
            "unet_steps": [],
            "vae": 0.0,
            "total": None,
        }
        self._last_decode = None

        generator = torch.Generator().manual_seed(SEED)
        t_total_start = time.perf_counter()
        with torch.no_grad():
            # output_type="pt" (not "latent") so the VAE actually decodes and the
            # timer fires; the postprocessed return value is ignored in favour of
            # the captured raw decode tensor.
            self.pipe(
                prompt=prompt,
                height=HEIGHT,
                width=WIDTH,
                num_inference_steps=num_inference_steps,
                guidance_scale=GUIDANCE_SCALE,
                generator=generator,
                output_type="pt",
                return_dict=False,
            )
        self._perf["total"] = time.perf_counter() - t_total_start

        return self._last_decode
