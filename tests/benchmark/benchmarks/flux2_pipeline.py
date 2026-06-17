# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""FLUX.2-dev — benchmark-side pipeline for the imagegen harness.

FLUX.2-dev is a flow-matching diffusion transformer (DiT), not a UNet model, and
it is far too large to fit on a single chip (~24B text encoder + ~32B
transformer). The compute-dominant component is the transformer / denoiser, so
this pipeline mirrors the proven composite recipe from the tt-forge-models
``flux2/pytorch/test_multichip.py``:

  * text encoder (Mistral3, ~24B)  -> CPU
  * transformer / denoiser (~32B)  -> DEVICE, tensor-parallel across all visible
                                      chips (SPMD mesh ``(1, num_devices)``),
                                      resident through every scheduler step
  * VAE decoder                    -> CPU

The text encoder and VAE stay on CPU only to keep this single-process composite
within device memory while the (sharded) denoiser is resident. The denoiser is
the sharding target and the on-device perf signal.

Per-component forward+sync times are collected into ``self._perf`` for the
imagegen harness to read after each ``generate()`` call. The harness expects the
keys ``te1``, ``te2``, ``unet_steps``, ``vae`` and ``total`` (shared with the
SDXL-style playground pipeline); for FLUX.2 the single text encoder maps to
``te1``, ``te2`` is unused (0.0), and each denoiser step is one ``unet_steps``
entry.
"""

import time
from typing import Optional

import numpy as np
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh

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


def _enable_spmd():
    """Enable torch_xla SPMD mode (mirrors infra.utilities.torch_multichip_utils)."""
    import os

    # tt-mlir's stablehlo pipeline expects shardy annotations from pytorch/xla.
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()


def _get_mesh(mesh_shape, mesh_names):
    num_devices = xr.global_runtime_device_count()
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, mesh_names)
    print(f"Created device mesh: {mesh_shape} with {num_devices} devices.")
    return mesh


class _DeviceDenoiser:
    """Routes Flux2Pipeline's transformer calls to the TP-sharded model on device.

    Mirrors ``test_multichip.DeviceDenoiser``: the transformer is moved to the
    device, its weights are partitioned per ``shard_transformer_specs`` and the
    module is compiled with the ``tt`` backend. Each call moves the (small)
    per-step activations to the device, runs the compiled denoiser, syncs and
    returns the result to CPU so the host-side scheduler loop can advance.

    The per-step device time is appended to the owning pipeline's current
    ``_perf["unet_steps"]`` so the harness sees only the steady-state pass.
    """

    def __init__(self, transformer, mesh, owner):
        self._dev = torch_xla.device()
        self._owner = owner
        self.config = transformer.config
        self.dtype = next(transformer.parameters()).dtype

        transformer = transformer.to(self._dev)
        if hasattr(transformer, "tie_weights"):
            transformer.tie_weights()
        for tensor, spec in shard_transformer_specs(transformer).items():
            xs.mark_sharding(tensor, mesh, spec)
        # Optional per-weight dtype override (perf-tuning knob). Applied AFTER
        # mark_sharding so the TP annotations stay on the leaf parameters
        # (torch parametrize keeps the sharded leaf as ``parametrizations.*.original``);
        # applied BEFORE torch.compile so the override custom_call is traced.
        overrides = getattr(owner.config, "weight_dtype_overrides", None)
        if overrides:
            from tt_torch.weight_dtype import apply_weight_dtype_overrides

            applied = apply_weight_dtype_overrides(transformer, overrides)
            print(f"Applied {len(applied)} weight dtype overrides to denoiser.")
        self._compiled = torch.compile(transformer, backend="tt")

    def __call__(self, **kwargs):
        moved = {
            k: (v.to(self._dev) if torch.is_tensor(v) else v)
            for k, v in kwargs.items()
        }
        t0 = time.perf_counter()
        out = self._compiled(**moved)
        torch_xla.sync()
        if isinstance(out, (tuple, list)):
            result = type(out)(
                o.cpu() if torch.is_tensor(o) else o for o in out
            )
        else:
            result = out.cpu()
        self._owner._perf["unet_steps"].append(time.perf_counter() - t0)
        return result


class Flux2Config:
    def __init__(
        self,
        compile_options: Optional[dict] = None,
        weight_dtype_overrides: Optional[object] = None,
    ):
        self.model_id = REPO_ID
        self.height = HEIGHT
        self.width = WIDTH
        self.guidance_scale = GUIDANCE_SCALE
        self.seed = SEED
        # Harness-set compile options (kept for parity with the other imagegen
        # pipelines; FLUX.2 does not switch options inline).
        self.compile_options = compile_options or {}
        # Optional per-weight dtype override applied to the on-device denoiser
        # (perf-tuning knob; e.g. "bfp_bf8" or {"default": "bfp_bf8"}).
        self.weight_dtype_overrides = weight_dtype_overrides


class Flux2Pipeline:
    """FLUX.2-dev composite pipeline: TP denoiser on device, text encoder + VAE on CPU."""

    def __init__(self, config: Flux2Config):
        self.config = config
        self._perf = {
            "te1": 0.0,
            "te2": 0.0,
            "unet_steps": [],
            "vae": 0.0,
            "total": None,
        }

    def setup(self):
        from diffusers import Flux2Pipeline as DiffusersFlux2Pipeline

        _enable_spmd()
        num_devices = xr.global_runtime_device_count()
        print(f"Visible TT devices: {num_devices}", flush=True)
        # All visible chips on the "model" axis — the transformer shard specs
        # only reference "model" (Megatron-style TP); batch axis is 1.
        self.mesh = _get_mesh((1, num_devices), MESH_NAMES)

        print(f"Loading Flux2Pipeline ({self.config.model_id}) on CPU ...", flush=True)
        self.pipe = DiffusersFlux2Pipeline.from_pretrained(
            self.config.model_id, torch_dtype=DTYPE
        )

        # Time the CPU text encoder and VAE decode by wrapping their forward
        # paths (best-effort; just delegates + records into self._perf).
        self._instrument_cpu_components()

        print("Placing denoiser on device (tensor-parallel) ...", flush=True)
        self.pipe.transformer = _DeviceDenoiser(self.pipe.transformer, self.mesh, self)

    def _instrument_cpu_components(self):
        orig_te_forward = self.pipe.text_encoder.forward

        def timed_te(*args, **kwargs):
            t0 = time.perf_counter()
            out = orig_te_forward(*args, **kwargs)
            self._perf["te1"] += time.perf_counter() - t0
            return out

        self.pipe.text_encoder.forward = timed_te

        orig_vae_decode = self.pipe.vae.decode

        def timed_vae(*args, **kwargs):
            t0 = time.perf_counter()
            out = orig_vae_decode(*args, **kwargs)
            self._perf["vae"] += time.perf_counter() - t0
            return out

        self.pipe.vae.decode = timed_vae

    def generate(self, prompt: str, num_inference_steps: int) -> torch.Tensor:
        # Reset per-generate timers so the harness reads only this pass.
        self._perf = {
            "te1": 0.0,
            "te2": 0.0,
            "unet_steps": [],
            "vae": 0.0,
            "total": None,
        }
        t_total_start = time.perf_counter()

        generator = torch.Generator().manual_seed(self.config.seed)
        with torch.no_grad():
            result = self.pipe(
                prompt=prompt,
                height=self.config.height,
                width=self.config.width,
                num_inference_steps=num_inference_steps,
                guidance_scale=self.config.guidance_scale,
                generator=generator,
                output_type="pt",
            )

        self._perf["total"] = time.perf_counter() - t_total_start

        # Flux2Pipeline (output_type="pt") returns images in [0, 1], shape
        # (B, 3, H, W). save_image() expects [-1, 1] (it applies /2 + 0.5), so
        # remap to that convention.
        image = result.images
        return image * 2.0 - 1.0
