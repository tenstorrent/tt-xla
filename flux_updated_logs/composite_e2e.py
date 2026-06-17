# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Composite FLUX.2-dev pipeline on OUR debug branch (akannan/bringup_flux2).

Mirrors the shared branch's test_multichip.py composition, but imports OUR
current submodule loaders (shard spec = the 1-D Megatron spec on this branch, the
one that gave PCC ~0.65 on the isolate test). PCC is IGNORED here on purpose — the
goal is only to confirm the full pipeline runs e2e and produces an image.

Component placement (same split as the shared branch):
  * text encoder (Mistral3, 24B)  -> CPU
  * transformer / denoiser (32B)  -> DEVICE (TT), tensor-parallel, mesh (1, N)
  * VAE decoder                   -> CPU

Resolution is env-overridable (FLUX2_HEIGHT / FLUX2_WIDTH / FLUX2_STEPS) so we can
later probe 1024x1024; defaults to the branch's 128x128 bringup size.

Run:
  source venv/activate
  HF_TOKEN=... timeout 2400 python flux_updated_logs/composite_e2e.py 2>&1 | tee flux_updated_logs/composite_e2e.log
"""

import os
import sys

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TT_XLA = os.path.abspath(os.path.join(THIS_DIR, ".."))
sys.path.insert(0, TT_XLA)
sys.path.insert(0, os.path.join(TT_XLA, "tests"))

import torch_xla  # noqa: E402
import torch_xla.distributed.spmd as xs  # noqa: E402
import torch_xla.runtime as xr  # noqa: E402
from infra.utilities.torch_multichip_utils import enable_spmd, get_mesh  # noqa: E402

from third_party.tt_forge_models.flux2.pytorch.src.model_utils import (  # noqa: E402
    DTYPE,
    GUIDANCE_SCALE,
    MESH_NAMES,
    PROMPT,
    REPO_ID,
    SEED,
    shard_transformer_specs,
)

HEIGHT = int(os.environ.get("FLUX2_HEIGHT", "128"))
WIDTH = int(os.environ.get("FLUX2_WIDTH", "128"))
NUM_INFERENCE_STEPS = int(os.environ.get("FLUX2_STEPS", "4"))
OUT_DIR = os.environ.get("FLUX2_OUT_DIR", THIS_DIR)


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


def main():
    from diffusers import Flux2Pipeline

    enable_spmd()
    num_devices = xr.global_runtime_device_count()
    print(f"Visible TT devices: {num_devices}", flush=True)
    mesh = get_mesh((1, num_devices), MESH_NAMES)

    print(f"Loading Flux2Pipeline ({REPO_ID}) on CPU ...", flush=True)
    pipe = Flux2Pipeline.from_pretrained(REPO_ID, torch_dtype=DTYPE)

    print("Placing denoiser on device (tensor-parallel) ...", flush=True)
    pipe.transformer = DeviceDenoiser(pipe.transformer, mesh)

    print(
        f"Generating {HEIGHT}x{WIDTH}, {NUM_INFERENCE_STEPS} steps, "
        f"guidance={GUIDANCE_SCALE} ...",
        flush=True,
    )
    generator = torch.Generator().manual_seed(SEED)
    result = pipe(
        prompt=PROMPT,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator,
    )
    image = result.images[0]

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "composite_generated.png")
    image.save(out_path)
    print(f"COMPOSITE SUCCESS -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
