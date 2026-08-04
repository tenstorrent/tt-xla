# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Wan 2.2 image-to-video pipeline running on the Tenstorrent backend.

Unlike the per-component Wan benchmarks (``tests/benchmark/test_wan.py``), this
drives the *whole* pipeline: UMT5 text encode -> multi-step denoise -> VAE
decode, producing one video per generation.

Design note — why diffusers drives the loop
-------------------------------------------
The A14B I2V pipeline has three pieces of correctness-critical math that are
easy to get subtly wrong:

  * the DiT takes 36 input channels (16 noisy latent + 16 image-condition
    latent + 4 mask), not the VAE's ``z_dim``;
  * A14B is a two-expert MoE — ``transformer`` (high noise) and
    ``transformer_2`` (low noise), switched partway through the schedule at
    ``boundary_ratio`` (0.9 for i2v);
  * the conditioning frame must stay fixed across denoise iterations.

Rather than reimplement that, this module keeps ``diffusers``'
:class:`WanImageToVideoPipeline` as the driver (diffusers >= 0.39 supports
``transformer_2`` / ``boundary_ratio`` / ``expand_timesteps`` natively) and only
swaps each *component* for a torch-compiled, device-resident proxy. So all
pipeline math stays upstream's, and we control execution + timing.

Per-stage timings land in ``self._perf`` using the same model-agnostic schema
the image-gen harness consumes:

    _perf = {
        "components": {<name>: seconds, ...},   # scalar per-stage times
        "steps": [seconds, ...],                # per denoise-step times
        "step_metric_name": "dit_step",
        "total": seconds,                       # full generate() wall time
    }
"""

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch_xla
import torch_xla.core.xla_model as xm
from infra.utilities.torch_multichip_utils import enable_spmd


class _TimedDeviceModule(torch.nn.Module):
    """Device-resident, torch-compiled stand-in for one pipeline component.

    Attribute access falls through to the original module, so diffusers keeps
    seeing everything it expects (``config``, ``dtype``, ``add_noise``, ...)
    while ``forward`` runs the compiled version on device and records its time.

    ``slot`` selects where the timing goes: a name records a scalar into
    ``_perf["components"][slot]``; ``None`` appends to ``_perf["steps"]`` (used
    for the DiT, which is called once per denoise step).
    """

    def __init__(self, raw, perf: dict, slot: Optional[str], run_context=None):
        super().__init__()
        # Bypass nn.Module.__setattr__ for the raw handle so it is not
        # registered as a submodule (it is already on device; re-registering
        # would make .to()/state_dict() walk multi-billion-parameter weights).
        object.__setattr__(self, "_raw", raw)
        object.__setattr__(self, "_perf", perf)
        object.__setattr__(self, "_slot", slot)
        object.__setattr__(self, "_run_context", run_context)
        object.__setattr__(self, "_compiled", torch.compile(raw, backend="tt"))

    def forward(self, *args, **kwargs):
        ctx = self._run_context() if self._run_context is not None else None
        t0 = time.perf_counter()
        if ctx is None:
            out = self._compiled(*args, **kwargs)
        else:
            with ctx:
                out = self._compiled(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        if self._slot is None:
            self._perf["steps"].append(elapsed)
        else:
            self._perf["components"][self._slot] = (
                self._perf["components"].get(self._slot, 0.0) + elapsed
            )
        return out

    def __getattr__(self, name):
        # nn.Module.__getattr__ only runs for misses on the instance dict, which
        # is exactly when we want to delegate to the wrapped module.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(object.__getattribute__(self, "_raw"), name)


@dataclass
class WanI2VConfig:
    """One Wan I2V pipeline configuration.

    Attributes:
        shared: The family's ``shared.py`` module (loaders, ``RESOLUTIONS``,
            ``wan22_mesh``, shard specs, ``make_synthetic_first_frame``).
        resolution: Key into ``shared.RESOLUTIONS`` ("480p" / "720p").
        prompt / negative_prompt: Text conditioning.
        guidance_scale: CFG scale for the high-noise expert.
        guidance_scale_2: CFG scale for the low-noise expert.
        sharded: Whether to run under SPMD sharding across ``wan22_mesh``.
        dit_patches: Monkey patches applied before building the DiT.
        dit_run_context: Zero-arg callable returning a context manager wrapping
            each DiT forward (``torch_function_override_disabled``).
        vae_run_context: Same, for the VAE decoder (``safe_xla_slicing``).
    """

    shared: object
    resolution: str
    prompt: str
    negative_prompt: str = ""
    guidance_scale: float = 3.5
    guidance_scale_2: float = 3.5
    sharded: bool = True
    dit_patches: tuple = ()
    dit_run_context: Optional[Callable] = None
    vae_run_context: Optional[Callable] = None


class WanI2VPipeline_TT:
    """Wan 2.2 I2V pipeline with every component executing on Tenstorrent."""

    def __init__(self, config: WanI2VConfig, compile_options: dict):
        self.config = config
        self._perf = {}
        self.mesh = None

        shared = config.shared
        for patch in config.dit_patches:
            patch()

        # SPMD must be enabled before any XLA op is issued.
        if config.sharded:
            self.mesh = shared.wan22_mesh()
            if len(self.mesh.device_ids) > 1:
                enable_spmd()
            else:
                self.mesh = None

        torch_xla.set_custom_compile_options(compile_options)
        self.device = xm.xla_device()

        self.shapes = shared.RESOLUTIONS[config.resolution]
        self._build()

    def _build(self) -> None:
        from diffusers import WanImageToVideoPipeline

        shared = self.config.shared

        # Two 14B experts: `transformer` (high noise) + `transformer_2` (low
        # noise). Both are resident simultaneously because diffusers switches
        # between them mid-schedule.
        raw_umt5 = shared.load_umt5().eval().bfloat16()
        raw_vae = shared.load_vae().eval().bfloat16()
        raw_dit_high = shared.load_dit(subfolder="transformer").eval().bfloat16()
        raw_dit_low = shared.load_dit(subfolder="transformer_2").eval().bfloat16()

        self._perf = {
            "components": {},
            "steps": [],
            "step_metric_name": "dit_step",
            "total": 0.0,
        }

        # Move to device, then shard (sharding must see XLA tensors).
        raw_umt5 = raw_umt5.to(self.device)
        raw_vae = raw_vae.to(self.device)
        raw_dit_high = raw_dit_high.to(self.device)
        raw_dit_low = raw_dit_low.to(self.device)

        if self.mesh is not None:
            shared.shard_umt5_specs(raw_umt5, self.mesh)
            shared.shard_vae_decoder_specs(raw_vae, self.mesh)
            for dit in (raw_dit_high, raw_dit_low):
                shared.shard_dit_specs(dit, self.mesh)
                shared.apply_dit_sp_activation_sharding(dit, self.mesh)

        cfg = self.config
        self.pipeline = WanImageToVideoPipeline(
            tokenizer=shared.load_tokenizer(),
            text_encoder=_TimedDeviceModule(raw_umt5, self._perf, "text_encoder"),
            vae=_TimedDeviceModule(raw_vae, self._perf, "vae", cfg.vae_run_context),
            scheduler=self._build_scheduler(),
            image_processor=None,
            image_encoder=None,
            transformer=_TimedDeviceModule(
                raw_dit_high, self._perf, None, cfg.dit_run_context
            ),
            transformer_2=_TimedDeviceModule(
                raw_dit_low, self._perf, None, cfg.dit_run_context
            ),
            boundary_ratio=shared.BOUNDARY_RATIO["i2v"],
            expand_timesteps=False,
        )

    def _build_scheduler(self):
        from diffusers import UniPCMultistepScheduler

        return UniPCMultistepScheduler(
            prediction_type="flow_prediction",
            use_flow_sigmas=True,
            num_train_timesteps=1000,
            flow_shift=5.0,
        )

    def first_frame(self) -> torch.Tensor:
        """Deterministic conditioning frame at the configured resolution."""
        return self.config.shared.make_synthetic_first_frame(
            self.shapes["video_h"], self.shapes["video_w"]
        )

    def generate(self, prompt: str, num_inference_steps: int):
        """Run one full text+image -> video generation.

        Returns the generated video frames. Resets ``_perf`` timings so each
        call reports only its own stage times.
        """
        self._perf["components"] = {}
        self._perf["steps"] = []

        cfg = self.config
        t_total = time.perf_counter()
        with torch.no_grad():
            output = self.pipeline(
                image=self.first_frame(),
                prompt=prompt,
                negative_prompt=cfg.negative_prompt or None,
                height=self.shapes["video_h"],
                width=self.shapes["video_w"],
                num_frames=self.shapes["num_frames"],
                num_inference_steps=num_inference_steps,
                guidance_scale=cfg.guidance_scale,
                guidance_scale_2=cfg.guidance_scale_2,
                output_type="pt",
                return_dict=False,
            )[0]
        self._perf["total"] = time.perf_counter() - t_total
        return output


def build_wan_i2v_pipeline(config: WanI2VConfig):
    """Factory matching the harness' ``build_pipeline_fn`` contract.

    Returns ``fn(compile_options) -> (pipeline, generate_fn)``.
    """

    def _build(compile_options: dict):
        pipeline = WanI2VPipeline_TT(config, compile_options)
        return pipeline, pipeline.generate

    return _build
