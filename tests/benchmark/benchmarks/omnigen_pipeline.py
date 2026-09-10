# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""OmniGen — benchmark-side pipeline for the imagegen harness.

OmniGen (``Shitao/OmniGen-v1-diffusers``) is a unified image-generation DiT with
a LLaMA-style backbone that embeds text tokens internally. The transformer is
the heavy net and runs **tensor-parallel across a multi-chip mesh** (Megatron-1D
on the ``"model"`` axis); the diffusion sampling loop, scheduler and VAE decode
stay on CPU. The end-to-end pipeline (device split + DiT sharding) lives in
``tt_forge_models`` and is the same one the nightly pipeline test drives.

Unlike the GLM-Image benchmark wrapper -- which had to reimplement ``generate``
because its base pipeline neither accepted a per-call step count nor populated a
harness-readable ``_perf`` -- the ``tt_forge_models`` ``OmniGenPipeline`` already
provides both:

  - ``generate(prompt, num_inference_steps=..., seed=...)`` takes the step count
    per call (the harness warms up with 1 step, then runs the full count), and
  - ``self._perf`` is populated with the model-agnostic schema the harness reads
    (``components`` / ``steps`` / ``step_metric_name`` / ``total``).

So this module only bridges the one gap the base pipeline leaves: it returns the
generated image in ``[0, 1]`` (diffusers ``output_type="pt"``), whereas the
harness's ``utils.save_image`` expects ``[-1, 1]``. The subclass converts the
range on the way out and supplies a default ``PROMPT`` (the base defines none),
without duplicating the model loading / sharding setup.
"""

from typing import Optional

import torch

from third_party.tt_forge_models.omnigen.pytorch.src.pipeline import (
    HEIGHT,
    WIDTH,
    OmniGenConfig,
)
from third_party.tt_forge_models.omnigen.pytorch.src.pipeline import (
    OmniGenPipeline as _BaseOmniGenPipeline,
)

# The base pipeline defines no default prompt/seed; the imagegen benchmark needs
# a prompt, so provide one here (matches the nightly pipeline test's prompt).
PROMPT = (
    "a photograph of an astronaut riding a horse in bright daylight, "
    "evenly lit, full frame"
)
SEED = 42

__all__ = ["OmniGenConfig", "OmniGenPipeline", "PROMPT", "SEED", "HEIGHT", "WIDTH"]


class OmniGenPipeline(_BaseOmniGenPipeline):
    """OmniGen pipeline adapted for the imagegen benchmark harness.

    Reuses the base pipeline's model loading, DiT tensor-parallel sharding
    (``setup``) and instrumented ``generate`` (which already accepts a per-call
    ``num_inference_steps`` and populates ``self._perf``). The only override is a
    range conversion on the returned image so ``utils.save_image`` renders it
    correctly.
    """

    @torch.no_grad()
    def generate(
        self,
        prompt: str = PROMPT,
        seed: Optional[int] = SEED,
        num_inference_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Run the base OmniGen t2i pipeline and return the image in ``[-1, 1]``.

        The base ``generate`` records per-stage timings into ``self._perf`` and
        returns a ``(1, 3, H, W)`` float tensor in ``[0, 1]``. The imagegen
        harness saves the steady-state image via ``utils.save_image``, which
        expects ``[-1, 1]`` (it applies ``x / 2 + 0.5``), so shift the range on
        the way out. Timings are untouched -- they come straight from the base.
        """
        image = super().generate(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            seed=seed,
        )
        return image * 2.0 - 1.0
