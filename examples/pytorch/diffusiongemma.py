# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Runnable DiffusionGemma 26B (26B-A4B-it) example on Tenstorrent: all four input paths.

The checkpoint takes text and images (no audio; its encoder does not take video), and emits
text either way. This runs every input path back to back:

  text            a short prompt (15 tokens)
  text (long)     the same path at the image cases' token count, for comparison
  image + text    an image prepended to the question -- the encoder's vision tower turns
                  it into up to 280 soft tokens inside the prompt
  image only      prompt="", so the image is the entire message

The pipeline implementation lives in ``tt_forge_models``; this is a thin runnable demo that
calls it. Both the encoder (prefill) and the decoder (denoising loop) run on the Tenstorrent
backend via ``torch.compile(backend="tt")``, staged so only one is device-resident at a time;
the host driver (sampler/stopping/cache/RNG) runs on CPU.

DiffusionGemma needs transformers>=5.11 but the env is pinned lower, so the run is wrapped in
``RequirementsManager`` (a tt-xla test util): it installs the loader's pinned version for the
run and rolls back on exit. Once the env is uplifted this wrapper can be dropped.

Run: python examples/pytorch/diffusiongemma.py
"""

import inspect
from contextlib import contextmanager

import pytest
import torch_xla.runtime as xr
from loguru import logger

from tests.runner.requirements import RequirementsManager
from third_party.tt_forge_models.diffusiongemma.pytorch import (
    loader as diffgemma_loader,
)
from third_party.tt_forge_models.diffusiongemma.pytorch.pipeline import (
    MAX_NEW_TOKENS,
    PROMPT,
    SEED,
    DiffusionGemmaConfig,
    DiffusionGemmaPipeline,
)

TEXT_LONG_TOKENS = 277  # matches the image cases
CASES = ("text", "text_long", "image", "image_only")


@contextmanager
def _pipeline():
    """Set up once; setup() is the expensive part and every case reuses it."""
    xr.set_device_type("TT")
    # transformers>=5.11 is required for DiffusionGemma; install the loader's pinned
    # version for this run only and roll back on exit.
    loader_path = inspect.getsourcefile(diffgemma_loader)
    with RequirementsManager.for_loader(loader_path, framework="torch"):
        p = DiffusionGemmaPipeline(
            config=DiffusionGemmaConfig(max_new_tokens=MAX_NEW_TOKENS, seed=SEED)
        )
        p.setup()
        yield p


def _generate(pipeline, case):
    if case == "text":
        return pipeline.generate(prompt=PROMPT)
    if case == "text_long":
        prompt, n = pipeline.loader.build_prompt(TEXT_LONG_TOKENS)
        logger.info("long text prompt is {} tokens", n)
        return pipeline.generate(prompt=prompt)
    if case == "image":
        return pipeline.generate(image=True)
    return pipeline.generate(image=True, prompt="")  # image_only


@pytest.fixture(scope="module")
def pipeline():
    with _pipeline() as p:
        yield p


@pytest.mark.parametrize("case", CASES)
def test_diffusiongemma(pipeline, case):
    """One case per test, so a failure names the path that broke."""
    out = _generate(pipeline, case)
    # The decode includes the prompt, so a non-empty string proves nothing on its
    # own; last_new_tokens is what the denoising loop actually produced.
    assert out.strip(), f"{case} produced no output"
    assert pipeline.last_new_tokens > 0, f"{case} generated no new tokens"
    logger.info("DiffusionGemma [{}] output:\n{}", case, out)


def main():
    with _pipeline() as p:
        outs = {case: _generate(p, case) for case in CASES}
    for case, out in outs.items():
        logger.info("DiffusionGemma [{}] output:\n{}", case, out)


if __name__ == "__main__":
    main()
