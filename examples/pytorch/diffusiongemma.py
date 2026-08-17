# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Runnable DiffusionGemma 26B (26B-A4B-it) text-to-text example on Tenstorrent.

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


def main():
    xr.set_device_type("TT")

    # transformers>=5.11 is required for DiffusionGemma; install the loader's pinned version for
    # this run only and roll back on exit (env stays clean for others).
    loader_path = inspect.getsourcefile(diffgemma_loader)
    with RequirementsManager.for_loader(loader_path, framework="torch"):
        pipeline = DiffusionGemmaPipeline(
            config=DiffusionGemmaConfig(max_new_tokens=MAX_NEW_TOKENS, seed=SEED)
        )
        pipeline.setup()
        text = pipeline.generate(prompt=PROMPT)

    logger.info("DiffusionGemma output:\n{}", text)


if __name__ == "__main__":
    main()
