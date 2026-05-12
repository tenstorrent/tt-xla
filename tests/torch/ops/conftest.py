# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for Pi0 op tests under ``tests/torch/ops/``."""

from __future__ import annotations

import inspect

import pytest

import third_party.tt_forge_models.pi_0.pytorch.loader as pi0_loader_module
from infra import Framework
from tests.runner.requirements import RequirementsManager
from third_party.tt_forge_models.pi_0.pytorch.loader import ModelLoader, ModelVariant


@pytest.fixture(scope="module")
def pi0_bundle():
    """Load policy + core + one real input batch (same path as ``test_models``).

    Returns ``(core, policy, bundle)`` where ``core`` is ``PI0Pytorch`` and ``policy``
    is the customized ``PI0Policy`` from ``pi_0/pytorch/src/model.py``.

    Uses ``RequirementsManager.for_loader`` on ``pi_0/pytorch/loader.py`` so
    ``requirements.txt`` / ``requirements.nodeps.txt`` match the full model test.
    Yields inside the context so packages are not rolled back until all Pi0 op
    tests in this directory that use this fixture finish.
    """
    loader_path = inspect.getsourcefile(pi0_loader_module)
    assert loader_path is not None
    with RequirementsManager.for_loader(loader_path, framework=str(Framework.TORCH)):
        loader = ModelLoader(ModelVariant.LIBERO_BASE)
        loader.load_model()
        bundle = loader.load_inputs()
        yield loader.pi_0.model, loader.pi_0, bundle
