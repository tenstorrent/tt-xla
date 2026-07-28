# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the MRv2 model architecture -> TT ModelState dispatch.

The v2 runner cannot use upstream's ``get_model_state_cls()`` (it returns the
GPU ModelState), so it resolves the TT state class through this registry.
"""

import pytest
import torch.nn as nn
from vllm_tt.model_state import TTModelState
from vllm_tt.model_state_registry import (
    _REGISTRY,
    get_tt_model_state_cls,
    register_tt_model_state,
)


class _Unregistered(nn.Module):
    pass


class _Dummy(nn.Module):
    pass


class _DummySubclass(_Dummy):
    pass


class _DummyState:
    pass


@pytest.fixture
def restore_registry():
    saved = dict(_REGISTRY)
    yield
    _REGISTRY.clear()
    _REGISTRY.update(saved)


@pytest.mark.push
@pytest.mark.cpu
def test_unregistered_model_falls_back_to_tt_model_state():
    assert get_tt_model_state_cls(_Unregistered()) is TTModelState


@pytest.mark.push
@pytest.mark.cpu
def test_registered_arch_resolves_to_its_state(restore_registry):
    register_tt_model_state("_Dummy", f"{__name__}:_DummyState")
    assert get_tt_model_state_cls(_Dummy()) is _DummyState


@pytest.mark.push
@pytest.mark.cpu
def test_subclass_of_registered_arch_matches_via_mro(restore_registry):
    register_tt_model_state("_Dummy", f"{__name__}:_DummyState")
    assert get_tt_model_state_cls(_DummySubclass()) is _DummyState


@pytest.mark.push
@pytest.mark.cpu
def test_subclass_registration_wins_over_base(restore_registry):
    register_tt_model_state("_Dummy", f"{__name__}:_DummyState")
    register_tt_model_state("_DummySubclass", f"{__name__}:TTModelState")
    # MRO order puts the subclass first.
    assert get_tt_model_state_cls(_DummySubclass()) is TTModelState


@pytest.mark.push
@pytest.mark.cpu
def test_diffusion_gemma_arch_is_registered():
    target = _REGISTRY["DiffusionGemmaForConditionalGeneration"]
    assert target == "vllm_tt.diffusion_gemma:TTDiffusionGemmaModelState"
