# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Pi0 ``sample_actions`` **denoise loop length** sanities on TT (``run_op_test``).

Same pipeline as ``modeling_pi0`` / ``pi0_pipeline_shared.sample_actions_lines_815_884``;
only ``num_steps`` changes (1, 2, or full ``config.num_inference_steps``).

Prefix-only block (~815--845) stays in ``test_pi0_pipeline_slices.py``.
"""

from __future__ import annotations

import pytest

from infra import Framework, run_op_test
from tests.torch.ops.pi0_pipeline_shared import (
    Pi0SampleActionsBlock815to884NumSteps,
    pi0_torch_cumsum_patch_like_model_py_forward,
    sample_actions_lines_815_884,
)
from utils import Category


@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_pi0_sample_actions_incremental_block_815_884_one_step(request, pi0_bundle):
    """``num_steps == 1`` (~847--884 once)."""
    core, _policy, (
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise,
    ) = pi0_bundle
    run_op_test(
        Pi0SampleActionsBlock815to884NumSteps(core, num_steps=1),
        [images, img_masks, lang_tokens, lang_masks, state, noise],
        framework=Framework.TORCH,
        request=request,
    )


@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_pi0_sample_actions_incremental_block_815_884_two_steps(request, pi0_bundle):
    """``num_steps == 2`` (two denoise iterations; bisects 1 vs full)."""
    core, _policy, (
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise,
    ) = pi0_bundle
    run_op_test(
        Pi0SampleActionsBlock815to884NumSteps(core, num_steps=2),
        [images, img_masks, lang_tokens, lang_masks, state, noise],
        framework=Framework.TORCH,
        request=request,
    )


@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_pi0_sample_actions_incremental_block_815_884_full_steps(request, pi0_bundle):
    """``num_steps=None`` → ``config.num_inference_steps``."""
    core, _policy, (
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise,
    ) = pi0_bundle
    run_op_test(
        Pi0SampleActionsBlock815to884NumSteps(core, num_steps=None),
        [images, img_masks, lang_tokens, lang_masks, state, noise],
        framework=Framework.TORCH,
        request=request,
    )


def test_pi0_incremental_two_steps_matches_core_sample_actions_cpu(pi0_bundle):
    """``num_steps=2`` incremental path matches ``core.sample_actions(..., num_steps=2)``."""
    core, _policy, (
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise,
    ) = pi0_bundle
    incremental = sample_actions_lines_815_884(
        core,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise,
        num_steps=2,
        kwargs={},
    )
    with pi0_torch_cumsum_patch_like_model_py_forward():
        reference = core.sample_actions(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            noise=noise,
            num_steps=2,
        )
    torch.testing.assert_close(incremental, reference, rtol=1e-5, atol=1e-5)
