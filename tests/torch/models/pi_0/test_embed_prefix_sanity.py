# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Sanity tests for embed_prefix and embed_prefix + make_att_2d_masks composition.

BACKGROUND:
  The PI0 inference path (sample_actions) hangs when embed_prefix and
  make_att_2d_masks are composed in a single XLA graph.  Adding
  xm.mark_step() between them breaks the graph into two compilations
  that each succeed.

  The existing test_make_att_2d_masks_sanity.py tests make_att_2d_masks
  in isolation with hardcoded static tensors.  While the *values* match
  what embed_prefix outputs (all-True pad_masks, all-False att_masks at
  shape [1, 816]), those tests cannot reproduce the hang because the
  static tensors carry no preceding XLA computation graph.

  These two tests use the *actual model and real dataset inputs* so that
  the XLA graph includes the full SigLIP vision encoder + language
  embedding from embed_prefix.

TEST 1 – test_embed_prefix_sanity:
  Runs embed_prefix alone.  Expected to PASS (no hang).

TEST 2 – test_embed_prefix_plus_make_att_2d_masks_sanity:
  Runs embed_prefix followed by make_att_2d_masks in the same graph.
  This is the combination that reproduces the hang during
  StableHLO → TTIR compilation.

TEST 3 – test_embed_prefix_plus_make_att_2d_masks_with_graph_break_sanity:
  Same as Test 2 but with an xm.mark_step() graph break between
  embed_prefix and make_att_2d_masks.  Expected to PASS, proving the
  hang only occurs when the two are compiled together in one graph.
"""

import pytest
import torch
from infra import Framework, run_op_test
from infra.evaluators import AllcloseConfig
from lerobot.policies.pi0.modeling_pi0 import make_att_2d_masks

from third_party.tt_forge_models.pi_0.pytorch import ModelLoader, ModelVariant





# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def pi0_model_and_inputs():
    """Load PI0 model (LIBERO_BASE) and real dataset inputs once per module."""
    loader = ModelLoader(ModelVariant.LIBERO_BASE)
    model = loader.load_model()
    images, img_masks, lang_tokens, lang_masks, state, noise = loader.load_inputs()
    return model, images, img_masks, lang_tokens, lang_masks


# ---------------------------------------------------------------------------
# Op wrappers
# ---------------------------------------------------------------------------
class EmbedPrefixOp(torch.nn.Module):
    """Wraps PI0Pytorch.embed_prefix so it can be tested via run_op_test.

    Inputs are passed as a flat arg list and reconstructed into the
    (images, img_masks, lang_tokens, lang_masks) structure that
    embed_prefix expects.
    """

    def __init__(self, pi0_pytorch_model, num_images):
        super().__init__()
        self.pi0 = pi0_pytorch_model
        self.num_images = num_images

    def forward(self, *args):
        n = self.num_images
        images = list(args[:n])
        img_masks = list(args[n : 2 * n])
        lang_tokens = args[2 * n]
        lang_masks = args[2 * n + 1]

        embs, pad_masks, att_masks = self.pi0.model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        return embs


class EmbedPrefixPlusMakeAtt2dMasksOp(torch.nn.Module):
    """Wraps embed_prefix + make_att_2d_masks in a single graph.

    This is the exact composition that hangs during StableHLO → TTIR
    compilation when there is no xm.mark_step() between the two.
    """

    def __init__(self, pi0_pytorch_model, num_images):
        super().__init__()
        self.pi0 = pi0_pytorch_model
        self.num_images = num_images

    def forward(self, *args):
        n = self.num_images
        images = list(args[:n])
        img_masks = list(args[n : 2 * n])
        lang_tokens = args[2 * n]
        lang_masks = args[2 * n + 1]

        embs, pad_masks, att_masks = self.pi0.model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        return att_2d_masks.to(torch.float32)


class EmbedPrefixPlusMakeAtt2dMasksWithGraphBreakOp(torch.nn.Module):
    """Same as EmbedPrefixPlusMakeAtt2dMasksOp but with xm.mark_step()
    between embed_prefix and make_att_2d_masks.

    The graph break forces two separate compilations (embed_prefix and
    make_att_2d_masks) that each succeed individually.  This proves the
    hang is caused by composing the two in a single compilation unit.
    """

    def __init__(self, pi0_pytorch_model, num_images):
        super().__init__()
        self.pi0 = pi0_pytorch_model
        self.num_images = num_images

    def forward(self, *args):
        n = self.num_images
        images = list(args[:n])
        img_masks = list(args[n : 2 * n])
        lang_tokens = args[2 * n]
        lang_masks = args[2 * n + 1]

        embs, pad_masks, att_masks = self.pi0.model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )

        try:
            import torch_xla.core.xla_model as xm
            xm.mark_step()
        except ImportError:
            pass

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        return att_2d_masks.to(torch.float32)


# ---------------------------------------------------------------------------
# Test 1: embed_prefix alone (expected to PASS)
# ---------------------------------------------------------------------------
@pytest.mark.single_device
def test_embed_prefix_sanity(pi0_model_and_inputs):
    """embed_prefix with real model inputs — should compile without hanging.

    Loads the full PI0 model (LIBERO_BASE), preprocesses a real dataset
    frame, and runs embed_prefix (SigLIP + language embedding + concat).
    Verifies correct output shape and numerical match between CPU and
    TT device.
    """
    model, images, img_masks, lang_tokens, lang_masks = pi0_model_and_inputs

    wrapper = EmbedPrefixOp(model, num_images=len(images))
    inputs = [*images, *img_masks, lang_tokens, lang_masks]

    cpu_out = wrapper(*inputs)
    assert cpu_out.ndim == 3, f"Expected 3D [B, N, D] output, got {cpu_out.ndim}D"
    assert cpu_out.shape[0] == 1, f"Expected batch size 1, got {cpu_out.shape[0]}"

    run_op_test(
        wrapper, inputs, framework=Framework.TORCH
    )


# ---------------------------------------------------------------------------
# Test 2: embed_prefix + make_att_2d_masks (expected to reproduce hang)
# ---------------------------------------------------------------------------
@pytest.mark.single_device
def test_embed_prefix_plus_make_att_2d_masks_sanity(pi0_model_and_inputs):
    """embed_prefix + make_att_2d_masks combined — reproduces the hang.

    This test composes embed_prefix and make_att_2d_masks in a single
    XLA graph, which is the exact scenario that causes the StableHLO →
    TTIR compilation to hang.  The hang is resolved by inserting
    xm.mark_step() between the two (see model.py workaround).

    If this test passes, the compiler hang has been fixed upstream.
    If it hangs, it confirms the issue still exists and the graph-break
    workaround is still necessary.
    """
    model, images, img_masks, lang_tokens, lang_masks = pi0_model_and_inputs

    wrapper = EmbedPrefixPlusMakeAtt2dMasksOp(model, num_images=len(images))
    inputs = [*images, *img_masks, lang_tokens, lang_masks]

    cpu_out = wrapper(*inputs)
    seq_len = cpu_out.shape[1]
    assert cpu_out.shape == (1, seq_len, seq_len), (
        f"Expected [1, {seq_len}, {seq_len}], got {cpu_out.shape}"
    )

    run_op_test(
        wrapper, inputs, framework=Framework.TORCH
    )


# ---------------------------------------------------------------------------
# Test 3: embed_prefix + make_att_2d_masks WITH graph break (expected to PASS)
# ---------------------------------------------------------------------------
@pytest.mark.single_device
def test_embed_prefix_plus_make_att_2d_masks_with_graph_break_sanity(
    pi0_model_and_inputs,
):
    """embed_prefix + make_att_2d_masks with xm.mark_step() — should PASS.

    Identical to test 2 except an xm.mark_step() graph break is inserted
    between embed_prefix and make_att_2d_masks.  This forces two separate
    XLA compilations that each succeed, proving the hang only occurs when
    the two subgraphs are compiled together as one.
    """
    model, images, img_masks, lang_tokens, lang_masks = pi0_model_and_inputs

    wrapper = EmbedPrefixPlusMakeAtt2dMasksWithGraphBreakOp(
        model, num_images=len(images)
    )
    inputs = [*images, *img_masks, lang_tokens, lang_masks]

    cpu_out = wrapper(*inputs)
    seq_len = cpu_out.shape[1]
    assert cpu_out.shape == (1, seq_len, seq_len), (
        f"Expected [1, {seq_len}, {seq_len}], got {cpu_out.shape}"
    )

    run_op_test(
        wrapper, inputs, framework=Framework.TORCH
    )
