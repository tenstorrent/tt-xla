# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Incremental layer-0 ``input_layernorm`` → ``LlamaAttention`` bisection tests.

Each stage adds the next attention op on top of layernorm in **one compiled TT graph**.

- ``test_*_model_faithful_*``: same graph as full model (LN + native ``self_attn``).
- ``test_*_bisect_profile_*``: decomposed stages to find which op class fails when fused
  after layernorm (stages 02–08 are **not** whole-model graphs).

Example::

    pytest -m "n150 and single_device" -s \\
      tests/torch/models/janus_pro_pcc_drop/test_image_token_decoder_layer0_ln_attn_bisect.py::test_image_token_decode_layer0_ln_attn_bisect_profile_pro_1b \\
      2>&1 | tee janus_logs_2/layer0_ln_attn_bisect.log
"""

from __future__ import annotations

import inspect

import pytest
import torch
import torch_xla.runtime as xr
from tests.runner.requirements import RequirementsManager
from tests.torch.models.janus_pro_pcc_drop.decoder_ln_attn_bisect import (
    LAYER0_LN_ATTN_BISECT_STAGES,
    LAYER0_LN_ATTN_BISECT_STAGE_IDS,
    make_layer0_ln_attn_bisect_wrapper,
    print_layer0_ln_attn_bisect_profile,
    print_layer0_ln_attn_model_faithful_profile,
)
from tests.torch.models.janus_pro_pcc_drop.decoder_op_test_utils import run_decoder_op_test
from tests.torch.models.janus_pro_pcc_drop.decoder_sanity import load_image_token_decode_bundle

import third_party.tt_forge_models.janus_pro.text_to_image.pytorch.loader as janus_loader
from third_party.tt_forge_models.janus_pro.text_to_image.pytorch import (
    ModelLoader,
    ModelVariant,
)


def _load_decode_bundle(variant_name: str) -> dict:
    loader_path = inspect.getsourcefile(janus_loader)
    with RequirementsManager.for_loader(loader_path, framework="torch"):
        from third_party.tt_forge_models.janus_pro.text_to_image.pytorch.src import (
            model_utils,
        )

        torch.manual_seed(42)
        model_utils._mmgpt_cache.clear()
        repo_id = ModelLoader(ModelVariant(variant_name))._repo_id()
        xr.set_device_type("TT")
        return load_image_token_decode_bundle(repo_id, dtype=torch.bfloat16)


@pytest.mark.model_test
@pytest.mark.single_device
@pytest.mark.n150
@pytest.mark.p150
def test_image_token_decode_layer0_ln_attn_model_faithful_profile_pro_1b():
    """
    Canonical repro: isolated LN + ``JanusLlamaDecoderLayer0LnAttnProfile`` stacked graph.

    ``fused_self_attn`` row should be ~**0.77** (not bisect single-output ~0.99).
    """
    bundle = _load_decode_bundle("Pro_1B")
    print_layer0_ln_attn_model_faithful_profile(
        bundle["llama_model"],
        bundle["inputs_embeds"],
        bundle["past_key_values"],
    )


@pytest.mark.model_test
@pytest.mark.single_device
@pytest.mark.n150
@pytest.mark.p150
def test_image_token_decode_layer0_ln_attn_bisect_profile_pro_1b():
    """
    Full bisect ladder including decomposed attention stages (02–08).

    Decomposed stages use different TT graphs than the whole model — for localization
    only. See ``test_*_model_faithful_*`` for e2e LN+attn graph.
    """
    bundle = _load_decode_bundle("Pro_1B")
    print_layer0_ln_attn_bisect_profile(
        bundle["llama_model"],
        bundle["inputs_embeds"],
        bundle["past_key_values"],
        label="Pro_1B decode layer0 ln→attn bisect",
    )


@pytest.mark.model_test
@pytest.mark.single_device
@pytest.mark.n150
@pytest.mark.p150
@pytest.mark.parametrize(
    "stop_after",
    LAYER0_LN_ATTN_BISECT_STAGE_IDS,
    ids=[stage.label for stage in LAYER0_LN_ATTN_BISECT_STAGES],
)
def test_image_token_decode_layer0_ln_attn_bisect_stage_pro_1b(stop_after: str):
    """Single incremental stage (parametrized); asserts PCC for that graph only."""
    bundle = _load_decode_bundle("Pro_1B")
    stage = next(s for s in LAYER0_LN_ATTN_BISECT_STAGES if s.stage_id == stop_after)
    wrapper = make_layer0_ln_attn_bisect_wrapper(
        bundle["llama_model"],
        bundle["past_key_values"],
        stop_after,
    )
    run_decoder_op_test(
        stage.label,
        wrapper,
        [bundle["inputs_embeds"]],
    )


@pytest.mark.model_test
@pytest.mark.single_device
@pytest.mark.n150
@pytest.mark.p150
def test_image_token_decode_layer0_ln_attn_bisect_layernorm_only_pro_1b():
    """Shortcut: layernorm-only stage (baseline — should pass)."""
    bundle = _load_decode_bundle("Pro_1B")
    wrapper = make_layer0_ln_attn_bisect_wrapper(
        bundle["llama_model"],
        bundle["past_key_values"],
        "layernorm",
    )
    run_decoder_op_test(
        "01_layernorm",
        wrapper,
        [bundle["inputs_embeds"]],
    )


@pytest.mark.model_test
@pytest.mark.single_device
@pytest.mark.n150
@pytest.mark.p150
def test_image_token_decode_layer0_ln_attn_bisect_native_self_attn_pro_1b():
    """Shortcut: LN + native ``self_attn`` (reproduces ~0.77 fused drop)."""
    bundle = _load_decode_bundle("Pro_1B")
    wrapper = make_layer0_ln_attn_bisect_wrapper(
        bundle["llama_model"],
        bundle["past_key_values"],
        "native_self_attn",
    )
    run_decoder_op_test(
        "09_ln_native_self_attn",
        wrapper,
        [bundle["inputs_embeds"]],
    )


@pytest.mark.model_test
@pytest.mark.single_device
@pytest.mark.n150
@pytest.mark.p150
def test_image_token_decode_decoder_layer0_ln_attn_serialize_pro_1b(request):
    """
    Canonical ~0.77 fused LN+attn graph — dump MLIR/TTNN with ``pytest --serialize``.

    Artifacts: ``output_artifact/<test_name>/`` (see docs/src/test_infra.md).
    """
    from infra import Framework, run_op_test
    from infra.evaluators import ComparisonConfig
    from tests.torch.models.janus_pro_pcc_drop.decoder_submodule_sanity import (
        JanusLlamaDecoderLayer0LnAttnProfile,
    )

    bundle = _load_decode_bundle("Pro_1B")
    wrapper = JanusLlamaDecoderLayer0LnAttnProfile(bundle["llama_model"])
    wrapper.past_key_values = bundle["past_key_values"]
    run_op_test(
        wrapper,
        [bundle["inputs_embeds"]],
        framework=Framework.TORCH,
        comparison_config=ComparisonConfig(assert_on_failure=False),
        request=request,
    )
