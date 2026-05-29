# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Isolated and combined ``LlamaModel`` submodule op tests on ImageTokenStep decode tensors.

Set ``JANUS_DECODER_PRINT_METRICS=1`` to print PCC, max/mean abs diff, and rel L2 after each test.

Isolated tests use exact captured inputs. Combined tests chain submodules or native layer loops
to locate interaction / depth PCC drops (issue #4968).

Layer-0 ``input_layernorm`` + ``self_attn`` fused tests: loader ``inputs_embeds`` vs saved
``hidden_before_input_layernorm`` (see ``test_*_ln_attn_*``).
"""

from __future__ import annotations

import inspect

import pytest
import torch
import torch_xla.runtime as xr
from tests.runner.requirements import RequirementsManager
from tests.torch.models.janus_pro_pcc_drop.decoder_op_test_utils import (
    run_decoder_op_test,
    run_decoder_stacked_stage_profile_op_test,
)
from tests.torch.models.janus_pro_pcc_drop.decoder_sanity import (
    JanusLlamaDecoderLayersLoop,
    JanusLlamaDecoderNativeLayerDecode,
    load_image_token_decode_bundle,
)
from tests.torch.models.janus_pro_pcc_drop.decoder_submodule_sanity import (
    DEFAULT_DECODE_LAYER_IDX,
    DecoderSubmoduleFixtures,
    JanusLlamaDecoderLayer0CombinedProfile,
    JanusLlamaDecoderLayer0LnAttnProfile,
    LAYER0_COMBINED_STAGE_NAMES,
    LAYER0_LN_ATTN_STAGE_NAMES_FROM_EMBEDS,
    LAYER0_LN_ATTN_STAGE_NAMES_FROM_HIDDEN,
    describe_llama_self_attn,
    print_cpu_layer0_ln_attn_reference,
    JanusLlamaMLPDecode,
    JanusLlamaRMSNormDecode,
    JanusLlamaRotaryEmbDecode,
    load_decoder_submodule_fixtures,
    make_layer0_self_attn_native_wrapper,
)
from tests.torch.models.janus_pro_pcc_drop.layer0_tensor_capture import load_saved_layer0_tensors

import third_party.tt_forge_models.janus_pro.text_to_image.pytorch.loader as janus_loader
from third_party.tt_forge_models.janus_pro.text_to_image.pytorch import (
    ModelLoader,
    ModelVariant,
)


def _load_fixtures(variant_name: str) -> DecoderSubmoduleFixtures:
    loader_path = inspect.getsourcefile(janus_loader)
    with RequirementsManager.for_loader(loader_path, framework="torch"):
        from third_party.tt_forge_models.janus_pro.text_to_image.pytorch.src import (
            model_utils,
        )

        torch.manual_seed(42)
        model_utils._mmgpt_cache.clear()

        repo_id = ModelLoader(ModelVariant(variant_name))._repo_id()
        xr.set_device_type("TT")
        return load_decoder_submodule_fixtures(
            repo_id, dtype=torch.bfloat16, layer_idx=DEFAULT_DECODE_LAYER_IDX
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


# --- Isolated submodule tests ---


@pytest.mark.model_test
@pytest.mark.single_device
@pytest.mark.n150
@pytest.mark.p150
def test_image_token_decode_rotary_emb_pro_1b():
    fx = _load_fixtures("Pro_1B")
    run_decoder_op_test(
        "rotary_emb",
        JanusLlamaRotaryEmbDecode(fx.rotary_emb),
        [fx.rotary_hidden_states, fx.rotary_position_ids],
    )


@pytest.mark.model_test
@pytest.mark.single_device
@pytest.mark.n150
@pytest.mark.p150
def test_image_token_decode_input_layernorm_layer0_pro_1b():
    fx = _load_fixtures("Pro_1B")
    run_decoder_op_test(
        "input_layernorm",
        JanusLlamaRMSNormDecode(fx.input_layernorm),
        [fx.input_layernorm_hidden],
    )


@pytest.mark.model_test
@pytest.mark.single_device
@pytest.mark.n150
@pytest.mark.p150
def test_image_token_decode_attention_layer0_pro_1b():
    """Real hidden after input_layernorm (capture path), native ``self_attn`` — not random."""
    bundle = _load_decode_bundle("Pro_1B")
    wrapper = make_layer0_self_attn_native_wrapper(bundle["llama_model"], bundle)
    fx = _load_fixtures("Pro_1B")
    run_decoder_op_test(
        "attention_captured_hidden",
        wrapper,
        [fx.attn_hidden_states],
    )


@pytest.mark.model_test
@pytest.mark.single_device
@pytest.mark.n150
@pytest.mark.p150
def test_image_token_decode_attention_layer0_saved_hidden_pro_1b():
    """``self_attn`` with hidden from ``save_layer0`` (``hidden_after_input_layernorm_layer0.pt``)."""
    bundle = _load_decode_bundle("Pro_1B")
    wrapper = make_layer0_self_attn_native_wrapper(bundle["llama_model"], bundle)
    saved = load_saved_layer0_tensors()
    hidden = saved["hidden_after_input_layernorm"].to(dtype=torch.bfloat16)
    run_decoder_op_test(
        "attention_saved_hidden",
        wrapper,
        [hidden],
    )


@pytest.mark.model_test
@pytest.mark.single_device
@pytest.mark.n150
@pytest.mark.p150
def test_image_token_decode_attention_layer0_random_hidden_pro_1b():
    """``self_attn`` with random hidden (same shape); decode KV/mask/RoPE still real."""
    bundle = _load_decode_bundle("Pro_1B")
    wrapper = make_layer0_self_attn_native_wrapper(bundle["llama_model"], bundle)
    torch.manual_seed(99)
    hidden = torch.randn(2, 1, 2048, dtype=torch.bfloat16) * 0.02
    run_decoder_op_test(
        "attention_random_hidden",
        wrapper,
        [hidden],
    )


@pytest.mark.model_test
@pytest.mark.single_device
@pytest.mark.n150
@pytest.mark.p150
def test_image_token_decode_post_attention_layernorm_layer0_pro_1b():
    fx = _load_fixtures("Pro_1B")
    run_decoder_op_test(
        "post_attention_layernorm",
        JanusLlamaRMSNormDecode(fx.post_attention_layernorm),
        [fx.post_attention_layernorm_hidden],
    )


@pytest.mark.model_test
@pytest.mark.single_device
@pytest.mark.n150
@pytest.mark.p150
def test_image_token_decode_mlp_layer0_pro_1b():
    fx = _load_fixtures("Pro_1B")
    run_decoder_op_test(
        "mlp",
        JanusLlamaMLPDecode(fx.mlp),
        [fx.mlp_hidden],
    )


@pytest.mark.model_test
@pytest.mark.single_device
@pytest.mark.n150
@pytest.mark.p150
def test_image_token_decode_final_norm_pro_1b():
    fx = _load_fixtures("Pro_1B")
    run_decoder_op_test(
        "final_norm",
        JanusLlamaRMSNormDecode(fx.final_norm),
        [fx.final_norm_hidden],
    )


# --- Combined tests (submodule chain + native layer loops) ---


def _print_llama_self_attn_forward_info(self_attn: torch.nn.Module) -> None:
    info = describe_llama_self_attn(self_attn)
    print("\n--- LlamaAttention forward (Janus layer-0 decode) ---")
    print(f"  class:              {info['class']}")
    print(f"  forward:            {info['forward_qualname']}")
    print(f"  source file:        {info['forward_file']}")
    print(f"  attn_implementation: {info['attn_implementation']!r}")
    print(
        f"  shapes:             hidden={info['hidden_size']} "
        f"heads={info['num_heads']} kv_heads={info['num_kv_heads']} "
        f"head_dim={info['head_dim']}"
    )
    print(f"  submodules:         {', '.join(info['submodules'])}")
    print("  forward stages:")
    for stage in info["forward_stages"]:
        print(f"    - {stage}")
    print("---\n")


def _run_layer0_ln_attn_profile(
    label: str,
    bundle: dict,
    *,
    entry: str,
    forward_input: torch.Tensor,
    stage_names: tuple[str, ...],
    assert_on_failure: bool = True,
) -> None:
    wrapper = JanusLlamaDecoderLayer0LnAttnProfile(
        bundle["llama_model"], entry=entry
    )
    wrapper.past_key_values = bundle["past_key_values"]
    if entry == "hidden":
        wrapper.inputs_embeds_for_decode_context = bundle["inputs_embeds"]
    run_decoder_stacked_stage_profile_op_test(
        label,
        wrapper,
        [forward_input],
        stage_names,
        assert_on_failure=assert_on_failure,
    )


@pytest.mark.model_test
@pytest.mark.single_device
@pytest.mark.n150
@pytest.mark.p150
def test_image_token_decode_decoder_layer0_input_layernorm_self_attn_loader_pro_1b():
    """Fused ``input_layernorm`` + ``self_attn``; entry = loader ``inputs_embeds`` (decode bundle)."""
    bundle = _load_decode_bundle("Pro_1B")
    _run_layer0_ln_attn_profile(
        "decoder_layer0_ln_attn_loader",
        bundle,
        entry="embeds",
        forward_input=bundle["inputs_embeds"],
        stage_names=LAYER0_LN_ATTN_STAGE_NAMES_FROM_EMBEDS,
    )


@pytest.mark.model_test
@pytest.mark.single_device
@pytest.mark.n150
@pytest.mark.p150
def test_image_token_decode_decoder_layer0_input_layernorm_self_attn_saved_pro_1b():
    """Same fused graph; entry = saved ``hidden_before_input_layernorm_layer0.pt``."""
    bundle = _load_decode_bundle("Pro_1B")
    saved = load_saved_layer0_tensors()
    hidden_before = saved["hidden_before_input_layernorm"].to(dtype=torch.bfloat16)
    _run_layer0_ln_attn_profile(
        "decoder_layer0_ln_attn_saved",
        bundle,
        entry="hidden",
        forward_input=hidden_before,
        stage_names=LAYER0_LN_ATTN_STAGE_NAMES_FROM_HIDDEN,
    )


def _run_layer0_ln_attn_detailed_profile(
    label: str,
    bundle: dict,
    *,
    entry: str,
    forward_input: torch.Tensor,
    stage_names: tuple[str, ...],
) -> None:
    llama_model = bundle["llama_model"]
    _print_llama_self_attn_forward_info(llama_model.layers[0].self_attn)
    print_cpu_layer0_ln_attn_reference(
        llama_model,
        bundle["inputs_embeds"],
        bundle["past_key_values"],
        label="CPU reference (q/k/v/RoPE/KV/eager/o_proj — not TT traced)",
    )
    _run_layer0_ln_attn_profile(
        label,
        bundle,
        entry=entry,
        forward_input=forward_input,
        stage_names=stage_names,
        assert_on_failure=False,
    )


@pytest.mark.model_test
@pytest.mark.single_device
@pytest.mark.n150
@pytest.mark.p150
def test_image_token_decode_decoder_layer0_input_layernorm_self_attn_detailed_loader_pro_1b():
    """
    (1) CPU reference: internal ``LlamaAttention`` stages (q/k/v, RoPE, KV, eager, o_proj).
    (2) TT op test: same fused graph as full model — ``input_layernorm`` + native ``self_attn``
    (reproduces ~0.77 ``self_attn`` PCC). Internal attn ops are not split in the TT graph.
    """
    bundle = _load_decode_bundle("Pro_1B")
    _run_layer0_ln_attn_detailed_profile(
        "decoder_layer0_ln_attn_detailed_loader",
        bundle,
        entry="embeds",
        forward_input=bundle["inputs_embeds"],
        stage_names=LAYER0_LN_ATTN_STAGE_NAMES_FROM_EMBEDS,
    )


@pytest.mark.model_test
@pytest.mark.single_device
@pytest.mark.n150
@pytest.mark.p150
def test_image_token_decode_decoder_layer0_submodules_combined_pro_1b():
    """Layer 0 combined chain; PCC/diff printed after each submodule stage."""
    bundle = _load_decode_bundle("Pro_1B")
    wrapper = JanusLlamaDecoderLayer0CombinedProfile(bundle["llama_model"])
    wrapper.past_key_values = bundle["past_key_values"]
    run_decoder_stacked_stage_profile_op_test(
        "decoder_layer0_submodules_combined",
        wrapper,
        [bundle["inputs_embeds"]],
        LAYER0_COMBINED_STAGE_NAMES,
    )


@pytest.mark.model_test
@pytest.mark.single_device
@pytest.mark.n150
@pytest.mark.p150
def test_image_token_decode_decoder_layer0_native_pro_1b():
    """Layer 0 via native ``LlamaDecoderLayer`` (same path as full ``LlamaModel`` decode)."""
    bundle = _load_decode_bundle("Pro_1B")
    wrapper = JanusLlamaDecoderNativeLayerDecode(bundle["llama_model"])
    wrapper.past_key_values = bundle["past_key_values"]
    run_decoder_op_test(
        "decoder_layer0_native",
        wrapper,
        [bundle["inputs_embeds"]],
    )


@pytest.mark.model_test
@pytest.mark.single_device
@pytest.mark.n150
@pytest.mark.p150
def test_image_token_decode_decoder_layers_5_native_pro_1b():
    """First 5 decoder layers (native loop), pre-norm output."""
    bundle = _load_decode_bundle("Pro_1B")
    wrapper = JanusLlamaDecoderLayersLoop(bundle["llama_model"], num_layers=5)
    wrapper.past_key_values = bundle["past_key_values"]
    run_decoder_op_test(
        "decoder_layers_5_native",
        wrapper,
        [bundle["inputs_embeds"]],
    )


@pytest.mark.model_test
@pytest.mark.single_device
@pytest.mark.n150
@pytest.mark.p150
def test_image_token_decode_decoder_full_with_norm_pro_1b():
    """All decoder layers + ``norm`` (matches failing full decoder op test)."""
    bundle = _load_decode_bundle("Pro_1B")
    wrapper = JanusLlamaDecoderLayersLoop(bundle["llama_model"], num_layers=None)
    wrapper.past_key_values = bundle["past_key_values"]
    run_decoder_op_test(
        "decoder_full_with_norm",
        wrapper,
        [bundle["inputs_embeds"]],
        assert_on_failure=False,
    )
