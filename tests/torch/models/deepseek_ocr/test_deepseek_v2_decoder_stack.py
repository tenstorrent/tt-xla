# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Sanity: isolate DeepSeek-OCR ``DeepseekV2Model`` decoder loop PCC (12 × ``DeepseekV2DecoderLayer`` + RMS norm).

Per bisection logs under ``deepseek_ocr_logs/``:
  - before_deepseekv2 / before_decoder_layers: PCC ≈ 1.0
  - before_lmhead (decoder + norm): PCC ≈ 0.95 vs required 0.99
  - whole_model (decoder + norm + lm_head): PCC ≈ 0.94

This test replicates the loop in ``modeling_deepseekv2.py`` (≈1581–1614) with real
``inputs_embeds`` / masks from the same OCR forward path as ``test_models.py``.

Run:

  cd /proj_sw/user_dev/ctr-akannan/20_may_yyz/tt-xla
  export DEEPSEEK_OCR_SKIP_SNAPSHOT_SYNC=1
  pytest tests/torch/models/deepseek_ocr/test_deepseek_v2_decoder_stack.py -v -s \\
    2>&1 | tee deepseek_ocr_logs/decoder_layers_sanity.log
"""

from __future__ import annotations

import inspect
import os
from contextlib import contextmanager
from typing import Iterator, Optional, Tuple

import pytest
import torch
import torch_xla.runtime as xr
from infra import Framework, run_op_test
from infra.evaluators.evaluation_config import ComparisonConfig, PccConfig
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from utils import Category

import third_party.tt_forge_models.deepseek.deepseek_ocr.pytorch.loader as deepseek_ocr_loader
from tests.runner.requirements import RequirementsManager
from third_party.tt_forge_models.deepseek.deepseek_ocr.pytorch.loader import ModelLoader

_LOADER_PATH = inspect.getsourcefile(deepseek_ocr_loader)
_DTYPE = torch.bfloat16
_REQUIRED_PCC = 0.99


class DeepseekV2DecoderStack(torch.nn.Module):
    """Mirrors ``DeepseekV2Model.forward`` decoder loop + final RMS norm."""

    def __init__(
        self,
        layers: torch.nn.ModuleList,
        norm: torch.nn.Module,
        num_layers: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.layers = layers
        self.norm = norm
        self.num_layers = num_layers if num_layers is not None else len(layers)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        output_attentions = False
        use_cache = False
        past_key_values = None

        for layer_idx, decoder_layer in enumerate(self.layers):
            if layer_idx >= self.num_layers:
                break
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs[0]

        return self.norm(hidden_states)


class DeepseekV2DecoderStackWithLmHead(torch.nn.Module):
    """Decoder stack + ``lm_head`` (whole-model logits path, ``DeepseekOCRForCausalLM``)."""

    def __init__(
        self,
        layers: torch.nn.ModuleList,
        norm: torch.nn.Module,
        lm_head: torch.nn.Module,
    ) -> None:
        super().__init__()
        self.decoder = DeepseekV2DecoderStack(layers, norm)
        self.lm_head = lm_head

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.decoder(hidden_states, attention_mask, position_ids)
        return self.lm_head(hidden_states).float()


def _deepseek_v2_model_class(module: torch.nn.Module) -> type:
    for cls in module.__class__.__mro__:
        if cls.__name__ == "DeepseekV2Model":
            return cls
    raise RuntimeError(
        f"DeepseekV2Model not found in MRO of {module.__class__.__name__}"
    )


@contextmanager
def _capture_at_deepseek_v2_forward(
    inner: torch.nn.Module,
) -> Iterator[dict]:
    """
    Wrap ``DeepseekV2Model.forward`` so we capture ``inputs_embeds`` when
    ``DeepseekOCRModel`` calls ``super().forward(...)`` (after vision + masked_scatter).

    A pre-hook on ``inner`` alone is wrong: it runs at ``DeepseekOCRModel.forward`` entry
    while ``inputs_embeds`` is still None.
    """
    captured: dict = {}
    v2_cls = _deepseek_v2_model_class(inner)
    original_forward = v2_cls.forward

    def wrapped_forward(module_self, *args, **kwargs):
        if module_self is inner:
            ie = kwargs.get("inputs_embeds")
            if ie is not None:
                captured["inputs_embeds"] = ie
        return original_forward(module_self, *args, **kwargs)

    v2_cls.forward = wrapped_forward
    try:
        yield captured
    finally:
        v2_cls.forward = original_forward


def _build_decoder_attention_inputs(
    ocr_inner: torch.nn.Module,
    inputs_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Same mask/position_ids path as ``DeepseekV2Model.forward`` (non-FA2)."""
    batch_size, seq_length = inputs_embeds.shape[:2]
    device = inputs_embeds.device

    position_ids = torch.arange(
        0,
        seq_length,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    if getattr(ocr_inner, "_use_flash_attention_2", False):
        mask = (
            attention_mask
            if (attention_mask is not None and 0 in attention_mask)
            else None
        )
        if mask is None:
            raise RuntimeError(
                "Flash-attn path not supported in this sanity test; expected 4D causal mask."
            )
        return mask, position_ids

    mask_4d = _prepare_4d_causal_attention_mask(
        attention_mask,
        (batch_size, seq_length),
        inputs_embeds,
        past_key_values_length=0,
    )
    return mask_4d, position_ids


def _run_ocr_forward_like_model_test(
    model: torch.nn.Module,
    inputs: dict,
) -> None:
    """Same call signature as ``DeepseekOCRForCausalLM.forward`` in model tests."""
    model(
        input_ids=inputs["input_ids"],
        attention_mask=None,
        images=inputs["images"],
        images_seq_mask=inputs["images_seq_mask"],
        images_spatial_crop=inputs["images_spatial_crop"],
        use_cache=False,
        return_dict=False,
    )


def _capture_inputs_embeds(model: torch.nn.Module, inputs: dict) -> torch.Tensor:
    inner = model.model
    with _capture_at_deepseek_v2_forward(inner) as captured:
        with torch.no_grad():
            _run_ocr_forward_like_model_test(model, inputs)

    if captured.get("inputs_embeds") is None:
        raise RuntimeError(
            "Failed to capture inputs_embeds at DeepseekV2Model.forward entry. "
            "If DeepseekOCRModel.forward returns before super(), use a stop after "
            "masked_scatter instead of a hook on the inner module."
        )
    return captured["inputs_embeds"]


def _build_decoder_stack_args(
    model: torch.nn.Module,
    inputs: dict,
) -> Tuple[DeepseekV2DecoderStack, tuple, dict]:
    inner = model.model
    inputs_embeds = _capture_inputs_embeds(model, inputs)
    attention_mask, position_ids = _build_decoder_attention_inputs(
        inner, inputs_embeds, attention_mask=None
    )

    stack = DeepseekV2DecoderStack(inner.layers, inner.norm).to(_DTYPE)
    stack.eval()

    args = (
        inputs_embeds.to(_DTYPE),
        attention_mask.to(_DTYPE) if attention_mask is not None else attention_mask,
        position_ids,
    )
    meta = {
        "batch_size": inputs_embeds.shape[0],
        "seq_len": inputs_embeds.shape[1],
        "hidden_size": inputs_embeds.shape[2],
        "mask_shape": tuple(attention_mask.shape) if attention_mask is not None else None,
        "position_ids_shape": tuple(position_ids.shape),
        "dtype": str(inputs_embeds.dtype),
    }
    return stack, args, meta


def _comparison_config() -> ComparisonConfig:
    config = ComparisonConfig()
    config.disable_all()
    config.pcc.enable()
    config.pcc.required_pcc = _REQUIRED_PCC
    return config


@pytest.fixture(scope="module")
def deepseek_ocr_session() -> Tuple[torch.nn.Module, dict]:
    """Keep transformers 4.46.3 for the whole module (matches model test in-process)."""
    os.environ.setdefault("DEEPSEEK_OCR_SKIP_SNAPSHOT_SYNC", "1")
    with RequirementsManager.for_loader(_LOADER_PATH):
        loader = ModelLoader()
        model = loader.load_model(dtype_override=_DTYPE)
        inputs = loader.load_inputs(dtype_override=_DTYPE)
        model.eval()
        yield model, inputs


@pytest.fixture(scope="module")
def ocr_decoder_real_inputs(
    deepseek_ocr_session: Tuple[torch.nn.Module, dict],
) -> Tuple[DeepseekV2DecoderStack, tuple, dict, torch.nn.Module]:
    model, inputs = deepseek_ocr_session
    stack, args, meta = _build_decoder_stack_args(model, inputs)
    return stack, args, meta, model


@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_deepseek_ocr_decoder_stack_real_inputs(
    ocr_decoder_real_inputs: Tuple[DeepseekV2DecoderStack, tuple, dict, torch.nn.Module],
) -> None:
    """
    Full 12-layer decoder + norm on real ``inputs_embeds`` (after vision + masked_scatter).

    Expect PCC ≈ ``before_lmhead`` (~0.95) and fail at 0.99 — same decoder pathology as
    whole-model, without vision or lm_head.
    """
    xr.set_device_type("TT")
    stack, args, meta, _model = ocr_decoder_real_inputs
    print(
        f"[deepseek_ocr_decoder_stack] real inputs: batch={meta['batch_size']} "
        f"seq_len={meta['seq_len']} hidden={meta['hidden_size']} "
        f"mask={meta['mask_shape']} dtype={meta['dtype']}",
        flush=True,
    )
    run_op_test(
        stack,
        list(args),
        comparison_config=_comparison_config(),
        framework=Framework.TORCH,
    )


@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_deepseek_ocr_decoder_stack_with_lm_head(
    ocr_decoder_real_inputs: Tuple[DeepseekV2DecoderStack, tuple, dict, torch.nn.Module],
) -> None:
    """
    Decoder + ``lm_head`` on real inputs — reproduces whole-model logits PCC (~0.94 at 0.99).
    """
    xr.set_device_type("TT")
    _stack, args, meta, model = ocr_decoder_real_inputs
    inner = model.model
    wrapper = DeepseekV2DecoderStackWithLmHead(
        inner.layers, inner.norm, model.lm_head
    ).to(_DTYPE)
    wrapper.eval()
    print(
        f"[deepseek_ocr_decoder_stack+lm_head] seq_len={meta['seq_len']} "
        f"vocab via lm_head out=({meta['batch_size']}, {meta['seq_len']}, vocab)",
        flush=True,
    )
    run_op_test(
        wrapper,
        list(args),
        comparison_config=_comparison_config(),
        framework=Framework.TORCH,
    )


@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_deepseek_ocr_decoder_stack_random_inputs(
    ocr_decoder_real_inputs: Tuple[DeepseekV2DecoderStack, tuple, dict, torch.nn.Module],
) -> None:
    """Same shapes/dtype as OCR decoder path; random Gaussian activations."""
    xr.set_device_type("TT")
    stack, _args, meta, _model = ocr_decoder_real_inputs
    batch, seq_len, hidden = meta["batch_size"], meta["seq_len"], meta["hidden_size"]

    gen = torch.Generator().manual_seed(0)
    hidden_states = torch.randn(batch, seq_len, hidden, dtype=_DTYPE, generator=gen)
    attention_mask = torch.zeros(batch, 1, seq_len, seq_len, dtype=_DTYPE)
    attention_mask.masked_fill_(
        ~torch.ones(seq_len, seq_len, dtype=torch.bool).tril(), float("-inf")
    )
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

    print(
        f"[deepseek_ocr_decoder_stack] random inputs: ({batch}, {seq_len}, {hidden}) "
        f"mask=({batch}, 1, {seq_len}, {seq_len}) dtype={_DTYPE}",
        flush=True,
    )
    run_op_test(
        stack,
        [hidden_states, attention_mask, position_ids],
        comparison_config=_comparison_config(),
        framework=Framework.TORCH,
    )


@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
@pytest.mark.parametrize("num_layers", [1, 2, 4, 6, 8, 10, 12])
def test_deepseek_ocr_decoder_stack_layer_accumulation(
    ocr_decoder_real_inputs: Tuple[DeepseekV2DecoderStack, tuple, dict, torch.nn.Module],
    num_layers: int,
) -> None:
    """Real ``inputs_embeds``; first ``num_layers`` blocks + norm (layer-wise accumulation)."""
    xr.set_device_type("TT")
    base_stack, args, meta, _model = ocr_decoder_real_inputs
    stack = DeepseekV2DecoderStack(
        base_stack.layers, base_stack.norm, num_layers=num_layers
    ).to(_DTYPE)
    stack.eval()

    config = _comparison_config()
    config.assert_on_failure = False

    print(
        f"[deepseek_ocr_decoder_stack] num_layers={num_layers}/{len(base_stack.layers)} "
        f"seq_len={meta['seq_len']}",
        flush=True,
    )
    run_op_test(
        stack,
        list(args),
        comparison_config=config,
        framework=Framework.TORCH,
    )
