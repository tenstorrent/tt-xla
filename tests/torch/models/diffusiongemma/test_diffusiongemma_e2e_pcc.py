# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""DiffusionGemma 26B -- e2e PCC: CPU-driven block-diffusion pipeline, TT-verified (no decode).

A single host-side driver runs generation, reusing the model's own helpers (sampler,
stopping, cache, RNG) so it is bit-identical to generate(). Only the two NN components run
on TT, each PCC-checked against CPU on the same on-trajectory inputs:
  * encoder prefill (model.model.encoder)
  * decoder forward (model.forward, input_ids=None -> decoder + lm_head)

The 26B model runs sharded (SPMD) via the loader's get_mesh_config / load_shard_spec
(MLP col->row + MoE experts; attention replicated).
"""

import math

import pytest
import torch
from transformers import DynamicCache
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from infra import RunMode
from infra.evaluators import PccConfig, TorchComparisonEvaluator
from infra.evaluators.evaluation_config import ComparisonConfig
from infra.utilities.torch_multichip_utils import enable_spmd, get_mesh
from loguru import logger
from tt_torch.moe_backend import TT_MOE_BACKEND_NAME  # importing registers the tt_moe backend
from utils import BringupStatus, Category, ModelGroup

from third_party.tt_forge_models.diffusiongemma.pytorch import ModelLoader

MAX_NEW_TOKENS = 512
SEED = 0
PCC_THRESHOLD = 0.90  # single-forward CPU-vs-TT bf16 gap is ~0.96-0.99


_PCC_EVALUATOR = TorchComparisonEvaluator(ComparisonConfig(assert_on_failure=False))
_PCC_CONFIG = PccConfig()


def _pcc(device_out, golden_out) -> float:
    return float(_PCC_EVALUATOR._compare_pcc(device_out, golden_out, _PCC_CONFIG))


def _to_device(obj, device):
    """Recursively move tensors in a tensor / dict / list / tuple to ``device``."""
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device(v, device) for v in obj)
    return obj


class _TTEncoder(torch.nn.Module):
    """torch.compile needs tensor I/O: returns last_hidden_state (cache updated in place)."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask, position_ids, past_key_values, mm_token_type_ids=None):
        return self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            mm_token_type_ids=mm_token_type_ids,
        ).last_hidden_state


class _TTDecoder(torch.nn.Module):
    """torch.compile needs tensor I/O: hardcodes input_ids=None (decoder path), returns logits."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, decoder_input_ids, decoder_position_ids, self_conditioning_logits, decoder_attention_mask, past_key_values):
        return self.model(
            input_ids=None,
            decoder_input_ids=decoder_input_ids,
            self_conditioning_logits=self_conditioning_logits,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            decoder_position_ids=decoder_position_ids,
        ).logits


@torch.no_grad()
def manual_generate(
    model,
    input_ids,
    attention_mask,
    max_new_tokens,
    *,
    encoder_forward=None,
    decoder_forward=None,
    **model_kwargs,
):
    """With-cache mimic of generate() (verified bit-identical). Only the encoder prefill and
    decoder forward are swappable (default to the model's own) for TT injection."""
    encoder_forward = encoder_forward or model.model.encoder
    decoder_forward = decoder_forward or model.forward

    gen_cfg, model_kwargs = model._prepare_generation_config(
        None, max_new_tokens=max_new_tokens, **model_kwargs
    )
    batch_size, cur_len = input_ids.shape
    max_length, max_new_tokens = model._prepare_generated_length(gen_cfg, cur_len)
    max_new_canvases = math.ceil(max_new_tokens / model.config.canvas_length)

    device = input_ids.device
    canvas_length = model.config.canvas_length
    finished_sequences = torch.zeros(batch_size, dtype=torch.bool, device=device)
    past_key_values = model._prepare_cache_for_generation(
        generation_config=gen_cfg, batch_size=batch_size, max_length=max_length - canvas_length
    )
    eos_tensor = (
        torch.tensor(gen_cfg.eos_token_id, device=device)
        if gen_cfg.eos_token_id is not None
        else None
    )
    encoder_position_ids = torch.arange(
        cur_len - input_ids.shape[1], cur_len, dtype=torch.int32, device=device
    ).unsqueeze(0)
    decoder_position_ids = torch.arange(
        cur_len, cur_len + canvas_length, dtype=torch.int32, device=device
    ).unsqueeze(0)
    decoder_attention_mask = torch.nn.functional.pad(
        attention_mask, (0, canvas_length), value=True
    )

    sampler = model._prepare_sampler(gen_cfg)
    logits_processor = model._prepare_logits_processor(gen_cfg, None)
    ar_stopping = model._prepare_ar_stopping_criteria(gen_cfg, None)
    diffusion_stopping = model._prepare_diffusion_stopping_criteria(gen_cfg)

    is_prefill = True
    for block in range(max_new_canvases):
        unprocessed_input_ids, encoder_mask_mapping = model._prepare_encoder_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_position_ids=encoder_position_ids,
            past_key_values=past_key_values,
            is_prefill=is_prefill,
            canvas_length=canvas_length,
            batch_size=batch_size,
            **model_kwargs,
        )
        encoder_outputs = encoder_forward(
            input_ids=unprocessed_input_ids,
            attention_mask=encoder_mask_mapping,
            past_key_values=past_key_values,
            position_ids=encoder_position_ids,
            **model_kwargs,
        )
        past_key_values = encoder_outputs.past_key_values
        is_prefill = False

        current_canvas, self_conditioning_logits, mask_mapping, finished_denoising = (
            model._prepare_denoiser_inputs(
                decoder_attention_mask=decoder_attention_mask,
                past_key_values=past_key_values,
                sampler=sampler,
                diffusion_stopping_criteria=diffusion_stopping,
                batch_size=batch_size,
                device=device,
                model_kwargs=model_kwargs,
            )
        )
        argmax_canvas = current_canvas

        for cur_step in reversed(range(1, gen_cfg.max_denoising_steps + 1)):
            current_canvas, argmax_canvas, self_conditioning_logits, finished_denoising = (
                model._denoising_step(
                    decoder_forward=decoder_forward,
                    current_canvas=current_canvas,
                    argmax_canvas=argmax_canvas,
                    input_ids=input_ids,
                    decoder_position_ids=decoder_position_ids,
                    self_conditioning_logits=self_conditioning_logits,
                    mask_mapping=mask_mapping,
                    past_key_values=past_key_values,
                    finished_denoising=finished_denoising,
                    cur_step=cur_step,
                    sampler=sampler,
                    logits_processor=logits_processor,
                    diffusion_stopping_criteria=diffusion_stopping,
                    **model_kwargs,
                )
            )
            if torch.all(finished_denoising):
                break

        logger.info("block {}/{} done", block + 1, max_new_canvases)
        input_ids = torch.cat([input_ids, argmax_canvas], dim=-1)
        input_ids, finished_sequences = model._finalize_canvas(
            input_ids=input_ids,
            finished_sequences=finished_sequences,
            generation_config=gen_cfg,
            stopping_criteria=ar_stopping,
            canvas_length=canvas_length,
            eos_tensor=eos_tensor,
        )
        if torch.all(finished_sequences):
            break
        (
            cur_len,
            decoder_attention_mask,
            attention_mask,
            encoder_position_ids,
            decoder_position_ids,
        ) = model._prepare_kwargs_for_next_canvas(
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
            past_key_values=past_key_values,
            canvas_length=canvas_length,
            cur_len=cur_len,
            is_compiling=False,
        )

    return input_ids


def _make_checked_forwards(cpu_model, tt_encoder, tt_decoder, pcc_records):
    """Encoder/decoder forwards that run CPU + TT, check PCC, and return the CPU output.

    Called by the driver every iteration (encoder once/block, decoder once/step); PCC is
    labelled (block, step). A separate TT cache lineage (tt_pkv) is prefilled by the encoder
    and read by the decoder. pcc_records entries are (component, block, step, pcc).
    """
    xla = xm.xla_device()
    tt_pkv = {"pkv": None}
    ctr = {"block": -1, "step": 0}

    def encoder_forward(**kw):
        ctr["block"] += 1
        ctr["step"] = 0
        cpu_out = cpu_model.model.encoder(**kw)
        if tt_pkv["pkv"] is None:
            tt_pkv["pkv"] = DynamicCache()
        tt_lhs = tt_encoder(
            _to_device(kw["input_ids"], xla),
            _to_device(kw["attention_mask"], xla),
            _to_device(kw["position_ids"], xla),
            tt_pkv["pkv"],
            _to_device(kw.get("mm_token_type_ids"), xla),
        )
        pcc = _pcc(_to_device(tt_lhs, "cpu"), cpu_out.last_hidden_state)
        pcc_records.append(("encoder", ctr["block"], 0, pcc))
        logger.info("[PCC] block={} encoder: pcc={:.6f}", ctr["block"], pcc)
        assert pcc >= PCC_THRESHOLD, f"encoder(block {ctr['block']}) PCC {pcc:.6f} < {PCC_THRESHOLD}"
        return cpu_out

    def decoder_forward(**kw):
        ctr["step"] += 1
        cpu_out = cpu_model.forward(**kw)  # no input_ids -> decoder path
        tt_logits = tt_decoder(
            _to_device(kw["decoder_input_ids"], xla),
            _to_device(kw["decoder_position_ids"], xla),
            _to_device(kw["self_conditioning_logits"], xla),
            _to_device(kw["decoder_attention_mask"], xla),
            tt_pkv["pkv"],
        )
        pcc = _pcc(_to_device(tt_logits, "cpu"), cpu_out.logits)
        pcc_records.append(("decoder", ctr["block"], ctr["step"], pcc))
        logger.info("[PCC] block={} step={} decoder: pcc={:.6f}", ctr["block"], ctr["step"], pcc)
        assert pcc >= PCC_THRESHOLD, (
            f"decoder(block {ctr['block']} step {ctr['step']}) PCC {pcc:.6f} < {PCC_THRESHOLD}"
        )
        return cpu_out

    return encoder_forward, decoder_forward


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.large
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name="DiffusionGemma_e2e_pcc",
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.FAILED_RUNTIME,
)
def test_diffusiongemma_e2e_pcc():
    """CPU-driven block-diffusion pipeline; every NN component verified against TT (no decode)."""
    xr.set_device_type("TT")
    enable_spmd()  # SPMD (Shardy) mode for the sharded 26B model

    loader = ModelLoader()

    # CPU model: drives the pipeline + golden for PCC.
    cpu_model = loader.load_model(dtype_override=torch.bfloat16)
    cpu_model.eval()
    cpu_model.config._experts_implementation = TT_MOE_BACKEND_NAME  # matches runner's inject_custom_moe
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)
    # generate()'s extra inputs (e.g. mm_token_type_ids), minus decoder_input_ids (loop inits its canvas).
    extra_kwargs = {
        k: v
        for k, v in inputs.items()
        if k not in ("input_ids", "attention_mask", "decoder_input_ids")
    }

    # TT model: NN forwards run sharded on device.
    tt_model = loader.load_model(dtype_override=torch.bfloat16)
    tt_model.eval()
    tt_model.config._experts_implementation = TT_MOE_BACKEND_NAME
    tt_model = tt_model.to(xm.xla_device())
    mesh_shape, mesh_names = loader.get_mesh_config(xr.global_runtime_device_count())
    mesh = get_mesh(mesh_shape, mesh_names)
    xs.set_global_mesh(mesh)  # tt_moe reads get_global_mesh() for the expert-parallel axis
    for tensor, spec in loader.load_shard_spec(tt_model).items():
        xs.mark_sharding(tensor, mesh, spec)

    # encoder_tt: encoder prefill graph. model_tt: decoder path (input_ids=None prunes the
    # encoder branch -> decoder + lm_head only, so the encoder is compiled once, not twice).
    encoder_tt = torch.compile(_TTEncoder(tt_model.model.encoder), backend="tt")
    model_tt = torch.compile(_TTDecoder(tt_model), backend="tt")

    pcc_records = []
    encoder_forward, decoder_forward = _make_checked_forwards(
        cpu_model, encoder_tt, model_tt, pcc_records
    )
    torch.manual_seed(SEED)
    manual_generate(
        cpu_model,
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=MAX_NEW_TOKENS,
        encoder_forward=encoder_forward,
        decoder_forward=decoder_forward,
        **extra_kwargs,
    )

    worst = min((p for *_, p in pcc_records), default=1.0)
    logger.info("per-iteration PCC: {} checks, worst={:.6f}", len(pcc_records), worst)
    assert worst >= PCC_THRESHOLD
