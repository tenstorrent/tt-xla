# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""DiffusionGemma 26B -- e2e PCC: CPU-driven block-diffusion pipeline, TT-verified (no decode).

A single host-side driver runs generation, reusing the model's own helpers (sampler,
stopping, cache, RNG) so it is bit-identical to generate(). Only the two NN components run
on TT, each PCC-checked against CPU on the same on-trajectory inputs:
  * encoder prefill (model.model.encoder)
  * decoder forward (model.forward, input_ids=None -> decoder + lm_head)

The driver and TT wrappers are shared with the demo -- imported from the tt_forge_models
pipeline (``manual_generate`` / ``_TTEncoder`` / ``_TTDecoder`` / device helpers); this file
adds only the per-step PCC staging. The 26B model runs sharded (SPMD) via the loader's
get_mesh_config / load_shard_spec (MLP col->row + MoE experts; attention replicated).
"""

import inspect

import pytest
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from infra import RunMode
from infra.evaluators import PccConfig, TorchComparisonEvaluator
from infra.evaluators.evaluation_config import ComparisonConfig
from infra.utilities.torch_multichip_utils import enable_spmd, get_mesh
from loguru import logger
from tt_torch.moe_backend import TT_MOE_BACKEND_NAME, register_tt_moe_backend
from utils import BringupStatus, Category, ModelGroup

from tests.runner.requirements import RequirementsManager
from third_party.tt_forge_models.diffusiongemma.pytorch import ModelLoader
from third_party.tt_forge_models.diffusiongemma.pytorch import (
    loader as diffgemma_loader,
)
from third_party.tt_forge_models.diffusiongemma.pytorch.loader import ModelVariant
from third_party.tt_forge_models.diffusiongemma.pytorch.pipeline import (
    TTDecoder as _TTDecoder,
)
from third_party.tt_forge_models.diffusiongemma.pytorch.pipeline import (
    TTEncoder as _TTEncoder,
)
from third_party.tt_forge_models.diffusiongemma.pytorch.pipeline import (
    cache_to_device as _cache_to_device,
)
from third_party.tt_forge_models.diffusiongemma.pytorch.pipeline import (
    free_tt_graphs as _free_tt_graphs,
)
from third_party.tt_forge_models.diffusiongemma.pytorch.pipeline import manual_generate
from third_party.tt_forge_models.diffusiongemma.pytorch.pipeline import (
    to_device as _to_device,
)

MAX_NEW_TOKENS = 256
SEED = 0
PCC_THRESHOLD = 0.95


_PCC_EVALUATOR = TorchComparisonEvaluator(ComparisonConfig(assert_on_failure=False))
_PCC_CONFIG = PccConfig()


def _pcc(device_out, golden_out) -> float:
    return float(_PCC_EVALUATOR._compare_pcc(device_out, golden_out, _PCC_CONFIG))


def _make_staged_forwards(cpu_model, mesh, pcc_records):
    """Staged both-on-TT: only ONE component resident on device at a time. encoder_forward loads
    the ENCODER variant (independent), prefills + PCC, frees it, then loads the decoder for the
    decode loop; the KV cache round-trips through host across the swap.
    pcc_records entries are (component, block, step, pcc)."""
    from transformers import DynamicCache  # 5.12.0, after the requirements swap

    xla = xm.xla_device()
    tt_pkv = {
        "pkv": None,
        "host": None,
    }  # "host": CPU cache handed from encoder to decoder
    ctr = {"block": -1, "step": 0}
    vocab_size = cpu_model.config.text_config.vocab_size
    stage = {
        "dec_tt": None,
        "dec_model": None,
    }  # decoder graph, resident during the decode loop

    def _load_sharded(variant):
        # Consumer applies the loader's spec: .to(device) + mark_sharding.
        loader = ModelLoader(variant)
        model = loader.load_model(dtype_override=torch.bfloat16)
        loader.config._experts_implementation = TT_MOE_BACKEND_NAME  # experts backend
        model = model.to(xla)
        xs.set_global_mesh(mesh)  # tt_moe reads get_global_mesh() for the EP axis
        for tensor, spec in loader.load_shard_spec(model).items():
            xs.mark_sharding(tensor, mesh, spec)
        return model

    def encoder_forward(**kw):
        ctr["block"] += 1
        ctr["step"] = 0
        # Free the previous block's decoder (if any) so only one model is device-resident.
        if stage["dec_tt"] is not None:
            stage["dec_tt"] = stage["dec_model"] = None
            _free_tt_graphs()
        # Load the encoder as an independent model, shard, compile; then CPU golden + TT prefill.
        enc_model = _load_sharded(ModelVariant.ENCODER)
        enc_tt = torch.compile(_TTEncoder(enc_model), backend="tt")
        cpu_out = cpu_model.model.encoder(**kw)
        pkv = DynamicCache()
        tt_lhs = enc_tt(
            _to_device(kw["input_ids"], xla),
            _to_device(kw["attention_mask"], xla),
            _to_device(kw["position_ids"], xla),
            pkv,
            _to_device(kw.get("mm_token_type_ids"), xla),
        )
        pcc = _pcc(_to_device(tt_lhs, "cpu"), cpu_out.last_hidden_state)
        pcc_records.append(("encoder", ctr["block"], 0, pcc))
        logger.info("[PCC] block={} encoder: pcc={:.6f}", ctr["block"], pcc)
        assert (
            pcc >= PCC_THRESHOLD
        ), f"encoder(block {ctr['block']}) PCC {pcc:.6f} < {PCC_THRESHOLD}"
        # Cache to host + FREE the encoder. The decoder is loaded lazily in decoder_forward,
        # so only one model is ever resident on device.
        xm.mark_step()
        tt_pkv["host"] = _cache_to_device(pkv, "cpu")
        del enc_tt, enc_model, pkv
        _free_tt_graphs()
        return cpu_out

    def decoder_forward(**kw):
        ctr["step"] += 1
        # First decode step: encoder is freed, so load the decoder now, vocab-shard lm_head/embed
        # (so decoder + logits fit), and restore the KV cache from host.
        if stage["dec_tt"] is None:
            dec_model = _load_sharded(ModelVariant.DIFFUSIONGEMMA_26B_A4B_IT)
            xs.mark_sharding(dec_model.lm_head.weight, mesh, ("model", None))
            stage["dec_model"] = dec_model
            stage["dec_tt"] = torch.compile(_TTDecoder(dec_model), backend="tt")
            tt_pkv["pkv"] = _cache_to_device(tt_pkv["host"], xla)
        # Consistent self-conditioning -> one TT graph across all steps (iter 1 logits=None ->
        # zeros + mask=False == the None branch; later -> real + mask=True).
        bs, canvas = kw["decoder_input_ids"].shape
        if kw.get("self_conditioning_logits") is None:
            kw["self_conditioning_logits"] = torch.zeros(
                bs, canvas, vocab_size, dtype=cpu_model.dtype
            )
            scm = torch.zeros(bs, dtype=torch.bool)
        else:
            scm = torch.ones(bs, dtype=torch.bool)
        kw["self_conditioning_mask"] = scm
        cpu_out = cpu_model.forward(**kw)  # no input_ids -> decoder path
        tt_logits = stage["dec_tt"](
            _to_device(kw["decoder_input_ids"], xla),
            _to_device(kw["decoder_position_ids"], xla),
            _to_device(kw["self_conditioning_logits"], xla),
            _to_device(kw["decoder_attention_mask"], xla),
            _to_device(scm, xla),
            tt_pkv["pkv"],
        )
        pcc = _pcc(_to_device(tt_logits, "cpu"), cpu_out.logits)
        pcc_records.append(("decoder", ctr["block"], ctr["step"], pcc))
        logger.info(
            "[PCC] block={} step={} decoder: pcc={:.6f}", ctr["block"], ctr["step"], pcc
        )
        assert (
            pcc >= PCC_THRESHOLD
        ), f"decoder(block {ctr['block']} step {ctr['step']}) PCC {pcc:.6f} < {PCC_THRESHOLD}"
        return cpu_out

    return encoder_forward, decoder_forward


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.large
@pytest.mark.llmbox
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name="DiffusionGemma_e2e",
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
def test_diffusiongemma_e2e():
    """CPU-driven block-diffusion pipeline; every NN component verified against TT (no decode)."""
    # transformers>=5.11 is required for DiffusionGemma; install it from the loader's
    # requirements.txt for this test only, roll back on exit (env stays clean for others).
    loader_path = inspect.getsourcefile(diffgemma_loader)
    with RequirementsManager.for_loader(loader_path, framework="torch"):
        xr.set_device_type("TT")
        enable_spmd()  # SPMD (Shardy) mode for the sharded 26B model
        # Re-register tt_moe: the transformers swap dropped its import-time registration.
        # See https://github.com/tenstorrent/tt-xla/pull/5424
        register_tt_moe_backend()

        # CPU model: drives the pipeline + golden for PCC (full model, host).
        cpu_loader = ModelLoader()
        cpu_model = cpu_loader.load_model(dtype_override=torch.bfloat16)
        cpu_model.eval()
        cpu_model.config._experts_implementation = (
            TT_MOE_BACKEND_NAME  # matches runner's inject_custom_moe
        )
        inputs = cpu_loader.load_inputs(dtype_override=torch.bfloat16)
        # generate()'s extra inputs (e.g. mm_token_type_ids), minus decoder_input_ids (loop inits its canvas).
        extra_kwargs = {
            k: v
            for k, v in inputs.items()
            if k not in ("input_ids", "attention_mask", "decoder_input_ids")
        }

        mesh_shape, mesh_names = cpu_loader.get_mesh_config(
            xr.global_runtime_device_count()
        )
        mesh = get_mesh(mesh_shape, mesh_names)

        # Staged both-on-TT: the encoder (ENCODER variant, independent) runs + is freed, then the
        # decoder is loaded -- only one component resident on device at a time (see loader variants).
        pcc_records = []
        encoder_forward, decoder_forward = _make_staged_forwards(
            cpu_model, mesh, pcc_records
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
        logger.info(
            "per-iteration PCC: {} checks, worst={:.6f}", len(pcc_records), worst
        )
        assert worst >= PCC_THRESHOLD
