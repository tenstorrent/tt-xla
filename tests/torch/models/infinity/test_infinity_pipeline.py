# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Infinity 2B — nightly e2e pipeline test.

Infinity is an autoregressive next-scale-prediction text-to-image model (not a
diffusion model): a generation is a Python loop over a fixed scale schedule -- a
transformer forward + multinomial sampling + BSQ-VAE code accumulation per scale,
then a single VAE decode. This reimplements the model's ``autoregressive_infer_cfg``
with an explicit CPU/TT device split:

  - Transformer on Tenstorrent, 8-way tensor-parallel sharded (mesh (1, 8),
    Megatron head-parallel attention from ``loader.load_shard_spec``).
  - T5-XL text encoder, multinomial sampling and BSQ-VAE decode stay on CPU.

Two correctness-critical choices:
  - SEQUENTIAL classifier-free guidance -- cond and uncond are run as two batch-1
    forwards per scale and combined on the logits. A batch-2 (stacked) forward
    makes the attention score matmul all-gather the heads (de-shard) and OOM at
    the last 1M scale; batch-1 keeps the score head-sharded.
  - fp32 LayerNorm -- every LayerNorm is computed via an explicit mean/var/rsqrt
    decomposition in fp32 (``_force_fp32_layernorm``). The bf16 fused
    ttnn.layer_norm loses precision on the mid/late layers' outlier activations,
    which autoregressive sampling amplifies into a noise image; the decomposition
    restores per-block PCC to ~1.0 and yields a coherent image. (A plain
    ``F.layer_norm(x.float())`` does not help -- it folds back to bf16.)
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from infra import RunMode
from infra.utilities.torch_multichip_utils import enable_spmd, get_mesh
from loguru import logger
from PIL import Image
from utils import BringupStatus, Category, ModelGroup

from third_party.tt_forge_models.config import Parallelism
from third_party.tt_forge_models.infinity.pytorch import ModelLoader, ModelVariant
from third_party.tt_forge_models.infinity.pytorch.src import model as _m

PROMPT = "A fantasy landscape with mountains and rivers"
SEED = 42
# Resolution preset: "1M" -> 1024x1024 (the model's native target);
# "0.25M" -> 512x512; "0.06M" -> 256x256. Output size is derived from the preset.
PN = "1M"
H_DIV_W = 1.0
HEIGHT, WIDTH = _m.dynamic_resolution_h_w[H_DIV_W][PN]["pixel"]
# Transformer weight dtype on TT (bf16 fits 1M in DRAM).
DTYPE = torch.bfloat16


def _force_fp32_layernorm(model):
    """Compute every nn.LayerNorm in fp32 via an explicit mean/var/rsqrt
    decomposition (NOT F.layer_norm, which folds back to a bf16 ttnn.layer_norm on
    TT). The fused bf16 LayerNorm loses precision on the mid/late layers' outlier
    activations -- the dominant TT accuracy loss for this model -- which the
    autoregressive sampling amplifies into a noise image. Normalizes over the last
    dim (normalized_shape is (C,) for every block here)."""
    for mod in model.modules():
        if isinstance(mod, nn.LayerNorm):

            def _fwd(x, m=mod):
                xf = x.float()
                mu = xf.mean(-1, keepdim=True)
                var = (xf - mu).pow(2).mean(-1, keepdim=True)
                y = (xf - mu) * torch.rsqrt(var + m.eps)
                if m.weight is not None:
                    y = y * m.weight.float()
                if m.bias is not None:
                    y = y + m.bias.float()
                return y.to(x.dtype)

            mod.forward = _fwd


class InfinityConfig:
    def __init__(
        self,
        cfg: float = 3.0,
        tau: float = 0.5,
        top_k: int = 900,
        top_p: float = 0.97,
        pn: str = PN,
        h_div_w: float = H_DIV_W,
        max_scales: Optional[int] = None,
        shard: bool = True,
        transformer_on_tt: bool = True,
    ):
        self.cfg = cfg
        self.tau = tau
        self.top_k = top_k
        self.top_p = top_p
        self.pn = pn
        self.h_div_w = h_div_w
        self.width = WIDTH
        self.height = HEIGHT
        # Scales (transformer passes) to run; None runs the full schedule. A
        # smaller value still decodes a full-resolution (coarse) image, since
        # every scale's codes are accumulated at the final resolution.
        self.max_scales = max_scales
        # 8-way Megatron tensor-parallel sharding (needed so the large-scale
        # attention does not OOM).
        self.shard = shard
        self.transformer_on_tt = transformer_on_tt


class InfinityPipeline:
    """Infinity 2B pipeline: transformer sharded on TT, sampling + VAE on CPU."""

    def __init__(self, config: InfinityConfig):
        self.config = config

    def setup(self):
        self.load_models()
        if self.config.transformer_on_tt:
            if self.config.shard:
                self.shard_to_tt()
            else:
                self.model = self.model.to(xm.xla_device())
        self.scale_schedule = self._build_scale_schedule()

    def load_models(self):
        # Loading the transformer side-loads the T5-XL tokenizer/encoder and the
        # BSQ-VAE onto the loader; both stay on CPU.
        self.loader = ModelLoader(ModelVariant.INFINITY_2B)
        self.model = self.loader.load_model(dtype_override=DTYPE)
        _force_fp32_layernorm(self.model)
        self.tokenizer = self.loader.tokenizer
        self.text_encoder = self.loader.text_encoder
        self.vae = self.loader.vae
        self.model_dtype = self.model.pos_start.dtype

    def shard_to_tt(self):
        # Enable SPMD, build the (1, 8) mesh, move the transformer to the XLA
        # device, then mark every weight in the Megatron shard spec.
        enable_spmd()
        mesh_shape, mesh_names = self.loader.get_mesh_config(
            xr.global_runtime_device_count()
        )
        self.mesh = get_mesh(mesh_shape, mesh_names)
        self.model = self.model.to(xm.xla_device())
        for tensor, spec in self.loader.load_shard_spec(self.model).items():
            xs.mark_sharding(tensor, self.mesh, spec)

    def _build_scale_schedule(self):
        sched = _m.dynamic_resolution_h_w[self.config.h_div_w][self.config.pn]["scales"]
        return [(1, h, w) for (_, h, w) in sched]

    @torch.no_grad()
    def generate(self, prompt: str, seed: Optional[int] = SEED) -> torch.Tensor:
        """Reimplements ``Infinity.autoregressive_infer_cfg`` with a CPU/TT split.

        - T5-XL text encode -> CPU
        - transformer conditioning, blocks, logits, word_embed -> TT
        - multinomial sampling -> CPU
        - BSQ-VAE indices->codes, residual accumulation, decode -> CPU
        """
        m = self.model
        vae = self.vae
        on_tt = self.config.transformer_on_tt

        # CPU <-> TT casts (no-ops when the transformer runs on CPU).
        tt_cast = lambda x: x.to(device=xm.xla_device()) if on_tt else x
        cpu_cast = lambda x: x.to("cpu") if on_tt else x

        scale_schedule = self.scale_schedule
        num_stages_minus_1 = len(scale_schedule) - 1
        tau_list = [self.config.tau] * len(scale_schedule)
        cfg_list = [self.config.cfg] * len(scale_schedule)
        B = 1

        # ── T5-XL text encode (CPU) ───────────────────────────────────
        logger.info("[STAGE] T5 text encode (CPU): start")
        kv_compact, lens, cu_seqlens_k, max_seqlen_k = _m.encode_prompt(
            self.tokenizer, self.text_encoder, prompt
        )
        logger.info("[STAGE] T5 text encode (CPU): done")

        # Seed the model's (CPU) multinomial sampling generator.
        if seed is not None:
            m.rng.manual_seed(seed)
            rng = m.rng
        else:
            rng = None

        # ── Classifier-free guidance: sequential cond + uncond passes ──
        # Two batch-1 forwards per scale, combined on the logits. A batch-2
        # (stacked) forward de-shards the attention score matmul -> OOM at 1M;
        # batch-1 keeps it head-sharded. cfg=1 -> single conditional pass.
        use_cfg = self.config.cfg != 1
        kv_branches = [kv_compact]
        if use_cfg:
            cfg_uncond = m.cfg_uncond.detach().to("cpu", dtype=kv_compact.dtype)
            kv_uncond = kv_compact.clone()
            total = 0
            for le in lens:
                kv_uncond[total : total + le] = cfg_uncond[:le]
                total += le
            kv_branches.append(kv_uncond)

        # ── Per-branch text conditioning projections (TT, batch=1) ─────
        cu_seqlens_k = tt_cast(cu_seqlens_k)

        def _conditioning(kv_raw):
            kv = m.text_norm(tt_cast(kv_raw.to(self.model_dtype)))
            sos = cond_BD = m.text_proj_for_sos((kv, cu_seqlens_k, max_seqlen_k))
            ca_kv = (m.text_proj_for_ca(kv), cu_seqlens_k, max_seqlen_k)
            sos_tok = sos.unsqueeze(1).expand(B, 1, -1) + m.pos_start.expand(B, 1, -1)
            with torch.amp.autocast("cuda", enabled=False):
                # bf16 throughout (no .float()): an f32 input to the bf16
                # shared_ada_lin Linear yields a mismatched-dtype dot that fails
                # HLO->MHLO conversion on TT.
                gss = m.shared_ada_lin(cond_BD).contiguous()
            return {"ca_kv": ca_kv, "cond_BD": cond_BD, "gss": gss, "sos": sos_tok}

        branches = [_conditioning(kv) for kv in kv_branches]

        # ── Next-scale prediction loop (packed recompute, stays sharded) ──
        # No KV cache: each scale rebuilds the full token sequence generated so far
        # and runs all blocks over it in ONE batch-1 forward per CFG branch, with a
        # block-causal attn_bias (each scale attends to itself + earlier scales).
        # The carried state is RAW token embeddings re-projected through the sharded
        # q/k/v weights every scale, so the attention score stays head-sharded. (A
        # KV cache instead de-shards: cached K/V cross the CPU sampling boundary
        # replicated and feed SDPA directly -> 16 heads on one device -> OOM.)
        n_run = len(scale_schedule)
        if self.config.max_scales is not None:
            n_run = min(self.config.max_scales, n_run)

        def _build_attn_bias(sched, ref):
            # Block-causal mask over the packed sequence: a position at scale i
            # attends to all positions at scales <= i. Shape [1, 1, L, L]. rope
            # uses the FULL schedule (its precomputed grid is keyed by the full
            # tuple) with scale_ind=0, slicing positions [0:L_cum].
            l_end = sum(int(np.prod(s)) for s in sched)
            d = torch.cat(
                [torch.full((int(np.prod(s)),), i) for i, s in enumerate(sched)]
            ).view(1, l_end, 1)
            bias = torch.where(d >= d.transpose(1, 2), 0.0, -torch.inf)
            return bias.reshape(1, 1, l_end, l_end).type_as(ref).to(ref.device)

        def _run_blocks(x_BLC, br, sub_sched, attn_bias):
            x_BLC = m.add_lvl_embeding_for_x_BLC(x_BLC, sub_sched)
            for b in m.block_chunks:
                for blk in b.module:
                    x_BLC = blk(
                        x=x_BLC,
                        cond_BD=br["gss"],
                        ca_kv=br["ca_kv"],
                        attn_bias_or_two_vector=attn_bias,
                        attn_fn=None,
                        scale_schedule=scale_schedule,
                        rope2d_freqs_grid=m.rope2d_freqs_grid,
                        scale_ind=0,
                    )
            return x_BLC

        # Shared per-scale RAW token inputs (scales 1..si-1); each branch prepends
        # its own SOS (conditioning-derived, so it differs per branch).
        shared_inputs = []
        summed_codes = 0
        for si, pn in enumerate(scale_schedule):
            if si >= n_run:
                break
            is_last_run = si == n_run - 1
            cfg = cfg_list[si]
            logger.info(f"[STEP] scale {si + 1}/{n_run} {tuple(pn)}")

            sub_sched = scale_schedule[: si + 1]
            L_si = int(np.prod(pn))
            attn_bias = None

            # --- one batch-1 forward per CFG branch (TT, sharded) -> logits (CPU) ---
            branch_logits = []
            for br in branches:
                x_BLC = torch.cat([br["sos"], *shared_inputs], dim=1)
                if attn_bias is None:
                    attn_bias = _build_attn_bias(sub_sched, x_BLC)
                x_BLC = _run_blocks(x_BLC, br, sub_sched, attn_bias)
                hidden_si = x_BLC[:, -L_si:]
                logits = m.get_logits(hidden_si, br["cond_BD"]).mul(1 / tau_list[si])
                branch_logits.append(cpu_cast(logits).float())

            # CFG combine on logits: cfg*cond + (1-cfg)*uncond.
            if use_cfg:
                logits_BlV = cfg * branch_logits[0] + (1 - cfg) * branch_logits[1]
            else:
                logits_BlV = branch_logits[0]

            # Bit-label codebook: every code is a sequence of binary bits.
            tmp_bs, tmp_seq_len = logits_BlV.shape[:2]
            logits_BlV = logits_BlV.reshape(tmp_bs, -1, 2)
            idx_Bld = _m.sample_with_top_k_top_p_also_inplace_modifying_logits_(
                logits_BlV,
                rng=rng,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                num_samples=1,
            )[:, :, 0]
            idx_Bld = idx_Bld.reshape(tmp_bs, tmp_seq_len, -1)

            # --- BSQ-VAE: indices -> codes, accumulate residual (CPU) ---
            assert pn[0] == 1
            idx_Bld = idx_Bld.reshape(B, pn[1], pn[2], -1).unsqueeze(1)  # (B,1,h,w,d)
            codes = vae.quantizer.lfq.indices_to_codes(idx_Bld, label_type="bit_label")
            if si != num_stages_minus_1:
                # Add this scale's contribution (always at the final resolution).
                summed_codes = summed_codes + F.interpolate(
                    codes, size=scale_schedule[-1], mode=vae.quantizer.z_interplote_up
                )
                # On the last executed scale there is no next pass to feed.
                if is_last_run:
                    break
                # Build the next scale's shared RAW input embedding and append it.
                next_stage = F.interpolate(
                    summed_codes,
                    size=scale_schedule[si + 1],
                    mode=vae.quantizer.z_interplote_up,
                )
                next_stage = next_stage.squeeze(-3)
                next_stage = next_stage.reshape(*next_stage.shape[:2], -1)
                next_stage = torch.permute(next_stage, [0, 2, 1])  # (B, L_next, d_vae)
                next_stage = tt_cast(next_stage.to(self.model_dtype))
                shared_inputs.append(m.word_embed(m.norm0_ve(next_stage)))
            else:
                summed_codes = summed_codes + codes

        # ── BSQ-VAE decode (CPU) -> RGB image in [-1, 1] ───────────────
        logger.info("[STAGE] BSQ-VAE decode (CPU): start")
        summed_codes = summed_codes.to("cpu")
        img = vae.decode(summed_codes.squeeze(-3))
        logger.info("[STAGE] BSQ-VAE decode (CPU): done")
        return img


def save_image(image: torch.Tensor, filepath: str = "output.png"):
    image = (
        (torch.clamp(image / 2 + 0.5, 0.0, 1.0) * 255.0).round().to(dtype=torch.uint8)
    )
    image_np = image.cpu().squeeze().numpy()
    assert image_np.ndim == 3, "Image must be 3D"
    if image_np.shape[0] == 3:
        image_np = image_np.transpose(1, 2, 0)
    Image.fromarray(image_np).save(filepath)


def run_infinity_pipeline(output_path: str = "infinity_2b_output.png"):
    """Run the full Infinity 2B pipeline (transformer 8-way TP on TT) and save."""
    pipeline = InfinityPipeline(config=InfinityConfig())
    pipeline.setup()
    img = pipeline.generate(prompt=PROMPT, seed=SEED)
    save_image(img, output_path)
    return output_path


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.llmbox
@pytest.mark.tensor_parallel
@pytest.mark.large
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name="Infinity_2B_Pipeline",
    model_group=ModelGroup.GENERALITY,
    parallelism=Parallelism.TENSOR_PARALLEL,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
def test_infinity_2b_pipeline():
    """Run the full Infinity 2B pipeline (transformer 8-way TP on TT, rest on CPU)."""
    xr.set_device_type("TT")

    output_path = "infinity_2b_output.png"
    output_file = Path(output_path)
    if output_file.exists():
        output_file.unlink()

    run_infinity_pipeline(output_path=output_path)

    assert output_file.exists(), f"Output image {output_path} was not created"

    with Image.open(output_path) as img:
        width, height = img.size
        assert width == WIDTH, f"Expected width {WIDTH}, got {width}"
        assert height == HEIGHT, f"Expected height {HEIGHT}, got {height}"

    logger.info(f"Output image saved to {output_path} ({width}x{height})")
