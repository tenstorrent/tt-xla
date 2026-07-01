# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Infinity 2B — benchmark-side pipeline for the imagegen harness.

Infinity is an autoregressive *next-scale-prediction* text-to-image model, not a
diffusion model: a single generation is a Python loop over a fixed scale schedule
(a transformer forward -> multinomial sampling -> BSQ-VAE code accumulation per
scale), followed by a single VAE decode. There are no per-step diffusion
components, so this pipeline reimplements the model's ``autoregressive_infer_cfg``
loop here (just as the SDXL pipeline reimplements its denoising loop) with an
explicit CPU/TT device split:

  - Transformer runs on Tenstorrent, 8-way tensor-parallel sharded (mesh (1, 8),
    Megatron-style head-parallel attention from ``loader.load_shard_spec``).
  - T5-XL text encoder, multinomial sampling, and BSQ-VAE decode stay on CPU.

Two correctness-critical choices (kept in sync with the e2e test in
tests/torch/models/infinity/test_infinity_pipeline.py):
  - SEQUENTIAL classifier-free guidance -- cond and uncond are run as two batch-1
    forwards per scale and combined on the logits. A batch-2 (stacked) forward
    makes the attention score matmul all-gather the heads (de-shard) and OOM at
    the last 1M scale.
  - fp32 LayerNorm -- every LayerNorm is computed via an explicit mean/var/rsqrt
    decomposition in fp32 (``_force_fp32_layernorm``); the bf16 fused
    ttnn.layer_norm loses precision on the mid/late layers' outlier activations,
    which autoregressive sampling amplifies into a noise image.

Per-stage forward+sync times are collected into ``self._perf`` for the harness to
read after each ``generate()`` call.
"""

import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from infra.utilities.torch_multichip_utils import enable_spmd, get_mesh
from loguru import logger

from third_party.tt_forge_models.infinity.pytorch import ModelLoader, ModelVariant
from third_party.tt_forge_models.infinity.pytorch.src import model as _m

# Resolution preset: "1M" -> 1024x1024 (the model's native target). Output size
# is derived from the preset.
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
        transformer_on_tt: bool = True,
        compile_options: Optional[dict] = None,
        cfg: float = 3.0,
        tau: float = 0.5,
        top_k: int = 900,
        top_p: float = 0.97,
        pn: str = PN,
        h_div_w: float = H_DIV_W,
    ):
        self.transformer_on_tt = transformer_on_tt
        # Harness-set compile options (kept for parity with the SDXL pipeline;
        # Infinity does not switch opt levels inline).
        self.compile_options = compile_options or {}
        self.cfg = cfg
        self.tau = tau
        self.top_k = top_k
        self.top_p = top_p
        self.pn = pn
        self.h_div_w = h_div_w
        self.width = WIDTH
        self.height = HEIGHT


class InfinityPipeline:
    """Infinity 2B pipeline: transformer sharded on TT, sampling + VAE on CPU."""

    def __init__(self, config: InfinityConfig):
        self.config = config

    def setup(self):
        self.load_models()
        if self.config.transformer_on_tt:
            self.shard_to_tt()
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
        # Replicates the runtime sharding the graph tester does (see
        # tests/infra/runners/torch_device_runner.py): enable SPMD, build the
        # (1, 8) mesh, move the transformer to the XLA device, then mark every
        # weight in the Megatron shard spec.
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

    def _encode_prompt(self, prompt: str):
        """Run T5-XL on CPU; returns (kv_compact, lens, cu_seqlens_k, max_seqlen_k)."""
        return _m.encode_prompt(self.tokenizer, self.text_encoder, prompt)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        num_inference_steps: Optional[int] = None,  # ignored: fixed by scale schedule
        seed: Optional[int] = 42,
    ) -> torch.Tensor:
        """Reimplements ``Infinity.autoregressive_infer_cfg`` with a CPU/TT split.

          - T5-XL text encode -> CPU
          - transformer conditioning, blocks, logits, word_embed -> TT (bf16)
          - multinomial sampling -> CPU
          - BSQ-VAE indices->codes, residual accumulation, decode -> CPU

        Classifier-free guidance is SEQUENTIAL (two batch-1 forwards per scale,
        combined on the logits) and the loop is packed-recompute (no KV cache), so
        the attention score stays head-sharded -- see the module docstring.
        ``_perf`` timers wrap the T5 encode, each per-scale transformer forward
        (both CFG branches + the TT->CPU sync), and the VAE decode.
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

        # Per-stage forward+sync times (reset every generate() call). Keys match
        # the shared imagegen harness (benchmarks/imagegen_benchmark.py): te1 = T5
        # encode (CPU), te2 = 0 (Infinity has a single text encoder), unet_steps =
        # per-scale transformer forward times (TT), vae = BSQ-VAE decode (CPU).
        self._perf = {
            "te1": None,
            "te2": 0.0,
            "unet_steps": [],
            "vae": None,
            "total": None,
        }
        t_total_start = time.perf_counter()

        if seed is None:
            rng = None
        else:
            m.rng.manual_seed(seed)
            rng = m.rng

        # ── T5-XL text encode (CPU) ───────────────────────────────────
        logger.info("[STAGE] T5 text encode: start")
        t0 = time.perf_counter()
        kv_compact, lens, cu_seqlens_k, max_seqlen_k = self._encode_prompt(prompt)
        self._perf["te1"] = time.perf_counter() - t0
        logger.info("[STAGE] T5 text encode: done")

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
        # block-causal attn_bias. The carried state is RAW token embeddings
        # re-projected through the sharded q/k/v weights every scale, so the
        # attention score stays head-sharded.
        def _build_attn_bias(sched, ref):
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
        logger.info(f"[STAGE] AR scale loop: start ({len(scale_schedule)} scales)")
        for si, pn in enumerate(scale_schedule):
            logger.info(f"[STEP] AR scale {si + 1}/{len(scale_schedule)} pn={pn}")
            cfg = cfg_list[si]
            sub_sched = scale_schedule[: si + 1]
            L_si = int(np.prod(pn))
            attn_bias = None

            # --- one batch-1 forward per CFG branch (TT, sharded) -> logits (CPU) ---
            # The per-scale timer covers both branches + the TT->CPU sync.
            t0 = time.perf_counter()
            branch_logits = []
            for br in branches:
                x_BLC = torch.cat([br["sos"], *shared_inputs], dim=1)
                if attn_bias is None:
                    attn_bias = _build_attn_bias(sub_sched, x_BLC)
                x_BLC = _run_blocks(x_BLC, br, sub_sched, attn_bias)
                hidden_si = x_BLC[:, -L_si:]
                logits = m.get_logits(hidden_si, br["cond_BD"]).mul(1 / tau_list[si])
                branch_logits.append(cpu_cast(logits).float())
            self._perf["unet_steps"].append(time.perf_counter() - t0)

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
        logger.info("[STAGE] AR scale loop: done")

        # ── BSQ-VAE decode (CPU) ──────────────────────────────────────
        logger.info("[STAGE] VAE decode: start")
        t0 = time.perf_counter()
        summed_codes = summed_codes.to("cpu")
        image = vae.decode(summed_codes.squeeze(-3))
        self._perf["vae"] = time.perf_counter() - t0
        logger.info("[STAGE] VAE decode: done")

        self._perf["total"] = time.perf_counter() - t_total_start
        # Returns the raw decode (B, 3, H, W) in [-1, 1]; the harness's save_image
        # does the [-1, 1] -> [0, 1] conversion.
        return image
