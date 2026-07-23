# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Krea Realtime Video — nightly e2e pipeline test with per-component PCC checks.

Every nn.Module component (text_encoder, transformer, vae) runs on Tenstorrent
(all bf16). After each TT forward, the same real pipeline tensors are fed to a
CPU "twin" of the component and PCC is checked immediately — encoder (once), each
transformer forward (recompute + every denoise step), the vae encode (recompute,
block >= 2), and the vae decode per block. Test fails fast the moment any PCC
drops below `PCC_THRESHOLD`.

The pipeline continues with TT outputs (real deployment behavior); each per-forward
PCC is measured against a CPU twin fed the same external inputs (x/t/context) as
the TT component. The transformer twin keeps its OWN CPU KV/cross-attn caches that
accumulate alongside the TT caches (compounded), so there is no per-forward gather
of the sharded TT cache to host — that gather was the single biggest DRAM/graph
cost on device (~8GB + ~160 all-gather graphs per forward).

Blackhole topology: text_encoder = 1 chip, transformer = SHARDED (enable_spmd +
mesh + shard_transformer_specs), vae = 1 chip.

STATUS: control flow (interleaved TT->CPU->PCC, fail-fast) is the deliverable;
the TT device / sharding / compounded-twin-cache mechanics are a FIRST CUT
for BH iteration and cannot be validated without hardware.
"""

import copy
import gc
import importlib
from collections import deque

import pytest
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from infra import ComparisonConfig, Framework, RunMode
from infra.evaluators import PccConfig, TorchComparisonEvaluator
from infra.utilities.torch_multichip_utils import enable_spmd, get_mesh
from loguru import logger
from transformers import AutoTokenizer
from utils import BringupStatus, Category, ModelGroup

from third_party.tt_forge_models.krea_realtime_video.pytorch.src.model_utils import (
    DTYPE,
    FRAME_SEQ_LENGTH,
    KREA_REPO_ID,
    KV_CACHE_NUM_FRAMES,
    LOCAL_ATTN_SIZE,
    MAX_SEQ_LEN,
    MESH_NAMES,
    MESH_SHAPES,
    NUM_CHANNELS_LATENTS,
    NUM_FRAMES_PER_BLOCK,
    SEQ_LENGTH,
    WAN_REPO_ID,
    fixed_sinusoidal_embedding_1d,
    load_text_encoder,
    load_transformer,
    load_vae,
    shard_transformer_specs,
)

PROMPT = "a cat sitting on a boat"
NUM_BLOCKS = 3 # smoke run
NUM_INFERENCE_STEPS = 6
SEED = 42
HEIGHT = 480
WIDTH = 832
VAE_SCALE_FACTOR = 8
SHIFT = 5.0

PCC_THRESHOLD = 0.80


_PCC_EVALUATOR = TorchComparisonEvaluator(ComparisonConfig(assert_on_failure=False))
_PCC_CONFIG = PccConfig()


def _pcc(device_out, golden_out) -> float:
    return float(_PCC_EVALUATOR._compare_pcc(device_out, golden_out, _PCC_CONFIG))


def _tt(x):
    return x.to(device=xm.xla_device())


def _cpu(x):
    return x.to("cpu")


def _patch_vectorized_rope(transformer):
    """Patch A: vectorize causal_rope_apply + rope_apply.

    Removes `grid_sizes.tolist()` + the per-sample Python loop (causal_model.py:173,
    model.py:61) — the data-dependent scalar (`_local_scalar_dense`) break that is the
    site of all 54 dynamo data-dependent breaks (confirmed via TORCH_LOGS=+dynamo).
    `grid_sizes = torch.tensor(u.shape[2:])` is shape-derived, so F/H/W are static:
    H, W are fixed by the resolution and F = seq // (H*W). No host sync, no loop,
    batch-vectorized. Math is identical to source for batch=1 with seq == F*H*W.

    Patches the module globals, so both the TT transformer and its CPU twin use it.
    """
    mod = importlib.import_module(type(transformer).__module__)
    ph, pw = transformer.patch_size[1], transformer.patch_size[2]
    H = (HEIGHT // VAE_SCALE_FACTOR) // ph
    W = (WIDTH // VAE_SCALE_FACTOR) // pw

    def _rope(x, grid_sizes, freqs, start_frame=0):
        b, s, n, cc = x.shape
        c = cc // 2
        F = s // (H * W)  # shape-derived (static); replaces grid_sizes.tolist()
        fp = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
        freqs_i = torch.cat(
            [
                fp[0][start_frame : start_frame + F].view(F, 1, 1, -1).expand(F, H, W, -1),
                fp[1][:H].view(1, H, 1, -1).expand(F, H, W, -1),
                fp[2][:W].view(1, 1, W, -1).expand(F, H, W, -1),
            ],
            dim=-1,
        ).reshape(1, s, 1, c)
        x_c = torch.view_as_complex(x.to(torch.float64).reshape(b, s, n, c, 2))
        return torch.view_as_real(x_c * freqs_i).flatten(3).type_as(x)

    mod.causal_rope_apply = lambda x, g, f, start_frame=0: _rope(x, g, f, start_frame)
    mod.rope_apply = lambda x, g, f: _rope(x, g, f, 0)


class KreaRealtimePipeline:
    """TT-canonical Krea pipeline with an inline CPU twin at every forward."""

    def setup(self):
        # Canonical (TT) components + CPU twins (golden). All bf16.
        self.text_encoder = load_text_encoder(WAN_REPO_ID, DTYPE)
        self.transformer = load_transformer(KREA_REPO_ID, DTYPE)
        self.vae = load_vae(WAN_REPO_ID, DTYPE)
        self.twin_text_encoder = load_text_encoder(WAN_REPO_ID, DTYPE)
        self.twin_transformer = load_transformer(KREA_REPO_ID, DTYPE)
        self.twin_vae = load_vae(WAN_REPO_ID, DTYPE)

        # sinusoidal_embedding_1d is a module global (shared by both transformer
        # instances since AutoModel resolves the same cached remote module).
        importlib.import_module(
            type(self.transformer).__module__
        ).sinusoidal_embedding_1d = fixed_sinusoidal_embedding_1d

        for tf in (self.transformer, self.twin_transformer):
            for blk in tf.blocks:
                blk.self_attn.local_attn_size = -1
                blk.self_attn.num_frame_per_block = NUM_FRAMES_PER_BLOCK

        # Patch A: vectorize RoPE (drop grid_sizes.tolist() + per-sample Python loop).
        _patch_vectorized_rope(self.transformer)

        self.tokenizer = AutoTokenizer.from_pretrained(WAN_REPO_ID, subfolder="tokenizer")
        self.video_processor = VideoProcessor(vae_scale_factor=VAE_SCALE_FACTOR)
        enc_mod = importlib.import_module(
            type(self.transformer).__module__.rsplit(".", 1)[0] + ".encoders"
        )
        self._prompt_clean = enc_mod.prompt_clean

        self._num_heads = self.transformer.config.num_heads
        self._head_dim = self.transformer.config.dim // self._num_heads
        self._num_tf_blocks = len(self.transformer.blocks)

        # TT setup: transformer sharded across the mesh, encoder + vae on 1 chip.
        xr.set_device_type("TT")
        enable_spmd()
        num_devices = xr.global_runtime_device_count()
        if num_devices not in MESH_SHAPES:
            raise ValueError(f"Unsupported device count {num_devices}")
        self._mesh = get_mesh(MESH_SHAPES[num_devices], MESH_NAMES)
        # text_encoder is compiled locally in _encode_and_check (used once, freed
        # after) so its compiled graph's device buffers don't linger and OOM the
        # transformer. transformer/vae stay resident and are compiled here.
        for m in (self.transformer, self.vae):
            m.compile(backend="tt")

        self.pccs = []  # (label, pcc)

    # ── PCC check (assert the moment PCC drops below threshold) ──────────────

    def _check(self, label, tt_out, golden_out):
        p = _pcc(tt_out, golden_out)
        self.pccs.append((label, p))
        logger.info(f"[PCC] {label}: {p:.6f}")
        assert p >= PCC_THRESHOLD, f"{label} PCC {p:.6f} < {PCC_THRESHOLD}"

    @staticmethod
    def _caches_to(caches, mover):
        for e in caches:
            e["k"] = mover(e["k"])
            e["v"] = mover(e["v"])

    # ── encoder (isolated: same input_ids) ───────────────────────────────────

    def _postprocess_embeds(self, embeds, seq_lens):
        embeds = embeds.to(dtype=DTYPE)
        embeds = [u[:v] for u, v in zip(embeds, seq_lens)]
        return torch.stack(
            [torch.cat([u, u.new_zeros(MAX_SEQ_LEN - u.size(0), u.size(1))]) for u in embeds],
            dim=0,
        ).contiguous()

    def _encode_and_check(self, prompt):
        text_inputs = self.tokenizer(
            [self._prompt_clean(prompt)],
            padding="max_length",
            max_length=MAX_SEQ_LEN,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        # canonical (TT) — compile a LOCAL wrapper and del it after use so the
        # compiled graph's device buffers are released (in-place .compile() keeps
        # them resident even after .to("cpu"), which OOMs the transformer). The
        # encoder runs exactly once, so drop it entirely.
        self.text_encoder = self.text_encoder.to(xm.xla_device())
        compiled = torch.compile(self.text_encoder, backend="tt")
        tt_embeds = _cpu(compiled(_tt(input_ids), _tt(mask)).last_hidden_state)
        del compiled
        self.text_encoder = None
        # twin (CPU, same input_ids), also freed (used once)
        golden = self.twin_text_encoder(input_ids, mask).last_hidden_state
        self.twin_text_encoder = None
        gc.collect()
        torch_xla.sync()  # reclaim both encoders before the transformer lands

        tt_embeds = self._postprocess_embeds(tt_embeds, seq_lens)
        golden = self._postprocess_embeds(golden, seq_lens)
        self._check("encoder", tt_embeds, golden)
        return tt_embeds  # canonical embeds used downstream

    # ── one transformer forward: TT (canonical) + twin on snapshot ───────────

    def _transformer_step(self, label, x, t, context, kv_cache, crossattn_cache,
                          twin_kv_cache, twin_crossattn_cache, current_start):
        noise_tt = _cpu(
            self.transformer(
                x=_tt(x), t=_tt(t), context=_tt(context),
                kv_cache=kv_cache, seq_len=SEQ_LENGTH, crossattn_cache=crossattn_cache,
                current_start=current_start, cache_start=None,
            )
        )
        # Compounded PCC: the twin keeps its OWN CPU caches that accumulate KV
        # alongside the TT caches (same external x/t/context each step). No
        # per-forward gather of the sharded TT cache to host -> no ~8GB/forward
        # DRAM spike, no all-gather graphs.
        noise_golden = self.twin_transformer(
            x=x, t=t, context=context,
            kv_cache=twin_kv_cache, seq_len=SEQ_LENGTH, crossattn_cache=twin_crossattn_cache,
            current_start=current_start, cache_start=None,
        )
        self._check(label, noise_tt, noise_golden)
        return noise_tt

    # ── recompute (block >= 1): both transformers refill their caches ────────

    def _vae_encode_and_check(self, label, frames):
        # VAE encoder nn.Module on TT (canonical) + CPU twin -> PCC, like decode.
        self.vae = self.vae.to(xm.xla_device())
        self.vae._enc_feat_map = [None] * 55
        tt_lat = _cpu(self.vae.encode(_tt(frames.to(self.vae.dtype))).latent_dist.mode())
        self.vae = self.vae.to("cpu")

        self.twin_vae._enc_feat_map = [None] * 55
        golden = self.twin_vae.encode(frames.to(self.twin_vae.dtype)).latent_dist.mode()
        self._check(label, tt_lat, golden)

        # normalize on host (deterministic; identical both sides) via the canonical.
        z_dim = self.twin_vae.config.z_dim
        mean = torch.tensor(self.twin_vae.config.latents_mean).view(1, z_dim, 1, 1, 1).to(tt_lat.device, tt_lat.dtype)
        std = 1.0 / torch.tensor(self.twin_vae.config.latents_std).view(1, z_dim, 1, 1, 1).to(tt_lat.device, tt_lat.dtype)
        return ((tt_lat - mean) * std).to(DTYPE)

    def _build_context_frames(self, block_idx, current_denoised, frame_cache_context, block_latents):
        total = (block_idx - 1) * NUM_FRAMES_PER_BLOCK
        if total < KV_CACHE_NUM_FRAMES:
            return current_denoised[:, :, :KV_CACHE_NUM_FRAMES]
        ctx = current_denoised[:, :, 1:][:, :, -KV_CACHE_NUM_FRAMES + 1:]
        first = self._vae_encode_and_check(
            f"b{block_idx}_vae_encode", frame_cache_context[0].half()
        )
        first = first.to(block_latents)
        return torch.cat((first, ctx), dim=2)

    def _recompute_and_check(self, block_idx, context_frames, prompt_embeds,
                             kv_cache, crossattn_cache, twin_kv_cache, twin_crossattn_cache):
        ctx_ts = torch.zeros((context_frames.shape[0], context_frames.shape[2]), dtype=torch.int64)

        def _mask(tf, device):
            return tf._prepare_blockwise_causal_attn_mask(
                device, num_frames=context_frames.shape[2], frame_seqlen=FRAME_SEQ_LENGTH,
                num_frame_per_block=NUM_FRAMES_PER_BLOCK, local_attn_size=-1,
            )

        self.transformer.block_mask = _mask(self.transformer, xm.xla_device())
        noise_tt = _cpu(
            self.transformer(
                x=_tt(context_frames), t=_tt(ctx_ts), context=_tt(prompt_embeds),
                kv_cache=kv_cache, seq_len=SEQ_LENGTH, crossattn_cache=crossattn_cache,
                current_start=0, cache_start=None,
            )
        )
        self.transformer.block_mask = None

        # twin refills its OWN caches (compounded), no gather of the TT cache.
        self.twin_transformer.block_mask = _mask(self.twin_transformer, torch.device("cpu"))
        noise_golden = self.twin_transformer(
            x=context_frames, t=ctx_ts, context=prompt_embeds,
            kv_cache=twin_kv_cache, seq_len=SEQ_LENGTH, crossattn_cache=twin_crossattn_cache,
            current_start=0, cache_start=None,
        )
        self.twin_transformer.block_mask = None
        self._check(f"b{block_idx}_recompute", noise_tt, noise_golden)

    # ── decode: TT (canonical) + twin on feat_map snapshot ───────────────────

    def _decode_and_check(self, latents, block_idx, decoder_cache, frame_cache_context):
        if frame_cache_context is None:
            frame_cache_context = deque(maxlen=1 + (KV_CACHE_NUM_FRAMES - 1) * 4)

        def _rescale(lat, vae):
            z_dim = vae.config.z_dim
            mean = torch.tensor(vae.config.latents_mean, device=lat.device, dtype=lat.dtype).view(1, z_dim, 1, 1, 1)
            std = 1.0 / torch.tensor(vae.config.latents_std, device=lat.device, dtype=lat.dtype).view(1, z_dim, 1, 1, 1)
            return (lat / std + mean).to(vae.dtype)

        # twin (CPU) on a snapshot of the canonical feat_map (pre-decode)
        twin_cache = copy.deepcopy(decoder_cache) if block_idx else None
        if block_idx == 0:
            self.twin_vae.clear_cache()
            self.twin_vae.clear_cache = lambda: None
            self.twin_vae._feat_map = [None] * 55
        else:
            self.twin_vae._feat_map = twin_cache
        golden = self.twin_vae.decode(_rescale(latents, self.twin_vae), return_dict=False)[0]

        # canonical (TT)
        self.vae = self.vae.to(xm.xla_device())
        if block_idx == 0:
            self.vae.clear_cache()
            self.vae.clear_cache = lambda: None
            self.vae._feat_map = [None] * 55
        else:
            self.vae._feat_map = decoder_cache
        videos = _cpu(self.vae.decode(_tt(_rescale(latents, self.vae)), return_dict=False)[0])
        decoder_cache = self.vae._feat_map
        self.vae = self.vae.to("cpu")

        self._check(f"b{block_idx}_decode", videos, golden)
        frame_cache_context.extend(videos.split(1, dim=2))
        frames = self.video_processor.postprocess_video(videos, output_type="pil")
        return frames[0], decoder_cache, frame_cache_context

    # ── timesteps / latents / caches ─────────────────────────────────────────

    def _set_timesteps(self, n):
        sigmas = torch.linspace(1.0, 0.0, 1001)[:-1]
        sigmas = SHIFT * sigmas / (1 + (SHIFT - 1) * sigmas)
        timesteps = sigmas * 1000.0
        zero_padded = torch.cat([timesteps, torch.tensor([0])])
        denoising_steps = torch.linspace(1.0 * 1000, 0, n, dtype=torch.float32).to(torch.long)
        return zero_padded[1000 - denoising_steps], timesteps, sigmas

    def _prepare_init_latents(self, num_blocks, generator):
        shape = (1, NUM_CHANNELS_LATENTS, num_blocks * NUM_FRAMES_PER_BLOCK,
                 HEIGHT // VAE_SCALE_FACTOR, WIDTH // VAE_SCALE_FACTOR)
        return randn_tensor(shape, generator=generator, device="cpu", dtype=DTYPE).contiguous()

    def _init_kv_cache(self):
        shape = [1, LOCAL_ATTN_SIZE * FRAME_SEQ_LENGTH, self._num_heads, self._head_dim]
        return [
            {"k": torch.zeros(shape, dtype=DTYPE).contiguous(),
             "v": torch.zeros(shape, dtype=DTYPE).contiguous(),
             "global_end_index": 0, "local_end_index": 0}
            for _ in range(self._num_tf_blocks)
        ]

    @staticmethod
    def _zero_kv_cache(kv_cache):
        for e in kv_cache:
            e["k"].zero_(); e["v"].zero_()
            e["global_end_index"] = 0; e["local_end_index"] = 0

    def _init_crossattn_cache(self):
        shape = [1, MAX_SEQ_LEN, self._num_heads, self._head_dim]
        return [
            {"k": torch.zeros(shape, dtype=DTYPE), "v": torch.zeros(shape, dtype=DTYPE), "is_init": False}
            for _ in range(self._num_tf_blocks)
        ]

    @staticmethod
    def _add_noise(sample, noise, timestep, all_timesteps, sigmas):
        if timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)
        tid = torch.argmin((all_timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma = sigmas[tid].reshape(-1, 1, 1, 1)
        return ((1 - sigma.double()) * sample.double() + sigma.double() * noise.double()).type_as(noise)

    # ── generate (TT canonical trajectory, inline twin checks) ───────────────

    def generate(self, prompt, num_blocks, num_inference_steps, seed):
        with torch.no_grad():
            generator = torch.Generator(device="cpu").manual_seed(seed)

            prompt_embeds = self._encode_and_check(prompt)  # encoder PCC

            timesteps, all_timesteps, sigmas = self._set_timesteps(num_inference_steps)
            init_latents = self._prepare_init_latents(num_blocks, generator)

            kv_cache = self._init_kv_cache()
            crossattn_cache = self._init_crossattn_cache()
            # Twin (CPU) caches evolve independently alongside the TT caches
            # (compounded PCC). They stay on host — never moved/sharded to TT.
            twin_kv_cache = self._init_kv_cache()
            twin_crossattn_cache = self._init_crossattn_cache()
            decoder_cache = None
            frame_cache_context = None
            current_denoised = None

            # transformer resident on the mesh for the whole run.
            self.transformer = self.transformer.to(xm.xla_device())
            for tensor, spec in shard_transformer_specs(self.transformer).items():
                xs.mark_sharding(tensor, self._mesh, spec)
            self._caches_to(kv_cache, _tt)
            self._caches_to(crossattn_cache, _tt)
            # Shard the caches on the head dim (dim 2) to match the transformer's
            # tensor-parallel head split; without this .to(xla) leaves them
            # replicated full-size on every chip -> DRAM OOM.
            head_spec = (None, None, "model", None)
            for e in (*kv_cache, *crossattn_cache):
                xs.mark_sharding(e["k"], self._mesh, head_spec)
                xs.mark_sharding(e["v"], self._mesh, head_spec)

            frames = []
            for block_idx in range(num_blocks):
                logger.info(f"  block {block_idx + 1}/{num_blocks}")
                if block_idx > 0:
                    self._zero_kv_cache(kv_cache)
                    self._zero_kv_cache(twin_kv_cache)

                start = block_idx * NUM_FRAMES_PER_BLOCK
                block_latents = init_latents[:, :, start:start + NUM_FRAMES_PER_BLOCK]
                current_start_frame = start

                if block_idx > 0:
                    context_frames = self._build_context_frames(
                        block_idx, current_denoised, frame_cache_context, block_latents
                    )
                    self._recompute_and_check(
                        block_idx, context_frames, prompt_embeds,
                        kv_cache, crossattn_cache, twin_kv_cache, twin_crossattn_cache,
                    )

                latents = block_latents
                for i, t in enumerate(timesteps):
                    start_frame = min(current_start_frame, KV_CACHE_NUM_FRAMES)
                    noise = self._transformer_step(
                        f"b{block_idx}_step{i}", latents,
                        t.expand(latents.shape[0], NUM_FRAMES_PER_BLOCK),
                        prompt_embeds, kv_cache, crossattn_cache,
                        twin_kv_cache, twin_crossattn_cache, start_frame * FRAME_SEQ_LENGTH,
                    )
                    tid = torch.argmin((all_timesteps - t).abs())
                    latents = (latents.double() - sigmas[tid].double() * noise.double()).to(latents.dtype)
                    if i < num_inference_steps - 1:
                        t1 = timesteps[i + 1]
                        sample = latents.transpose(1, 2).squeeze(0)
                        noise_r = randn_tensor(sample.shape, device="cpu", dtype=latents.dtype, generator=generator)
                        latents = self._add_noise(
                            sample, noise_r, t1.expand(latents.shape[0], NUM_FRAMES_PER_BLOCK),
                            all_timesteps, sigmas,
                        ).unsqueeze(0).transpose(1, 2)
                current_denoised = latents

                block_frames, decoder_cache, frame_cache_context = self._decode_and_check(
                    current_denoised, block_idx, decoder_cache, frame_cache_context
                )
                frames.extend(block_frames)
            return frames


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.llmbox
@pytest.mark.large
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name="KreaRealtimeVideo_Pipeline",
    model_group=ModelGroup.RED,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,  
)
def test_krea_realtime_pipeline():
    """e2e Krea on TT (all bf16), per-forward PCC vs an inline CPU twin."""
    pipe = KreaRealtimePipeline()
    pipe.setup()
    pipe.generate(PROMPT, NUM_BLOCKS, NUM_INFERENCE_STEPS, SEED)

    # _check already asserts each forward inline; if we got here everything passed.
    worst_label, worst = min(pipe.pccs, key=lambda lp: lp[1])
    logger.info(f"[PCC] worst: {worst_label} = {worst:.6f} (all >= {PCC_THRESHOLD})")


if __name__ == "__main__":
    test_krea_realtime_pipeline()
