# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Krea Realtime Video — CPU-only pipeline parity check.

First step of the e2e TT bringup: prove that a *hand-written* orchestration
(``KreaRealtimePipeline``) reproduces the stock ModularPipeline output on CPU,
so we know the orchestration itself introduces no deviation before any component
is moved to TT.

Structure mirrors tests/torch/models/playground_v2_5/test_playground_v2_5_pipeline.py
(a manual pipeline that owns the loop and calls each nn.Module directly) but
CPU-only — no torch_xla, no ``.compile(backend="tt")``. Every component
(text_encoder / transformer / vae) is a discrete, independently-placeable unit;
the ``# [TT offload point]`` comments mark where a later TT version would move a
module to ``xm.xla_device()`` around its forward.

    original :  stock ModularPipeline.load_components()   (reference / golden)
    ours     :  KreaRealtimePipeline                      (our orchestration)

The manual orchestration is a faithful port of the remote modular blocks
(text_encoder → before_denoise → denoise → decode), run once per block over 9
blocks with persistent KV-cache / streaming-VAE state. The model internals
(transformer forward, VAE encode/decode) are called unchanged; only the
host-side orchestration is reimplemented. Source line refs are in the method
docstrings (before_denoise.py / denoise.py / decoders.py / encoders.py of the
krea remote code).

NOTE: full 9-block run is ~15 min/block on CPU, run TWICE (~4.5 h total). Lower
NUM_BLOCKS to iterate faster (3 covers every code path).
"""

import gc
import importlib
from collections import deque
from typing import List, Optional

import numpy as np
import torch
from diffusers import ModularPipeline
from diffusers.modular_pipelines import PipelineState
from diffusers.utils import export_to_video
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from loguru import logger
from tqdm import tqdm
from transformers import AutoTokenizer

from third_party.tt_forge_models.krea_realtime_video.pytorch.src.model_utils import (
    DTYPE,  # torch.bfloat16 — all components stay bf16
    FRAME_SEQ_LENGTH,
    KREA_REPO_ID,
    KV_CACHE_NUM_FRAMES,
    LOCAL_ATTN_SIZE,  # kv_cache_num_frames + num_frames_per_block = 6
    MAX_SEQ_LEN,
    NUM_CHANNELS_LATENTS,
    NUM_FRAMES_PER_BLOCK,
    SEQ_LENGTH,
    WAN_REPO_ID,
    fixed_sinusoidal_embedding_1d,
    load_text_encoder,
    load_transformer,
    load_vae,
)

# ── Run configuration ──────────────────────────────────────────────────────

REPO_ID = KREA_REPO_ID  # "krea/krea-realtime-video"
PROMPT = "a cat sitting on a boat"
NUM_BLOCKS = 9 # lower to 3 for a fast smoke run (covers all code paths)
NUM_INFERENCE_STEPS = 6
SEED = 42
FPS = 24
HEIGHT = 480
WIDTH = 832
VAE_SCALE_FACTOR = 8
SHIFT = 5.0  # flow-matching sigma shift (before_denoise.py:389)
STRENGTH = 1.0
DEVICE = "cpu"

HEAVY_COMPONENTS = {"transformer", "text_encoder", "vae"}

# CPU-vs-CPU with identical math should be pixel-exact. Allowed max abs uint8
# pixel diff before the check fails (0 = require identical). An autoregressive
# divergence compounds across blocks, so it shows up as a large diff, not a
# stable ±1 — strict is a safe default.
MAX_PIXEL_DIFF = 0


# ═══════════════════════════════════════════════════════════════════════════
#  Original pipeline (reference / golden) — stock ModularPipeline
# ═══════════════════════════════════════════════════════════════════════════


def _apply_cpu_patches(pipe: ModularPipeline) -> None:
    """Patch out the two hardcoded-CUDA / hardcoded-fp16 spots (stock pipeline)."""
    transformer_module = type(pipe.transformer).__module__

    importlib.import_module(
        transformer_module
    ).sinusoidal_embedding_1d = fixed_sinusoidal_embedding_1d

    before_denoise = importlib.import_module(
        transformer_module.rsplit(".", 1)[0] + ".before_denoise"
    )
    cls = before_denoise.WanRTRecomputeKVCache
    if not getattr(cls.prepare_latents, "_cpu_patched", False):
        orig_prepare_latents = cls.prepare_latents

        def prepare_latents(self, components, frames):
            return orig_prepare_latents(self, components, frames.to(components.vae.dtype))

        prepare_latents._cpu_patched = True
        cls.prepare_latents = prepare_latents


def run_original_pipeline() -> list:
    """Stock ModularPipeline with natively-loaded components (the golden)."""
    pipe = ModularPipeline.from_pretrained(REPO_ID, trust_remote_code=True)
    pipe.load_components(
        trust_remote_code=True,
        torch_dtype={"default": DTYPE, "vae": DTYPE},
    )
    _apply_cpu_patches(pipe)

    frames = []
    state = PipelineState()
    generator = torch.Generator(device=pipe.device).manual_seed(SEED)
    for block_idx in tqdm(range(NUM_BLOCKS), desc="original"):
        logger.info(f"  [original] block {block_idx + 1}/{NUM_BLOCKS}")
        state = pipe(
            state,
            prompt=[PROMPT],
            num_inference_steps=NUM_INFERENCE_STEPS,
            num_blocks=NUM_BLOCKS,
            block_idx=block_idx,
            generator=generator,
        )
        frames.extend(state.values["videos"][0])

    del pipe
    gc.collect()
    return frames


# ═══════════════════════════════════════════════════════════════════════════
#  Our manual pipeline — hand-written orchestration, CPU-only
# ═══════════════════════════════════════════════════════════════════════════


class KreaRealtimePipeline:
    """Hand-written CPU orchestration of Krea Realtime (Wan 2.1 14B, autoregressive).

    Reproduces the remote modular blocks' per-block flow without ModularPipeline:
        text_encode (once) → [per block] extract → kv-cache setup/recompute →
        denoise loop → streaming VAE decode.

    Components are loaded via the tt-forge loaders and kept as discrete modules so
    a later TT version can offload each around its forward (see [TT offload point]).
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    # ── setup ──────────────────────────────────────────────────────────────

    def setup(self):
        # Raw components (NOT the CausalWanWrapper/VAEDecoderWrapper — we need the
        # full interfaces: transformer kv_cache/current_start, vae encode+decode).
        self.text_encoder = load_text_encoder(WAN_REPO_ID, DTYPE)  # UMT5EncoderModel
        self.transformer = load_transformer(KREA_REPO_ID, DTYPE)  # CausalWanModel
        self.vae = load_vae(WAN_REPO_ID, DTYPE)  # AutoencoderKLWan

        # Drop the hardcoded-CUDA sinusoidal embedding (CausalWanWrapper does this
        # in __init__; we use the raw model, so patch its module global ourselves).
        importlib.import_module(
            type(self.transformer).__module__
        ).sinusoidal_embedding_1d = fixed_sinusoidal_embedding_1d

        # Per-block self-attention attributes SetupKVCache sets each block
        # (constant, so set once). before_denoise.py:1126-1129.
        for blk in self.transformer.blocks:
            blk.self_attn.local_attn_size = -1
            blk.self_attn.num_frame_per_block = NUM_FRAMES_PER_BLOCK

        self.tokenizer = AutoTokenizer.from_pretrained(
            WAN_REPO_ID, subfolder="tokenizer"
        )
        self.video_processor = VideoProcessor(vae_scale_factor=VAE_SCALE_FACTOR)

        # prompt_clean from the remote encoders module → identical text cleaning.
        enc_mod = importlib.import_module(
            type(self.transformer).__module__.rsplit(".", 1)[0] + ".encoders"
        )
        self._prompt_clean = enc_mod.prompt_clean

        self._num_heads = self.transformer.config.num_heads
        self._head_dim = self.transformer.config.dim // self._num_heads
        self._num_tf_blocks = len(self.transformer.blocks)

    # ── text encoding (encoders.py:_get_t5_prompt_embeds) — run once ────────

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        prompt = [self._prompt_clean(prompt)]
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=MAX_SEQ_LEN,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        # [TT offload point] text_encoder
        embeds = self.text_encoder(
            input_ids.to(self.device), mask.to(self.device)
        ).last_hidden_state
        embeds = embeds.to(dtype=self.text_encoder.dtype)

        embeds = [u[:v] for u, v in zip(embeds, seq_lens)]
        embeds = torch.stack(
            [
                torch.cat([u, u.new_zeros(MAX_SEQ_LEN - u.size(0), u.size(1))])
                for u in embeds
            ],
            dim=0,
        )
        return embeds.contiguous()

    # ── timesteps / sigmas (before_denoise.py:386-409) ──────────────────────

    def _set_timesteps(self, num_inference_steps: int):
        sigmas = torch.linspace(1.0, 0.0, 1001)[:-1]  # len 1000, fp32 CPU
        sigmas = SHIFT * sigmas / (1 + (SHIFT - 1) * sigmas)
        timesteps = sigmas.to(self.transformer.device) * 1000.0  # len 1000
        zero_padded = torch.cat(
            [timesteps, torch.tensor([0], device=self.transformer.device)]
        )  # len 1001
        denoising_steps = (
            torch.linspace(STRENGTH * 1000, 0, num_inference_steps, dtype=torch.float32)
            .to(torch.long)
        )
        out_timesteps = zero_padded[1000 - denoising_steps]  # len num_inference_steps
        return out_timesteps, timesteps, sigmas  # timesteps, all_timesteps, sigmas

    # ── initial latents (before_denoise.py:496-533) — drawn once ────────────

    def _prepare_init_latents(self, num_blocks: int, generator: torch.Generator):
        shape = (
            1,
            NUM_CHANNELS_LATENTS,
            num_blocks * NUM_FRAMES_PER_BLOCK,  # 27 for 9 blocks
            HEIGHT // VAE_SCALE_FACTOR,  # 60
            WIDTH // VAE_SCALE_FACTOR,  # 104
        )
        latents = randn_tensor(
            shape,
            generator=generator,
            device=self.transformer.device,
            dtype=self.transformer.dtype,
        )
        return latents.contiguous()

    # ── KV / cross-attn caches (before_denoise.py:_initialize_*) ────────────

    def _init_kv_cache(self):
        kv_size = LOCAL_ATTN_SIZE * FRAME_SEQ_LENGTH  # 6 * 1560 = 9360
        shape = [1, kv_size, self._num_heads, self._head_dim]
        return [
            {
                "k": torch.zeros(shape, dtype=self.transformer.dtype).contiguous(),
                "v": torch.zeros(shape, dtype=self.transformer.dtype).contiguous(),
                "global_end_index": 0,
                "local_end_index": 0,
            }
            for _ in range(self._num_tf_blocks)
        ]

    @staticmethod
    def _zero_kv_cache(kv_cache):
        # SetupKVCache zeroes k/v in place every block >= 1 (before_denoise.py:160-165).
        for entry in kv_cache:
            entry["k"].zero_()
            entry["v"].zero_()
            entry["global_end_index"] = 0
            entry["local_end_index"] = 0

    def _init_crossattn_cache(self):
        shape = [1, MAX_SEQ_LEN, self._num_heads, self._head_dim]  # [1, 512, ...]
        return [
            {
                "k": torch.zeros(shape, dtype=self.transformer.dtype),
                "v": torch.zeros(shape, dtype=self.transformer.dtype),
                "is_init": False,
            }
            for _ in range(self._num_tf_blocks)
        ]

    # ── VAE encode of a cached pixel frame (before_denoise.py:1199-1212) ────

    def _vae_encode_latent(self, frames: torch.Tensor) -> torch.Tensor:
        self.vae._enc_feat_map = [None] * 55
        # Cast to the VAE dtype (the remote code hardcodes .half(); we align to
        # vae.dtype instead — same fix as the stock-pipeline monkey patch).
        # [TT offload point] vae (encode)
        enc = self.vae.encode(frames.to(self.vae.dtype))
        latents = enc.latent_dist.mode()  # retrieve_latents(sample_mode="argmax")
        z_dim = self.vae.config.z_dim
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
            1, z_dim, 1, 1, 1
        ).to(latents.device, latents.dtype)
        latents = (latents - latents_mean) * latents_std
        return latents.to(self.transformer.dtype)

    # ── KV-cache recompute (before_denoise.py:1145-1269) — block >= 1 ───────

    def _recompute_kv_cache(
        self,
        block_idx: int,
        current_denoised_latents: torch.Tensor,
        frame_cache_context: deque,
        prompt_embeds: torch.Tensor,
        kv_cache,
        crossattn_cache,
        block_latents: torch.Tensor,
    ):
        total = (block_idx - 1) * NUM_FRAMES_PER_BLOCK
        if total < KV_CACHE_NUM_FRAMES:  # block_idx == 1: first 3 denoised frames
            context_frames = current_denoised_latents[:, :, :KV_CACHE_NUM_FRAMES]
        else:  # block_idx >= 2: attention-sink first frame + last 2 prev frames
            ctx = current_denoised_latents[:, :, 1:][:, :, -KV_CACHE_NUM_FRAMES + 1:]
            first_frame_latent = self._vae_encode_latent(
                frame_cache_context[0].half()
            )
            first_frame_latent = first_frame_latent.to(block_latents)
            context_frames = torch.cat((first_frame_latent, ctx), dim=2)

        block_mask = self.transformer._prepare_blockwise_causal_attn_mask(
            self.transformer.device,
            num_frames=context_frames.shape[2],
            frame_seqlen=FRAME_SEQ_LENGTH,
            num_frame_per_block=NUM_FRAMES_PER_BLOCK,
            local_attn_size=-1,
        )
        self.transformer.block_mask = block_mask
        context_timestep = torch.zeros(
            (context_frames.shape[0], context_frames.shape[2]),
            device=self.transformer.device,
            dtype=torch.int64,
        )
        # [TT offload point] transformer (cache prefill over context frames)
        self.transformer(
            x=context_frames.to(self.transformer.dtype),
            t=context_timestep,
            context=prompt_embeds.to(self.transformer.dtype),
            kv_cache=kv_cache,
            seq_len=SEQ_LENGTH,
            crossattn_cache=crossattn_cache,
            current_start=0,
            cache_start=None,
        )
        self.transformer.block_mask = None

    # ── flow-matching re-noise (denoise.py:195-211) ─────────────────────────

    @staticmethod
    def _add_noise(sample, noise, timestep, all_timesteps, sigmas):
        if timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)
        timestep_id = torch.argmin(
            (all_timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1
        )
        sigma = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        sample = (1 - sigma.double()) * sample.double() + sigma.double() * noise.double()
        return sample.type_as(noise)

    # ── denoise loop for one block (denoise.py:98-280) ──────────────────────

    def _denoise_block(
        self,
        latents,
        timesteps,
        all_timesteps,
        sigmas,
        num_steps,
        prompt_embeds,
        kv_cache,
        crossattn_cache,
        current_start_frame,
        generator,
    ):
        for i, t in enumerate(timesteps):
            # WanRTLoopDenoiser: transformer forward with persistent KV cache.
            start_frame = min(current_start_frame, KV_CACHE_NUM_FRAMES)
            # [TT offload point] transformer (denoise step)
            noise_pred = self.transformer(
                x=latents,
                t=t.expand(latents.shape[0], NUM_FRAMES_PER_BLOCK),
                context=prompt_embeds,
                kv_cache=kv_cache,
                seq_len=SEQ_LENGTH,
                crossattn_cache=crossattn_cache,
                current_start=start_frame * FRAME_SEQ_LENGTH,
                cache_start=None,
            )

            # WanRTLoopAfterDenoiser: flow-match euler step (double precision).
            timestep_id = torch.argmin((all_timesteps - t).abs())
            sigma_t = sigmas[timestep_id]
            latents = (
                latents.double() - sigma_t.double() * noise_pred.double()
            ).to(latents.dtype)

            # Re-noise for the next step (skip after the final step).
            if i < num_steps - 1:
                t1 = timesteps[i + 1]
                sample = latents.transpose(1, 2).squeeze(0)
                noise = randn_tensor(
                    sample.shape,
                    device=latents.device,
                    dtype=latents.dtype,
                    generator=generator,
                )
                latents = (
                    self._add_noise(
                        sample,
                        noise,
                        t1.expand(latents.shape[0], NUM_FRAMES_PER_BLOCK),
                        all_timesteps,
                        sigmas,
                    )
                    .unsqueeze(0)
                    .transpose(1, 2)
                )
        return latents  # current_denoised_latents

    # ── streaming VAE decode for one block (decoders.py:104-153) ────────────

    def _decode_block(self, latents, block_idx, decoder_cache, frame_cache_context):
        if frame_cache_context is None:
            frame_cache_len = 1 + (KV_CACHE_NUM_FRAMES - 1) * 4  # 9
            frame_cache_context = deque(maxlen=frame_cache_len)

        if block_idx == 0:
            self.vae.clear_cache()
            self.vae.clear_cache = lambda: None  # keep feat_map across blocks
            self.vae._feat_map = [None] * 55
        else:
            self.vae._feat_map = decoder_cache

        lat = latents.to(self.vae.device)
        z_dim = self.vae.config.z_dim
        mean = torch.tensor(
            self.vae.config.latents_mean, device=lat.device, dtype=lat.dtype
        ).view(1, z_dim, 1, 1, 1)
        std = 1.0 / torch.tensor(
            self.vae.config.latents_std, device=lat.device, dtype=lat.dtype
        ).view(1, z_dim, 1, 1, 1)
        lat = lat / std + mean
        lat = lat.to(self.vae.dtype)

        # [TT offload point] vae (decode)
        videos = self.vae.decode(lat, return_dict=False)[0]

        decoder_cache = self.vae._feat_map
        frame_cache_context.extend(videos.split(1, dim=2))
        frames = self.video_processor.postprocess_video(videos, output_type="pil")
        return frames[0], decoder_cache, frame_cache_context

    # ── top-level generate ──────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        num_blocks: int,
        num_inference_steps: int,
        seed: Optional[int] = None,
    ) -> list:
        with torch.no_grad():
            generator = torch.Generator(device=self.device)
            if seed is not None:
                generator.manual_seed(seed)

            prompt_embeds = self._encode_prompt(prompt)  # once (block 0 equivalent)
            timesteps, all_timesteps, sigmas = self._set_timesteps(num_inference_steps)
            init_latents = self._prepare_init_latents(num_blocks, generator)

            kv_cache = self._init_kv_cache()
            crossattn_cache = self._init_crossattn_cache()
            decoder_cache = None
            frame_cache_context = None
            current_denoised_latents = None

            frames = []
            for block_idx in tqdm(range(num_blocks), desc="ours"):
                logger.info(f"  [ours] block {block_idx + 1}/{num_blocks}")

                if block_idx > 0:
                    self._zero_kv_cache(kv_cache)

                # Extract this block's 3 latent frames from the full buffer.
                start = block_idx * NUM_FRAMES_PER_BLOCK
                block_latents = init_latents[:, :, start:start + NUM_FRAMES_PER_BLOCK]
                current_start_frame = start

                if block_idx > 0:
                    self._recompute_kv_cache(
                        block_idx,
                        current_denoised_latents,
                        frame_cache_context,
                        prompt_embeds,
                        kv_cache,
                        crossattn_cache,
                        block_latents,
                    )

                current_denoised_latents = self._denoise_block(
                    block_latents,
                    timesteps,
                    all_timesteps,
                    sigmas,
                    num_inference_steps,
                    prompt_embeds,
                    kv_cache,
                    crossattn_cache,
                    current_start_frame,
                    generator,
                )

                block_frames, decoder_cache, frame_cache_context = self._decode_block(
                    current_denoised_latents, block_idx, decoder_cache, frame_cache_context
                )
                frames.extend(block_frames)

            return frames


def run_our_pipeline() -> list:
    """Our hand-written CPU orchestration."""
    pipeline = KreaRealtimePipeline(device=DEVICE)
    pipeline.setup()
    frames = pipeline.generate(
        prompt=PROMPT,
        num_blocks=NUM_BLOCKS,
        num_inference_steps=NUM_INFERENCE_STEPS,
        seed=SEED,
    )
    del pipeline
    gc.collect()
    return frames


# ═══════════════════════════════════════════════════════════════════════════
#  Video comparison (the video-gen analogue of ref.py's image compare)
# ═══════════════════════════════════════════════════════════════════════════


def _frames_to_numpy(frames: list) -> np.ndarray:
    """Stack a list of frames (PIL.Image | np.ndarray | torch.Tensor) → (T,H,W,C)."""
    arrs = []
    for f in frames:
        if isinstance(f, np.ndarray):
            arrs.append(f)
        elif torch.is_tensor(f):
            arrs.append(f.detach().cpu().float().numpy())
        else:  # PIL.Image.Image
            arrs.append(np.asarray(f))
    return np.stack(arrs, axis=0)


def compare_videos(frames_a, frames_b, name_a="original", name_b="ours") -> int:
    """Report per-video pixel parity and return the max absolute pixel diff."""
    a = _frames_to_numpy(frames_a).astype(np.int64)
    b = _frames_to_numpy(frames_b).astype(np.int64)

    print(f"\n{'=' * 64}")
    print(f"  {name_a}: {len(frames_a)} frames, array {a.shape}")
    print(f"  {name_b}: {len(frames_b)} frames, array {b.shape}")

    if a.shape != b.shape:
        print("  RESULT: SHAPE MISMATCH")
        print(f"{'=' * 64}\n")
        return 2**31 - 1  # force failure

    diff = np.abs(a - b)
    max_diff = int(diff.max())

    if max_diff == 0:
        print("  RESULT: pixel-identical across all frames!")
        print(f"{'=' * 64}\n")
        return 0

    per_frame_max = diff.reshape(diff.shape[0], -1).max(axis=1)
    worst = int(per_frame_max.argmax())
    print(f"  Max pixel diff   : {max_diff}")
    print(f"  Mean pixel diff  : {diff.mean():.6f}")
    print(f"  Pixels identical : {100.0 * (diff == 0).mean():.4f}%")
    print(f"  Worst frame      : {worst} (max diff {int(per_frame_max[worst])})")
    print(f"{'=' * 64}\n")
    return max_diff


# ═══════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════


def test_krea_realtime_cpu_verify():
    logger.info(f"--- original ModularPipeline ({NUM_BLOCKS} blocks) ---")
    frames_org = run_original_pipeline()
    export_to_video(frames_org, "krea_verify_original.mp4", fps=FPS)
    logger.info("saved krea_verify_original.mp4")

    logger.info(f"--- ours: KreaRealtimePipeline ({NUM_BLOCKS} blocks) ---")
    frames_our = run_our_pipeline()
    export_to_video(frames_our, "krea_verify_ours.mp4", fps=FPS)
    logger.info("saved krea_verify_ours.mp4")

    max_diff = compare_videos(frames_org, frames_our)
    assert max_diff <= MAX_PIXEL_DIFF, (
        f"CPU parity max pixel diff {max_diff} > {MAX_PIXEL_DIFF}"
    )


if __name__ == "__main__":
    test_krea_realtime_cpu_verify()
