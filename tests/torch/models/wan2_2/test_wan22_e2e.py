# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Wan 2.2 TI2V-5B end-to-end smoke test.

Runs the full pipeline (text + optional first-frame image -> video) and
writes a playable .mp4 under `tests/torch/models/wan2_2/generated/`.
No quality assertions; the only check is that the output file exists
and is non-empty.

Edit the module-level constants below to change behavior. Each TT_*
flag routes one component onto TT hardware; all default to CPU.
"""

from pathlib import Path

import torch

from .shared import (
    LATENT_CHANNELS,
    RESOLUTIONS,
    UMT5Wrapper,
    VAEDecoderWrapper,
    VAEEncoderWrapper,
    WanDiTWrapper,
    load_dit,
    load_tokenizer,
    load_umt5,
    load_vae,
    prompt_clean,
    run_component,
    shard_dit_specs,
    shard_umt5_specs,
    shard_vae_decoder_specs,
    shard_vae_encoder_specs,
)

# ---------------------------------------------------------------------------
# Configuration — edit these and re-run.
# ---------------------------------------------------------------------------

MODE = "t2v"  # "t2v" or "i2v"
RESOLUTION = "480p"  # "480p" or "720p"
NUM_STEPS = 4  # denoising steps
GUIDANCE_SCALE = 1.0  # 1.0 disables classifier-free guidance

TT_TEXT_ENCODER = False
TT_VAE_ENCODER = False  # only used when MODE == "i2v"
TT_DIT = False
TT_VAE_DECODER = False

PROMPT = (
    "A gentle ocean wave rolling across a quiet sandy beach at sunrise, "
    "cinematic, high detail"
)
NEGATIVE_PROMPT = "low quality, blurry, distorted, watermark, text"
SEED = 42
FPS = 16

# ---------------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------------

_OUT_DIR = Path(__file__).parent / "generated"


def _devtag() -> str:
    flags = {
        "textenc": TT_TEXT_ENCODER,
        "vaeenc": TT_VAE_ENCODER and MODE == "i2v",
        "dit": TT_DIT,
        "vaedec": TT_VAE_DECODER,
    }
    on = [name for name, v in flags.items() if v]
    if not on:
        return "cpu"
    if len(on) == 4 or (MODE == "t2v" and len(on) == 3 and "vaeenc" not in on):
        return "tt-all"
    return "tt-" + "-".join(on)


def _output_path() -> Path:
    name = f"wan22_{MODE}_{RESOLUTION}_steps{NUM_STEPS}_{_devtag()}.mp4"
    return _OUT_DIR / name


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def _encode_prompt(tokenizer, encoder_wrapper, text: str) -> torch.Tensor:
    ids = tokenizer(
        prompt_clean(text),
        padding="max_length",
        max_length=512,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids = ids["input_ids"]
    attention_mask = ids["attention_mask"]
    seq_lens = attention_mask.gt(0).sum(dim=1).long()

    embeds = run_component(
        encoder_wrapper,
        [input_ids, attention_mask],
        on_tt=TT_TEXT_ENCODER,
        shard_spec_fn=(lambda m: shard_umt5_specs(m.encoder)),
    )

    # Zero out positions past the real token length. UMT5 produces non-zero
    # values at pad positions via residuals/FFN; DiT cross-attention is
    # trained expecting zero there. Mirrors diffusers _get_t5_prompt_embeds
    # (pipeline_wan.py:186-190) and Wan repo T5EncoderModel.__call__.
    max_len = embeds.shape[1]
    trimmed = [e[:n] for e, n in zip(embeds, seq_lens)]
    return torch.stack(
        [torch.cat([e, e.new_zeros(max_len - e.size(0), e.size(1))]) for e in trimmed],
        dim=0,
    )


def _init_latents(shapes: dict, generator: torch.Generator) -> torch.Tensor:
    return torch.randn(
        1,
        LATENT_CHANNELS,
        shapes["latent_frames"],
        shapes["latent_h"],
        shapes["latent_w"],
        dtype=torch.float32,
        generator=generator,
    )


def _make_first_frame(shapes: dict, generator: torch.Generator) -> torch.Tensor:
    """Return a deterministic RGB first-frame image as (1, 3, 1, H, W) in [-1, 1]."""
    h, w = shapes["video_h"], shapes["video_w"]
    x = torch.randn(1, 3, 1, h, w, dtype=torch.bfloat16, generator=generator)
    # Clamp to a sane image-like range; values outside [-1, 1] confuse the VAE.
    return x.clamp(-1.0, 1.0)


def _encode_first_frame(vae, image: torch.Tensor) -> torch.Tensor:
    enc_wrapper = VAEEncoderWrapper(vae).eval().bfloat16()
    return run_component(
        enc_wrapper,
        [image],
        on_tt=TT_VAE_ENCODER,
        shard_spec_fn=(lambda m: shard_vae_encoder_specs(m.vae)),
    )


def _denoise(
    dit_wrapper: WanDiTWrapper,
    latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    negative_embeds,
    scheduler,
    shapes: dict,
    image_latent: torch.Tensor | None,
) -> torch.Tensor:
    t_dim, h_dim, w_dim = (
        shapes["latent_frames"],
        shapes["latent_h"],
        shapes["latent_w"],
    )
    num_patches = t_dim * (h_dim // 2) * (w_dim // 2)

    # For i2v (Wan 2.2 TI2V expand_timesteps=True): first latent frame is
    # the conditioning image and should receive timestep 0, not t. Mirrors
    # diffusers pipeline_wan_i2v.py:757-764 — first_frame_mask[0][0][:,::2,::2]*t.
    if image_latent is not None:
        ts_mask = torch.ones(t_dim, h_dim // 2, w_dim // 2, dtype=torch.float32)
        ts_mask[0] = 0
        ts_mask = ts_mask.flatten()  # shape (num_patches,)
    else:
        ts_mask = None

    latents_fp32 = latents  # keep float32 authoritative copy

    for t in scheduler.timesteps:
        if ts_mask is not None:
            timestep = (ts_mask * float(t)).unsqueeze(0)
        else:
            timestep = torch.full((1, num_patches), float(t), dtype=torch.float32)

        # DiT has bf16 weights — cast latents at the boundary; keep the
        # float32 copy for scheduler.step precision.
        latents_bf16 = latents_fp32.to(torch.bfloat16)

        v_pos = run_component(
            dit_wrapper,
            [latents_bf16, timestep, prompt_embeds],
            on_tt=TT_DIT,
            shard_spec_fn=(lambda m: shard_dit_specs(m.dit)),
        )
        if negative_embeds is not None:
            v_neg = run_component(
                dit_wrapper,
                [latents_bf16, timestep, negative_embeds],
                on_tt=TT_DIT,
                shard_spec_fn=(lambda m: shard_dit_specs(m.dit)),
            )
            velocity = v_neg + GUIDANCE_SCALE * (v_pos - v_neg)
        else:
            velocity = v_pos

        # Cast velocity back to float32 for numerically-careful scheduler.step.
        latents_fp32 = scheduler.step(
            velocity.to(torch.float32), t, latents_fp32
        ).prev_sample

        if image_latent is not None:
            # Keep the conditioning frame fixed across iterations.
            latents_fp32[:, :, 0:1, :, :] = image_latent.to(torch.float32)

    return latents_fp32


def _decode_and_save(vae, latents: torch.Tensor, out_path: Path) -> None:
    import numpy as np
    from diffusers.utils import export_to_video

    latents_mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(latents.dtype)
    )
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(
        1, vae.config.z_dim, 1, 1, 1
    ).to(latents.dtype)
    latents_unscaled = latents / latents_std + latents_mean

    decoder_wrapper = VAEDecoderWrapper(vae).eval().bfloat16()
    pixels = run_component(
        decoder_wrapper,
        [latents_unscaled],
        on_tt=TT_VAE_DECODER,
        shard_spec_fn=(lambda m: shard_vae_decoder_specs(m.vae)),
    )

    pixels = pixels.float().clamp(-1.0, 1.0)
    pixels = (pixels + 1.0) / 2.0
    # (1, 3, T, H, W) -> (T, H, W, 3)
    frames = pixels[0].permute(1, 2, 3, 0).contiguous().cpu().numpy()
    frames = (frames * 255.0).round().astype(np.uint8)
    frames_list = [frames[i] for i in range(frames.shape[0])]

    export_to_video(frames_list, str(out_path), fps=FPS)


def test_wan22_e2e():
    out_path = _output_path()
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    _run(out_path)

    assert out_path.exists(), f"Output video not produced at {out_path}"
    assert out_path.stat().st_size > 0, f"Output video is empty: {out_path}"


def _run(out_path: Path) -> None:
    from diffusers import UniPCMultistepScheduler

    from .shared import MODEL_ID

    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)
    shapes = RESOLUTIONS[RESOLUTION]

    tokenizer = load_tokenizer()
    text_encoder = UMT5Wrapper(load_umt5()).eval().bfloat16()

    prompt_embeds = _encode_prompt(tokenizer, text_encoder, PROMPT)
    assert prompt_embeds.shape == (1, 512, 4096)

    if GUIDANCE_SCALE > 1.0:
        negative_embeds = _encode_prompt(tokenizer, text_encoder, NEGATIVE_PROMPT)
    else:
        negative_embeds = None

    del text_encoder  # free memory before loading DiT

    latents = _init_latents(shapes, generator)

    image_latent = None
    if MODE == "i2v":
        vae_for_encode = load_vae()
        image = _make_first_frame(shapes, generator)
        image_latent = _encode_first_frame(vae_for_encode, image)
        assert image_latent.shape == (
            1,
            LATENT_CHANNELS,
            1,
            shapes["latent_h"],
            shapes["latent_w"],
        ), image_latent.shape
        latents[:, :, 0:1, :, :] = image_latent
        del vae_for_encode

    scheduler = UniPCMultistepScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    scheduler.set_timesteps(num_inference_steps=NUM_STEPS)

    dit_wrapper = WanDiTWrapper(load_dit()).eval().bfloat16()
    latents = _denoise(
        dit_wrapper,
        latents,
        prompt_embeds,
        negative_embeds,
        scheduler,
        shapes,
        image_latent,
    )
    del dit_wrapper

    vae = load_vae()
    _decode_and_save(vae, latents, out_path)
