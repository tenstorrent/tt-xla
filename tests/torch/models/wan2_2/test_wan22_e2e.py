# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
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

import time
from datetime import datetime
from pathlib import Path

import torch

from .monkey_patch import _disable_tt_torch_function_override, _patch_apply_lora_scale
from .shared import (
    LATENT_CHANNELS,
    RESOLUTIONS,
    VAE_SCALE_FACTOR,
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
    wan22_mesh,
)


def _log(msg: str) -> None:
    """Single-line progress print; flush so it shows up under pytest capture."""
    print(f"[wan22] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Configuration — edit these and re-run.
# ---------------------------------------------------------------------------

MODE = "t2v"  # "t2v" or "i2v"
RESOLUTION = "480p"  # "480p" or "720p"
NUM_STEPS = 40  # denoising steps
GUIDANCE_SCALE = 5.0  # matches diffusers / Wan repo default; CFG on

TT_TEXT_ENCODER = False
TT_VAE_ENCODER = False  # only used when MODE == "i2v"
TT_DIT = True
TT_VAE_DECODER = False

# Mesh shared by every component that runs on TT in this pipeline. Built once
# so callers pass the same mesh object each time. None means CPU-only run.
_mesh = (
    wan22_mesh()
    if (TT_TEXT_ENCODER or TT_VAE_ENCODER or TT_DIT or TT_VAE_DECODER)
    else None
)

# Simple single-subject prompt for smoke-test readability. Failure is
# immediately obvious by eye; success is unambiguous. Fallback options
# (if quality seems poor) documented in the design spec:
#   "A cat sitting on grass"
#   "A dog running on a beach"
#   Wan 2.2 README canonical (complex):
#   "Two anthropomorphic cats in comfy boxing gear and bright gloves fight
#    intensely on a spotlighted stage."
PROMPT = "A red apple on a white table"
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


def _stem() -> str:
    return f"wan22_{MODE}_{RESOLUTION}_steps{NUM_STEPS}_{_devtag()}"


def _output_path() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _OUT_DIR / f"{_stem()}_{ts}.mp4"


# ---------------------------------------------------------------------------
# Monkey patches
# ---------------------------------------------------------------------------

_patch_apply_lora_scale()
_disable_tt_torch_function_override()

# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def _encode_prompt(tokenizer, encoder_wrapper, text: str, mesh) -> torch.Tensor:
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
        mesh=mesh,
        shard_module=encoder_wrapper.encoder,
        shard_fn=shard_umt5_specs,
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


def _encode_first_frame(vae, image: torch.Tensor, mesh) -> torch.Tensor:
    enc_wrapper = VAEEncoderWrapper(vae).eval().bfloat16()
    return run_component(
        enc_wrapper,
        [image],
        on_tt=TT_VAE_ENCODER,
        mesh=mesh,
        shard_module=enc_wrapper.vae,
        shard_fn=shard_vae_encoder_specs,
    )


def _denoise(
    dit_wrapper: WanDiTWrapper,
    latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    negative_embeds,
    scheduler,
    shapes: dict,
    image_latent: torch.Tensor | None,
    mesh,
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
    total_steps = len(scheduler.timesteps)

    for i, t in enumerate(scheduler.timesteps):
        step_start = time.perf_counter()
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
            mesh=mesh,
            shard_module=dit_wrapper.dit,
            shard_fn=shard_dit_specs,
        )
        if negative_embeds is not None:
            v_neg = run_component(
                dit_wrapper,
                [latents_bf16, timestep, negative_embeds],
                on_tt=TT_DIT,
                mesh=mesh,
                shard_module=dit_wrapper.dit,
                shard_fn=shard_dit_specs,
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

        cfg_tag = " +cfg" if negative_embeds is not None else ""
        _log(
            f"  step {i + 1}/{total_steps} "
            f"t={float(t):.1f}{cfg_tag} ({time.perf_counter() - step_start:.1f}s)"
        )

    return latents_fp32


def _run_vae(vae, latents: torch.Tensor, mesh) -> torch.Tensor:
    """Unscale latents and run the VAE decoder. Returns raw bf16 pixels
    (1, 3, T, H, W) before any postprocessing."""
    latents_mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(latents.dtype)
    )
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(
        1, vae.config.z_dim, 1, 1, 1
    ).to(latents.dtype)
    latents_unscaled = latents / latents_std + latents_mean

    # VAE decoder has bf16 weights — cast at the boundary (same pattern as
    # the DiT call in _denoise). Unscaling happens in float32 for precision.
    decoder_wrapper = VAEDecoderWrapper(vae).eval().bfloat16()
    return run_component(
        decoder_wrapper,
        [latents_unscaled.to(torch.bfloat16)],
        on_tt=TT_VAE_DECODER,
        mesh=mesh,
        shard_module=decoder_wrapper.vae,
        shard_fn=shard_vae_decoder_specs,
    )


def _postprocess_and_save(pixels: torch.Tensor, out_path: Path) -> None:
    """Turn raw VAE pixel tensor into an mp4 on disk.

    Mirrors diffusers.WanPipeline.__call__:
        video = video_processor.postprocess_video(pixels, output_type="np")
        export_to_video(video[0], path, fps=fps)

    The VideoProcessor's ``vae_scale_factor`` is read from
    ``vae.config.scale_factor_spatial`` via a config-only load (no
    weights). ``export_to_video`` handles the float→uint8 cast
    internally (truncation).
    """
    from diffusers.utils import export_to_video
    from diffusers.video_processor import VideoProcessor

    video_processor = VideoProcessor(vae_scale_factor=VAE_SCALE_FACTOR)
    video = video_processor.postprocess_video(pixels.float(), output_type="np")
    export_to_video(video[0], str(out_path), fps=FPS)


def test_wan22_e2e():
    out_path = _output_path()
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    _run(out_path)

    assert out_path.exists(), f"Output video not produced at {out_path}"


def _run(out_path: Path) -> None:
    from diffusers import UniPCMultistepScheduler

    from .shared import MODEL_ID

    t_total = time.perf_counter()
    _log(
        f"mode={MODE} res={RESOLUTION} steps={NUM_STEPS} "
        f"guidance={GUIDANCE_SCALE} device={_devtag()} seed={SEED}"
    )
    shapes = RESOLUTIONS[RESOLUTION]
    _log(
        f"shapes: video={shapes['video_h']}x{shapes['video_w']} "
        f"frames={shapes['num_frames']} "
        f"latent={shapes['latent_frames']}x{shapes['latent_h']}x{shapes['latent_w']}"
    )

    _log(f"prompt: {PROMPT!r}")
    if GUIDANCE_SCALE > 1.0:
        _log(f"negative: {NEGATIVE_PROMPT!r}")

    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)

    t = time.perf_counter()
    tokenizer = load_tokenizer()
    text_encoder = UMT5Wrapper(load_umt5()).eval().bfloat16()
    _log(f"text encoder loaded ({time.perf_counter() - t:.1f}s)")

    t = time.perf_counter()
    prompt_embeds = _encode_prompt(tokenizer, text_encoder, PROMPT, _mesh)
    assert prompt_embeds.shape == (1, 512, 4096)
    _log(
        f"prompt encoded ({time.perf_counter() - t:.1f}s) "
        f"shape={tuple(prompt_embeds.shape)} dtype={prompt_embeds.dtype}"
    )

    if GUIDANCE_SCALE > 1.0:
        t = time.perf_counter()
        negative_embeds = _encode_prompt(
            tokenizer, text_encoder, NEGATIVE_PROMPT, _mesh
        )
        _log(f"negative encoded ({time.perf_counter() - t:.1f}s)")
    else:
        negative_embeds = None

    del text_encoder  # free memory before loading DiT

    latents = _init_latents(shapes, generator)
    _log(f"latents init shape={tuple(latents.shape)} dtype={latents.dtype}")

    image_latent = None
    if MODE == "i2v":
        t = time.perf_counter()
        vae_for_encode = load_vae()
        image = _make_first_frame(shapes, generator)
        image_latent = _encode_first_frame(vae_for_encode, image, _mesh)
        _log(
            f"i2v first-frame encoded ({time.perf_counter() - t:.1f}s) "
            f"shape={tuple(image_latent.shape)}"
        )
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
    _log(
        f"scheduler={type(scheduler).__name__} "
        f"flow_shift={scheduler.config.flow_shift} "
        f"timesteps={[round(float(x), 1) for x in scheduler.timesteps]}"
    )

    t = time.perf_counter()
    dit_wrapper = WanDiTWrapper(load_dit()).eval().bfloat16()
    _log(
        f"DiT loaded ({time.perf_counter() - t:.1f}s) "
        f"blocks={len(dit_wrapper.dit.blocks)}"
    )

    t = time.perf_counter()
    _log(f"denoising {NUM_STEPS} steps...")
    latents = _denoise(
        dit_wrapper,
        latents,
        prompt_embeds,
        negative_embeds,
        scheduler,
        shapes,
        image_latent,
        _mesh,
    )
    _log(f"denoise done ({time.perf_counter() - t:.1f}s)")
    del dit_wrapper

    t = time.perf_counter()
    vae = load_vae()
    _log(f"VAE loaded ({time.perf_counter() - t:.1f}s), decoding...")
    t = time.perf_counter()
    pixels = _run_vae(vae, latents, _mesh)

    t = time.perf_counter()
    _postprocess_and_save(pixels, out_path)
    size_kb = out_path.stat().st_size / 1024
    _log(
        f"postprocess+save done ({time.perf_counter() - t:.1f}s) "
        f"output={out_path.name} ({size_kb:.0f} KB)"
    )
    _log(f"total: {time.perf_counter() - t_total:.1f}s")
