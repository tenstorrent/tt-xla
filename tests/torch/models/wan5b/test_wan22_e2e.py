# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Wan 2.2 TI2V-5B end-to-end smoke test.

Runs the full pipeline (text + optional first-frame image -> video) and
writes a playable .mp4 under `tests/torch/models/wan5b/generated/`.
No assertions: the test passes if the full pipeline runs to completion
(any compilation or runtime failure raises and fails the test).

Edit the module-level constants below to change behavior. Each TT_*
flag routes one component onto TT hardware; all default to CPU.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pytest
import torch
from diffusers import UniPCMultistepScheduler

from .monkey_patch import (
    _patch_wan_resample_avoid_4d_fold,
    _patch_wan_resample_rep_sentinel,
    _patch_wan_time_embedder_dtype_probe,
    safe_xla_slicing,
    torch_function_override_disabled,
)
from .shared import (
    LATENT_CHANNELS,
    MODEL_ID,
    RESOLUTIONS,
    VAE_SCALE_FACTOR,
    UMT5Wrapper,
    VAEDecoderWrapper,
    VAEEncoderWrapper,
    WanDiTWrapper,
    load_dit,
    load_first_frame_image,
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

# ---------------------------------------------------------------------------
# Configuration — edit these and re-run.
# ---------------------------------------------------------------------------


SEED = 42
NUM_STEPS = 40  # denoising steps
GUIDANCE_SCALE = 5.0  # matches diffusers / Wan repo default; CFG on
FPS = 16

PROMPT_T2V = "A red apple on a white table"
NEG_T2V = "low quality, blurry, distorted, watermark, text"
PROMPT_I2V = "smiling and jumping into camera for a hug"
NEG_I2V = "distorted face, bad hands, extra limbs, blurry, low quality"

TT_TEXT_ENCODER = True
TT_VAE_ENCODER = True  # only used when mode == "i2v"
TT_DIT = True
TT_VAE_DECODER = True

# True uses the runtime-detected mesh from wan22_mesh(). Only affects _mesh.
MULTI_CHIP = True


def _build_mesh():
    if not (TT_TEXT_ENCODER or TT_VAE_ENCODER or TT_DIT or TT_VAE_DECODER):
        return None
    if MULTI_CHIP:
        return wan22_mesh()
    return None


# ---------------------------------------------------------------------------
# Per-component perf timings
# ---------------------------------------------------------------------------


@dataclass
class _ComponentTimings:
    cold: float | None = None
    warm: list[float] = field(default_factory=list)


@dataclass
class _Timings:
    text_encoder: _ComponentTimings = field(default_factory=_ComponentTimings)
    dit_pair: _ComponentTimings = field(default_factory=_ComponentTimings)
    vae_encoder: _ComponentTimings = field(default_factory=_ComponentTimings)
    vae_decoder: _ComponentTimings = field(default_factory=_ComponentTimings)


def _record(slot: _ComponentTimings, secs: float) -> None:
    """First call → cold; subsequent calls → warm. Uses ``None`` as sentinel
    so that a genuinely fast (~0 s) first measurement is still recorded
    as cold rather than being treated as 'no measurement yet'."""
    if slot.cold is None:
        slot.cold = secs
    else:
        slot.warm.append(secs)


def _print_perf_summary(t: _Timings) -> None:
    rows = [
        ("text_encoder", t.text_encoder),
        ("DiT (CFG pair)", t.dit_pair),
        ("VAE encoder", t.vae_encoder),
        ("VAE decoder", t.vae_decoder),
    ]
    print()
    print("=" * 78)
    print("PERF SUMMARY  (cold = first call incl. compile; warm = subsequent calls)")
    print("-" * 78)
    print(
        f"  {'component':<18} {'cold (s)':>12} {'warm avg (s)':>14} "
        f"{'warm runs':>11} {'speedup':>10}"
    )
    for label, slot in rows:
        if slot.cold is None and not slot.warm:
            continue  # component not exercised in this run
        cold = slot.cold if slot.cold is not None else 0.0
        warm_avg = sum(slot.warm) / len(slot.warm) if slot.warm else 0.0
        speedup = (cold / warm_avg) if warm_avg > 0 else float("inf")
        speedup_str = f"{speedup:>9.1f}x" if speedup != float("inf") else "       inf"
        print(
            f"  {label:<18} {cold:>12.3f} {warm_avg:>14.3f} "
            f"{len(slot.warm):>11d} {speedup_str}"
        )
    print("=" * 78, flush=True)


# ---------------------------------------------------------------------------
# Components container + builder
# ---------------------------------------------------------------------------


@dataclass
class _Components:
    """Model wrappers and shared resources, built once per test and reused
    across both ``_run`` invocations so warmup-phase compiles are reused on
    the run phase."""

    tokenizer: object
    text_encoder: "UMT5Wrapper"
    dit_wrapper: "WanDiTWrapper"
    decoder_wrapper: "VAEDecoderWrapper"
    vae_encoder: "VAEEncoderWrapper | None"  # only built for i2v
    mesh: object


def _build_components(mode: str) -> _Components:
    text_encoder = UMT5Wrapper(load_umt5()).eval().bfloat16()
    dit_wrapper = WanDiTWrapper(load_dit()).eval().bfloat16()
    decoder_wrapper = VAEDecoderWrapper(load_vae()).eval().bfloat16()
    vae_encoder = (
        VAEEncoderWrapper(load_vae()).eval().bfloat16() if mode == "i2v" else None
    )
    return _Components(
        tokenizer=load_tokenizer(),
        text_encoder=text_encoder,
        dit_wrapper=dit_wrapper,
        decoder_wrapper=decoder_wrapper,
        vae_encoder=vae_encoder,
        mesh=_build_mesh(),
    )


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


# First-frame conditioning image for i2v. Resized via cover-fit + center
# crop to the active resolution. Ignored when mode == "t2v".
IMAGE_PATH = Path(__file__).parent / "reze.jpg"
_OUT_DIR = Path(__file__).parent / "generated"


def _devtag(mode: str) -> str:
    flags = {
        "textenc": TT_TEXT_ENCODER,
        "vaeenc": TT_VAE_ENCODER and mode == "i2v",
        "dit": TT_DIT,
        "vaedec": TT_VAE_DECODER,
    }
    on = [name for name, v in flags.items() if v]
    if not on:
        return "cpu"
    if len(on) == 4 or (mode == "t2v" and len(on) == 3 and "vaeenc" not in on):
        return "tt-all"
    return "tt-" + "-".join(on)


def _stem(mode: str, resolution: str) -> str:
    base = f"wan22_{mode}_{resolution}_steps{NUM_STEPS}_{_devtag(mode)}"
    if mode == "i2v":
        base += f"_{IMAGE_PATH.stem}"
    return base


def _output_path(mode: str, resolution: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _OUT_DIR / f"{_stem(mode, resolution)}_{ts}.mp4"


# ---------------------------------------------------------------------------
# Monkey patches
# ---------------------------------------------------------------------------

_patch_wan_resample_rep_sentinel()
_patch_wan_resample_avoid_4d_fold()
_patch_wan_time_embedder_dtype_probe()

# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def _run_e2e(mode: str, resolution: str, prompt: str, negative_prompt: str) -> None:
    out_path = _output_path(mode, resolution)
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    components = _build_components(mode)
    timings = _Timings()
    try:
        _run(
            components,
            timings,
            out_path,
            mode,
            resolution,
            prompt,
            negative_prompt,
            warmup=True,
        )
        _run(
            components,
            timings,
            out_path,
            mode,
            resolution,
            prompt,
            negative_prompt,
            warmup=False,
        )
    finally:
        _print_perf_summary(timings)


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.qb2_blackhole
@pytest.mark.lb_blackhole
@pytest.mark.bh_galaxy
def test_wan22_t2v_480p_e2e():
    _run_e2e("t2v", "480p", PROMPT_T2V, NEG_T2V)


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.qb2_blackhole
@pytest.mark.lb_blackhole
@pytest.mark.bh_galaxy
def test_wan22_t2v_720p_e2e():
    _run_e2e("t2v", "720p", PROMPT_T2V, NEG_T2V)


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.qb2_blackhole
@pytest.mark.lb_blackhole
@pytest.mark.bh_galaxy
def test_wan22_i2v_480p_e2e():
    _run_e2e("i2v", "480p", PROMPT_I2V, NEG_I2V)


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.qb2_blackhole
@pytest.mark.lb_blackhole
@pytest.mark.bh_galaxy
def test_wan22_i2v_720p_e2e():
    _run_e2e("i2v", "720p", PROMPT_I2V, NEG_I2V)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _log(msg: str) -> None:
    """Single-line progress print; flush so it shows up under pytest capture."""
    print(f"[wan22] {msg}", flush=True)


def _encode_prompt(
    tokenizer,
    encoder_wrapper,
    text: str,
    mesh,
    timings_slot: _ComponentTimings,
) -> torch.Tensor:
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

    t0 = time.perf_counter()
    embeds = run_component(
        encoder_wrapper,
        [input_ids, attention_mask],
        on_tt=TT_TEXT_ENCODER,
        mesh=mesh,
        shard_module=encoder_wrapper.encoder,
        shard_fn=shard_umt5_specs,
    )
    _record(timings_slot, time.perf_counter() - t0)

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


def _denoise(
    dit_wrapper: WanDiTWrapper,
    latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    negative_embeds,
    scheduler,
    shapes: dict,
    image_latent: torch.Tensor | None,
    mesh,
    timings_slot: _ComponentTimings,
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

        time_start = time.perf_counter()
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

        cfg_tag = " +cfg" if negative_embeds is not None else ""
        cfg_pair_secs = time.perf_counter() - time_start
        _log(f"velocity computed ({cfg_pair_secs:.3f}s){cfg_tag}")
        _record(timings_slot, cfg_pair_secs)

        # Cast velocity back to float32 for numerically-careful scheduler.step.
        time_start = time.perf_counter()
        latents_fp32 = scheduler.step(
            velocity.to(torch.float32), t, latents_fp32
        ).prev_sample
        _log(f"scheduler step done ({time.perf_counter() - time_start:.3f}s)")

        if image_latent is not None:
            # Keep the conditioning frame fixed across iterations.
            latents_fp32[:, :, 0:1, :, :] = image_latent.to(torch.float32)

        _log(
            f"  step {i + 1}/{total_steps} "
            f"t={float(t):.1f}{cfg_tag} ({time.perf_counter() - step_start:.3f}s)"
        )

    return latents_fp32


def _run_vae(
    decoder_wrapper: "VAEDecoderWrapper",
    latents: torch.Tensor,
    mesh,
    timings_slot: _ComponentTimings,
) -> torch.Tensor:
    """Unscale latents and run the VAE decoder. Returns raw bf16 pixels
    (1, 3, T, H, W) before any postprocessing."""
    vae = decoder_wrapper.vae
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
    t0 = time.perf_counter()
    with safe_xla_slicing():
        result = run_component(
            decoder_wrapper,
            [latents_unscaled.to(torch.bfloat16)],
            on_tt=TT_VAE_DECODER,
            mesh=mesh,
            shard_module=decoder_wrapper.vae,
            shard_fn=shard_vae_decoder_specs,
        )
    _record(timings_slot, time.perf_counter() - t0)
    return result


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


def _run(
    components: _Components,
    timings: _Timings,
    out_path: Path,
    mode: str,
    resolution: str,
    prompt: str,
    negative_prompt: str,
    warmup: bool,
) -> None:
    try:
        import imageio_ffmpeg  # noqa: F401  # required by export_to_video at end of pipeline
    except ImportError as e:
        raise RuntimeError(
            "imageio-ffmpeg is required to export the final mp4. "
            "Install with: pip install imageio-ffmpeg"
        ) from e

    # Warmup runs a single denoising step to populate the compile cache for
    # all components, then skips the mp4 export. Shapes match the real run
    # so the cache hits on the next call.
    phase = "warmup" if warmup else "run"
    num_steps = 1 if warmup else NUM_STEPS

    t_total = time.perf_counter()
    _log(
        f"phase={phase} mode={mode} res={resolution} steps={num_steps} "
        f"guidance={GUIDANCE_SCALE} device={_devtag(mode)} seed={SEED}"
    )
    shapes = RESOLUTIONS[resolution]
    _log(
        f"shapes: video={shapes['video_h']}x{shapes['video_w']} "
        f"frames={shapes['num_frames']} "
        f"latent={shapes['latent_frames']}x{shapes['latent_h']}x{shapes['latent_w']}"
    )
    _log(f"prompt: {prompt!r}")
    if GUIDANCE_SCALE > 1.0:
        _log(f"negative: {negative_prompt!r}")

    _mesh = components.mesh
    _log(f"mesh={_mesh}")

    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)

    t = time.perf_counter()
    prompt_embeds = _encode_prompt(
        components.tokenizer,
        components.text_encoder,
        prompt,
        _mesh,
        timings.text_encoder,
    )
    _log(
        f"prompt encoded ({time.perf_counter() - t:.3f}s) "
        f"shape={tuple(prompt_embeds.shape)} dtype={prompt_embeds.dtype}"
    )
    assert prompt_embeds.shape == (1, 512, 4096)

    negative_embeds = None
    if GUIDANCE_SCALE > 1.0:
        t = time.perf_counter()
        negative_embeds = _encode_prompt(
            components.tokenizer,
            components.text_encoder,
            negative_prompt,
            _mesh,
            timings.text_encoder,
        )
        _log(f"negative encoded ({time.perf_counter() - t:.3f}s)")

    latents = _init_latents(shapes, generator)
    _log(f"latents init shape={tuple(latents.shape)} dtype={latents.dtype}")

    image_latent = None
    if mode == "i2v":
        assert components.vae_encoder is not None, "i2v requires components.vae_encoder"
        image = load_first_frame_image(IMAGE_PATH, shapes["video_h"], shapes["video_w"])
        _log(
            f"i2v first-frame loaded image={IMAGE_PATH.name} shape={tuple(image.shape)}"
        )

        t = time.perf_counter()
        image_latent = run_component(
            components.vae_encoder,
            [image],
            on_tt=TT_VAE_ENCODER,
            mesh=_mesh,
            shard_module=components.vae_encoder.vae,
            shard_fn=shard_vae_encoder_specs,
        )
        _record(timings.vae_encoder, time.perf_counter() - t)
        _log(
            f"i2v first-frame encoded ({time.perf_counter() - t:.3f}s) "
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

    scheduler = UniPCMultistepScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    scheduler.set_timesteps(num_inference_steps=num_steps)
    _log(
        f"scheduler={type(scheduler).__name__} "
        f"flow_shift={scheduler.config.flow_shift} "
        f"timesteps={[round(float(x), 1) for x in scheduler.timesteps]}"
    )

    _log(f"denoising {num_steps} steps...")

    t = time.perf_counter()
    with torch_function_override_disabled():
        latents = _denoise(
            components.dit_wrapper,
            latents,
            prompt_embeds,
            negative_embeds,
            scheduler,
            shapes,
            image_latent,
            _mesh,
            timings.dit_pair,
        )
        _log(f"denoise done ({time.perf_counter() - t:.3f}s)")

        t = time.perf_counter()
        pixels = _run_vae(
            components.decoder_wrapper,
            latents,
            _mesh,
            timings.vae_decoder,
        )
        _log(f"decoder done ({time.perf_counter() - t:.3f}s)")

    if not warmup:
        _postprocess_and_save(pixels, out_path)
    _log(f"{phase} total: {time.perf_counter() - t_total:.3f}s")
