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

import time
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


def _log(msg: str) -> None:
    """Single-line progress print; flush so it shows up under pytest capture."""
    print(f"[wan22] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Configuration — edit these and re-run.
# ---------------------------------------------------------------------------

MODE = "t2v"  # "t2v" or "i2v"
RESOLUTION = "480p"  # "480p" or "720p"
NUM_STEPS = 40  # denoising steps (bump to 50 for quality check)
GUIDANCE_SCALE = 5.0  # matches diffusers / Wan repo default; CFG on

TT_TEXT_ENCODER = False
TT_VAE_ENCODER = False  # only used when MODE == "i2v"
TT_DIT = True
TT_VAE_DECODER = False

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


def _output_path() -> Path:
    name = f"wan22_{MODE}_{RESOLUTION}_steps{NUM_STEPS}_{_devtag()}.mp4"
    return _OUT_DIR / name


# ---------------------------------------------------------------------------
# Monkey patches
# ---------------------------------------------------------------------------


def _patch_apply_lora_scale() -> None:
    """Make `@apply_lora_scale` a pass-through.

    `diffusers.utils.peft_utils.apply_lora_scale` wraps the DiT forward in
    a helper that calls `scale_lora_layers` + `unscale_lora_layers`, each
    of which is a graph break. This test loads plain weights via
    `load_dit()` – no LoRA adapters exist, so the wrapper is pure
    overhead.
    """
    from diffusers.utils import peft_utils

    def noop_decorator(kwargs_name: str = "joint_attention_kwargs"):
        def decorator(forward_fn):
            return forward_fn

        return decorator

    peft_utils.apply_lora_scale = noop_decorator

    # The WanTransformer3DModel.forward in diffusers is decorated at class
    # definition time, so the patch above only affects future imports.
    # Rebind the already-decorated forward to the underlying function.
    from diffusers.models.transformers.transformer_wan import WanTransformer3DModel

    wrapped = WanTransformer3DModel.forward
    underlying = getattr(wrapped, "__wrapped__", None)
    if underlying is not None:
        WanTransformer3DModel.forward = underlying


def _disable_tt_torch_function_override() -> None:
    """Pop `TorchFunctionOverride` off the global TorchFunctionMode stack.

    `tt_torch/torch_overrides.py` enters a `TorchFunctionMode` at import
    time. Its body is gated by `torch.compiler.is_compiling()` and does
    nothing on the compile path, but the mode still sits on dynamo's
    function-mode stack and forces a `__torch_function__` trace for every
    matmul / linear encountered during tracing.
    """
    try:
        import tt_torch.torch_overrides as overrides
    except ImportError:
        return

    mode = getattr(overrides, "torch_function_override", None)
    if mode is None:
        return

    try:
        mode.__exit__(None, None, None)
    except Exception:
        # Mode wasn't on the stack or was already popped – ignore.
        pass


_patch_apply_lora_scale()
_disable_tt_torch_function_override()

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

        cfg_tag = " +cfg" if negative_embeds is not None else ""
        _log(
            f"  step {i + 1}/{total_steps} "
            f"t={float(t):.1f}{cfg_tag} ({time.perf_counter() - step_start:.1f}s)"
        )

    return latents_fp32


def _decode_and_save(vae, latents: torch.Tensor, out_path: Path) -> None:
    import numpy as np
    from diffusers.image_processor import VaeImageProcessor
    from diffusers.utils import export_to_video
    from diffusers.video_processor import VideoProcessor

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
    pixels = run_component(
        decoder_wrapper,
        [latents_unscaled.to(torch.bfloat16)],
        on_tt=TT_VAE_DECODER,
        shard_spec_fn=(lambda m: shard_vae_decoder_specs(m.vae)),
    )

    video_processor = VideoProcessor(
        vae_scale_factor=VaeImageProcessor().vae_scale_factor
    )
    frames_np = video_processor.postprocess_video(pixels.float(), output_type="np")[
        0
    ]  # (T, H, W, C), [0,1]
    frames_list = [(frame * 255.0).round().astype("uint8") for frame in frames_np]

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

    t_total = time.perf_counter()
    _log(
        f"mode={MODE} res={RESOLUTION} steps={NUM_STEPS} "
        f"guidance={GUIDANCE_SCALE} device={_devtag()} seed={SEED}"
    )
    _log(f"prompt: {PROMPT!r}")
    if GUIDANCE_SCALE > 1.0:
        _log(f"negative: {NEGATIVE_PROMPT!r}")

    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)
    shapes = RESOLUTIONS[RESOLUTION]
    _log(
        f"shapes: video={shapes['video_h']}x{shapes['video_w']} "
        f"frames={shapes['num_frames']} "
        f"latent={shapes['latent_frames']}x{shapes['latent_h']}x{shapes['latent_w']}"
    )

    t = time.perf_counter()
    tokenizer = load_tokenizer()
    text_encoder = UMT5Wrapper(load_umt5()).eval().bfloat16()
    _log(f"text encoder loaded ({time.perf_counter() - t:.1f}s)")

    t = time.perf_counter()
    prompt_embeds = _encode_prompt(tokenizer, text_encoder, PROMPT)
    assert prompt_embeds.shape == (1, 512, 4096)
    _log(
        f"prompt encoded ({time.perf_counter() - t:.1f}s) "
        f"shape={tuple(prompt_embeds.shape)} dtype={prompt_embeds.dtype}"
    )

    if GUIDANCE_SCALE > 1.0:
        t = time.perf_counter()
        negative_embeds = _encode_prompt(tokenizer, text_encoder, NEGATIVE_PROMPT)
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
        image_latent = _encode_first_frame(vae_for_encode, image)
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
    )
    _log(f"denoise done ({time.perf_counter() - t:.1f}s)")
    del dit_wrapper

    t = time.perf_counter()
    vae = load_vae()
    _log(f"VAE loaded ({time.perf_counter() - t:.1f}s), decoding...")
    t = time.perf_counter()
    _decode_and_save(vae, latents, out_path)
    size_kb = out_path.stat().st_size / 1024
    _log(
        f"decode+save done ({time.perf_counter() - t:.1f}s) "
        f"output={out_path.name} ({size_kb:.0f} KB)"
    )
    _log(f"total: {time.perf_counter() - t_total:.1f}s")


# ---------------------------------------------------------------------------
# Reference test — diffusers.WanPipeline end-to-end for side-by-side
# comparison against the hand-rolled pipeline above. Same config constants
# (PROMPT, SEED, NUM_STEPS, ...) so outputs are directly comparable.
# ---------------------------------------------------------------------------


def _hf_output_path() -> Path:
    name = f"wan22_hf_{MODE}_{RESOLUTION}_steps{NUM_STEPS}.mp4"
    return _OUT_DIR / name


def test_wan22_hf_pipeline():
    """Reference run via diffusers.WanPipeline. CPU only, t2v only."""
    assert MODE == "t2v", "HF reference test currently supports MODE='t2v' only"

    out_path = _hf_output_path()
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    _run_hf_pipeline(out_path)

    assert out_path.exists(), f"Output video not produced at {out_path}"
    assert out_path.stat().st_size > 0, f"Output video is empty: {out_path}"


def _run_hf_pipeline(out_path: Path) -> None:
    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.utils import export_to_video

    from .shared import MODEL_ID

    t_total = time.perf_counter()
    _log(
        f"[hf] mode={MODE} res={RESOLUTION} steps={NUM_STEPS} "
        f"guidance={GUIDANCE_SCALE} seed={SEED}"
    )
    _log(f"[hf] prompt: {PROMPT!r}")
    _log(f"[hf] negative: {NEGATIVE_PROMPT!r}")

    shapes = RESOLUTIONS[RESOLUTION]
    _log(
        f"[hf] shapes: video={shapes['video_h']}x{shapes['video_w']} "
        f"frames={shapes['num_frames']}"
    )

    # Follow the example in diffusers/pipelines/wan/pipeline_wan.py docstring:
    # VAE in float32, transformer + text encoder in bf16.
    t = time.perf_counter()
    vae = AutoencoderKLWan.from_pretrained(
        MODEL_ID, subfolder="vae", torch_dtype=torch.float32
    )
    pipe = WanPipeline.from_pretrained(MODEL_ID, vae=vae, torch_dtype=torch.bfloat16)
    _log(
        f"[hf] pipeline loaded ({time.perf_counter() - t:.1f}s) "
        f"scheduler={type(pipe.scheduler).__name__}"
    )

    generator = torch.Generator().manual_seed(SEED)

    t = time.perf_counter()
    output = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        height=shapes["video_h"],
        width=shapes["video_w"],
        num_frames=shapes["num_frames"],
        num_inference_steps=NUM_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator,
    ).frames[0]
    _log(f"[hf] pipeline call done ({time.perf_counter() - t:.1f}s)")

    t = time.perf_counter()
    export_to_video(output, str(out_path), fps=FPS)
    size_kb = out_path.stat().st_size / 1024
    _log(
        f"[hf] saved ({time.perf_counter() - t:.1f}s) "
        f"output={out_path.name} ({size_kb:.0f} KB)"
    )
    _log(f"[hf] total: {time.perf_counter() - t_total:.1f}s")
