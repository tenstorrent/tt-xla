# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Compare rewritten WanT2VPipeline against forge-style DiffusionPipeline.from_pretrained.

Tests two loading styles and two dtypes:
  1. float32: forge-style vs rewritten (component-by-component)
  2. bfloat16: forge-style vs rewritten

Usage:
    cd examples/pytorch
    pytest test_wan_forge_comparison.py -v -s
"""

import time

import pytest
import torch
from diffusers import DiffusionPipeline, UniPCMultistepScheduler
from diffusers.video_processor import VideoProcessor

MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
SEED = 42
PROMPT = "A cat sitting on a sunny windowsill"
NEGATIVE_PROMPT = ""
NUM_STEPS = 2
GUIDANCE_SCALE = 5.0
MAX_SEQ_LEN = 512
DEVICE = "cpu"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _run_forge_pipeline(dtype):
    """Load via DiffusionPipeline.from_pretrained (same as forge ModelLoader) and run."""
    print(f"\n  [forge] Loading DiffusionPipeline at {dtype}...")
    t0 = time.time()
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map=DEVICE,
        low_cpu_mem_usage=True,
    )
    if dtype is not None:
        pipe = pipe.to(dtype=dtype)
    print(f"  [forge] Loaded in {time.time() - t0:.1f}s  type={type(pipe).__name__}")

    generator = torch.Generator(device="cpu").manual_seed(SEED)

    t0 = time.time()
    result = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        guidance_scale=GUIDANCE_SCALE,
        num_inference_steps=NUM_STEPS,
        generator=generator,
        max_sequence_length=MAX_SEQ_LEN,
        output_type="latent",
    )
    elapsed = time.time() - t0

    latents = result.frames
    print(f"  [forge] shape={latents.shape}  dtype={latents.dtype}  time={elapsed:.1f}s")
    return latents, pipe


def _run_rewritten_pipeline(dtype, components=None):
    """Load rewritten WanT2VPipeline (component-by-component or inject) and run."""
    from wan_t2v_pipeline import WanConfig, WanT2VPipeline

    config = WanConfig(device=DEVICE, dtype=dtype)
    pipe = WanT2VPipeline(config=config)

    if components is not None:
        pipe.text_encoder = components["text_encoder"]
        pipe.transformer = components["transformer"]
        pipe.vae = components["vae"]
        pipe.tokenizer = components["tokenizer"]
        pipe.scheduler = UniPCMultistepScheduler.from_pretrained(
            MODEL_ID, subfolder="scheduler"
        )
        pipe.vae_scale_factor_temporal = pipe.vae.config.scale_factor_temporal
        pipe.vae_scale_factor_spatial = pipe.vae.config.scale_factor_spatial
        pipe.video_processor = VideoProcessor(
            vae_scale_factor=pipe.vae_scale_factor_spatial
        )
        print(f"  [rewritten] Using injected components at {dtype}")
    else:
        print(f"\n  [rewritten] Loading components at {dtype}...")
        t0 = time.time()
        pipe.setup()
        print(f"  [rewritten] Loaded in {time.time() - t0:.1f}s")

    t0 = time.time()
    latents = pipe.generate(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        guidance_scale=GUIDANCE_SCALE,
        num_inference_steps=NUM_STEPS,
        seed=SEED,
        max_sequence_length=MAX_SEQ_LEN,
        output_type="latent",
    )
    elapsed = time.time() - t0

    print(f"  [rewritten] shape={latents.shape}  dtype={latents.dtype}  time={elapsed:.1f}s")
    return latents


def _compare(name, a, b, atol=1e-5, rtol=1e-5):
    """Assert shapes match and values are close; print stats."""
    print(f"\n  --- {name} ---")
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"
    assert a.dtype == b.dtype, f"Dtype mismatch: {a.dtype} vs {b.dtype}"

    abs_diff = (a.float() - b.float()).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()
    print(f"  Max  abs diff: {max_diff:.2e}")
    print(f"  Mean abs diff: {mean_diff:.2e}")

    is_close = torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol)
    print(f"  allclose(atol={atol}): {is_close}")
    return is_close, max_diff


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------
@pytest.fixture(scope="module")
def forge_f32_result():
    """Run forge-style pipeline at float32, return (latents, pipe_components)."""
    print("\n" + "=" * 60)
    print("FORGE-STYLE PIPELINE — float32")
    print("=" * 60)
    latents, pipe = _run_forge_pipeline(torch.float32)
    components = {
        "text_encoder": pipe.text_encoder,
        "transformer": pipe.transformer,
        "vae": pipe.vae,
        "tokenizer": pipe.tokenizer,
    }
    return latents, components


@pytest.fixture(scope="module")
def rewritten_f32_result(forge_f32_result):
    """Run rewritten pipeline at float32 using forge-loaded components."""
    print("\n" + "=" * 60)
    print("REWRITTEN PIPELINE — float32 (forge components)")
    print("=" * 60)
    _, components = forge_f32_result
    return _run_rewritten_pipeline(torch.float32, components=components)


@pytest.fixture(scope="module")
def forge_bf16_result():
    """Run forge-style pipeline at bfloat16."""
    print("\n" + "=" * 60)
    print("FORGE-STYLE PIPELINE — bfloat16")
    print("=" * 60)
    latents, pipe = _run_forge_pipeline(torch.bfloat16)
    components = {
        "text_encoder": pipe.text_encoder,
        "transformer": pipe.transformer,
        "vae": pipe.vae,
        "tokenizer": pipe.tokenizer,
    }
    return latents, components


@pytest.fixture(scope="module")
def rewritten_bf16_result(forge_bf16_result):
    """Run rewritten pipeline at bfloat16 using forge-loaded components."""
    print("\n" + "=" * 60)
    print("REWRITTEN PIPELINE — bfloat16 (forge components)")
    print("=" * 60)
    _, components = forge_bf16_result
    return _run_rewritten_pipeline(torch.bfloat16, components=components)


# ------------------------------------------------------------------
# Tests — float32 forge vs rewritten
# ------------------------------------------------------------------
def test_forge_vs_rewritten_f32_shapes(forge_f32_result, rewritten_f32_result):
    forge_lat, _ = forge_f32_result
    assert forge_lat.shape == rewritten_f32_result.shape
    assert forge_lat.dtype == rewritten_f32_result.dtype


def test_forge_vs_rewritten_f32_close(forge_f32_result, rewritten_f32_result):
    forge_lat, _ = forge_f32_result
    is_close, max_diff = _compare(
        "forge f32 vs rewritten f32", forge_lat, rewritten_f32_result, atol=1e-5
    )
    assert is_close, f"float32: max diff = {max_diff:.2e}"


def test_forge_vs_rewritten_f32_exact(forge_f32_result, rewritten_f32_result):
    forge_lat, _ = forge_f32_result
    exact = torch.equal(forge_lat, rewritten_f32_result)
    if not exact:
        diff = (forge_lat - rewritten_f32_result).abs().max().item()
        pytest.xfail(f"Not bit-exact (max diff = {diff:.2e})")


# ------------------------------------------------------------------
# Tests — bfloat16 forge vs rewritten
# ------------------------------------------------------------------
def test_forge_vs_rewritten_bf16_shapes(forge_bf16_result, rewritten_bf16_result):
    forge_lat, _ = forge_bf16_result
    assert forge_lat.shape == rewritten_bf16_result.shape


def test_forge_vs_rewritten_bf16_close(forge_bf16_result, rewritten_bf16_result):
    forge_lat, _ = forge_bf16_result
    is_close, max_diff = _compare(
        "forge bf16 vs rewritten bf16",
        forge_lat, rewritten_bf16_result,
        atol=1e-3, rtol=1e-3,
    )
    assert is_close, f"bfloat16: max diff = {max_diff:.2e}"


def test_forge_vs_rewritten_bf16_exact(forge_bf16_result, rewritten_bf16_result):
    forge_lat, _ = forge_bf16_result
    exact = torch.equal(forge_lat.float(), rewritten_bf16_result.float())
    if not exact:
        diff = (forge_lat.float() - rewritten_bf16_result.float()).abs().max().item()
        pytest.xfail(f"bf16 not bit-exact (max diff = {diff:.2e})")


# ------------------------------------------------------------------
# Cross-dtype test — informational
# ------------------------------------------------------------------
# def test_f32_vs_bf16_drift(forge_f32_result, forge_bf16_result):
#     """Measure how much bfloat16 drifts from float32 (informational, xfail ok)."""
#     f32_lat, _ = forge_f32_result
#     bf16_lat, _ = forge_bf16_result
#     abs_diff = (f32_lat.float() - bf16_lat.float()).abs()
#     max_diff = abs_diff.max().item()
#     mean_diff = abs_diff.mean().item()
#     print(f"\n  f32 vs bf16 drift: max={max_diff:.2e}  mean={mean_diff:.2e}")
#     if max_diff > 1.0:
#         pytest.xfail(f"Expected large f32-vs-bf16 drift (max={max_diff:.2e})")
