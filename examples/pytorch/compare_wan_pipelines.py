# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Compare rewritten WanT2VPipeline against original diffusers WanPipeline.

Loads model weights once via a module-scoped fixture, runs both pipelines
with identical settings, and asserts latent outputs match.

Usage:
    cd examples/pytorch
    pytest compare_wan_pipelines.py -v -s
"""

import time

import pytest
import torch
from diffusers import AutoencoderKLWan, WanPipeline, WanTransformer3DModel, UniPCMultistepScheduler
from diffusers.video_processor import VideoProcessor
from transformers import AutoTokenizer, UMT5EncoderModel

MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
SEED = 42
PROMPT = "A cat sitting on a sunny windowsill"
NEGATIVE_PROMPT = ""
NUM_STEPS = 2
GUIDANCE_SCALE = 5.0
MAX_SEQ_LEN = 512
DTYPE = torch.float32
DEVICE = "cpu"


# ------------------------------------------------------------------
# Fixtures – load heavy components once per module
# ------------------------------------------------------------------
@pytest.fixture(scope="module")
def shared_components():
    """Load all model components once, shared across every test in this file."""
    print("\nLoading model components (shared)...")
    t0 = time.time()

    text_encoder = UMT5EncoderModel.from_pretrained(
        MODEL_ID, subfolder="text_encoder",
        torch_dtype=DTYPE, device_map=DEVICE, low_cpu_mem_usage=True,
    )
    transformer = WanTransformer3DModel.from_pretrained(
        MODEL_ID, subfolder="transformer",
        torch_dtype=DTYPE, device_map=DEVICE, low_cpu_mem_usage=True,
    )
    vae = AutoencoderKLWan.from_pretrained(
        MODEL_ID, subfolder="vae",
        torch_dtype=DTYPE, device_map=DEVICE, low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")

    print(f"  Loaded in {time.time() - t0:.1f}s")
    return {
        "text_encoder": text_encoder,
        "transformer": transformer,
        "vae": vae,
        "tokenizer": tokenizer,
    }


@pytest.fixture(scope="module")
def original_latents(shared_components):
    """Run the original diffusers WanPipeline and return latent output."""
    print("\nRunning ORIGINAL diffusers WanPipeline...")
    scheduler = UniPCMultistepScheduler.from_pretrained(
        MODEL_ID, subfolder="scheduler"
    )
    pipe = WanPipeline(
        tokenizer=shared_components["tokenizer"],
        text_encoder=shared_components["text_encoder"],
        vae=shared_components["vae"],
        transformer=shared_components["transformer"],
        scheduler=scheduler,
        expand_timesteps=True,
    )
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
    print(f"  shape={latents.shape}  dtype={latents.dtype}  time={elapsed:.1f}s")
    return latents


@pytest.fixture(scope="module")
def rewritten_latents(shared_components):
    """Run the rewritten WanT2VPipeline and return latent output."""
    print("\nRunning REWRITTEN WanT2VPipeline...")
    from wan_t2v_pipeline import WanConfig, WanT2VPipeline

    config = WanConfig(device=DEVICE)
    pipe = WanT2VPipeline(config=config)

    pipe.text_encoder = shared_components["text_encoder"]
    pipe.transformer = shared_components["transformer"]
    pipe.vae = shared_components["vae"]
    pipe.tokenizer = shared_components["tokenizer"]
    pipe.scheduler = UniPCMultistepScheduler.from_pretrained(
        MODEL_ID, subfolder="scheduler"
    )
    vae = shared_components["vae"]
    pipe.vae_scale_factor_temporal = vae.config.scale_factor_temporal
    pipe.vae_scale_factor_spatial = vae.config.scale_factor_spatial
    pipe.video_processor = VideoProcessor(
        vae_scale_factor=pipe.vae_scale_factor_spatial
    )

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

    print(f"  shape={latents.shape}  dtype={latents.dtype}  time={elapsed:.1f}s")
    return latents


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------
def test_shapes_match(original_latents, rewritten_latents):
    """Both pipelines must produce tensors of the same shape and dtype."""
    assert original_latents.shape == rewritten_latents.shape, (
        f"Shape mismatch: original {original_latents.shape} "
        f"vs rewritten {rewritten_latents.shape}"
    )
    assert original_latents.dtype == rewritten_latents.dtype, (
        f"Dtype mismatch: original {original_latents.dtype} "
        f"vs rewritten {rewritten_latents.dtype}"
    )


def test_latents_close(original_latents, rewritten_latents):
    """Latent outputs must be numerically close (atol=1e-5)."""
    abs_diff = (original_latents - rewritten_latents).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()

    print(f"\n  Max  abs diff: {max_diff:.2e}")
    print(f"  Mean abs diff: {mean_diff:.2e}")

    assert torch.allclose(
        original_latents, rewritten_latents, atol=1e-5, rtol=1e-5
    ), f"Latents differ beyond 1e-5 tolerance (max diff = {max_diff:.2e})"


def test_latents_exact(original_latents, rewritten_latents):
    """Check for bit-exact match (informational – xfail if not exact)."""
    exact = torch.equal(original_latents, rewritten_latents)
    if not exact:
        abs_diff = (original_latents - rewritten_latents).abs()
        pytest.xfail(
            f"Not bit-exact (max diff = {abs_diff.max().item():.2e}), "
            f"but allclose may still pass"
        )
