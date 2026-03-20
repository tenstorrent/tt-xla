# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Isolated TT sanity tests for each Wan NN module: text encoder, transformer, VAE.

Each test loads ONE module on TT, runs a forward pass with pipeline-consistent
shapes, and verifies the output shape/dtype.  Only one model is on TT at a time
to avoid OOM from loading multiple large models simultaneously.

All input shapes and dtypes are derived from the loaded model's config,
matching exactly how the TT pipeline (wan_t2v_tt_pipeline.py) calls each module.

Usage:
    cd /proj_sw/user_dev/akannan_new/19_mar_bgd/tt-xla
    pytest examples/pytorch/test_wan_tt_modules.py -v -s
    pytest examples/pytorch/test_wan_tt_modules.py -v -s -k text_encoder
    pytest examples/pytorch/test_wan_tt_modules.py -v -s -k transformer
    pytest examples/pytorch/test_wan_tt_modules.py -v -s -k vae
"""

import time

import pytest
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from diffusers import AutoencoderKLWan, WanTransformer3DModel, UniPCMultistepScheduler
from transformers import AutoTokenizer, UMT5EncoderModel
from wan_t2v_tt_pipeline import _patch_transformer_for_tt, _patch_vae_for_tt

# ── Shared configuration ──────────────────────────────────────────────
MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
PROMPT = "A cat sitting on a sunny windowsill"
MAX_SEQ_LEN = 512
SEED = 42

HEIGHT = 480
WIDTH = 832
NUM_FRAMES = 81

VAE_SCALE_TEMPORAL = 4
VAE_SCALE_SPATIAL = 8


def _derive_spatial_shapes(patch_size=(1, 2, 2)):
    """Compute spatial/temporal shapes from shared config + model patch_size."""
    num_frames = NUM_FRAMES
    if num_frames % VAE_SCALE_TEMPORAL != 1:
        num_frames = num_frames // VAE_SCALE_TEMPORAL * VAE_SCALE_TEMPORAL + 1
    num_frames = max(num_frames, 1)

    h_mult = VAE_SCALE_SPATIAL * patch_size[1]
    w_mult = VAE_SCALE_SPATIAL * patch_size[2]
    height = HEIGHT // h_mult * h_mult
    width = WIDTH // w_mult * w_mult

    num_latent_frames = (num_frames - 1) // VAE_SCALE_TEMPORAL + 1
    latent_h = height // VAE_SCALE_SPATIAL
    latent_w = width // VAE_SCALE_SPATIAL

    return {
        "num_frames": num_frames,
        "height": height,
        "width": width,
        "num_latent_frames": num_latent_frames,
        "latent_h": latent_h,
        "latent_w": latent_w,
    }


# ── Setup ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def tt_setup():
    xr.set_device_type("TT")
    torch_xla.set_custom_compile_options({"optimization_level": 1})


# ======================================================================
# Test 1: Text Encoder (UMT5) on TT
# ======================================================================

def test_text_encoder_on_tt(tt_setup):
    """Load UMT5EncoderModel on TT, run tokenize -> encode, check output shape.

    Pipeline usage (wan_t2v_tt_pipeline.py):
      - Model loaded with torch_dtype=torch.bfloat16
      - input_ids: int64 (1, 512) -> xla_device
      - attention_mask: int64 (1, 512) -> xla_device
      - Output: last_hidden_state -> cpu, cast to float32
    """
    print(f"\n--- Text Encoder test ---")
    print(f"  max_seq_len={MAX_SEQ_LEN}")

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(
        MODEL_ID,
        subfolder="text_encoder",
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    hidden_dim = text_encoder.config.d_model
    print(f"  text_encoder.config.d_model = {hidden_dim}")

    text_encoder.compile(backend="tt")
    text_encoder = text_encoder.to(xm.xla_device())
    print(f"  Loaded + compiled in {time.time() - t0:.1f}s")

    text_inputs = tokenizer(
        [PROMPT],
        padding="max_length",
        max_length=MAX_SEQ_LEN,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids      # (1, 512), int64
    attn_mask = text_inputs.attention_mask  # (1, 512), int64

    print(f"  input_ids:  {input_ids.shape}  dtype={input_ids.dtype}")
    print(f"  attn_mask:  {attn_mask.shape}  dtype={attn_mask.dtype}")

    input_ids_tt = input_ids.to(xm.xla_device())
    attn_mask_tt = attn_mask.to(xm.xla_device())

    t0 = time.time()
    with torch.no_grad():
        output = text_encoder(input_ids_tt, attn_mask_tt).last_hidden_state
    elapsed = time.time() - t0

    output_cpu = output.to("cpu").to(dtype=torch.float32)
    print(f"  output:     {output_cpu.shape}  dtype={output_cpu.dtype}  time={elapsed:.1f}s")

    assert output_cpu.shape == (1, MAX_SEQ_LEN, hidden_dim), (
        f"Expected (1, {MAX_SEQ_LEN}, {hidden_dim}), got {output_cpu.shape}"
    )
    assert torch.isfinite(output_cpu).all(), "Output contains NaN/Inf"
    print("  PASSED")


# ======================================================================
# Test 2: Transformer (WanTransformer3DModel) on TT
# ======================================================================

def test_transformer_on_tt(tt_setup):
    """Load WanTransformer3DModel on TT, run one forward pass, check output shape.

    Pipeline usage (wan_t2v_tt_pipeline.py):
      - Model loaded with torch_dtype=torch.bfloat16
      - hidden_states: float32 on CPU -> tt_cast (bfloat16, xla)
        shape: (1, config.in_channels, num_latent_frames, latent_h, latent_w)
      - timestep: float32 on CPU -> tt_cast (bfloat16, xla)
        shape: (1, timestep_seq_len) via expand_timesteps
      - encoder_hidden_states: float32 on CPU -> tt_cast (bfloat16, xla)
        shape: (1, 512, config.text_dim)
    """
    print(f"\n--- Transformer test ---")

    t0 = time.time()
    transformer = WanTransformer3DModel.from_pretrained(
        MODEL_ID,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )

    in_channels = transformer.config.in_channels
    text_dim = transformer.config.text_dim
    patch_size = tuple(transformer.config.patch_size)

    print(f"  transformer.config.in_channels = {in_channels}")
    print(f"  transformer.config.text_dim    = {text_dim}")
    print(f"  transformer.config.patch_size  = {patch_size}")

    shapes = _derive_spatial_shapes(patch_size=patch_size)
    latent_shape = (
        1,
        in_channels,
        shapes["num_latent_frames"],
        shapes["latent_h"],
        shapes["latent_w"],
    )
    print(f"  latent_shape = {latent_shape}")

    _patch_transformer_for_tt(transformer)
    transformer.compile(backend="tt")
    transformer = transformer.to(xm.xla_device())
    print(f"  Loaded + compiled in {time.time() - t0:.1f}s")

    generator = torch.Generator(device="cpu").manual_seed(SEED)

    hidden_states = torch.randn(latent_shape, generator=generator, dtype=torch.float32)

    latent_mask = torch.ones_like(hidden_states)
    t_val = torch.tensor(999.0)
    temp_ts = (latent_mask[0][0][:, ::2, ::2] * t_val).flatten()
    timestep = temp_ts.unsqueeze(0)  # (1, timestep_seq_len)

    encoder_hidden_states = torch.randn(
        1, MAX_SEQ_LEN, text_dim, generator=generator, dtype=torch.float32
    )

    print(f"  hidden_states:           {hidden_states.shape}  dtype={hidden_states.dtype}")
    print(f"  timestep:                {timestep.shape}  dtype={timestep.dtype}")
    print(f"  encoder_hidden_states:   {encoder_hidden_states.shape}  dtype={encoder_hidden_states.dtype}")

    hidden_states_tt = hidden_states.to(dtype=torch.bfloat16).to(xm.xla_device())
    timestep_tt = timestep.to(dtype=torch.bfloat16).to(xm.xla_device())
    enc_hidden_tt = encoder_hidden_states.to(dtype=torch.bfloat16).to(xm.xla_device())

    t0 = time.time()
    with torch.no_grad():
        output = transformer(
            hidden_states=hidden_states_tt,
            timestep=timestep_tt,
            encoder_hidden_states=enc_hidden_tt,
            return_dict=False,
        )[0]
    elapsed = time.time() - t0

    output_cpu = output.to("cpu").to(dtype=torch.float32)
    print(f"  output:                  {output_cpu.shape}  dtype={output_cpu.dtype}  time={elapsed:.1f}s")

    assert output_cpu.shape == latent_shape, (
        f"Expected {latent_shape}, got {output_cpu.shape}"
    )
    assert torch.isfinite(output_cpu).all(), "Output contains NaN/Inf"
    print("  PASSED")


# ======================================================================
# Test 3: VAE decoder (AutoencoderKLWan) on TT
# ======================================================================

def test_vae_on_tt(tt_setup):
    """Load AutoencoderKLWan on TT, run VAE decode on dummy latents, check output.

    The stock VAE _decode processes frames one-at-a-time with feat_cache,
    which uses x[:, :, -CACHE_T:, :, :] (CACHE_T=2) on single-frame tensors.
    The TT XLA backend rejects the out-of-bounds negative start index.

    Fix: _patch_vae_for_tt replaces _decode with a version that passes all
    frames at once (feat_cache=None), using the causal conv's full left-padding
    instead. Mathematically equivalent, TT-compatible.

    Pipeline usage (wan_t2v_tt_pipeline.py):
      - Model loaded with torch_dtype=torch.float32
      - _patch_vae_for_tt applied before compile
      - Input latents: float32
        shape: (1, vae.config.z_dim, num_latent_frames, latent_h, latent_w)
      - Latents normalized with vae.config.latents_mean / latents_std
      - Output: (1, 3, num_frames, height, width)
    """
    print(f"\n--- VAE Decoder test (TT) ---")

    t0 = time.time()
    vae = AutoencoderKLWan.from_pretrained(
        MODEL_ID,
        subfolder="vae",
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )

    z_dim = vae.config.z_dim
    print(f"  vae.config.z_dim = {z_dim}")

    shapes = _derive_spatial_shapes()
    latent_shape = (1, z_dim, shapes["num_latent_frames"], shapes["latent_h"], shapes["latent_w"])
    print(f"  latent_shape = {latent_shape}")

    _patch_vae_for_tt(vae)
    vae.compile(backend="tt")
    vae = vae.to(xm.xla_device())
    print(f"  Loaded + patched + compiled in {time.time() - t0:.1f}s")

    generator = torch.Generator(device="cpu").manual_seed(SEED)
    latents = torch.randn(latent_shape, generator=generator, dtype=torch.float32)

    latents_mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, z_dim, 1, 1, 1)
        .to(latents.dtype)
    )
    latents_std = (
        1.0
        / torch.tensor(vae.config.latents_std)
        .view(1, z_dim, 1, 1, 1)
        .to(latents.dtype)
    )
    latents = latents / latents_std + latents_mean

    print(f"  latents (normalized):    {latents.shape}  dtype={latents.dtype}")

    latents_tt = latents.to(xm.xla_device())

    t0 = time.time()
    with torch.no_grad():
        decoded = vae.decode(latents_tt, return_dict=False)[0]
    elapsed = time.time() - t0

    decoded_cpu = decoded.to("cpu").to(dtype=torch.float32)
    print(f"  decoded output:          {decoded_cpu.shape}  dtype={decoded_cpu.dtype}  time={elapsed:.1f}s")

    expected_shape = (1, 3, shapes["num_frames"], shapes["height"], shapes["width"])
    assert decoded_cpu.shape == expected_shape, (
        f"Expected {expected_shape}, got {decoded_cpu.shape}"
    )
    assert torch.isfinite(decoded_cpu).all(), "Output contains NaN/Inf"
    print("  PASSED")
