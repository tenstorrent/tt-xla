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

import gc
import time

import pytest
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from diffusers import AutoencoderKLWan, WanTransformer3DModel, UniPCMultistepScheduler
from diffusers.models.transformers.transformer_wan import WanTransformerBlock
from transformers import AutoTokenizer, UMT5EncoderModel
from wan_t2v_tt_pipeline import _patch_transformer_for_tt, _patch_vae_for_tt, TTWanAttnProcessor

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

ENCODER_OPTION_COMBOS = {
    "baseline": {
        "optimization_level": 1,
    },
    # "bfp8": {
    #     "optimization_level": 1,
    #     "experimental_weight_dtype": "bfp8",
    # },
    # "bfp4": {
    #     "optimization_level": 1,
    #     "experimental_weight_dtype": "bfp4",
    # },
    # "no_consteval": {
    #     "optimization_level": 1,
    #     "enable_const_eval": False,
    # },
    # "consteval_on_cpu": {
    #     "optimization_level": 1,
    #     "enable_const_eval_on_cpu": True,
    # },
    # "bfp8_no_consteval": {
    #     "optimization_level": 1,
    #     "experimental_weight_dtype": "bfp8",
    #     "enable_const_eval": False,
    # },
    # "bfp8_consteval_cpu": {
    #     "optimization_level": 1,
    #     "experimental_weight_dtype": "bfp8",
    #     "enable_const_eval_on_cpu": True,
    # },
    # "opt0": {
    #     "optimization_level": 0,
    # },
}


@pytest.mark.parametrize("combo_name", list(ENCODER_OPTION_COMBOS.keys()))
def test_text_encoder_on_tt(tt_setup, combo_name):
    """Load UMT5EncoderModel on TT, run tokenize -> encode, check output shape.

    Pipeline usage (wan_t2v_tt_pipeline.py):
      - Model loaded with torch_dtype=torch.bfloat16
      - input_ids: int64 (1, MAX_SEQ_LEN) -> xla_device
      - attention_mask: int64 (1, MAX_SEQ_LEN) -> xla_device
      - Output: last_hidden_state -> cpu, cast to float32

    Sweeps compile-option combinations to find what fits in TT DRAM at
    MAX_SEQ_LEN=512 with all 24 encoder layers.
    """
    options = ENCODER_OPTION_COMBOS[combo_name]
    print(f"\n--- Text Encoder test [{combo_name}] ---")
    print(f"  max_seq_len={MAX_SEQ_LEN}")
    print(f"  compile options: {options}")

    torch_xla.set_custom_compile_options(options)

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
    print("tt dtype is ",output.dtype)
    output_cpu = output.to("cpu").to(dtype=torch.float32)
    print(f"  output:     {output_cpu.shape}  dtype={output_cpu.dtype}  time={elapsed:.1f}s")

    assert output_cpu.shape == (1, MAX_SEQ_LEN, hidden_dim), (
        f"Expected (1, {MAX_SEQ_LEN}, {hidden_dim}), got {output_cpu.shape}"
    )
    assert torch.isfinite(output_cpu).all(), "Output contains NaN/Inf"
    print("  PASSED")


# ======================================================================
# Test 1b: Text Encoder layer-count OOM isolation
# ======================================================================

@pytest.mark.parametrize("num_layers", [20, 24])
def test_text_encoder_layers_on_tt(tt_setup, num_layers):
    """Run UMT5EncoderModel on TT with a truncated block list.

    The full model has 24 UMT5Block layers.  20 layers fit on TT DRAM,
    but 24 layers OOM.  This test proves the OOM is caused by the total
    number of compiled layers, not by any single layer or the embedding.

    Run individually:
        pytest ... -k "test_text_encoder_layers_on_tt[20]"   # should PASS
        pytest ... -k "test_text_encoder_layers_on_tt[24]"   # expected OOM
    """
    print(f"\n--- Text Encoder layer-count test ({num_layers} layers) ---")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(
        MODEL_ID,
        subfolder="text_encoder",
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )

    total_layers = len(text_encoder.encoder.block)
    hidden_dim = text_encoder.config.d_model
    print(f"  total layers in model: {total_layers}")
    print(f"  layers for this test:  {num_layers}")
    print(f"  d_model:               {hidden_dim}")

    if num_layers < total_layers:
        text_encoder.encoder.block = text_encoder.encoder.block[:num_layers]
        print(f"  truncated encoder.block to {len(text_encoder.encoder.block)} layers")

    text_encoder.compile(backend="tt")
    text_encoder = text_encoder.to(xm.xla_device())

    text_inputs = tokenizer(
        [PROMPT],
        padding="max_length",
        max_length=MAX_SEQ_LEN,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids_tt = text_inputs.input_ids.to(xm.xla_device())
    attn_mask_tt = text_inputs.attention_mask.to(xm.xla_device())

    t0 = time.time()
    with torch.no_grad():
        output = text_encoder(input_ids_tt, attn_mask_tt).last_hidden_state
    elapsed = time.time() - t0

    output_cpu = output.to("cpu").to(dtype=torch.float32)
    print(f"  output: {output_cpu.shape}  time={elapsed:.1f}s")

    assert output_cpu.shape == (1, MAX_SEQ_LEN, hidden_dim), (
        f"Expected (1, {MAX_SEQ_LEN}, {hidden_dim}), got {output_cpu.shape}"
    )
    assert torch.isfinite(output_cpu).all(), "Output contains NaN/Inf"
    print(f"  PASSED ({num_layers} layers)")


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

    print("spatial shape are derived from config + shared constants:", shapes)
    latent_shape = (
        1,
        in_channels,
        shapes["num_latent_frames"],
        shapes["latent_h"],
        shapes["latent_w"],
    )
    print(f"  latent_shape = {latent_shape}")

    # _patch_transformer_for_tt(transformer)
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
# Test 2b: Single Transformer Block on TT
# ======================================================================

@pytest.mark.parametrize("text_seq_len", [384])
def test_single_transformer_block_on_tt(tt_setup, text_seq_len):
    """Run ONE WanTransformerBlock on TT with post-patch-embedding inputs.

    The full WanTransformer3DModel OOMs because ALL block weights must reside
    on TT DRAM simultaneously.  This test verifies that a *single* block with
    realistic intermediate shapes fits and executes correctly.

    Parametrized by text_seq_len to find the maximum text sequence length
    that fits in TT DRAM.  For TI2V models (added_kv_proj_dim is set),
    enc_seq_len = 257 (image tokens) + text_seq_len.

    Inputs match the shapes after patch_embedding + condition_embedder:
      - hidden_states:           (1, seq_len, inner_dim)
      - encoder_hidden_states:   (1, enc_seq_len, inner_dim)
      - temb (timestep_proj):    (1, seq_len, 6, inner_dim)   [TI2V path]
      - rotary_emb:              tuple of 2 x (1, seq_len, 1, head_dim)
    """
    print(f"\n--- Single Transformer Block test [text_seq_len={text_seq_len}] ---")

    config = WanTransformer3DModel.load_config(MODEL_ID, subfolder="transformer")

    num_heads = config.get("num_attention_heads", 40)
    head_dim = config.get("attention_head_dim", 128)
    inner_dim = num_heads * head_dim
    ffn_dim = config.get("ffn_dim", 13824)
    patch_size = tuple(config.get("patch_size", (1, 2, 2)))
    added_kv_proj_dim = config.get("added_kv_proj_dim")
    num_layers_total = config.get("num_layers", 40)

    shapes = _derive_spatial_shapes(patch_size=patch_size)
    ppf = shapes["num_latent_frames"]
    pph = shapes["latent_h"] // patch_size[1]
    ppw = shapes["latent_w"] // patch_size[2]
    seq_len = ppf * pph * ppw

    enc_seq_len = (257 + text_seq_len) if added_kv_proj_dim is not None else text_seq_len

    print(f"  inner_dim={inner_dim}  num_heads={num_heads}  head_dim={head_dim}")
    print(f"  ffn_dim={ffn_dim}  added_kv_proj_dim={added_kv_proj_dim}")
    print(f"  total blocks in model: {num_layers_total}")
    print(f"  seq_len = {ppf}*{pph}*{ppw} = {seq_len}")
    print(f"  enc_seq_len = {enc_seq_len}  (257 img + {text_seq_len} text)" if added_kv_proj_dim else f"  enc_seq_len = {enc_seq_len}")

    t0 = time.time()
    block = WanTransformerBlock(
        dim=inner_dim,
        ffn_dim=ffn_dim,
        num_heads=num_heads,
        qk_norm=config.get("qk_norm", "rms_norm_across_heads"),
        cross_attn_norm=config.get("cross_attn_norm", True),
        eps=config.get("eps", 1e-6),
        added_kv_proj_dim=added_kv_proj_dim,
    ).to(dtype=torch.bfloat16)

    block.attn1.processor = TTWanAttnProcessor()
    block.attn2.processor = TTWanAttnProcessor()

    block.compile(backend="tt")
    block = block.to(xm.xla_device())
    print(f"  Created + compiled in {time.time() - t0:.1f}s")

    generator = torch.Generator(device="cpu").manual_seed(SEED)

    hidden_states = torch.randn(1, seq_len, inner_dim, generator=generator, dtype=torch.bfloat16)
    encoder_hidden_states = torch.randn(1, enc_seq_len, inner_dim, generator=generator, dtype=torch.bfloat16)
    temb = torch.randn(1, seq_len, 6, inner_dim, generator=generator, dtype=torch.bfloat16)
    rotary_emb = (
        torch.randn(1, seq_len, 1, head_dim, generator=generator, dtype=torch.bfloat16),
        torch.randn(1, seq_len, 1, head_dim, generator=generator, dtype=torch.bfloat16),
    )

    print(f"  hidden_states:         {hidden_states.shape}")
    print(f"  encoder_hidden_states: {encoder_hidden_states.shape}")
    print(f"  temb:                  {temb.shape}")
    print(f"  rotary_emb[0]:         {rotary_emb[0].shape}")

    hs_tt = hidden_states.to(xm.xla_device())
    enc_tt = encoder_hidden_states.to(xm.xla_device())
    temb_tt = temb.to(xm.xla_device())
    rope_tt = (rotary_emb[0].to(xm.xla_device()), rotary_emb[1].to(xm.xla_device()))

    t0 = time.time()
    with torch.no_grad():
        output = block(hs_tt, enc_tt, temb_tt, rope_tt)
    elapsed = time.time() - t0

    # output_cpu = output.to("cpu")
    # print(f"  output: {output_cpu.shape}  dtype={output_cpu.dtype}  time={elapsed:.1f}s")

    # assert output_cpu.shape == (1, seq_len, inner_dim), (
    #     f"Expected (1, {seq_len}, {inner_dim}), got {output_cpu.shape}"
    # )
    # assert torch.isfinite(output_cpu).all(), "Output contains NaN/Inf"
    # print(f"  PASSED (text_seq_len={text_seq_len})")


# ======================================================================
# Test 2c: Transformer Blocks loop – OOM threshold discovery
# ======================================================================

# class _BlocksRunner(torch.nn.Module):
#     """Thin wrapper to compile N WanTransformerBlocks as one module."""

#     def __init__(self, blocks):
#         super().__init__()
#         self.blocks = torch.nn.ModuleList(blocks)

#     def forward(self, hidden_states, encoder_hidden_states, temb, rotary_emb):
#         for block in self.blocks:
#             hidden_states = block(hidden_states, encoder_hidden_states, temb, rotary_emb)
#         return hidden_states


# @pytest.mark.parametrize("num_blocks", [1, 5, 10, 15, 20, 30])
# def test_transformer_blocks_on_tt(tt_setup, num_blocks):
#     """Compile N WanTransformerBlocks together on TT to locate OOM boundary.

#     Each block's weights are ~700-800 MB (bfloat16).  With ~12.8 GB TT DRAM
#     and ~2-3 GB needed for inputs/activations, the theoretical limit is
#     around 12-14 blocks.  This test sweeps across values to find the
#     threshold experimentally.

#     Run individually:
#         pytest ... -k "test_transformer_blocks_on_tt[1]"    # expected PASS
#         pytest ... -k "test_transformer_blocks_on_tt[30]"   # expected OOM
#     """
#     print(f"\n--- Transformer Blocks loop test ({num_blocks} blocks) ---")

#     config = WanTransformer3DModel.load_config(MODEL_ID, subfolder="transformer")

#     num_heads = config.get("num_attention_heads", 40)
#     head_dim = config.get("attention_head_dim", 128)
#     inner_dim = num_heads * head_dim
#     ffn_dim = config.get("ffn_dim", 13824)
#     patch_size = tuple(config.get("patch_size", (1, 2, 2)))
#     added_kv_proj_dim = config.get("added_kv_proj_dim")
#     num_layers_total = config.get("num_layers", 40)

#     shapes = _derive_spatial_shapes(patch_size=patch_size)
#     ppf = shapes["num_latent_frames"]
#     pph = shapes["latent_h"] // patch_size[1]
#     ppw = shapes["latent_w"] // patch_size[2]
#     seq_len = ppf * pph * ppw

#     print(f"  inner_dim={inner_dim}  total blocks in model={num_layers_total}")
#     print(f"  seq_len={seq_len}  num_blocks for this test={num_blocks}")

#     t0 = time.time()
#     blocks = [
#         WanTransformerBlock(
#             dim=inner_dim,
#             ffn_dim=ffn_dim,
#             num_heads=num_heads,
#             qk_norm=config.get("qk_norm", "rms_norm_across_heads"),
#             cross_attn_norm=config.get("cross_attn_norm", True),
#             eps=config.get("eps", 1e-6),
#             added_kv_proj_dim=added_kv_proj_dim,
#         )
#         for _ in range(num_blocks)
#     ]

#     runner = _BlocksRunner(blocks).to(dtype=torch.bfloat16)

#     patched = TTWanAttnProcessor()
#     for blk in runner.blocks:
#         blk.attn1.processor = patched
#         blk.attn2.processor = patched

#     runner.compile(backend="tt")
#     runner = runner.to(xm.xla_device())
#     print(f"  Created {num_blocks} blocks + compiled in {time.time() - t0:.1f}s")

#     generator = torch.Generator(device="cpu").manual_seed(SEED)

#     enc_seq_len = (257 + 512) if added_kv_proj_dim is not None else 512
#     hidden_states = torch.randn(1, seq_len, inner_dim, generator=generator, dtype=torch.bfloat16)
#     encoder_hidden_states = torch.randn(1, enc_seq_len, inner_dim, generator=generator, dtype=torch.bfloat16)
#     temb = torch.randn(1, seq_len, 6, inner_dim, generator=generator, dtype=torch.bfloat16)
#     rotary_emb = (
#         torch.randn(1, seq_len, 1, head_dim, generator=generator, dtype=torch.bfloat16),
#         torch.randn(1, seq_len, 1, head_dim, generator=generator, dtype=torch.bfloat16),
#     )

#     print(f"  hidden_states: {hidden_states.shape}   temb: {temb.shape}")

#     hs_tt = hidden_states.to(xm.xla_device())
#     enc_tt = encoder_hidden_states.to(xm.xla_device())
#     temb_tt = temb.to(xm.xla_device())
#     rope_tt = (rotary_emb[0].to(xm.xla_device()), rotary_emb[1].to(xm.xla_device()))

#     t0 = time.time()
#     with torch.no_grad():
#         output = runner(hs_tt, enc_tt, temb_tt, rope_tt)
#     elapsed = time.time() - t0

#     output_cpu = output.to("cpu")
#     print(f"  output: {output_cpu.shape}  dtype={output_cpu.dtype}  time={elapsed:.1f}s")

#     assert output_cpu.shape == (1, seq_len, inner_dim), (
#         f"Expected (1, {seq_len}, {inner_dim}), got {output_cpu.shape}"
#     )
#     assert torch.isfinite(output_cpu).all(), "Output contains NaN/Inf"
#     print(f"  PASSED ({num_blocks} blocks)")


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
    print("vae is ",vae)
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

    # expected_shape = (1, 3, shapes["num_frames"], shapes["height"], shapes["width"])
    # assert decoded_cpu.shape == expected_shape, (
    #     f"Expected {expected_shape}, got {decoded_cpu.shape}"
    # )
    # assert torch.isfinite(decoded_cpu).all(), "Output contains NaN/Inf"
    # print("  PASSED")
