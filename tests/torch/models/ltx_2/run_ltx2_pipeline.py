# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LTX-2 Full Pipeline — text/image to video+audio on TT hardware.

Model: Lightricks/LTX-2 (31.6B parameters total)
Hardware: Blackhole Quietbox (bhqb) — 4 x p150 (32 GiB DRAM each, 128 GiB total)

Strategy: 4-chip tensor parallelism with sequential offload
  Phase 1: Text encoding (Gemma 3 sharded) + Connectors -> conditioning embeddings
  Phase 2: Denoising loop (Transformer sharded) -> denoised latents
  Phase 3: Decoding (VAE + Vocoder replicated) -> video frames + audio waveform

Memory analysis (per device with 4-chip TP + sequential offload):
  Phase 1 peak: 24.4/4 + 2.67 = 8.77 GiB -> 23.23 GiB remaining for activations
  Phase 2 peak: 35.17/4 = 8.79 GiB -> 23.21 GiB remaining for activations
  Phase 3 peak: 2.27 GiB (replicated) -> 29.73 GiB remaining for activations

Known blockers (from bringup report):
  - Transformer: KeyError c_lifted_tensor_1 (graph partitioner shared tensor bug)
  - Video VAE Decoder: Conv3d temporal padding, channel alignment (48 % 32 != 0)
  - Latent Upsampler: Conv3d symmetric padding requirement
  - Text Connectors: Dynamo graph break during end-to-end tracing

Usage:
  python run_ltx2_pipeline.py --prompt "A cat playing piano" --num_inference_steps 5
"""

import argparse
import copy
import math
import os
import time

import numpy as np
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh


# ---------------------------------------------------------------------------
# Constants matching the LTX-2 architecture
# ---------------------------------------------------------------------------
VAE_SPATIAL_COMPRESSION = 32
VAE_TEMPORAL_COMPRESSION = 8
AUDIO_VAE_TEMPORAL_COMPRESSION = 4
AUDIO_VAE_MEL_COMPRESSION = 4
AUDIO_SAMPLING_RATE = 16000
AUDIO_HOP_LENGTH = 160
TRANSFORMER_PATCH_SIZE = 1
TRANSFORMER_PATCH_SIZE_T = 1


# ---------------------------------------------------------------------------
# Sharding helpers
# ---------------------------------------------------------------------------
def shard_gemma3_text_encoder(model, mesh):
    """Apply tensor-parallel sharding to the Gemma 3 language model layers."""
    shard_specs = {}
    for layer in model.model.language_model.layers:
        # Attention: column-parallel Q/K/V, row-parallel O
        shard_specs[layer.self_attn.q_proj.weight] = ("model", None)
        shard_specs[layer.self_attn.k_proj.weight] = ("model", None)
        shard_specs[layer.self_attn.v_proj.weight] = ("model", None)
        shard_specs[layer.self_attn.o_proj.weight] = (None, "model")
        # MLP: column-parallel gate/up, row-parallel down
        shard_specs[layer.mlp.gate_proj.weight] = ("model", None)
        shard_specs[layer.mlp.up_proj.weight] = ("model", None)
        shard_specs[layer.mlp.down_proj.weight] = (None, "model")

    for tensor, shard_spec in shard_specs.items():
        xs.mark_sharding(tensor, mesh, shard_spec)


def shard_transformer(model, mesh):
    """Apply tensor-parallel sharding to the dual-stream DiT transformer."""
    shard_specs = {}

    for block in model.transformer_blocks:
        # Video self-attention (32 heads, 4096 dim)
        shard_specs[block.attn1.to_q.weight] = ("model", None)
        shard_specs[block.attn1.to_k.weight] = ("model", None)
        shard_specs[block.attn1.to_v.weight] = ("model", None)
        shard_specs[block.attn1.to_out[0].weight] = (None, "model")

        # Audio self-attention (32 heads, 2048 dim)
        shard_specs[block.audio_attn1.to_q.weight] = ("model", None)
        shard_specs[block.audio_attn1.to_k.weight] = ("model", None)
        shard_specs[block.audio_attn1.to_v.weight] = ("model", None)
        shard_specs[block.audio_attn1.to_out[0].weight] = (None, "model")

        # Video cross-attention (text -> video)
        shard_specs[block.attn2.to_q.weight] = ("model", None)
        shard_specs[block.attn2.to_k.weight] = ("model", None)
        shard_specs[block.attn2.to_v.weight] = ("model", None)
        shard_specs[block.attn2.to_out[0].weight] = (None, "model")

        # Audio cross-attention (text -> audio)
        shard_specs[block.audio_attn2.to_q.weight] = ("model", None)
        shard_specs[block.audio_attn2.to_k.weight] = ("model", None)
        shard_specs[block.audio_attn2.to_v.weight] = ("model", None)
        shard_specs[block.audio_attn2.to_out[0].weight] = (None, "model")

        # Cross-modal: audio-to-video
        shard_specs[block.audio_to_video_attn.to_q.weight] = ("model", None)
        shard_specs[block.audio_to_video_attn.to_k.weight] = ("model", None)
        shard_specs[block.audio_to_video_attn.to_v.weight] = ("model", None)
        shard_specs[block.audio_to_video_attn.to_out[0].weight] = (None, "model")

        # Cross-modal: video-to-audio
        shard_specs[block.video_to_audio_attn.to_q.weight] = ("model", None)
        shard_specs[block.video_to_audio_attn.to_k.weight] = ("model", None)
        shard_specs[block.video_to_audio_attn.to_v.weight] = ("model", None)
        shard_specs[block.video_to_audio_attn.to_out[0].weight] = (None, "model")

        # Video FFN: 4096 -> 16384 -> 4096 (GEGLU)
        shard_specs[block.ff.net[0].proj.weight] = ("model", None)
        shard_specs[block.ff.net[2].weight] = (None, "model")

        # Audio FFN: 2048 -> 8192 -> 2048
        shard_specs[block.audio_ff.net[0].proj.weight] = ("model", None)
        shard_specs[block.audio_ff.net[2].weight] = (None, "model")

    for tensor, shard_spec in shard_specs.items():
        xs.mark_sharding(tensor, mesh, shard_spec)


# ---------------------------------------------------------------------------
# Latent packing / unpacking (from diffusers LTX2Pipeline)
# ---------------------------------------------------------------------------
def pack_latents(latents, patch_size=1, patch_size_t=1):
    """Pack [B, C, F, H, W] -> [B, S, D] for transformer input."""
    batch_size, channels, num_frames, height, width = latents.shape
    latents = latents.reshape(
        batch_size, channels,
        num_frames // patch_size_t, patch_size_t,
        height // patch_size, patch_size,
        width // patch_size, patch_size,
    )
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
    return latents


def unpack_latents(latents, num_frames, height, width, patch_size=1, patch_size_t=1):
    """Unpack [B, S, D] -> [B, C, F, H, W]."""
    batch_size = latents.size(0)
    latents = latents.reshape(batch_size, num_frames, height, width, -1, patch_size_t, patch_size, patch_size)
    latents = latents.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)
    return latents


def pack_audio_latents(latents):
    """Pack [B, C, L, M] -> [B, S, D] for transformer input."""
    return latents.permute(0, 2, 1, 3).flatten(2, 3)


def unpack_audio_latents(latents, latent_length, num_mel_bins):
    """Unpack [B, S, D] -> [B, C, L, M]."""
    latents = latents.unflatten(2, (-1, num_mel_bins)).transpose(1, 2)
    return latents


def normalize_latents(latents, latents_mean, latents_std, scaling_factor=1.0):
    """Normalize latents: (latent - mean) * scaling_factor / std."""
    latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents = (latents - latents_mean) * scaling_factor / latents_std
    return latents


def denormalize_latents(latents, latents_mean, latents_std, scaling_factor=1.0):
    """Denormalize latents: latent * std / scaling_factor + mean."""
    latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents = latents * latents_std / scaling_factor + latents_mean
    return latents


def normalize_audio_latents(latents, latents_mean, latents_std):
    """Normalize audio latents."""
    latents_mean = latents_mean.to(latents.device, latents.dtype)
    latents_std = latents_std.to(latents.device, latents.dtype)
    return (latents - latents_mean) / latents_std


def denormalize_audio_latents(latents, latents_mean, latents_std):
    """Denormalize audio latents."""
    latents_mean = latents_mean.to(latents.device, latents.dtype)
    latents_std = latents_std.to(latents.device, latents.dtype)
    return (latents * latents_std) + latents_mean


def pack_text_embeds(text_hidden_states, sequence_lengths, padding_side="left", scale_factor=8, eps=1e-6):
    """Pack and normalize text encoder hidden states.

    Args:
        text_hidden_states: [B, seq_len, hidden_dim, num_layers] from Gemma3
        sequence_lengths: [B] number of valid (non-padded) tokens per batch
        padding_side: "left" or "right"
        scale_factor: scaling factor for normalized hidden states

    Returns:
        [B, seq_len, hidden_dim * num_layers] normed and flattened
    """
    batch_size, seq_len, hidden_dim, num_layers = text_hidden_states.shape

    # Create mask for non-padded positions
    if padding_side == "left":
        # Left-padded: valid tokens are at the right end
        positions = torch.arange(seq_len, device=text_hidden_states.device).unsqueeze(0)
        pad_lengths = seq_len - sequence_lengths.unsqueeze(1)
        mask = positions >= pad_lengths  # [B, seq_len]
    else:
        positions = torch.arange(seq_len, device=text_hidden_states.device).unsqueeze(0)
        mask = positions < sequence_lengths.unsqueeze(1)

    mask = mask.unsqueeze(-1).unsqueeze(-1)  # [B, seq_len, 1, 1]

    # Per-layer, per-batch normalization (only over non-padded positions)
    masked_states = text_hidden_states * mask.float()
    # Sum over sequence dim for mean computation
    sum_states = masked_states.sum(dim=1, keepdim=True)  # [B, 1, hidden_dim, num_layers]
    count = mask.float().sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1, 1, 1]
    mean = sum_states / count

    diff = (text_hidden_states - mean) * mask.float()
    var = (diff ** 2).sum(dim=1, keepdim=True) / count.clamp(min=1)
    std = (var + eps).sqrt()

    normalized = (text_hidden_states - mean) / std * scale_factor * mask.float()

    # Flatten layers into feature dim: [B, seq_len, hidden_dim * num_layers]
    return normalized.reshape(batch_size, seq_len, hidden_dim * num_layers)


def calculate_shift(seq_len, base_seq_len=1024, max_seq_len=4096, base_shift=0.95, max_shift=2.05):
    """Calculate dynamic shift for flow matching scheduler."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = seq_len * m + b
    return mu


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------
class LTX2TTPipeline:
    """LTX-2 pipeline for TT hardware with sequential offload and 4-chip TP."""

    def __init__(self, use_sequential_offload=True):
        self.use_sequential_offload = use_sequential_offload
        self.device = None
        self.mesh = None

    def setup(self):
        """Initialize SPMD, create mesh, load all components."""
        os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
        xr.use_spmd()

        self.num_devices = xr.global_runtime_device_count()
        assert self.num_devices >= 4, f"Pipeline requires 4+ devices, found {self.num_devices}"

        mesh_shape = (1, self.num_devices)
        device_ids = np.array(range(self.num_devices))
        self.mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))
        self.device = torch_xla.device()

        self._load_tokenizer()
        self._load_scheduler()

    def _load_tokenizer(self):
        from transformers import GemmaTokenizerFast

        print("Loading tokenizer...")
        self.tokenizer = GemmaTokenizerFast.from_pretrained(
            "Lightricks/LTX-2", subfolder="tokenizer"
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _load_scheduler(self):
        from diffusers import FlowMatchEulerDiscreteScheduler

        print("Loading scheduler...")
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            "Lightricks/LTX-2", subfolder="scheduler"
        )

    # ------ Phase 1: Text Encoding ------

    def _load_phase1(self):
        """Load text encoder + connectors to device with sharding."""
        from transformers import Gemma3ForConditionalGeneration
        from diffusers.pipelines.ltx2 import LTX2TextConnectors
        import safetensors.torch
        from huggingface_hub import hf_hub_download

        print("Phase 1: Loading text encoder (Gemma 3, 24.4 GiB)...")
        self.text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
            "Lightricks/LTX-2",
            subfolder="text_encoder",
            torch_dtype=torch.bfloat16,
            use_cache=False,
        ).eval()
        self.text_encoder = self.text_encoder.to(self.device)
        shard_gemma3_text_encoder(self.text_encoder, self.mesh)
        self.compiled_text_encoder = torch.compile(self.text_encoder, backend="tt")

        print("Phase 1: Loading text connectors (2.67 GiB)...")
        self.connectors = LTX2TextConnectors(
            caption_channels=3840,
            text_proj_in_factor=49,
            video_connector_num_attention_heads=30,
            video_connector_attention_head_dim=128,
            video_connector_num_layers=2,
            video_connector_num_learnable_registers=None,
            audio_connector_num_attention_heads=30,
            audio_connector_attention_head_dim=128,
            audio_connector_num_layers=2,
            audio_connector_num_learnable_registers=None,
            connector_rope_base_seq_len=4096,
            rope_theta=10000.0,
            rope_double_precision=True,
            causal_temporal_positioning=False,
            rope_type="interleaved",
        )
        weights_path = hf_hub_download(
            "Lightricks/LTX-2",
            "connectors/diffusion_pytorch_model.safetensors",
        )
        state_dict = safetensors.torch.load_file(weights_path)
        filtered = {k: v for k, v in state_dict.items() if "learnable_registers" not in k}
        self.connectors.load_state_dict(filtered, strict=False)
        self.connectors = self.connectors.to(torch.bfloat16).eval()
        self.connectors = self.connectors.to(self.device)
        self.compiled_connectors = torch.compile(self.connectors, backend="tt")

    def _offload_phase1(self):
        """Move phase 1 models off device."""
        if self.use_sequential_offload:
            print("Phase 1: Offloading text encoder + connectors...")
            self.text_encoder = self.text_encoder.to("cpu")
            self.connectors = self.connectors.to("cpu")
            del self.compiled_text_encoder
            del self.compiled_connectors
            torch_xla.sync(wait=True)

    def encode_prompt(self, prompt, negative_prompt="", max_sequence_length=256):
        """Phase 1: Encode text prompt into conditioning embeddings.

        Returns:
            connector_prompt_embeds: [B, seq_len, caption_channels] video conditioning
            connector_audio_prompt_embeds: [B, seq_len, caption_channels] audio conditioning
            connector_attention_mask: [B, seq_len] attention mask
        """
        self._load_phase1()

        prompts = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompts)

        # Encode positive prompt
        prompt_embeds, prompt_mask = self._encode_single_prompt(
            prompts, max_sequence_length
        )

        # Encode negative prompt for CFG
        neg_prompts = [negative_prompt] * batch_size if isinstance(negative_prompt, str) else negative_prompt
        neg_embeds, neg_mask = self._encode_single_prompt(
            neg_prompts, max_sequence_length
        )

        # Concatenate [negative, positive] for CFG
        prompt_embeds = torch.cat([neg_embeds, prompt_embeds], dim=0)
        prompt_mask = torch.cat([neg_mask, prompt_mask], dim=0)

        # Run connectors
        additive_mask = (1 - prompt_mask.to(prompt_embeds.dtype)) * -1000000.0
        connector_prompt_embeds, connector_audio_prompt_embeds, connector_mask = self.compiled_connectors(
            prompt_embeds, additive_mask, additive_mask=True
        )
        torch_xla.sync(wait=True)

        self._offload_phase1()

        return connector_prompt_embeds, connector_audio_prompt_embeds, connector_mask

    def _encode_single_prompt(self, prompts, max_sequence_length):
        """Encode a list of prompts using the Gemma 3 text encoder."""
        inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self.compiled_text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        # Stack all hidden states: [B, seq_len, hidden_dim, num_layers]
        hidden_states = torch.stack(outputs.hidden_states, dim=-1)
        sequence_lengths = attention_mask.sum(dim=-1)

        # Pack and normalize text embeddings
        prompt_embeds = pack_text_embeds(
            hidden_states,
            sequence_lengths,
            padding_side=self.tokenizer.padding_side,
        )
        prompt_embeds = prompt_embeds.to(dtype=torch.bfloat16)

        return prompt_embeds, attention_mask

    # ------ Phase 2: Denoising ------

    def _load_phase2(self):
        """Load transformer to device with sharding."""
        from diffusers import LTX2VideoTransformer3DModel

        print("Phase 2: Loading transformer (35.17 GiB)...")
        self.transformer = LTX2VideoTransformer3DModel.from_pretrained(
            "Lightricks/LTX-2",
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )
        self.transformer.config.rope_type = "interleaved"
        self.transformer = self.transformer.eval()
        self.transformer = self.transformer.to(self.device)
        shard_transformer(self.transformer, self.mesh)
        self.compiled_transformer = torch.compile(self.transformer, backend="tt")

    def _offload_phase2(self):
        """Move phase 2 model off device."""
        if self.use_sequential_offload:
            print("Phase 2: Offloading transformer...")
            self.transformer = self.transformer.to("cpu")
            del self.compiled_transformer
            torch_xla.sync(wait=True)

    def denoise(
        self,
        connector_prompt_embeds,
        connector_audio_prompt_embeds,
        connector_mask,
        height=512,
        width=320,
        num_frames=49,
        frame_rate=24.0,
        num_inference_steps=5,
        guidance_scale=4.0,
        generator=None,
    ):
        """Phase 2: Run the denoising loop.

        Returns:
            latents: [B, C, F, H, W] denoised video latents
            audio_latents: [B, C, L, M] denoised audio latents
        """
        self._load_phase2()

        batch_size = 1

        # Compute latent dimensions
        latent_num_frames = (num_frames - 1) // VAE_TEMPORAL_COMPRESSION + 1
        latent_height = height // VAE_SPATIAL_COMPRESSION
        latent_width = width // VAE_SPATIAL_COMPRESSION
        video_seq_len = latent_num_frames * latent_height * latent_width

        # Audio latent dimensions
        duration_s = num_frames / frame_rate
        audio_latents_per_second = AUDIO_SAMPLING_RATE / AUDIO_HOP_LENGTH / float(AUDIO_VAE_TEMPORAL_COMPRESSION)
        audio_num_frames = round(duration_s * audio_latents_per_second)
        latent_mel_bins = 64 // AUDIO_VAE_MEL_COMPRESSION  # 64 mel bins / 4 = 16

        # Initialize random latents
        latents_shape = (batch_size, 128, latent_num_frames, latent_height, latent_width)
        latents_5d = torch.randn(latents_shape, generator=generator, dtype=torch.float32)
        latents = pack_latents(latents_5d, TRANSFORMER_PATCH_SIZE, TRANSFORMER_PATCH_SIZE_T)
        latents = latents.to(device=self.device, dtype=torch.float32)

        audio_latents_shape = (batch_size, 8, audio_num_frames, latent_mel_bins)
        audio_latents_4d = torch.randn(audio_latents_shape, generator=generator, dtype=torch.float32)
        audio_latents = pack_audio_latents(audio_latents_4d)
        audio_latents = audio_latents.to(device=self.device, dtype=torch.float32)

        # Prepare timesteps with dynamic shifting
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        mu = calculate_shift(video_seq_len)
        audio_scheduler = copy.deepcopy(self.scheduler)
        audio_scheduler.set_timesteps(sigmas=sigmas, mu=mu)
        self.scheduler.set_timesteps(sigmas=sigmas, mu=mu)
        timesteps = self.scheduler.timesteps.to(self.device)

        # Pre-compute positional embeddings
        video_coords = self.transformer.rope.prepare_video_coords(
            batch_size * 2, latent_num_frames, latent_height, latent_width,
            self.device, fps=frame_rate,
        )
        audio_coords = self.transformer.audio_rope.prepare_audio_coords(
            batch_size * 2, audio_num_frames, self.device,
        )

        # Denoising loop
        print(f"Phase 2: Denoising ({num_inference_steps} steps)...")
        for i, t in enumerate(timesteps):
            print(f"  Step {i+1}/{num_inference_steps}")

            # CFG: duplicate latents for unconditional + conditional
            latent_model_input = torch.cat([latents] * 2).to(torch.bfloat16)
            audio_latent_model_input = torch.cat([audio_latents] * 2).to(torch.bfloat16)
            timestep = t.expand(latent_model_input.shape[0])

            with torch.no_grad():
                noise_pred_video, noise_pred_audio = self.compiled_transformer(
                    hidden_states=latent_model_input,
                    audio_hidden_states=audio_latent_model_input,
                    encoder_hidden_states=connector_prompt_embeds,
                    audio_encoder_hidden_states=connector_audio_prompt_embeds,
                    timestep=timestep,
                    encoder_attention_mask=connector_mask,
                    audio_encoder_attention_mask=connector_mask,
                    num_frames=latent_num_frames,
                    height=latent_height,
                    width=latent_width,
                    fps=frame_rate,
                    audio_num_frames=audio_num_frames,
                    video_coords=video_coords,
                    audio_coords=audio_coords,
                    return_dict=False,
                )
            torch_xla.sync(wait=True)

            noise_pred_video = noise_pred_video.float()
            noise_pred_audio = noise_pred_audio.float()

            # Apply classifier-free guidance
            npv_uncond, npv_cond = noise_pred_video.chunk(2)
            noise_pred_video = npv_uncond + guidance_scale * (npv_cond - npv_uncond)

            npa_uncond, npa_cond = noise_pred_audio.chunk(2)
            noise_pred_audio = npa_uncond + guidance_scale * (npa_cond - npa_uncond)

            # Scheduler step
            latents = self.scheduler.step(noise_pred_video, t, latents, return_dict=False)[0]
            audio_latents = audio_scheduler.step(noise_pred_audio, t, audio_latents, return_dict=False)[0]

        # Unpack latents back to spatial format
        latents = unpack_latents(latents, latent_num_frames, latent_height, latent_width,
                                 TRANSFORMER_PATCH_SIZE, TRANSFORMER_PATCH_SIZE_T)

        # Unpack audio latents
        audio_latents = unpack_audio_latents(audio_latents, audio_num_frames, latent_mel_bins)

        self._offload_phase2()

        return latents, audio_latents, latent_num_frames, latent_height, latent_width, audio_num_frames, latent_mel_bins

    # ------ Phase 3: Decoding ------

    def _load_phase3(self):
        """Load VAE decoders + vocoder to device (replicated, no sharding needed)."""
        from diffusers import AutoencoderKLLTX2Video
        from diffusers.models.autoencoders import AutoencoderKLLTX2Audio
        from diffusers.pipelines.ltx2 import LTX2Vocoder

        print("Phase 3: Loading Video VAE decoder (1.14 GiB)...")
        self.vae = AutoencoderKLLTX2Video.from_pretrained(
            "Lightricks/LTX-2",
            subfolder="vae",
            torch_dtype=torch.bfloat16,
        ).eval()
        self.vae = self.vae.to(self.device)
        self.compiled_vae_decoder = torch.compile(self.vae.decoder, backend="tt")

        print("Phase 3: Loading Audio VAE decoder (0.10 GiB)...")
        self.audio_vae = AutoencoderKLLTX2Audio.from_pretrained(
            "Lightricks/LTX-2",
            subfolder="audio_vae",
            torch_dtype=torch.bfloat16,
        ).eval()
        self.audio_vae = self.audio_vae.to(self.device)
        self.compiled_audio_vae_decoder = torch.compile(self.audio_vae.decoder, backend="tt")

        print("Phase 3: Loading Vocoder (0.10 GiB)...")
        self.vocoder = LTX2Vocoder.from_pretrained(
            "Lightricks/LTX-2",
            subfolder="vocoder",
            torch_dtype=torch.bfloat16,
        ).eval()
        self.vocoder = self.vocoder.to(self.device)
        self.compiled_vocoder = torch.compile(self.vocoder, backend="tt")

    def _offload_phase3(self):
        """Move phase 3 models off device."""
        if self.use_sequential_offload:
            print("Phase 3: Offloading decoders...")
            self.vae = self.vae.to("cpu")
            self.audio_vae = self.audio_vae.to("cpu")
            self.vocoder = self.vocoder.to("cpu")
            del self.compiled_vae_decoder, self.compiled_audio_vae_decoder, self.compiled_vocoder
            torch_xla.sync(wait=True)

    def decode(self, latents, audio_latents):
        """Phase 3: Decode latents to video frames and audio waveform.

        Returns:
            video: [B, 3, T, H, W] decoded video frames
            audio: [B, 2, samples] stereo audio waveform
        """
        self._load_phase3()

        # Denormalize video latents
        latents = denormalize_latents(
            latents,
            self.vae.latents_mean,
            self.vae.latents_std,
            self.vae.config.scaling_factor,
        )

        # Denormalize audio latents
        audio_latents = denormalize_audio_latents(
            audio_latents,
            self.audio_vae.latents_mean,
            self.audio_vae.latents_std,
        )

        # Decode video
        print("Phase 3: Decoding video...")
        latents = latents.to(torch.bfloat16)
        with torch.no_grad():
            # Use the full VAE decode (handles timestep conditioning)
            if self.vae.config.timestep_conditioning:
                decode_timestep = torch.tensor([0.0], device=self.device, dtype=torch.bfloat16)
            else:
                decode_timestep = None
            video = self.vae.decode(latents, decode_timestep, return_dict=False)[0]
        torch_xla.sync(wait=True)
        print(f"  Video output shape: {video.shape}")

        # Decode audio: VAE decode -> mel spectrogram -> vocoder -> waveform
        print("Phase 3: Decoding audio...")
        audio_latents = audio_latents.to(torch.bfloat16)
        with torch.no_grad():
            mel_spectrogram = self.audio_vae.decode(audio_latents, return_dict=False)[0]
            audio = self.compiled_vocoder(mel_spectrogram)
        torch_xla.sync(wait=True)
        print(f"  Audio output shape: {audio.shape}")

        self._offload_phase3()

        return video, audio

    # ------ Full generation ------

    def generate(
        self,
        prompt="A cat playing piano in a jazz club",
        negative_prompt="",
        height=512,
        width=320,
        num_frames=49,
        frame_rate=24.0,
        num_inference_steps=5,
        guidance_scale=4.0,
        max_sequence_length=256,
        seed=42,
    ):
        """Run full text-to-video+audio generation pipeline."""
        generator = torch.Generator(device="cpu")
        if seed is not None:
            generator.manual_seed(seed)

        print(f"\n{'='*60}")
        print(f"LTX-2 Pipeline: {height}x{width}, {num_frames} frames @ {frame_rate} fps")
        print(f"Prompt: {prompt}")
        print(f"Steps: {num_inference_steps}, Guidance: {guidance_scale}")
        print(f"Devices: {self.num_devices} x p150, Sequential offload: {self.use_sequential_offload}")
        print(f"{'='*60}\n")

        total_start = time.time()

        # Phase 1: Text encoding
        phase1_start = time.time()
        (
            connector_prompt_embeds,
            connector_audio_prompt_embeds,
            connector_mask,
        ) = self.encode_prompt(prompt, negative_prompt, max_sequence_length)
        phase1_time = time.time() - phase1_start
        print(f"Phase 1 complete: {phase1_time:.2f}s")
        print(f"  Video conditioning: {connector_prompt_embeds.shape}")
        print(f"  Audio conditioning: {connector_audio_prompt_embeds.shape}")

        # Phase 2: Denoising
        phase2_start = time.time()
        (
            latents,
            audio_latents,
            latent_num_frames,
            latent_height,
            latent_width,
            audio_num_frames,
            latent_mel_bins,
        ) = self.denoise(
            connector_prompt_embeds,
            connector_audio_prompt_embeds,
            connector_mask,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        phase2_time = time.time() - phase2_start
        print(f"Phase 2 complete: {phase2_time:.2f}s")
        print(f"  Video latents: {latents.shape}")
        print(f"  Audio latents: {audio_latents.shape}")

        # Phase 3: Decoding
        phase3_start = time.time()
        video, audio = self.decode(latents, audio_latents)
        phase3_time = time.time() - phase3_start
        print(f"Phase 3 complete: {phase3_time:.2f}s")

        total_time = time.time() - total_start
        print(f"\n{'='*60}")
        print(f"Total generation time: {total_time:.2f}s")
        print(f"  Phase 1 (text encoding):  {phase1_time:.2f}s")
        print(f"  Phase 2 (denoising):      {phase2_time:.2f}s")
        print(f"  Phase 3 (decoding):       {phase3_time:.2f}s")
        print(f"{'='*60}\n")

        return video, audio


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_ltx2_pipeline(
    prompt="A cinematic shot of a futuristic city at sunset",
    num_inference_steps=5,
    height=512,
    width=320,
    num_frames=49,
):
    xr.set_device_type("TT")

    pipeline = LTX2TTPipeline(use_sequential_offload=True)
    pipeline.setup()

    # Warm-up pass (compilation)
    print("\n=== WARM-UP PASS (compilation) ===")
    video, audio = pipeline.generate(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        height=height,
        width=width,
        num_frames=num_frames,
        seed=42,
    )

    # Timed pass
    print("\n=== TIMED PASS ===")
    video, audio = pipeline.generate(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        height=height,
        width=width,
        num_frames=num_frames,
        seed=42,
    )

    return video, audio


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LTX-2 pipeline on TT hardware")
    parser.add_argument("--prompt", type=str, default="A cinematic shot of a futuristic city at sunset")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--num_inference_steps", type=int, default=5)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--frame_rate", type=float, default=24.0)
    parser.add_argument("--guidance_scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_offload", action="store_true", help="Disable sequential offload")
    args = parser.parse_args()

    xr.set_device_type("TT")

    pipeline = LTX2TTPipeline(use_sequential_offload=not args.no_offload)
    pipeline.setup()

    # Warm-up pass
    print("\n=== WARM-UP PASS (compilation) ===")
    pipeline.generate(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.num_inference_steps,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )

    # Timed pass
    print("\n=== TIMED PASS ===")
    video, audio = pipeline.generate(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.num_inference_steps,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )

    print(f"Final video shape: {video.shape}")
    print(f"Final audio shape: {audio.shape}")
