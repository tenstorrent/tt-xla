# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen-Image (Qwen/Qwen-Image) text-to-image pipeline for TT hardware.

Runs the ~27B parameter flow-matching diffusion model on Tenstorrent devices
using SPMD tensor parallelism for the 40.9 GB MMDiT transformer.

Architecture:
  - Text Encoder (Qwen2.5-VL-7B, 16.6 GB): runs on CPU
  - Transformer (MMDiT, 40.9 GB, 60 layers, 24 heads): compiled on TT with TP
  - VAE Decoder (AutoencoderKLQwenImage, 0.254 GB): runs on CPU
  - Scheduler (FlowMatchEulerDiscreteScheduler): runs on CPU

Hardware requirements (from bringup report):
  - Recommended: bhqb / ge (4 x p150, 128 GB total DRAM)
  - Minimum viable: p300 (2 x p150, 64 GB total DRAM) with sequential offloading
  - NOT feasible on any single-device configuration

Inference pipeline flow:
  1. Text encoder produces embeddings (CPU) -> offloadable after completion
  2. Transformer runs iterative denoising (default 50 steps) on TT -> memory-dominant
  3. VAE decoder converts latents to pixel space (CPU) -> lightweight final step

Requires: diffusers >= 0.33.0, transformers >= 4.49.0

Usage:
  python run_qwen_image.py --prompt "a cat sitting on a windowsill"
  python run_qwen_image.py --prompt "a sunset over mountains" --resolution 1024
  python run_qwen_image.py --use-dummy-inputs  # skip text encoder, use random embeddings
"""

import argparse
import math
import os
import time

import numpy as np
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh

from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers import QwenImageTransformer2DModel
from diffusers import AutoencoderKLQwenImage
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer
from PIL import Image


MODEL_ID = "Qwen/Qwen-Image"

# The pipeline wraps prompts in a chat template. The first 34 tokens (system
# prefix) are dropped from the text encoder output before passing to the
# transformer, matching the diffusers QwenImagePipeline behavior.
PROMPT_TEMPLATE = (
    "<|im_start|>system\n"
    "Describe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:"
    "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)
PROMPT_TEMPLATE_PREFIX_LEN = 34
TOKENIZER_MAX_LENGTH = 1024


class QwenImageTransformerWrapper(torch.nn.Module):
    """Wraps QwenImageTransformer2DModel to return a plain tensor.

    The raw transformer returns a Transformer2DModelOutput dataclass, which
    causes graph breaks with torch.compile. This wrapper extracts the sample
    tensor directly, matching the pattern used in mochi_dit_sharded.py.
    """

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor,
        img_shapes: list,
        guidance: torch.Tensor = None,
    ) -> torch.Tensor:
        output = self.transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            img_shapes=img_shapes,
            guidance=guidance,
            return_dict=False,
        )
        return output[0]


def apply_transformer_sharding(transformer, mesh):
    """Apply Megatron-style tensor parallelism to the MMDiT transformer.

    Sharding strategy (matching qwen3_tp.py pattern):
      - Column-parallel: Q/K/V projections, MLP gate+up (GEGLU)
        Splits output dimension across devices. Each device computes distinct
        output chunks with no communication required.
      - Row-parallel: output projections, MLP down
        Splits input dimension across devices. Each device computes partial
        results that are all-reduced at the end.

    The MMDiT has dual-stream (image + text) attention and MLP in each block.
    Both streams are sharded identically.

    From bringup report:
      - 24 attention heads / 4 devices = 6 heads/device (clean division)
      - 60 layers / 4 devices = 15 layers/device (clean division)
      - ~95-98% of transformer weights are shardable
    """
    shard_specs = {}

    for name, param in transformer.named_parameters():
        # Column-parallel: Q/K/V projections for both image and text streams.
        if any(k in name for k in [
            "to_q.weight", "to_k.weight", "to_v.weight",
            "add_q_proj.weight", "add_k_proj.weight", "add_v_proj.weight",
        ]):
            shard_specs[param] = ("model", None)

        # Column-parallel: MLP gate+up (GEGLU) for both streams.
        elif any(k in name for k in [
            ".ff.net.0.proj.weight",
            ".ff_context.net.0.proj.weight",
        ]):
            shard_specs[param] = ("model", None)

        # Row-parallel: attention output projections for both streams.
        elif any(k in name for k in [
            "to_out.0.weight",
            "to_add_out.weight",
        ]):
            shard_specs[param] = (None, "model")

        # Row-parallel: MLP down projections for both streams.
        elif any(k in name for k in [
            ".ff.net.2.weight",
            ".ff_context.net.2.weight",
        ]):
            shard_specs[param] = (None, "model")

    for tensor, spec in shard_specs.items():
        xs.mark_sharding(tensor, mesh, spec)

    return len(shard_specs)


def calculate_shift(
    image_seq_len,
    base_seq_len=256,
    max_seq_len=8192,
    base_shift=None,
    max_shift=None,
):
    """Compute dynamic time-shift mu for flow-matching scheduler.

    The shift adjusts the noise schedule based on the image sequence length.
    For Qwen-Image defaults (base_shift == max_shift == ln(3)), this returns
    a constant ln(3) ~ 1.099 regardless of resolution.
    """
    if base_shift is None:
        base_shift = math.log(3)
    if max_shift is None:
        max_shift = math.log(3)

    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def pack_latents(latents, height, width):
    """Pack spatial latents into sequence format via 2x2 patch grouping.

    Input:  (B, C, H, W)  -- spatial latent representation
    Output: (B, H*W/4, C*4) -- sequence of 2x2 patches

    This matches the QwenImagePipeline._pack_latents method. The transformer
    operates on the packed sequence representation.
    """
    B, C, H, W = latents.shape
    latents = latents.reshape(B, C, H // 2, 2, W // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)  # (B, H//2, W//2, C, 2, 2)
    latents = latents.reshape(B, (H // 2) * (W // 2), C * 4)
    return latents


def unpack_latents(latents, height, width, channels=16):
    """Unpack sequence latents back to spatial format.

    Input:  (B, H*W/4, C*4) -- sequence representation
    Output: (B, C, H, W)    -- spatial latent representation

    Reverses pack_latents, matching QwenImagePipeline._unpack_latents.
    """
    B = latents.shape[0]
    h, w = height // 2, width // 2
    latents = latents.reshape(B, h, w, channels, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)  # (B, C, h, 2, w, 2)
    latents = latents.reshape(B, channels, height, width)
    return latents


class QwenImagePipelineTT:
    """Qwen-Image text-to-image pipeline with MMDiT transformer on TT hardware.

    This pipeline manually orchestrates the diffusion process, loading the
    transformer onto TT devices with SPMD tensor parallelism while keeping
    the text encoder and VAE on CPU. This follows the same pattern as
    sdxl-pipeline.py and sd_v1_4_pipeline.py in examples/pytorch/.
    """

    def __init__(
        self,
        resolution=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        seed=42,
        use_dummy_inputs=False,
        offload_text_encoder=True,
    ):
        self.resolution = resolution
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.seed = seed
        self.use_dummy_inputs = use_dummy_inputs
        self.offload_text_encoder = offload_text_encoder

        self.vae_scale_factor = 8
        self.latent_channels = 16
        # Align latent dims to vae_scale_factor * 2 (for 2x2 packing).
        self.latent_height = 2 * (resolution // (self.vae_scale_factor * 2))
        self.latent_width = 2 * (resolution // (self.vae_scale_factor * 2))

    def setup(self):
        """Load all model components and compile transformer on TT."""
        print("=" * 60)
        print("Qwen-Image Pipeline Setup")
        print(f"  Model: {MODEL_ID}")
        print(f"  Resolution: {self.resolution}x{self.resolution}")
        print(f"  Latent size: {self.latent_height}x{self.latent_width}")
        print("=" * 60)

        # Enable SPMD for tensor parallelism.
        os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
        xr.use_spmd()
        self.num_devices = xr.global_runtime_device_count()
        print(f"TT devices available: {self.num_devices}")

        # Create SPMD mesh for tensor parallelism.
        device_ids = np.array(range(self.num_devices))
        mesh_shape = (1, self.num_devices)
        self.mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))
        self.device = torch_xla.device()

        self._load_scheduler()
        self._load_text_encoder()
        self._load_transformer()
        self._load_vae()

        print("Setup complete.\n")

    def _load_scheduler(self):
        """Load FlowMatchEulerDiscreteScheduler (CPU, negligible memory)."""
        print("Loading scheduler...")
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            MODEL_ID, subfolder="scheduler"
        )

    def _load_text_encoder(self):
        """Load Qwen2.5-VL-7B text encoder + tokenizer (CPU, 16.6 GB)."""
        if self.use_dummy_inputs:
            print("Using dummy text embeddings (skipping text encoder load).")
            self.tokenizer = None
            self.text_encoder = None
            return

        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, subfolder="tokenizer"
        )

        print("Loading text encoder (Qwen2.5-VL-7B, 16.6 GB) on CPU...")
        self.text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            subfolder="text_encoder",
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )
        self.text_encoder.eval()

    def _load_transformer(self):
        """Load MMDiT transformer, apply TP sharding, compile on TT."""
        print("Loading transformer (MMDiT, 40.9 GB)...")
        transformer = QwenImageTransformer2DModel.from_pretrained(
            MODEL_ID,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )
        transformer.eval()

        self.has_guidance_embeds = getattr(
            transformer.config, "guidance_embeds", False
        )

        # Wrap to return plain tensor (avoid dataclass graph breaks).
        self.transformer_wrapper = QwenImageTransformerWrapper(transformer)
        self.transformer_wrapper.eval()

        # Move to TT device.
        print("Moving transformer to TT device...")
        self.transformer_wrapper = self.transformer_wrapper.to(self.device)

        # Validate head divisibility for tensor parallelism.
        num_heads = transformer.config.num_attention_heads
        if num_heads % self.num_devices != 0:
            raise ValueError(
                f"Attention heads ({num_heads}) must be divisible by device "
                f"count ({self.num_devices}) for head-parallel sharding. "
                f"Valid device counts for {num_heads} heads: "
                f"{[d for d in range(1, num_heads+1) if num_heads % d == 0]}"
            )

        # Apply Megatron-style tensor parallelism.
        print(
            f"Applying tensor parallelism across {self.num_devices} devices "
            f"({num_heads} heads -> {num_heads // self.num_devices} heads/device)..."
        )
        num_sharded = apply_transformer_sharding(
            self.transformer_wrapper.transformer, self.mesh
        )
        print(f"  Sharded {num_sharded} parameter tensors.")

        # Compile on TT backend.
        print("Compiling transformer on TT backend...")
        self.compiled_transformer = torch.compile(
            self.transformer_wrapper, backend="tt"
        )

    def _load_vae(self):
        """Load AutoencoderKLQwenImage VAE decoder (CPU, 0.254 GB)."""
        print("Loading VAE decoder (0.254 GB) on CPU...")
        self.vae = AutoencoderKLQwenImage.from_pretrained(
            MODEL_ID,
            subfolder="vae",
            torch_dtype=torch.float32,
        )
        self.vae.eval()

    def encode_prompt(self, prompt):
        """Encode text prompt using Qwen2.5-VL-7B on CPU.

        Returns:
            prompt_embeds: (1, seq_len, 3584) text embeddings
            prompt_mask: (1, seq_len) boolean attention mask
        """
        if self.use_dummy_inputs:
            prompt_embeds = torch.randn(
                1, 256, 3584, dtype=torch.bfloat16
            )
            prompt_mask = torch.ones(1, 256, dtype=torch.bool)
            return prompt_embeds, prompt_mask

        formatted_prompt = PROMPT_TEMPLATE.format(prompt)
        max_length = TOKENIZER_MAX_LENGTH + PROMPT_TEMPLATE_PREFIX_LEN

        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
            truncation=True,
        )

        with torch.no_grad():
            outputs = self.text_encoder(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                output_hidden_states=True,
            )

        # Use last hidden state, drop the system prompt prefix tokens.
        hidden_states = outputs.hidden_states[-1]
        prompt_embeds = hidden_states[:, PROMPT_TEMPLATE_PREFIX_LEN:]
        prompt_mask = inputs.attention_mask[:, PROMPT_TEMPLATE_PREFIX_LEN:]

        return prompt_embeds.to(torch.bfloat16), prompt_mask.bool()

    def prepare_latents(self):
        """Create initial noise latents in packed sequence format.

        Returns:
            latents: (1, H*W/4, C*4) packed noise latents
        """
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)

        shape = (1, self.latent_channels, self.latent_height, self.latent_width)
        latents = torch.randn(shape, generator=generator, dtype=torch.bfloat16)

        # Pack into sequence format for the transformer.
        latents = pack_latents(latents, self.latent_height, self.latent_width)
        return latents

    def setup_timesteps(self, num_steps):
        """Configure scheduler timesteps with dynamic time shifting.

        Uses exponential time shifting based on image sequence length,
        matching the QwenImagePipeline behavior.
        """
        sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)
        image_seq_len = (self.latent_height // 2) * (self.latent_width // 2)

        # Read shift params from scheduler config, fall back to defaults.
        cfg = self.scheduler.config
        mu = calculate_shift(
            image_seq_len,
            base_seq_len=getattr(cfg, "base_image_seq_len", 256),
            max_seq_len=getattr(cfg, "max_image_seq_len", 8192),
            base_shift=getattr(cfg, "base_shift", math.log(3)),
            max_shift=getattr(cfg, "max_shift", math.log(3)),
        )

        self.scheduler.set_timesteps(sigmas=sigmas, mu=mu)
        return self.scheduler.timesteps

    def generate(self, prompt="a photo of a cat", num_steps=None):
        """Run the full text-to-image generation pipeline.

        Args:
            prompt: Text description of the image to generate.
            num_steps: Number of denoising steps (overrides self.num_inference_steps).

        Returns:
            image: (1, 3, H, W) decoded image tensor.
        """
        if num_steps is None:
            num_steps = self.num_inference_steps

        print(f"\nGenerating: '{prompt}'")
        print(
            f"  Resolution: {self.resolution}x{self.resolution}, "
            f"Steps: {num_steps}, Guidance: {self.guidance_scale}"
        )

        # --- Step 1: Text encoding (CPU) ---
        t0 = time.time()
        prompt_embeds, prompt_mask = self.encode_prompt(prompt)
        t_text = time.time() - t0
        print(f"  Text encoding: {t_text:.2f}s")

        # Optionally free text encoder memory after encoding.
        if self.offload_text_encoder and self.text_encoder is not None:
            del self.text_encoder
            self.text_encoder = None
            import gc
            gc.collect()
            print("  Text encoder offloaded from CPU memory.")

        # --- Step 2: Prepare latents ---
        latents = self.prepare_latents()

        # --- Step 3: Configure scheduler timesteps ---
        timesteps = self.setup_timesteps(num_steps)

        # Guidance tensor for guidance-distilled variants.
        guidance = None
        if self.has_guidance_embeds:
            guidance = torch.full(
                [1], self.guidance_scale, dtype=torch.bfloat16
            )

        # img_shapes for RoPE position embeddings: (temporal, H_latent, W_latent).
        img_shapes = [(1, self.latent_height, self.latent_width)]

        # --- Step 4: Denoising loop (transformer on TT) ---
        print(f"  Denoising ({num_steps} steps)...")
        t0 = time.time()

        with torch.no_grad():
            for i, t in enumerate(timesteps):
                if i % 10 == 0 or i == num_steps - 1:
                    print(f"    Step {i + 1}/{num_steps}")

                # Prepare timestep: expand to batch and divide by 1000
                # (flow-matching convention used by QwenImagePipeline).
                timestep = t.expand(latents.shape[0]).to(torch.bfloat16) / 1000.0

                # Move inputs to TT device.
                latents_tt = latents.to(dtype=torch.bfloat16, device=self.device)
                timestep_tt = timestep.to(self.device)
                embeds_tt = prompt_embeds.to(
                    dtype=torch.bfloat16, device=self.device
                )
                mask_tt = prompt_mask.to(self.device)

                guidance_tt = None
                if guidance is not None:
                    guidance_tt = guidance.to(self.device)

                # Run transformer on TT.
                noise_pred = self.compiled_transformer(
                    hidden_states=latents_tt,
                    timestep=timestep_tt,
                    encoder_hidden_states=embeds_tt,
                    encoder_hidden_states_mask=mask_tt,
                    img_shapes=img_shapes,
                    guidance=guidance_tt,
                )

                # Move prediction back to CPU for scheduler step.
                noise_pred = noise_pred.cpu().to(torch.bfloat16)

                # Scheduler step (CPU).
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

        t_denoise = time.time() - t0
        print(
            f"  Denoising: {t_denoise:.2f}s "
            f"({t_denoise / num_steps:.2f}s/step)"
        )

        # --- Step 5: Unpack latents and VAE decode (CPU) ---
        t0 = time.time()
        latents = unpack_latents(
            latents, self.latent_height, self.latent_width, self.latent_channels
        )
        latents = latents.to(torch.float32)

        with torch.no_grad():
            image = self.vae.decode(latents, return_dict=False)[0]
            # Squeeze temporal dimension if present (3D VAE outputs B,C,T,H,W).
            if image.ndim == 5:
                image = image[:, :, 0]

        t_vae = time.time() - t0
        print(f"  VAE decode: {t_vae:.2f}s")

        return image


def save_image(image, filepath="output.png"):
    """Post-process and save generated image tensor to file."""
    image = (torch.clamp(image / 2 + 0.5, 0.0, 1.0) * 255.0).to(torch.uint8)
    image_np = image.cpu().squeeze().numpy()
    if image_np.ndim == 3 and image_np.shape[0] == 3:
        image_np = image_np.transpose(1, 2, 0)  # CHW -> HWC
    Image.fromarray(image_np).save(filepath)
    print(f"Image saved to: {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Qwen-Image text-to-image on TT hardware"
    )
    parser.add_argument(
        "--prompt", type=str, default="a photo of a cat",
        help="Text prompt for image generation",
    )
    parser.add_argument(
        "--resolution", type=int, default=512, choices=[512, 1024],
        help="Output image resolution (512 or 1024)",
    )
    parser.add_argument(
        "--steps", type=int, default=50,
        help="Number of denoising steps",
    )
    parser.add_argument(
        "--guidance-scale", type=float, default=7.5,
        help="Guidance scale for generation",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output", type=str, default="qwen_image_output.png",
        help="Output image file path",
    )
    parser.add_argument(
        "--use-dummy-inputs", action="store_true",
        help="Use random embeddings instead of loading the 16.6 GB text encoder",
    )
    parser.add_argument(
        "--no-offload-text-encoder", action="store_true",
        help="Keep text encoder in CPU memory after encoding (default: offload)",
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=2,
        help="Number of denoising steps for the warmup/compilation pass",
    )
    parser.add_argument(
        "--optimization-level", type=int, default=1,
        help="TT-MLIR optimization level (0-2)",
    )
    args = parser.parse_args()

    # Set TT device type and compilation options.
    xr.set_device_type("TT")
    torch_xla.set_custom_compile_options(
        {"optimization_level": args.optimization_level}
    )

    pipeline = QwenImagePipelineTT(
        resolution=args.resolution,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        use_dummy_inputs=args.use_dummy_inputs,
        offload_text_encoder=not args.no_offload_text_encoder,
    )
    pipeline.setup()

    # --- Warmup pass (triggers compilation) ---
    print("=" * 60)
    print("WARMUP PASS (compilation)")
    print("=" * 60)
    t0 = time.time()
    _ = pipeline.generate(prompt=args.prompt, num_steps=args.warmup_steps)
    t_warmup = time.time() - t0
    print(f"Warmup total: {t_warmup:.2f}s\n")

    # --- Real inference pass ---
    print("=" * 60)
    print("INFERENCE PASS")
    print("=" * 60)
    t0 = time.time()
    image = pipeline.generate(prompt=args.prompt, num_steps=args.steps)
    t_inference = time.time() - t0
    print(f"Inference total: {t_inference:.2f}s\n")

    save_image(image, args.output)

    print("=" * 60)
    print("Summary")
    print(f"  Warmup (compilation): {t_warmup:.2f}s")
    print(f"  Inference ({args.steps} steps): {t_inference:.2f}s")
    print(f"  Output: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
