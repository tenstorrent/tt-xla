# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from typing import Optional

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from diffusers import AutoencoderKL, PNDMScheduler, UNet2DConditionModel
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer


class SD14Config:
    def __init__(self, device="cpu", vae_on_tt=False, clip_on_tt=True):
        self.model_id = "CompVis/stable-diffusion-v1-4"
        self.width = 512
        self.height = 512
        self.latents_width = self.width // 8
        self.latents_height = self.height // 8
        self.device = device
        self.vae_on_tt = vae_on_tt
        self.clip_on_tt = clip_on_tt


class SD14Pipeline:
    """Pipeline for text-to-image generation with Stable Diffusion 1.4."""

    def __init__(self, config: SD14Config):
        self.config = config
        self.device = config.device
        self.model_id = config.model_id
        self.latents_width = config.latents_width
        self.latents_height = config.latents_height
        self.vae_on_tt = config.vae_on_tt
        self.clip_on_tt = config.clip_on_tt

    def setup(self):
        self.load_models()
        self.load_scheduler()
        self.load_tokenizer()

    def load_models(self):
        self.vae = AutoencoderKL.from_pretrained(
            self.model_id,
            subfolder="vae",
            torch_dtype=torch.float32,
            device_map=self.device,
        )
        if self.vae_on_tt:
            self.vae.compile(backend="tt")
            self.vae = self.vae.to(xm.xla_device())

        self.unet = UNet2DConditionModel.from_pretrained(
            self.model_id,
            subfolder="unet",
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )
        self.unet.compile(backend="tt")
        self.unet = self.unet.to(xm.xla_device())

        self.text_encoder = CLIPTextModel.from_pretrained(
            self.model_id,
            subfolder="text_encoder",
            torch_dtype=torch.bfloat16 if self.clip_on_tt else torch.float16,
            device_map=self.device,
        )

        if self.clip_on_tt:
            self.text_encoder.compile(backend="tt")
            self.text_encoder = self.text_encoder.to(xm.xla_device())

    def load_scheduler(self):
        self.scheduler = PNDMScheduler.from_pretrained(
            self.model_id, subfolder="scheduler"
        )

    def load_tokenizer(self):
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.model_id, subfolder="tokenizer"
        )

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        cfg_scale: float = 7.5,
        num_inference_steps: int = 50,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate an image from a text prompt. Returns tensor (B, 3, H, W)."""

        batch_size = 1 if isinstance(prompt, str) else len(prompt)

        tt_cast = lambda x: (
            x.to(dtype=torch.bfloat16).to(device=xm.xla_device())
            if x.device == torch.device("cpu")
            else x.to(dtype=torch.bfloat16)
        )
        cpu_cast = lambda x: x.to("cpu").to(dtype=torch.float16)

        with torch.no_grad():
            generator = torch.Generator(device="cpu")
            if seed is not None:
                generator.manual_seed(seed)
            else:
                generator.seed()

            # --- Text encoding (CLIP) ---
            negative_prompt = negative_prompt or ""

            cond_tokens = self.tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            uncond_tokens = self.tokenizer.batch_encode_plus(
                [negative_prompt], padding="max_length", max_length=77
            ).input_ids

            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long).to(
                device=self.device
            )
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long).to(
                device=self.device
            )

            if self.clip_on_tt:
                cond_tokens = cond_tokens.to(device=xm.xla_device())
                uncond_tokens = uncond_tokens.to(device=xm.xla_device())

            cond_hidden_state = self.text_encoder(cond_tokens)[0]  # (B, 77, 768)
            uncond_hidden_state = self.text_encoder(uncond_tokens)[0]  # (B, 77, 768)

            if self.clip_on_tt:
                cond_hidden_state = cpu_cast(cond_hidden_state)
                uncond_hidden_state = cpu_cast(uncond_hidden_state)

            encoder_hidden_states = torch.cat(
                [uncond_hidden_state, cond_hidden_state], dim=0
            )  # (2B, 77, 768)

            # --- Prepare timesteps ---
            self.scheduler.set_timesteps(num_inference_steps)

            # --- Prepare latents ---
            latent_shape = (batch_size, 4, self.latents_height, self.latents_width)
            latents = torch.randn(
                latent_shape, generator=generator, dtype=torch.float16
            ).to(device=self.device)
            latents = latents * self.scheduler.init_noise_sigma

            # --- Denoising loop (UNet on TT) ---
            for i, timestep in enumerate(self.scheduler.timesteps):

                model_input = torch.cat([latents] * 2)
                model_input = self.scheduler.scale_model_input(model_input, timestep)

                # CPU → TT
                model_input = tt_cast(model_input)
                timestep_tt = tt_cast(timestep.unsqueeze(0))
                encoder_hidden_states = tt_cast(encoder_hidden_states)

                unet_output = self.unet(
                    model_input,
                    timestep_tt,
                    encoder_hidden_states,
                ).sample

                # TT → CPU
                unet_output = cpu_cast(unet_output)

                # CFG blending (CPU)
                uncond_output, cond_output = unet_output.chunk(2)
                model_output = uncond_output + (cond_output - uncond_output) * cfg_scale

                # Scheduler step (CPU)
                latents = cpu_cast(latents)
                latents = self.scheduler.step(
                    model_output, timestep, latents
                ).prev_sample

            # --- VAE decode ---
            latents = latents / self.vae.config.scaling_factor
            latents = latents.to(dtype=torch.float32)
            if self.vae_on_tt:
                latents = latents.to(device=xm.xla_device())
            images = self.vae.decode(latents).sample
            if self.vae_on_tt:
                images = cpu_cast(images)

            return images


def save_image(image: torch.Tensor, filepath: str = "output.png"):
    """Rescale, reshape and save the image from pipeline output."""
    image = (
        (torch.clamp(image / 2 + 0.5, 0.0, 1.0) * 255.0).round().to(dtype=torch.uint8)
    )
    image_np = image.cpu().squeeze().numpy()
    assert image_np.ndim == 3, "Image must be 3D"
    if image_np.shape[0] == 3:
        image_np = image_np.transpose(1, 2, 0)
    Image.fromarray(image_np).save(filepath)


def run_sd14_pipeline(output_path: str = "output.png", num_inference_steps: int = 50):
    """Run SD 1.4 pipeline and save output image."""
    torch_xla.set_custom_compile_options({"optimization_level": 1})

    config = SD14Config(device="cpu")
    pipeline = SD14Pipeline(config=config)
    pipeline.setup()

    img = pipeline.generate(
        prompt="a photo of a cat",
        negative_prompt="",
        cfg_scale=7.5,
        num_inference_steps=num_inference_steps,
        seed=42,
    )

    save_image(img, output_path)
    return output_path


def test_sd14_pipeline():
    """Test SD 1.4 pipeline generates valid output image."""
    xr.set_device_type("TT")

    output_path = "test_sd14_output.png"
    output_file = Path(output_path)
    if output_file.exists():
        output_file.unlink()

    try:
        run_sd14_pipeline(output_path=output_path, num_inference_steps=50)

        assert output_file.exists(), f"Output image {output_path} was not created"

        with Image.open(output_path) as img:
            width, height = img.size
            assert width == 512, f"Expected width 512, got {width}"
            assert height == 512, f"Expected height 512, got {height}"

        print(f"Output image created with resolution {width}x{height}")

    finally:
        if output_file.exists():
            output_file.unlink()
            print(f"Cleaned up {output_path}")


if __name__ == "__main__":
    test_sd14_pipeline()
