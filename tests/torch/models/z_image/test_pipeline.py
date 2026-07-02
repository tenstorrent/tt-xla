# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Z-Image — end-to-end text-to-image pipeline (Tongyi-MAI/Z-Image).

Stitches the three validated single-chip components into the full generation
flow, mirroring diffusers ``ZImagePipeline.__call__``:

    Qwen3 text encoder -> ZImageTransformer2DModel denoising loop (CFG +
    FlowMatchEulerDiscreteScheduler) -> AutoencoderKL decode -> PIL image.

Each component compiles with the ``tt`` backend and runs on a single
Blackhole chip. Source inference parameters (prompt, 1280x720, 4 steps,
guidance_scale=4.0) are used so the run produces a realistic image, matching
the pattern of the FLUX.1 / FLUX.2 / Janus-Pro component-based model tests.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.image_processor import VaeImageProcessor

from third_party.tt_forge_models.z_image.pytorch.src.model_utils import (
    CFG_NORMALIZATION,
    DTYPE,
    GUIDANCE_SCALE,
    HEIGHT,
    LATENT_CHANNELS,
    MAX_SEQUENCE_LENGTH,
    NEGATIVE_PROMPT,
    NUM_INFERENCE_STEPS,
    PROMPT,
    REPO_ID,
    SEED,
    VAE_SCALE_FACTOR,
    WIDTH,
    load_text_encoder,
    load_tokenizer,
    load_transformer,
    load_vae,
)


# --- diffusers.pipelines.z_image.pipeline_z_image helpers (inlined) ---------


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def retrieve_timesteps(scheduler, num_inference_steps, device, **kwargs):
    scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
    return scheduler.timesteps, num_inference_steps


# --- TT-compilable component wrappers (tensor in / tensor out) ---------------


class TextEncoderWrapper(torch.nn.Module):
    """Qwen3 encoder -> penultimate hidden state (hidden_states[-2])."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        return out.hidden_states[-2]


class CondTransformerWrapper(torch.nn.Module):
    """Single (conditional) transformer pass; cap_feats is (L, D)."""

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(self, latents, timestep, cap_feats):
        x_list = list(latents.unsqueeze(2).unbind(dim=0))
        t = timestep.reshape(-1).to(dtype=latents.dtype)
        out = self.transformer(x_list, t, [cap_feats], return_dict=False)[0]
        return torch.stack([o.float() for o in out], dim=0)


class VaeDecodeWrapper(torch.nn.Module):
    """Undo latent scaling, then AutoencoderKL.decode -> pixels."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latents):
        z = latents.to(dtype=self.vae.dtype)
        z = (z / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        return self.vae.decode(z, return_dict=False)[0]


# --- Pipeline ---------------------------------------------------------------


class ZImagePipeline:
    """Self-contained Z-Image text-to-image pipeline for TT bring-up."""

    def __init__(self, run_on_tt: bool = True, dtype: torch.dtype = DTYPE):
        self.run_on_tt = run_on_tt
        self.dtype = dtype
        self.vae_scale_factor = VAE_SCALE_FACTOR

    def setup(self):
        self.tokenizer = load_tokenizer()
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            REPO_ID, subfolder="scheduler"
        )

        text_encoder = TextEncoderWrapper(load_text_encoder(self.dtype)).eval()
        transformer = load_transformer(self.dtype)
        # Complex-valued RoPE only legalizes at batch=1 on the pinned tt-mlir
        # (batch=2 broadcast fails, tt-mlir #8874), so classifier-free guidance
        # runs cond and uncond as two separate batch=1 passes (as WAN does).
        self.transformer = CondTransformerWrapper(transformer).eval()
        vae = load_vae(self.dtype)
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor * 2
        )
        vae_decoder = VaeDecodeWrapper(vae).eval()

        if self.run_on_tt:
            device = xm.xla_device()
            text_encoder.compile(backend="tt")
            self.transformer.compile(backend="tt")
            vae_decoder.compile(backend="tt")
            self.text_encoder = text_encoder.to(device)
            self.transformer = self.transformer.to(device)
            self.vae_decoder = vae_decoder.to(device)
            self._device = device
        else:
            self.text_encoder = text_encoder
            self.vae_decoder = vae_decoder
            self._device = torch.device("cpu")

    def _to_tt(self, x):
        return x.to(device=self._device) if self.run_on_tt else x

    @staticmethod
    def _to_cpu(x):
        return x.to("cpu")

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        """Tokenize (chat template) -> penultimate hidden state, mask-trimmed."""
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        text_inputs = self.tokenizer(
            [text],
            padding="max_length",
            max_length=MAX_SEQUENCE_LENGTH,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = self._to_tt(text_inputs.input_ids)
        attention_mask = self._to_tt(text_inputs.attention_mask.bool())

        hidden = self.text_encoder(input_ids, attention_mask)
        hidden = self._to_cpu(hidden)
        mask = text_inputs.attention_mask[0].bool()
        # Ragged, mask-trimmed embedding for this prompt: (valid_len, dim).
        return hidden[0][mask].to(self.dtype)

    def _transformer_step(self, latents, timestep, cap_feats):
        """One batch=1 transformer pass; returns CPU fp32 (1, C, 1, H, W)."""
        out = self.transformer(
            self._to_tt(latents),
            self._to_tt(timestep),
            self._to_tt(cap_feats),
        )
        return self._to_cpu(out).float()

    def generate(
        self,
        prompt: str = PROMPT,
        negative_prompt: str = NEGATIVE_PROMPT,
        height: int = HEIGHT,
        width: int = WIDTH,
        num_inference_steps: int = NUM_INFERENCE_STEPS,
        guidance_scale: float = GUIDANCE_SCALE,
        cfg_normalization: bool = CFG_NORMALIZATION,
        seed: int = SEED,
        output_type: str = "pil",
    ):
        do_cfg = guidance_scale > 0
        cpu = torch.device("cpu")

        with torch.no_grad():
            # 1. Text encoding (Qwen3, penultimate layer).
            cap_pos = self._encode_prompt(prompt)
            cap_neg = self._encode_prompt(negative_prompt) if do_cfg else None

            # 2. Latents (fp32 on CPU; scheduler math stays fp32).
            latent_h = 2 * (int(height) // (self.vae_scale_factor * 2))
            latent_w = 2 * (int(width) // (self.vae_scale_factor * 2))
            generator = torch.Generator(device="cpu").manual_seed(seed)
            latents = torch.randn(
                (1, LATENT_CHANNELS, latent_h, latent_w),
                generator=generator,
                dtype=torch.float32,
                device=cpu,
            )

            # 3. Timesteps with resolution-dependent shift (mu).
            image_seq_len = (latent_h // 2) * (latent_w // 2)
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.15),
            )
            self.scheduler.sigma_min = 0.0
            set_ts_kwargs = {}
            if "mu" in inspect.signature(self.scheduler.set_timesteps).parameters:
                set_ts_kwargs["mu"] = mu
            timesteps, _ = retrieve_timesteps(
                self.scheduler, num_inference_steps, cpu, **set_ts_kwargs
            )
            self.scheduler.set_begin_index(0)

            # 4. Denoising loop.
            for i, t in enumerate(timesteps):
                timestep = t.expand(1)
                timestep = (1000 - timestep) / 1000
                latent_input = latents.to(self.dtype)
                timestep_input = timestep.to(self.dtype)

                pos = self._transformer_step(latent_input, timestep_input, cap_pos)

                if do_cfg:
                    neg = self._transformer_step(
                        latent_input, timestep_input, cap_neg
                    )
                    pred = pos + guidance_scale * (pos - neg)
                    if cfg_normalization and float(cfg_normalization) > 0.0:
                        ori = torch.linalg.vector_norm(pos)
                        new = torch.linalg.vector_norm(pred)
                        max_norm = ori * float(cfg_normalization)
                        if new > max_norm:
                            pred = pred * (max_norm / new)
                    noise_pred = pred
                else:
                    noise_pred = pos

                noise_pred = noise_pred.squeeze(2)
                noise_pred = -noise_pred
                latents = self.scheduler.step(
                    noise_pred.to(torch.float32), t, latents, return_dict=False
                )[0]
                print(f"  denoise step {i + 1}/{num_inference_steps}")

            if output_type == "latent":
                return latents

            # 5. VAE decode (scaling folded into the wrapper).
            image = self.vae_decoder(self._to_tt(latents))
            image = self._to_cpu(image).float()
            return self.image_processor.postprocess(image, output_type=output_type)


def run_zimage_pipeline(
    output_path: str = "zimage_output.png",
    num_inference_steps: int = NUM_INFERENCE_STEPS,
    output_type: str = "pil",
    run_on_tt: bool = True,
):
    torch_xla.set_custom_compile_options({"optimization_level": 1})

    pipeline = ZImagePipeline(run_on_tt=run_on_tt)
    pipeline.setup()

    result = pipeline.generate(
        num_inference_steps=num_inference_steps,
        output_type=output_type,
    )

    if output_type == "latent":
        print(f"Latent output shape: {result.shape}")
        return result

    image = result[0]
    image.save(output_path)
    print(f"Image saved to {output_path} ({image.size})")
    return result


@pytest.mark.model_test
@pytest.mark.single_device
def test_pipeline():
    """Full Z-Image text-to-image e2e on a single Blackhole chip.

    optimization_level=1 keeps GroupNorm as native ttnn.group_norm so the VAE
    decode at 1280x720 does not OOM (issue #4755).
    """
    xr.set_device_type("TT")
    torch.manual_seed(SEED)

    output_path = "test_zimage_output.png"
    output_file = Path(output_path)
    if output_file.exists():
        output_file.unlink()

    images = run_zimage_pipeline(output_path=output_path, output_type="pil")

    assert images is not None, "Pipeline returned None"
    assert len(images) == 1, f"Expected 1 image, got {len(images)}"
    from PIL import Image

    assert isinstance(images[0], Image.Image), "Output is not a PIL image"
    assert output_file.exists(), "Output image was not saved"
    print("Z-Image e2e pipeline test passed.")


if __name__ == "__main__":
    test_pipeline()
