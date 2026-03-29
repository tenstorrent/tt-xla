# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Marigold Normals model loader implementation for monocular surface normal estimation.
"""
import torch
from diffusers import MarigoldNormalsPipeline
from datasets import load_dataset
from typing import Optional

from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel


class ModelVariant(StrEnum):
    """Available Marigold Normals model variants."""

    V0_1 = "v0.1"


class ModelLoader(ForgeModel):
    """Marigold Normals model loader implementation."""

    _VARIANTS = {
        ModelVariant.V0_1: ModelConfig(
            pretrained_model_name="prs-eth/marigold-normals-v0-1",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V0_1

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="MarigoldNormals",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_DEPTH_EST,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self, dtype_override=None):
        pretrained_model_name = self._variant_config.pretrained_model_name
        pipe = MarigoldNormalsPipeline.from_pretrained(
            pretrained_model_name, torch_dtype=dtype_override or torch.float32
        )
        pipe.to("cpu")

        for module in [pipe.unet, pipe.vae, pipe.text_encoder]:
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

        self.pipeline = pipe
        return pipe

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.pipeline is None:
            self._load_pipeline(dtype_override=dtype_override)

        return self.pipeline.unet

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.pipeline is None:
            self._load_pipeline(dtype_override=dtype_override)

        pipe = self.pipeline
        dtype = dtype_override or torch.float32

        # Encode a sample image through the VAE to get the image latent
        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"].resize((768, 768))
        image_tensor = (
            torch.tensor([list(image.getdata())], dtype=dtype)
            .reshape(1, image.size[1], image.size[0], 3)
            .permute(0, 3, 1, 2)
            / 255.0
        )
        image_tensor = image_tensor * 2.0 - 1.0

        with torch.no_grad():
            image_latent = pipe.vae.encode(image_tensor.to(dtype)).latent_dist.mean
            image_latent = image_latent * pipe.vae.config.scaling_factor

        # Generate noise latent and concatenate with image latent for 8-channel input
        noise_latent = torch.randn_like(image_latent)
        latent_model_input = torch.cat([image_latent, noise_latent], dim=1)

        # Prepare timestep
        pipe.scheduler.set_timesteps(10)
        timestep = pipe.scheduler.timesteps[0].expand(batch_size)

        # Encode empty prompt for unconditional generation
        text_inputs = pipe.tokenizer(
            "",
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            encoder_hidden_states = pipe.text_encoder(text_inputs.input_ids)[0].to(
                dtype
            )

        if batch_size > 1:
            latent_model_input = latent_model_input.repeat_interleave(batch_size, dim=0)
            encoder_hidden_states = encoder_hidden_states.repeat_interleave(
                batch_size, dim=0
            )

        return [latent_model_input, timestep, encoder_hidden_states]
