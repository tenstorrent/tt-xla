# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Stable Diffusion UNET model loader implementation
"""

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
import torch
from diffusers import (
    UNet2DConditionModel,
    LMSDiscreteScheduler,
)
from transformers import CLIPTextModel, CLIPTokenizer
from typing import Optional


class ModelVariant(StrEnum):
    """Available Stable Diffusion UNet model variants."""

    BASE = "base"


class ModelLoader(ForgeModel):
    """Stable Diffusion UNet model loader implementation."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="CompVis/stable-diffusion-v1-4",
        )
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant_name: Optional variant name string. If None, uses 'base'.

        Returns:
            ModelInfo: Information about the model and variant
        """

        return ModelInfo(
            model="stable_diffusion_unet",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_IMAGE_TTT,  # FIXME: Update to text to image
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the UNet model along with required components.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use torch.bfloat16.

        Returns:
            UNet2DConditionModel: The pre-trained UNet model.
        """
        dtype = dtype_override or torch.bfloat16

        # Load the pre-trained model and tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.text_encoder = CLIPTextModel.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        unet = UNet2DConditionModel.from_pretrained(
            self._variant_config.pretrained_model_name,
            subfolder="unet",
            torch_dtype=dtype,
        )
        self.scheduler = LMSDiscreteScheduler.from_pretrained(
            self._variant_config.pretrained_model_name, subfolder="scheduler"
        )

        # in_channels is needed in load_inputs so we store it here
        self.in_channels = unet.in_channels
        return unet

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the UNet model.

        Args:
            dtype_override: Optional torch.dtype to override the input dtype.
            batch_size: Optional batch size for the inputs.

        Returns:
            dict: Dictionary containing sample, timestep, and encoder_hidden_states.
        """
        dtype = dtype_override or torch.bfloat16

        # Prepare the text prompt
        prompt = ["A fantasy landscape with mountains and rivers"] * batch_size
        text_input = self.tokenizer(prompt, return_tensors="pt")
        text_embeddings = self.text_encoder(text_input.input_ids)[0]

        # Generate noise
        height, width = 512, 512  # Output image size
        # Use the stored in_channels from the loaded model
        latents = torch.randn((batch_size, self.in_channels, height // 8, width // 8))

        # Set number of diffusion steps
        num_inference_steps = 1
        self.scheduler.set_timesteps(num_inference_steps)

        # Scale the latent noise to match the model's expected input
        latents = latents * self.scheduler.init_noise_sigma

        # Get the model's predicted noise
        latent_model_input = self.scheduler.scale_model_input(latents, 0)
        arguments = {
            "sample": latent_model_input.to(dtype),
            "timestep": 0,
            "encoder_hidden_states": text_embeddings.to(dtype),
        }
        return arguments
