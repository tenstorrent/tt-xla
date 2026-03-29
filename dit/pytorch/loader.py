# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DiT (Diffusion Transformer) model loader implementation
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
from diffusers import DiTPipeline
from typing import Optional


class ModelVariant(StrEnum):
    """Available DiT model variants."""

    XL_2_256 = "XL-2-256"


class ModelLoader(ForgeModel):
    """DiT (Diffusion Transformer) model loader implementation."""

    _VARIANTS = {
        ModelVariant.XL_2_256: ModelConfig(
            pretrained_model_name="facebook/DiT-XL-2-256",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.XL_2_256

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
            variant: Optional variant name string. If None, uses default.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="DiT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the DiT transformer model.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use torch.bfloat16.

        Returns:
            DiTTransformer2DModel: The pre-trained DiT transformer model.
        """
        dtype = dtype_override or torch.bfloat16

        pipe = DiTPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )

        self.scheduler = pipe.scheduler
        self.in_channels = pipe.transformer.config.in_channels
        return pipe.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the DiT transformer model.

        Args:
            dtype_override: Optional torch.dtype to override the input dtype.
            batch_size: Optional batch size for the inputs.

        Returns:
            dict: Dictionary containing hidden_states, timestep, and class_labels.
        """
        dtype = dtype_override or torch.bfloat16

        # DiT-XL-2-256 operates on 256x256 images, latent space is 32x32
        latent_size = 256 // 8
        latents = torch.randn(
            (batch_size, self.in_channels, latent_size, latent_size), dtype=dtype
        )

        # Set up scheduler and get a timestep
        num_inference_steps = 1
        self.scheduler.set_timesteps(num_inference_steps)
        timestep = self.scheduler.timesteps[0]

        # Class-conditional: use ImageNet class label (e.g., 207 = golden retriever)
        class_labels = torch.tensor([207] * batch_size)

        return {
            "hidden_states": latents,
            "timestep": timestep,
            "class_labels": class_labels,
        }
