# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from ...config import ModelConfig, StrEnum
from ...base import ForgeModel
from typing import Optional
from ...config import ModelInfo, ModelGroup, ModelTask, ModelSource, Framework
import torch
from diffusers import UNet2DConditionModel


class ModelVariant(StrEnum):
    """Available UNet for Conditional Generation model variants."""

    BASE = "base"


class ModelLoader(ForgeModel):
    """UNet for Conditional Generation model loader implementation."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="stabilityai/stable-diffusion-xl-base-1.0",
        )
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        """
        Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        return ModelInfo(
            model="unet_for_conditional_generation",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the UNet for Conditional Generation model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).
        """
        dtype = dtype_override or torch.bfloat16

        model = UNet2DConditionModel.from_pretrained(
            self._variant_config.pretrained_model_name,
            subfolder="unet",
            torch_dtype=dtype,
            variant="fp16",
        )

        self.in_channels = model.in_channels
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the UNet for Conditional Generation model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Optional batch size for the inputs.
        """
        dtype = dtype_override or torch.bfloat16

        sample = torch.rand((1, 4, 64, 64), dtype=torch.bfloat16)
        timestep = torch.randint(0, 1000, (1,))
        encoder_hidden_states = torch.rand((1, 77, 2048), dtype=torch.bfloat16)
        added_cond_kwargs = {
            "text_embeds": torch.rand(
                (1, 1280), dtype=torch.bfloat16
            ),  # Pooled text embeddings
            "time_ids": torch.rand((1, 6), dtype=torch.bfloat16),  # Time conditioning
        }
        return {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "added_cond_kwargs": added_cond_kwargs,
        }
