# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FramePack model loader implementation for image-to-video generation
"""
import torch
from typing import Optional

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from .src.model_utils import load_pipe, framepack_preprocessing


class ModelVariant(StrEnum):
    """Available FramePack model variants."""

    FRAMEPACK_F1_I2V_HY = "F1_I2V_HY_20250503"


class ModelLoader(ForgeModel):
    """FramePack model loader implementation for image-to-video generation tasks."""

    _VARIANTS = {
        ModelVariant.FRAMEPACK_F1_I2V_HY: ModelConfig(
            pretrained_model_name="lllyasviel/FramePack_F1_I2V_HY_20250503",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FRAMEPACK_F1_I2V_HY

    prompt = "A cat walking gracefully across a sunlit garden"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="FramePack",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the FramePack pipeline for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            HunyuanVideoFramepackPipeline: The FramePack pipeline instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.pipeline = load_pipe(pretrained_model_name)

        if dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype_override)

        return self.pipeline

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the FramePack model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            list: Input tensors that can be fed to the transformer model:
                - hidden_states (torch.Tensor): Latent video input
                - timestep (torch.Tensor): Timestep tensor
                - encoder_hidden_states (torch.Tensor): Encoded prompt embeddings
                - encoder_attention_mask (torch.Tensor): Attention mask for encoder states
                - pooled_projections (torch.Tensor): Pooled prompt embeddings
                - image_embeds (torch.Tensor): Image conditioning embeddings
                - indices_latents (torch.Tensor): Frame index tensor
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        (
            hidden_states,
            timestep,
            encoder_hidden_states,
            encoder_attention_mask,
            pooled_projections,
            image_embeds,
            indices_latents,
        ) = framepack_preprocessing(self.pipeline, self.prompt)

        if dtype_override:
            hidden_states = hidden_states.to(dtype_override)
            timestep = timestep.to(dtype_override)
            encoder_hidden_states = encoder_hidden_states.to(dtype_override)
            encoder_attention_mask = encoder_attention_mask.to(dtype_override)
            pooled_projections = pooled_projections.to(dtype_override)
            image_embeds = image_embeds.to(dtype_override)

        return [
            hidden_states,
            timestep,
            encoder_hidden_states,
            encoder_attention_mask,
            pooled_projections,
            image_embeds,
            indices_latents,
        ]
