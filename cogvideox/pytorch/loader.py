# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CogVideoX model loader implementation for text-to-video generation
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
from .src.model_utils import load_pipe, cogvideox_preprocessing


class ModelVariant(StrEnum):
    """Available CogVideoX model variants."""

    COGVIDEOX_2B = "2b"


class ModelLoader(ForgeModel):
    """CogVideoX model loader implementation for text-to-video generation tasks."""

    _VARIANTS = {
        ModelVariant.COGVIDEOX_2B: ModelConfig(
            pretrained_model_name="zai-org/CogVideoX-2b",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.COGVIDEOX_2B

    prompt = "A panda sitting on a wooden stool in a bamboo forest"

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
            model="CogVideoX",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the CogVideoX pipeline for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            CogVideoXPipeline: The CogVideoX pipeline instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.pipeline = load_pipe(pretrained_model_name)

        if dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype_override)

        return self.pipeline

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the CogVideoX model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            list: Input tensors that can be fed to the transformer model:
                - hidden_states (torch.Tensor): Latent video input
                - timestep (torch.Tensor): Timestep tensor
                - encoder_hidden_states (torch.Tensor): Encoded prompt embeddings
                - image_rotary_emb (tuple): Rotary positional embeddings
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        (
            hidden_states,
            timestep,
            encoder_hidden_states,
            image_rotary_emb,
        ) = cogvideox_preprocessing(self.pipeline, self.prompt)

        if dtype_override:
            hidden_states = hidden_states.to(dtype_override)
            timestep = timestep.to(dtype_override)
            encoder_hidden_states = encoder_hidden_states.to(dtype_override)
            if image_rotary_emb is not None:
                image_rotary_emb = tuple(t.to(dtype_override) for t in image_rotary_emb)

        return [hidden_states, timestep, encoder_hidden_states, image_rotary_emb]
