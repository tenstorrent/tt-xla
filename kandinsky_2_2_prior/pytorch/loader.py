# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Kandinsky 2.2 Prior model loader implementation.

Kandinsky 2.2 Prior is a transformer-based model that maps CLIP text embeddings
to CLIP image embeddings for use in the Kandinsky 2.2 text-to-image pipeline.

Available variants:
- KANDINSKY_2_2_PRIOR: kandinsky-community/kandinsky-2-2-prior text-to-image prior
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
from .src.model_utils import load_pipe, kandinsky_prior_preprocessing


class ModelVariant(StrEnum):
    """Available Kandinsky 2.2 Prior model variants."""

    KANDINSKY_2_2_PRIOR = "kandinsky-2-2-prior"


class ModelLoader(ForgeModel):
    """Kandinsky 2.2 Prior model loader implementation."""

    _VARIANTS = {
        ModelVariant.KANDINSKY_2_2_PRIOR: ModelConfig(
            pretrained_model_name="kandinsky-community/kandinsky-2-2-prior",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.KANDINSKY_2_2_PRIOR

    prompt = "An astronaut riding a green horse"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Kandinsky 2.2 Prior",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Kandinsky 2.2 Prior transformer.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            PriorTransformer: The Kandinsky 2.2 Prior transformer instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.pipeline = load_pipe(pretrained_model_name)

        if dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype_override)

        return self.pipeline

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Kandinsky 2.2 Prior model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            list: Input tensors that can be fed to the model:
                - hidden_states (torch.Tensor): Noised image embeddings
                - timestep (torch.Tensor): Timestep tensor
                - proj_embedding (torch.Tensor): Projected text embeddings
                - encoder_hidden_states (torch.Tensor): Text encoder hidden states
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        (
            hidden_states,
            timestep,
            proj_embedding,
            encoder_hidden_states,
        ) = kandinsky_prior_preprocessing(self.pipeline, self.prompt)

        if dtype_override:
            hidden_states = hidden_states.to(dtype_override)
            timestep = timestep.to(dtype_override)
            proj_embedding = proj_embedding.to(dtype_override)
            encoder_hidden_states = encoder_hidden_states.to(dtype_override)

        return [hidden_states, timestep, proj_embedding, encoder_hidden_states]
