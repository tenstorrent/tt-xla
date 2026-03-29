# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SEDD (Score Entropy Discrete Diffusion) model loader implementation.

SEDD is a discrete diffusion model for text generation based on the paper
"Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution".
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
from .src import SEDD


class ModelVariant(StrEnum):
    """Available SEDD model variants."""

    SEDD_MEDIUM = "Medium"


class ModelLoader(ForgeModel):
    """SEDD model loader for discrete diffusion text generation."""

    _VARIANTS = {
        ModelVariant.SEDD_MEDIUM: ModelConfig(
            pretrained_model_name="louaaron/sedd-medium",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SEDD_MEDIUM

    # Model uses GPT-2 tokenizer vocabulary (50,257 tokens)
    VOCAB_SIZE = 50257
    SEQ_LEN = 1024

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="SEDD",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model = SEDD.from_pretrained(pretrained_model_name)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        # Token indices: random tokens from the GPT-2 vocabulary
        indices = torch.randint(0, self.VOCAB_SIZE, (1, self.SEQ_LEN), dtype=torch.long)

        # Sigma: diffusion timestep (scalar per batch element)
        sigma = torch.tensor([1.0])

        return [indices, sigma]
