# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Enformer model loader implementation for gene expression prediction.
"""
import torch
from typing import Optional

from enformer_pytorch import from_pretrained

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
    """Available Enformer model variants."""

    OFFICIAL_ROUGH = "EleutherAI/enformer-official-rough"


class ModelLoader(ForgeModel):
    """Enformer model loader for gene expression prediction from DNA sequences."""

    _VARIANTS = {
        ModelVariant.OFFICIAL_ROUGH: ModelConfig(
            pretrained_model_name="EleutherAI/enformer-official-rough",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.OFFICIAL_ROUGH

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Enformer",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.ATOMIC_ML,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the Enformer model from HuggingFace.

        Returns:
            torch.nn.Module: The Enformer model instance.
        """
        model = from_pretrained(self._variant_config.pretrained_model_name)

        if dtype_override is not None:
            model = model.to(dtype=dtype_override)

        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Prepare a random DNA sequence input for Enformer.

        Returns:
            torch.Tensor: Sequence tensor of shape (1, 196608) with nucleotide indices.
        """
        torch.manual_seed(42)
        # DNA nucleotide indices: 0=A, 1=C, 2=G, 3=T, 4=N
        sequence = torch.randint(0, 5, (1, 196_608))
        return sequence
