# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Flashzoi model loader implementation for genomic prediction.
"""
import torch
from typing import Optional

from borzoi_pytorch import Borzoi

from third_party.tt_forge_models.config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    ModelConfig,
)
from third_party.tt_forge_models.base import ForgeModel


class ModelVariant(StrEnum):
    """Available Flashzoi model variants for genomic prediction."""

    FLASHZOI_REPLICATE_0 = "johahi/flashzoi-replicate-0"


class ModelLoader(ForgeModel):
    """Flashzoi model loader implementation for genomic prediction."""

    _VARIANTS = {
        ModelVariant.FLASHZOI_REPLICATE_0: ModelConfig(
            pretrained_model_name="johahi/flashzoi-replicate-0",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FLASHZOI_REPLICATE_0

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Flashzoi",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = Borzoi.from_pretrained(model_name, **model_kwargs)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        # One-hot encoded DNA sequence of shape (N, 4, L)
        # Channels represent nucleotides: A, C, G, T
        seq_length = 524288
        x = torch.zeros(1, 4, seq_length)
        # Create a simple repeating ACGT pattern
        positions = torch.arange(seq_length)
        x[0, 0, positions % 4 == 0] = 1.0  # A
        x[0, 1, positions % 4 == 1] = 1.0  # C
        x[0, 2, positions % 4 == 2] = 1.0  # G
        x[0, 3, positions % 4 == 3] = 1.0  # T

        if dtype_override is not None:
            x = x.to(dtype_override)

        return {"x": x}

    def decode_output(self, outputs, inputs=None):
        # Output shape: (N, num_tracks, num_bins)
        # num_tracks = 7611 (human) or 2608 (mouse)
        if isinstance(outputs, (tuple, list)):
            predictions = outputs[0]
        else:
            predictions = outputs

        return predictions
