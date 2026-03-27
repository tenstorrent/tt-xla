# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Borzoi model loader implementation for genomic sequence prediction.

Borzoi is a convolutional neural network that predicts RNA-seq coverage
from DNA sequence at 32bp resolution. It takes 524kb input DNA sequences
and outputs predicted gene expression tracks.
"""
import torch
from transformers import AutoModel, AutoConfig
from typing import Optional

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
    """Available Borzoi model variants for sequence prediction."""

    BORZOI_REPLICATE_0 = "johahi/borzoi-replicate-0"


class ModelLoader(ForgeModel):
    """Borzoi model loader implementation for genomic sequence prediction."""

    _VARIANTS = {
        ModelVariant.BORZOI_REPLICATE_0: ModelConfig(
            pretrained_model_name="johahi/borzoi-replicate-0",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BORZOI_REPLICATE_0

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Borzoi",
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

        model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        # Borzoi expects one-hot encoded DNA sequences of shape
        # (batch_size, seq_len, 4) where 4 channels represent A, C, G, T.
        # The model's expected input length is 524288 bp.
        seq_len = 524288

        # Generate a random one-hot encoded DNA sequence
        random_indices = torch.randint(0, 4, (1, seq_len))
        one_hot = torch.zeros(1, seq_len, 4)
        one_hot.scatter_(2, random_indices.unsqueeze(-1), 1.0)

        if dtype_override is not None:
            one_hot = one_hot.to(dtype=dtype_override)

        return {"sequence": one_hot}

    def decode_output(self, outputs, inputs=None):
        if isinstance(outputs, (tuple, list)):
            predictions = outputs[0]
        elif hasattr(outputs, "logits"):
            predictions = outputs.logits
        else:
            predictions = outputs

        print(f"Output shape: {predictions.shape}")
        print(
            f"Output range: [{predictions.min().item():.4f}, {predictions.max().item():.4f}]"
        )

        return predictions
