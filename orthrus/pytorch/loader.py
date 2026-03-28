# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Orthrus RNA foundation model loader implementation.

Orthrus is a Mamba-based RNA foundation model pre-trained with contrastive
learning on 45M+ mature RNA transcripts. It generates sequence embeddings
for spliced mature RNA transcripts.
"""

import torch
import numpy as np
from transformers import AutoModel
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


class ModelVariant(StrEnum):
    """Available Orthrus model variants."""

    LARGE_6_TRACK = "large-6-track"


class ModelLoader(ForgeModel):
    """Orthrus RNA foundation model loader implementation."""

    _VARIANTS = {
        ModelVariant.LARGE_6_TRACK: ModelConfig(
            pretrained_model_name="quietflamingo/orthrus-large-6-track",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LARGE_6_TRACK

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Orthrus",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Orthrus model instance."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        model = AutoModel.from_pretrained(
            pretrained_model_name,
            trust_remote_code=True,
        )
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Load sample inputs for the Orthrus model.

        Orthrus 6-track expects input of shape (batch, seq_len, 6) where the
        6 channels are: 4 one-hot nucleotide channels + CDS track + splice site track.
        It also requires a lengths tensor indicating the sequence length.
        """
        seq_len = 128
        n_tracks = 6

        # Generate a synthetic 6-track RNA input
        # First 4 channels: one-hot encoded nucleotide sequence (A, C, G, T)
        seq_ohe = np.zeros((seq_len, 4), dtype=np.float32)
        nucleotide_indices = np.random.randint(0, 4, size=seq_len)
        seq_ohe[np.arange(seq_len), nucleotide_indices] = 1.0

        # Channel 5: CDS (coding sequence) indicator track
        cds = np.zeros((seq_len, 1), dtype=np.float32)
        cds[10:100] = 1.0

        # Channel 6: Splice site indicator track
        splice = np.zeros((seq_len, 1), dtype=np.float32)
        splice[10] = 1.0
        splice[99] = 1.0

        model_input = np.hstack((seq_ohe, cds, splice))
        model_input = torch.tensor(model_input).unsqueeze(0)

        lengths = torch.tensor([seq_len], dtype=torch.float32)

        if dtype_override is not None:
            model_input = model_input.to(dtype_override)
            lengths = lengths.to(dtype_override)

        return model_input, lengths
