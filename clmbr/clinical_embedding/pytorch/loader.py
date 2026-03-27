# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CLMBR-T-Base model loader implementation for clinical embedding generation.

CLMBR (Clinical Language Model-Based Representations) generates dense patient
embeddings from structured electronic health record (EHR) data sequences.
The model uses a transformer architecture with hierarchical token embeddings
and is designed for downstream clinical prediction tasks.
"""
import torch
from typing import Optional

from femr.models.transformer import FEMRModel

from ....base import ForgeModel
from ....config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    ModelConfig,
)


class ModelVariant(StrEnum):
    """Available CLMBR model variants for clinical embedding generation."""

    CLMBR_T_BASE = "StanfordShahLab/clmbr-t-base"


class ModelLoader(ForgeModel):
    """CLMBR model loader for clinical embedding generation from EHR data."""

    _VARIANTS = {
        ModelVariant.CLMBR_T_BASE: ModelConfig(
            pretrained_model_name="StanfordShahLab/clmbr-t-base",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CLMBR_T_BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="CLMBR",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = FEMRModel.from_pretrained(model_name, **model_kwargs)
        model.eval()
        self.model = model
        return model

    def load_inputs(self, dtype_override=None):
        # CLMBR operates on structured EHR event sequences represented as
        # hierarchical token embeddings. Create synthetic inputs matching
        # the expected batch format.
        seq_len = 32
        num_tokens = 64

        batch = {
            "transformer": {
                "hierarchical_tokens": torch.randint(
                    0, 32768, (num_tokens,), dtype=torch.int32
                ),
                "token_indices": torch.arange(
                    0, num_tokens + 1, num_tokens // seq_len, dtype=torch.int32
                ),
                "hierarchical_weights": torch.ones(num_tokens, dtype=torch.float16),
                "ages": torch.rand(seq_len, dtype=torch.float32) * 36500,
                "time_data": torch.rand(seq_len, 5, dtype=torch.float16),
                "subject_lengths": torch.tensor([seq_len], dtype=torch.int32),
                "timestamps": torch.randint(
                    0, 1_000_000_000, (seq_len,), dtype=torch.int64
                ),
                "label_indices": torch.tensor([seq_len - 1], dtype=torch.int32),
                "valid_tokens": torch.ones(seq_len, dtype=torch.bool),
            },
            "subject_ids": torch.zeros(seq_len, dtype=torch.int64),
        }

        return batch

    def unpack_forward_output(self, fwd_output):
        if isinstance(fwd_output, tuple):
            # FEMRModel returns (loss, result_dict)
            _, result_dict = fwd_output
            if isinstance(result_dict, dict) and "representations" in result_dict:
                return result_dict["representations"].flatten()
            return fwd_output[0]
        return fwd_output
