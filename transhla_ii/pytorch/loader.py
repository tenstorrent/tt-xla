# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TransHLA_II model loader for HLA class II epitope prediction.
"""
import torch
from transformers import AutoTokenizer, AutoModel
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

# Tokenizer model for ESM2 embeddings used by TransHLA_II
ESM2_TOKENIZER = "facebook/esm2_t33_650M_UR50D"

# Sequence length expected by TransHLA_II (including special tokens)
PADDED_LENGTH = 23


class ModelVariant(StrEnum):
    """Available TransHLA_II model variants."""

    TRANSHLA_II = "SkywalkerLu/TransHLA_II"


class ModelLoader(ForgeModel):
    """TransHLA_II model loader for HLA class II epitope prediction."""

    _VARIANTS = {
        ModelVariant.TRANSHLA_II: ModelConfig(
            pretrained_model_name="SkywalkerLu/TransHLA_II",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TRANSHLA_II

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="TransHLA_II",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(ESM2_TOKENIZER)
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        # Sample HLA-II peptide sequences (13-21 amino acids)
        peptide = "KMIYSYSSHAASSL"

        encoding = self.tokenizer(peptide)["input_ids"]

        # Pad to expected length with pad token (1)
        padding_length = PADDED_LENGTH - len(encoding)
        if padding_length > 0:
            encoding.extend([1] * padding_length)

        return torch.tensor([encoding])

    def decode_output(self, outputs, inputs=None):
        if isinstance(outputs, tuple):
            probs = outputs[0]
        else:
            probs = outputs

        # Column index 1 is the epitope probability
        predictions = (probs[:, 1] >= 0.5).int()
        labels = ["non-epitope", "epitope"]
        return [labels[p] for p in predictions]
