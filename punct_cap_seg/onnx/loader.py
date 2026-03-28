# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Punctuation, capitalization, and segmentation model loader (ONNX).

Loads the 1-800-BAD-CODE/punct_cap_seg_47_language model, a multi-task
token classification model that restores punctuation, true-casing, and
sentence boundaries for 47 languages.
"""

import onnx
import torch
import numpy as np
from huggingface_hub import hf_hub_download

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

_REPO_ID = "1-800-BAD-CODE/punct_cap_seg_47_language"
_ONNX_FILENAME = "punct_cap_seg_47lang.onnx"
_SPE_FILENAME = "spe_unigram_64k_lowercase_47lang.model"
_MAX_LENGTH = 128


class ModelVariant(StrEnum):
    """Available model variants."""

    PUNCT_CAP_SEG_47LANG = "punct_cap_seg_47lang"


class ModelLoader(ForgeModel):
    """Punctuation, capitalization, and segmentation ONNX model loader."""

    _VARIANTS = {
        ModelVariant.PUNCT_CAP_SEG_47LANG: ModelConfig(
            pretrained_model_name=_REPO_ID,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.PUNCT_CAP_SEG_47LANG

    def __init__(self, variant=None):
        super().__init__(variant)
        self.sample_text = "hello friend how's it going it's snowing outside"

    @classmethod
    def _get_model_info(cls, variant_name=None):
        if variant_name is None:
            variant_name = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="PunctCapSeg",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TOKEN_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.ONNX,
        )

    def load_model(self, **kwargs):
        onnx_path = hf_hub_download(repo_id=_REPO_ID, filename=_ONNX_FILENAME)
        model = onnx.load(onnx_path)
        return model

    def load_inputs(self, **kwargs):
        import sentencepiece as spm

        spe_path = hf_hub_download(repo_id=_REPO_ID, filename=_SPE_FILENAME)
        sp = spm.SentencePieceProcessor(model_file=spe_path)

        token_ids = sp.encode(self.sample_text.lower(), out_type=int)

        # Pad or truncate to max_length
        if len(token_ids) > _MAX_LENGTH:
            token_ids = token_ids[:_MAX_LENGTH]
        attention_mask = [1] * len(token_ids) + [0] * (_MAX_LENGTH - len(token_ids))
        token_ids = token_ids + [0] * (_MAX_LENGTH - len(token_ids))

        input_ids = torch.tensor([token_ids], dtype=torch.long)
        attention_mask = torch.tensor([attention_mask], dtype=torch.long)

        return {"input_ids": input_ids, "attention_mask": attention_mask}
