# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Multilingual CLIP text encoder loader for multilingual text embedding generation.

Uses M-CLIP (Multilingual CLIP) which extends CLIP text encoders to 50+ languages
via XLM-RoBERTa with a linear projection into CLIP embedding space.
"""

import torch
from multilingual_clip import pt_multilingual_clip
from transformers import AutoTokenizer
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
    """Available Multilingual CLIP model variants."""

    XLM_ROBERTA_LARGE_VIT_B_16PLUS = "XLM-Roberta-Large-Vit-B-16Plus"


class MultilingualCLIPTextEncoder(torch.nn.Module):
    """Wrapper around M-CLIP model that accepts pre-tokenized inputs.

    The original M-CLIP forward() takes raw text strings and a tokenizer.
    This wrapper accepts input_ids and attention_mask tensors directly,
    making it compatible with the ForgeModel interface.
    """

    def __init__(self, mclip_model):
        super().__init__()
        self.transformer = mclip_model.transformer
        self.linear_transformation = mclip_model.LinearTransformation

    def forward(self, input_ids, attention_mask):
        embs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)[0]
        embs = (embs * attention_mask.unsqueeze(2)).sum(dim=1) / attention_mask.sum(
            dim=1
        )[:, None]
        return self.linear_transformation(embs)


class ModelLoader(ForgeModel):
    """Multilingual CLIP text encoder loader for embedding generation."""

    _VARIANTS = {
        ModelVariant.XLM_ROBERTA_LARGE_VIT_B_16PLUS: ModelConfig(
            pretrained_model_name="M-CLIP/XLM-Roberta-Large-Vit-B-16Plus",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.XLM_ROBERTA_LARGE_VIT_B_16PLUS

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="multilingual-clip",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self._variant_config.pretrained_model_name
            )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer()

        pretrained_model_name = self._variant_config.pretrained_model_name

        mclip_model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(
            pretrained_model_name
        )

        model = MultilingualCLIPTextEncoder(mclip_model)

        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer()

        sentences = [
            "Three blind horses listening to Mozart.",
            "Älgen är skogens konung!",
        ]

        inputs = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        if batch_size > 1:
            for key in inputs:
                if torch.is_tensor(inputs[key]):
                    inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def unpack_forward_output(self, fwd_output):
        if isinstance(fwd_output, torch.Tensor):
            return fwd_output.flatten()

        if isinstance(fwd_output, (tuple, list)):
            tensors = [t.flatten() for t in fwd_output if isinstance(t, torch.Tensor)]
            if tensors:
                return torch.cat(tensors, dim=0)

        return fwd_output
