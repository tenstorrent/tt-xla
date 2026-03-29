# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CDS-BART-denoising model loader implementation for feature extraction.
"""
import torch
from typing import Optional

from transformers import AutoTokenizer, BartModel
from third_party.tt_forge_models.config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    LLMModelConfig,
)
from third_party.tt_forge_models.base import ForgeModel


class ModelVariant(StrEnum):
    """Available CDS-BART-denoising model variants for feature extraction."""

    CDS_BART_DENOISING = "CDS_BART_Denoising"


class ModelLoader(ForgeModel):
    """CDS-BART-denoising model loader implementation for feature extraction."""

    _VARIANTS = {
        ModelVariant.CDS_BART_DENOISING: LLMModelConfig(
            pretrained_model_name="mogam-ai/CDS-BART-denoising",
            max_length=512,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CDS_BART_DENOISING

    # Sample mRNA coding sequence (CDS) for inference
    sample_text = "ATGGGCAGCAGCCCCAGCAAGAGCACCAGCGGCGGCAGCGAGGACCTGGGCAGCCTG"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="CDS-BART-denoising",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        if self.tokenizer is None:
            model_name = self._variant_config.pretrained_model_name
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer()

        model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = BartModel.from_pretrained(model_name, **model_kwargs)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        max_length = self._variant_config.max_length

        inputs = self.tokenizer(
            self.sample_text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, outputs):
        if isinstance(outputs, (tuple, list)):
            last_hidden_state = outputs[0]
        elif hasattr(outputs, "last_hidden_state"):
            last_hidden_state = outputs.last_hidden_state
        else:
            last_hidden_state = outputs

        # Use mean pooling over sequence dimension for feature extraction
        embedding = last_hidden_state.mean(dim=1)
        return embedding

    def unpack_forward_output(self, fwd_output):
        tensors = []

        if hasattr(fwd_output, "last_hidden_state"):
            tensors.append(fwd_output.last_hidden_state.flatten())
        if (
            hasattr(fwd_output, "encoder_last_hidden_state")
            and fwd_output.encoder_last_hidden_state is not None
        ):
            tensors.append(fwd_output.encoder_last_hidden_state.flatten())

        if tensors:
            return torch.cat(tensors, dim=0)
        return fwd_output
