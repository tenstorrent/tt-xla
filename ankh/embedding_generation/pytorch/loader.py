# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Ankh model loader implementation for protein embedding generation.
"""
import torch
from transformers import AutoTokenizer, T5EncoderModel
from typing import Optional

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Ankh model variants."""

    ANKH_LARGE = "ElnaggarLab/ankh-large"


class ModelLoader(ForgeModel):
    """Ankh model loader implementation for protein embedding generation."""

    _VARIANTS = {
        ModelVariant.ANKH_LARGE: ModelConfig(
            pretrained_model_name="ElnaggarLab/ankh-large",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ANKH_LARGE

    # Sample protein sequence for testing
    sample_sequence = "MKALCLLLLPVLGLLVSSKTLCSMEEAINERIQEVAGSLIFRAISSIGLECQ"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Ankh",
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

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = T5EncoderModel.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        # Ankh tokenizer expects protein sequences split into individual amino acids
        protein_chars = list(self.sample_sequence)

        inputs = self.tokenizer(
            protein_chars,
            add_special_tokens=True,
            is_split_into_words=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, outputs, inputs=None):
        if inputs is None:
            inputs = self.load_inputs()

        attention_mask = inputs["attention_mask"]

        if isinstance(outputs, (tuple, list)):
            token_embeddings = outputs[0]
        elif hasattr(outputs, "last_hidden_state"):
            token_embeddings = outputs.last_hidden_state
        else:
            token_embeddings = outputs

        # Mean pooling over non-padding tokens
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sentence_embeddings = torch.sum(
            token_embeddings * input_mask_expanded, 1
        ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return sentence_embeddings
