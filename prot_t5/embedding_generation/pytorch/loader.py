# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ProtT5 model loader implementation for protein sequence embedding generation.
"""
import re
import torch
from transformers import T5EncoderModel, T5Tokenizer
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
    """Available ProtT5 model variants for embedding generation."""

    PROT_T5_XL_HALF_UNIREF50_ENC = "Rostlab/prot_t5_xl_half_uniref50-enc"


class ModelLoader(ForgeModel):
    """ProtT5 model loader implementation for protein sequence embedding generation."""

    _VARIANTS = {
        ModelVariant.PROT_T5_XL_HALF_UNIREF50_ENC: ModelConfig(
            pretrained_model_name="Rostlab/prot_t5_xl_half_uniref50-enc",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.PROT_T5_XL_HALF_UNIREF50_ENC

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ProtT5",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = T5Tokenizer.from_pretrained(
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
        model.float()
        model.eval()

        return model

    def _preprocess_sequence(self, sequence):
        """Preprocess a protein sequence for tokenization.

        Replaces rare/ambiguous amino acids and adds whitespace between residues.
        """
        sequence = re.sub(r"[UZOB]", "X", sequence)
        return " ".join(list(sequence))

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        # Sample protein sequence for testing
        sequence = self._preprocess_sequence("MGSSHHHHHHSSGLVPRGSHM")

        inputs = self.tokenizer(
            sequence,
            return_tensors="pt",
            add_special_tokens=True,
            padding="longest",
        )

        return inputs

    def decode_output(self, outputs, inputs=None):
        """Decode model output by extracting per-protein embedding via mean pooling."""
        if inputs is None:
            inputs = self.load_inputs()

        attention_mask = inputs["attention_mask"]

        if hasattr(outputs, "last_hidden_state"):
            token_embeddings = outputs.last_hidden_state
        elif isinstance(outputs, (tuple, list)):
            token_embeddings = outputs[0]
        else:
            token_embeddings = outputs

        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        per_protein_embedding = torch.sum(
            token_embeddings * input_mask_expanded, 1
        ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return per_protein_embedding
