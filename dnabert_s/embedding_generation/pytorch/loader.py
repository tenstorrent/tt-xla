# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DNABERT-S model loader implementation for embedding generation.
"""
import sys
import torch
import huggingface_hub
from transformers import AutoTokenizer, AutoModel, AutoConfig
from typing import Optional

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
    """Available DNABERT-S model variants for embedding generation."""

    DNABERT_S = "zhihan1996/DNABERT-S"


class ModelLoader(ForgeModel):
    """DNABERT-S model loader implementation for embedding generation."""

    _VARIANTS = {
        ModelVariant.DNABERT_S: LLMModelConfig(
            pretrained_model_name="zhihan1996/DNABERT-S",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DNABERT_S

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="DNABERT-S",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        if self.tokenizer is None:
            model_name = self._variant_config.pretrained_model_name
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer()

        model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        # DNABERT-S uses custom BERT layers with ALiBi tensors that fail when
        # constructed on the meta device (used by from_pretrained internally).
        # Work around by constructing the model on CPU then loading weights.
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_config(config, trust_remote_code=True)
        weights_file = huggingface_hub.hf_hub_download(model_name, "pytorch_model.bin")
        state_dict = torch.load(weights_file, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        # Disable Triton flash attention so the model uses the PyTorch
        # fallback path, which works on CPU and non-CUDA devices.
        bert_layers_mod = sys.modules.get(type(model.encoder).__module__)
        if bert_layers_mod is not None:
            bert_layers_mod.flash_attn_qkvpacked_func = None

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        # Sample DNA sequence
        dna_sequence = "ACTGACTGACTGACTGACTGACTGACTGACTG"

        max_length = self._variant_config.max_length
        inputs = self.tokenizer(
            dna_sequence,
            padding="max_length",
            truncation=True,
            max_length=max_length,
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

        # Mean pooling
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sentence_embeddings = torch.sum(
            token_embeddings * input_mask_expanded, 1
        ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return sentence_embeddings
