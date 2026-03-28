# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Multilingual CLIP text encoder model loader for sentence embedding generation.

Uses XLM-RoBERTa with a linear projection to produce CLIP-compatible text embeddings
supporting 48 languages. Based on the M-CLIP project.
"""

import json

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoModel, AutoTokenizer
from typing import Optional

from ...config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    LLMModelConfig,
)
from ...base import ForgeModel


class MultilingualCLIPTextEncoder(nn.Module):
    """XLM-RoBERTa transformer with a linear projection to CLIP embedding space."""

    def __init__(self, transformer, linear_proj):
        super().__init__()
        self.transformer = transformer
        self.LinearTransformation = linear_proj

    def forward(self, input_ids, attention_mask):
        output = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        embs = output[0]
        mask_expanded = attention_mask.unsqueeze(-1).expand(embs.size()).float()
        pooled = torch.sum(embs * mask_expanded, dim=1) / mask_expanded.sum(
            dim=1
        ).clamp(min=1e-9)
        return self.LinearTransformation(pooled)


class ModelVariant(StrEnum):
    """Available Multilingual CLIP model variants."""

    XLM_ROBERTA_LARGE_VIT_B_32 = "M-CLIP/XLM-Roberta-Large-Vit-B-32"


class ModelLoader(ForgeModel):
    """Multilingual CLIP text encoder model loader."""

    _VARIANTS = {
        ModelVariant.XLM_ROBERTA_LARGE_VIT_B_32: LLMModelConfig(
            pretrained_model_name="M-CLIP/XLM-Roberta-Large-Vit-B-32",
            max_length=512,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.XLM_ROBERTA_LARGE_VIT_B_32

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None
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
            model_name = self._variant_config.pretrained_model_name
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer()

        model_name = self._variant_config.pretrained_model_name

        # Load the M-CLIP config to get architecture parameters
        config_path = hf_hub_download(repo_id=model_name, filename="config.json")
        with open(config_path) as f:
            mclip_config = json.load(f)

        model_base = mclip_config["modelBase"]
        transformer_dims = mclip_config["transformerDimensions"]
        num_dims = mclip_config["numDims"]

        # Load the checkpoint containing both transformer and projection weights
        ckpt_path = hf_hub_download(repo_id=model_name, filename="pytorch_model.bin")
        state_dict = torch.load(ckpt_path, map_location="cpu")

        # Separate transformer and linear projection weights
        transformer_sd = {}
        linear_sd = {}
        for key, value in state_dict.items():
            if key.startswith("transformer."):
                transformer_sd[key[len("transformer.") :]] = value
            elif key.startswith("LinearTransformation."):
                linear_sd[key[len("LinearTransformation.") :]] = value

        # Build the transformer from config and load weights
        transformer_config = AutoConfig.from_pretrained(model_base)
        if dtype_override is not None:
            transformer_config.torch_dtype = dtype_override
        transformer = AutoModel.from_config(transformer_config)
        transformer.load_state_dict(transformer_sd)

        # Build the linear projection and load weights
        linear_proj = nn.Linear(transformer_dims, num_dims)
        linear_proj.load_state_dict(linear_sd)

        model = MultilingualCLIPTextEncoder(transformer, linear_proj)

        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        self.model = model

        return model

    def load_inputs(self, dtype_override=None, sentence=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        if sentence is None:
            sentence = "Three blind horses listening to Mozart."

        max_length = getattr(self._variant_config, "max_length", 512)

        inputs = self.tokenizer(
            sentence,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, outputs, inputs=None):
        return outputs

    def unpack_forward_output(self, fwd_output):
        if isinstance(fwd_output, torch.Tensor):
            return fwd_output.flatten()

        tensors = []
        if hasattr(fwd_output, "last_hidden_state"):
            tensors.append(fwd_output.last_hidden_state.flatten())
        if (
            hasattr(fwd_output, "pooler_output")
            and fwd_output.pooler_output is not None
        ):
            tensors.append(fwd_output.pooler_output.flatten())

        if tensors:
            return torch.cat(tensors, dim=0)
        return fwd_output
