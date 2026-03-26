# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TabSTAR model loader implementation for tabular classification.

TabSTAR (Tabular model with Semantically Target-Aware Representations) is a
foundation model for tabular data that uses a pretrained text encoder (e5-small-v2)
combined with numerical fusion and interaction encoding layers.
"""
import numpy as np
import torch
import torch.nn as nn
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
    """Available TabSTAR model variants."""

    TABSTAR = "alana89/TabSTAR"


class TabSTARWrapper(nn.Module):
    """Wrapper around TabStarModel that accepts tensor inputs for XLA tracing.

    The original TabStarModel.forward() takes numpy arrays and an int, which
    is not compatible with XLA tracing. This wrapper pre-tokenizes text inputs
    and passes tensors through the model components directly.
    """

    def __init__(self, model, d_output=2):
        super().__init__()
        self.text_encoder = model.text_encoder
        self.numerical_fusion = model.numerical_fusion
        self.tabular_encoder = model.tabular_encoder
        self.d_output = d_output
        if d_output == 1:
            self.head = model.reg_head
        else:
            self.head = model.cls_head

    def forward(self, input_ids, attention_mask, x_num):
        """Run forward pass with tensor inputs.

        Args:
            input_ids: Tokenized text input IDs (batch_size * seq_len, max_token_len)
            attention_mask: Attention mask for tokenized text (batch_size * seq_len, max_token_len)
            x_num: Numerical features (batch_size, seq_len)

        Returns:
            Tensor: Prediction scores (batch_size, d_output)
        """
        batch_size = x_num.shape[0]
        seq_len = x_num.shape[1]

        # Get text embeddings from the encoder
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Take [CLS] token representation
        embeddings = outputs.last_hidden_state[:, 0, :]
        # Reshape to (batch_size, seq_len, d_model)
        embeddings = embeddings.view(batch_size, seq_len, -1)

        # Fuse text and numerical features
        fused = self.numerical_fusion(textual_embeddings=embeddings, x_num=x_num)

        # Encode interactions
        encoded = self.tabular_encoder(fused)

        # Extract target tokens and predict
        target_tokens = encoded[:, : self.d_output]
        scores = self.head(target_tokens)
        scores = scores.squeeze(dim=-1)

        return scores


class ModelLoader(ForgeModel):
    """TabSTAR model loader implementation for tabular classification."""

    _VARIANTS = {
        ModelVariant.TABSTAR: ModelConfig(
            pretrained_model_name="alana89/TabSTAR",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TABSTAR

    # Sample tabular data: columns are text features describing a movie
    # The model expects (batch_size, seq_len) text array and numerical array
    # We use d_output=2 for binary classification (e.g., is_drama or not)
    _D_OUTPUT = 2
    _SAMPLE_TEXTS = [
        [
            "query: classify drama",
            "query: The Shawshank Redemption",
            "query: 1994",
            "query: drama",
        ],
        [
            "query: classify drama",
            "query: The Dark Knight",
            "query: 2008",
            "query: action",
        ],
    ]
    _SAMPLE_NUMS = [
        [0.0, 0.0, 1994.0, 0.0],
        [0.0, 0.0, 2008.0, 0.0],
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="TabSTAR",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load TabSTAR model wrapped for tensor-based inference.

        Returns:
            torch.nn.Module: The wrapped TabSTAR model instance.
        """
        from tabstar.arch.arch import TabStarModel

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        tabstar_model = TabStarModel.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )

        self.tokenizer = tabstar_model.tokenizer

        wrapper = TabSTARWrapper(tabstar_model, d_output=self._D_OUTPUT)
        wrapper.eval()
        return wrapper

    def load_inputs(self, dtype_override=None):
        """Prepare sample inputs for the TabSTAR model.

        Returns:
            list: [input_ids, attention_mask, x_num] tensors.
        """
        if self.tokenizer is None:
            self.load_model(dtype_override=dtype_override)

        # Flatten text array for tokenization
        texts_flat = [t for row in self._SAMPLE_TEXTS for t in row]
        encoded = self.tokenizer(
            texts_flat,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        dtype = dtype_override if dtype_override is not None else torch.float32
        x_num = torch.tensor(self._SAMPLE_NUMS, dtype=dtype)

        return [encoded["input_ids"], encoded["attention_mask"], x_num]
