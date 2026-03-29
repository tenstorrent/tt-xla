# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Supertonic 2 ONNX model loader for text-to-speech tasks.

Loads the Supertone/supertonic-2 TTS model, a lightweight 66M-parameter
text-to-speech model supporting 5 languages (en, ko, es, pt, fr) with
2-step and 5-step inference modes.
"""

import json

import numpy as np
import onnx
import torch
from huggingface_hub import hf_hub_download
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

_REPO_ID = "Supertone/supertonic-2"
_TEXT_ENCODER_ONNX = "onnx/text_encoder.onnx"
_VOICE_STYLE_FILE = "voice_styles/F1.json"
_UNICODE_INDEXER_FILE = "onnx/unicode_indexer.json"


class ModelVariant(StrEnum):
    """Available Supertonic 2 model variants."""

    TEXT_ENCODER = "TextEncoder"


class ModelLoader(ForgeModel):
    """Supertonic 2 ONNX model loader for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.TEXT_ENCODER: ModelConfig(
            pretrained_model_name=_REPO_ID,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TEXT_ENCODER

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.sample_text = "Hello, how are you today?"

    @classmethod
    def _get_model_info(cls, variant=None):
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Supertonic2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.ONNX,
        )

    def _load_unicode_indexer(self):
        """Load the unicode indexer mapping for text preprocessing."""
        indexer_path = hf_hub_download(repo_id=_REPO_ID, filename=_UNICODE_INDEXER_FILE)
        with open(indexer_path, "r") as f:
            return json.load(f)

    def _text_to_ids(self, text, language="en"):
        """Convert text to token IDs using the unicode indexer.

        Args:
            text: Input text string.
            language: Language code (en, ko, es, pt, fr).

        Returns:
            List of integer token IDs.
        """
        indexer = self._load_unicode_indexer()
        # Wrap text in language tags as required by the model
        tagged_text = f"<{language}>{text}</{language}>"
        token_ids = []
        i = 0
        while i < len(tagged_text):
            # Try to match multi-character tokens (like language tags)
            matched = False
            for length in range(min(8, len(tagged_text) - i), 0, -1):
                substr = tagged_text[i : i + length]
                if substr in indexer:
                    token_ids.append(indexer[substr])
                    i += length
                    matched = True
                    break
            if not matched:
                # Use character-level fallback
                char = tagged_text[i]
                if char in indexer:
                    token_ids.append(indexer[char])
                i += 1
        return token_ids

    def load_model(self, **kwargs):
        """Load and return the Supertonic 2 text encoder ONNX model.

        Returns:
            onnx.ModelProto: The loaded ONNX text encoder model.
        """
        onnx_path = hf_hub_download(repo_id=_REPO_ID, filename=_TEXT_ENCODER_ONNX)
        model = onnx.load(onnx_path)
        return model

    def load_inputs(self, **kwargs):
        """Load and return sample inputs for the Supertonic 2 text encoder.

        Returns:
            dict: Input tensors for the text encoder model.
        """
        # Load voice style embeddings
        style_path = hf_hub_download(repo_id=_REPO_ID, filename=_VOICE_STYLE_FILE)
        with open(style_path, "r") as f:
            voice_style = json.load(f)

        # Extract style_ttl tensor from voice style JSON
        style_ttl_data = voice_style["style_ttl"]
        style_ttl_shape = style_ttl_data["shape"]
        style_ttl = np.array(style_ttl_data["data"], dtype=np.float32).reshape(
            style_ttl_shape
        )

        # Convert text to token IDs
        token_ids = self._text_to_ids(self.sample_text)
        seq_len = len(token_ids)

        text_ids = torch.tensor([token_ids], dtype=torch.long)
        text_mask = torch.ones(1, seq_len, dtype=torch.long)
        style_ttl = torch.from_numpy(style_ttl)

        return {
            "text_ids": text_ids,
            "style_ttl": style_ttl,
            "text_mask": text_mask,
        }
