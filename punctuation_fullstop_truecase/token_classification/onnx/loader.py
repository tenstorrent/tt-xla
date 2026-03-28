# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Punctuation Fullstop Truecase model loader implementation for token classification.

This model performs punctuation restoration, true-casing, and sentence boundary
detection using a custom Transformer encoder with three classification heads
exported as ONNX.
"""

import numpy as np
import onnx
from huggingface_hub import hf_hub_download

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
)


class ModelLoader(ForgeModel):
    """Punctuation Fullstop Truecase model loader implementation."""

    def __init__(self, variant=None):
        super().__init__(variant)
        self.repo_id = "1-800-BAD-CODE/punctuation_fullstop_truecase_english"
        self.onnx_filename = "punct_cap_seg_en.onnx"
        self.max_length = 256
        self.sample_text = "hey how are you doing today i wanted to talk about the meeting we had yesterday it was really interesting"

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="PunctuationFullstopTruecase",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TOKEN_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.ONNX,
        )

    def load_model(self, **kwargs):
        """Load the ONNX model from HuggingFace Hub.

        Returns:
            onnx.ModelProto: The ONNX model instance.
        """
        model_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=self.onnx_filename,
        )
        model = onnx.load(model_path)
        return model

    def load_inputs(self, **kwargs):
        """Prepare sample input for the punctuation model.

        The model expects integer token IDs as input with shape (batch, seq_len).
        We generate a simple dummy input of the correct shape since the actual
        SentencePiece tokenizer is not required for compilation testing.

        Returns:
            numpy.ndarray: Input token IDs array.
        """
        input_ids = np.ones((1, self.max_length), dtype=np.int64)
        return input_ids
