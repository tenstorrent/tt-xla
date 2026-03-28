# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GLiClass model loader implementation for zero-shot text classification.
"""
from gliclass import GLiClassModel
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
    """Available GLiClass model variants for zero-shot text classification."""

    INSTRUCT_LARGE_V1_0 = "Instruct_Large_v1.0"


class ModelLoader(ForgeModel):
    """GLiClass model loader implementation for zero-shot text classification tasks."""

    _VARIANTS = {
        ModelVariant.INSTRUCT_LARGE_V1_0: ModelConfig(
            pretrained_model_name="knowledgator/gliclass-instruct-large-v1.0",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.INSTRUCT_LARGE_V1_0

    sample_text = "NASA launched a new Mars rover to search for signs of ancient life."
    candidate_labels = ["space", "politics", "sports", "technology", "health"]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="GLiClass",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = GLiClassModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()
        self.model = model

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # GLiClass expects text and labels concatenated with a separator
        # Format: "text [SEP] label1 [SEP] label2 ..."
        separator = " [SEP] "
        input_text = (
            self.sample_text + separator + separator.join(self.candidate_labels)
        )

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        return inputs

    def decode_output(self, outputs, inputs=None):
        logits = outputs[0]
        # GLiClass outputs logits for each candidate label
        num_labels = len(self.candidate_labels)
        label_logits = logits[0, :num_labels]
        predicted_idx = label_logits.argmax().item()
        predicted_label = self.candidate_labels[predicted_idx]

        return predicted_label
