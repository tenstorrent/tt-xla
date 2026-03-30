# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Unstuffer model loader implementation for sequence classification.

Loads the jondurbin/unstuffer-v0.2 model, a RoBERTa-large based binary
classifier for detecting AI-generated filler content.
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ....base import ForgeModel
from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)

REPO_ID = "jondurbin/unstuffer-v0.2"


class ModelVariant(StrEnum):
    """Available Unstuffer model variants."""

    UNSTUFFER_V0_2 = "unstuffer_v0_2"


class ModelLoader(ForgeModel):
    """Unstuffer model loader for sequence classification."""

    _VARIANTS = {
        ModelVariant.UNSTUFFER_V0_2: LLMModelConfig(
            pretrained_model_name=REPO_ID,
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.UNSTUFFER_V0_2

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.max_length = self._variant_config.max_length
        self.tokenizer = None
        self.text = "This is an important document that contains relevant information."

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = "base"

        return ModelInfo(
            model="Unstuffer",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load Unstuffer model for sequence classification."""

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, **model_kwargs
        )
        model.eval()
        self.model = model
        return model

    def load_inputs(self, dtype_override=None):
        """Prepare sample input for binary classification."""
        if self.tokenizer is None:
            self.load_model(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out, framework_model=None):
        """Decode the model output for sequence classification."""
        predicted_class_id = co_out[0].argmax().item()
        model = framework_model if framework_model is not None else self.model
        if model and hasattr(model, "config") and hasattr(model.config, "id2label"):
            predicted_label = model.config.id2label[predicted_class_id]
            print(f"Predicted Label: {predicted_label}")
        else:
            print(f"Predicted class ID: {predicted_class_id}")
