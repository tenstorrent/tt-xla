# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
AfroLID model loader implementation for sequence classification (language identification).
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ....config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    LLMModelConfig,
)
from ....base import ForgeModel


class ModelVariant(StrEnum):
    """Available AfroLID sequence classification model variants."""

    AFROLID_1_5 = "UBC-NLP/afrolid_1.5"


class ModelLoader(ForgeModel):
    """AfroLID model loader for sequence classification (language identification)."""

    _VARIANTS = {
        ModelVariant.AFROLID_1_5: LLMModelConfig(
            pretrained_model_name="UBC-NLP/afrolid_1.5",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.AFROLID_1_5

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.max_length = self._variant_config.max_length
        self.tokenizer = None
        self.text = "Bawo ni, kilode ti o fi pe to bayii?"

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = "1.5"

        return ModelInfo(
            model="AfroLID",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load AfroLID model for sequence classification from Hugging Face."""

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
        """Prepare sample input for AfroLID language identification."""
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
        """Decode the model output for language identification."""
        predicted_class_id = co_out[0].argmax().item()
        model = framework_model if framework_model is not None else self.model
        if model and hasattr(model, "config") and hasattr(model.config, "id2label"):
            predicted_language = model.config.id2label[predicted_class_id]
            print(f"Predicted Language: {predicted_language}")
        else:
            print(f"Predicted class ID: {predicted_class_id}")
