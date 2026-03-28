# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
NCI Binary Detector model loader implementation for sequence classification.
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
    """Available NCI Binary Detector model variants."""

    NCI_BINARY_DETECTOR_V2 = "synapti/nci-binary-detector-v2"


class ModelLoader(ForgeModel):
    """NCI Binary Detector model loader for sequence classification (propaganda detection)."""

    _VARIANTS = {
        ModelVariant.NCI_BINARY_DETECTOR_V2: LLMModelConfig(
            pretrained_model_name="synapti/nci-binary-detector-v2",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NCI_BINARY_DETECTOR_V2

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.max_length = self._variant_config.max_length
        self.tokenizer = None
        self.text = "The Federal Reserve announced a 0.25% rate increase."

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = "v2"

        return ModelInfo(
            model="NCI_Binary_Detector",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load NCI Binary Detector model for sequence classification."""

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
        """Prepare sample input for NCI Binary Detector sequence classification."""
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
        """Decode the model output for binary propaganda detection."""
        predicted_class_id = co_out[0].argmax().item()
        model = framework_model if framework_model is not None else self.model
        if model and hasattr(model, "config") and hasattr(model.config, "id2label"):
            predicted_category = model.config.id2label[predicted_class_id]
            print(f"Predicted Label: {predicted_category}")
        else:
            print(f"Predicted class ID: {predicted_class_id}")
