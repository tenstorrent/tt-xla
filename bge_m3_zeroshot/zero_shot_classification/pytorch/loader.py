# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BGE-M3 zero-shot classification model loader implementation.

Uses NLI-based zero-shot classification with entailment/not_entailment labels.
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
    """Available BGE-M3 zero-shot classification model variants."""

    BGE_M3_ZEROSHOT_V2_0 = "MoritzLaurer/bge-m3-zeroshot-v2.0"


class ModelLoader(ForgeModel):
    """BGE-M3 zero-shot classification model loader."""

    _VARIANTS = {
        ModelVariant.BGE_M3_ZEROSHOT_V2_0: LLMModelConfig(
            pretrained_model_name="MoritzLaurer/bge-m3-zeroshot-v2.0",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BGE_M3_ZEROSHOT_V2_0

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.max_length = self._variant_config.max_length
        self.tokenizer = None
        self.premise = "Angela Merkel is a politician in Germany and leader of the CDU."
        self.hypothesis = "This text is about politics."

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = "base"

        return ModelInfo(
            model="BGE-M3-Zeroshot",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load BGE-M3 zero-shot classification model from Hugging Face."""

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
        """Prepare sample NLI input pair for zero-shot classification."""
        if self.tokenizer is None:
            self.load_model(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.premise,
            self.hypothesis,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out, framework_model=None):
        """Decode the model output for zero-shot classification."""
        predicted_class_id = co_out[0].argmax().item()
        model = framework_model if framework_model is not None else self.model
        if model and hasattr(model, "config") and hasattr(model.config, "id2label"):
            predicted_label = model.config.id2label[predicted_class_id]
            print(f"Predicted: {predicted_label}")
        else:
            print(f"Predicted class ID: {predicted_class_id}")
