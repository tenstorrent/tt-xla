# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Impresso Language Identifier model loader implementation for language identification.
"""

from transformers import AutoModelForTokenClassification
from typing import Optional

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
    """Available Impresso Language Identifier model variants."""

    IMPRESSO_LANGUAGE_IDENTIFIER = "impresso-project/language-identifier"


class ModelLoader(ForgeModel):
    """Impresso Language Identifier model loader for language identification."""

    _VARIANTS = {
        ModelVariant.IMPRESSO_LANGUAGE_IDENTIFIER: LLMModelConfig(
            pretrained_model_name="impresso-project/language-identifier",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.IMPRESSO_LANGUAGE_IDENTIFIER

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.text = (
            "En l'an 1348, au plus fort des ravages de la peste noire à travers "
            "l'Europe, le Royaume de France se trouvait à la fois au bord du "
            "désespoir et face à une opportunité."
        )

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = "base"

        return ModelInfo(
            model="Impresso-LangIdent",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load Impresso Language Identifier model from Hugging Face."""

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForTokenClassification.from_pretrained(
            self.model_name, trust_remote_code=True, **model_kwargs
        )
        model.eval()
        self.model = model
        return model

    def load_inputs(self, dtype_override=None):
        """Prepare sample input for language identification.

        This model accepts raw text strings as input rather than tokenized tensors.
        """
        return {"input_ids": self.text}

    def decode_output(self, co_out, framework_model=None):
        """Decode the model output for language identification."""
        predictions, probabilities = co_out
        label = predictions[0][0].replace("__label__", "")
        confidence = float(probabilities[0][0])
        print(f"Detected language: {label} (confidence: {confidence:.2f})")
