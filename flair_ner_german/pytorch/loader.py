# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Flair NER German model loader implementation for German named entity recognition.
"""

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
    """Available Flair NER German model variants."""

    NER_GERMAN = "NER_German"


class ModelLoader(ForgeModel):
    """Flair NER German model loader implementation."""

    _VARIANTS = {
        ModelVariant.NER_GERMAN: ModelConfig(
            pretrained_model_name="flair/ner-german",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NER_GERMAN

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.sample_text = "George Washington ging nach Washington"

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Flair_NER_German",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TOKEN_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from flair.models import SequenceTagger

        tagger = SequenceTagger.load(self.model_name)
        self.model = tagger

        if dtype_override is not None:
            tagger = tagger.to(dtype_override)

        tagger.eval()
        return tagger

    def load_inputs(self, dtype_override=None):
        from flair.data import Sentence

        sentence = Sentence(self.sample_text)
        return [sentence]

    def decode_output(self, co_out):
        from flair.data import Sentence

        sentence = Sentence(self.sample_text)
        self.model.predict(sentence)

        entities = []
        for entity in sentence.get_spans("ner"):
            entities.append(
                {
                    "text": entity.text,
                    "label": entity.get_label("ner").value,
                    "score": entity.get_label("ner").score,
                }
            )

        print(f"Context: {self.sample_text}")
        print(f"Entities: {entities}")
        return entities
