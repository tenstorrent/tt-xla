# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Flair NER model loader implementation for multilingual named entity recognition.
"""

from typing import Optional

from third_party.tt_forge_models.config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    ModelConfig,
    StrEnum,
)
from third_party.tt_forge_models.base import ForgeModel


class ModelVariant(StrEnum):
    """Available Flair NER model variants."""

    NER_MULTI_FAST = "ner-multi-fast"


class ModelLoader(ForgeModel):
    """Flair NER model loader implementation for multilingual named entity recognition."""

    _VARIANTS = {
        ModelVariant.NER_MULTI_FAST: ModelConfig(
            pretrained_model_name="flair/ner-multi-fast",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NER_MULTI_FAST

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.sample_text = "George Washington went to Washington"

    @classmethod
    def _get_model_info(cls, variant_name=None):
        if variant_name is None:
            variant_name = "ner-multi-fast"
        return ModelInfo(
            model="Flair",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TOKEN_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, **kwargs):
        from flair.models import SequenceTagger

        model = SequenceTagger.load(self.model_name)
        self.model = model
        model.eval()
        return model

    def load_inputs(self, **kwargs):
        from flair.data import Sentence

        sentence = Sentence(self.sample_text)
        self.sentence = sentence
        return sentence

    def decode_output(self, co_out):
        entities = self.sentence.get_spans("ner")
        print(f"Context: {self.sample_text}")
        print(f"Entities: {entities}")
        return entities
