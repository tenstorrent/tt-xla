# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Flair NER model loader implementation for German named entity recognition.
"""

from third_party.tt_forge_models.config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    ModelConfig,
)
from third_party.tt_forge_models.base import ForgeModel


class ModelVariant(StrEnum):
    """Available Flair NER model variants."""

    NER_GERMAN_LARGE = "ner-german-large"


class ModelLoader(ForgeModel):
    """Flair NER model loader implementation for German NER."""

    _VARIANTS = {
        ModelVariant.NER_GERMAN_LARGE: ModelConfig(
            pretrained_model_name="flair/ner-german-large",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NER_GERMAN_LARGE

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.sample_text = "George Washington ging nach Washington"

    @classmethod
    def _get_model_info(cls, variant_name=None):
        if variant_name is None:
            variant_name = "ner_german_large"
        return ModelInfo(
            model="Flair",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TOKEN_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from flair.models import SequenceTagger

        model = SequenceTagger.load(self.model_name)
        if dtype_override is not None:
            model = model.to(dtype_override)
        self.model = model
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        from flair.data import Sentence

        sentence = Sentence(self.sample_text)
        return [sentence]

    def decode_output(self, co_out):
        from flair.data import Sentence

        sentence = Sentence(self.sample_text)
        self.model.predict(sentence)
        entities = [str(entity) for entity in sentence.get_spans("ner")]

        print(f"Context: {self.sample_text}")
        print(f"NER Entities: {entities}")
