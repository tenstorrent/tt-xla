# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Flair NER Multi model loader implementation for multilingual named entity recognition.

This model uses the Flair library's SequenceTagger (BiLSTM-CRF) architecture
with stacked Flair and GloVe embeddings. It recognizes four entity types
(PER, LOC, ORG, MISC) across multiple languages.
"""

from flair.data import Sentence
from flair.models import SequenceTagger
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

    NER_MULTI = "flair/ner-multi"


class ModelLoader(ForgeModel):
    """Flair NER Multi model loader for multilingual named entity recognition."""

    _VARIANTS = {
        ModelVariant.NER_MULTI: ModelConfig(
            pretrained_model_name="flair/ner-multi",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NER_MULTI

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.sample_text = "George Washington went to Washington"
        self.model = None

    @classmethod
    def _get_model_info(cls, variant_name=None):
        if variant_name is None:
            variant_name = "ner_multi"
        return ModelInfo(
            model="Flair",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TOKEN_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        tagger = SequenceTagger.load(self.model_name)
        tagger.eval()
        self.model = tagger
        return tagger

    def load_inputs(self, dtype_override=None):
        if self.model is None:
            self.load_model(dtype_override=dtype_override)

        sentence = Sentence(self.sample_text)
        return [sentence]

    def decode_output(self, co_out):
        sentence = Sentence(self.sample_text)
        self.model.predict(sentence)
        entities = sentence.get_spans("ner")
        for entity in entities:
            print(
                f"{entity.text} [{entity.get_label('ner').value}]"
                f" ({entity.get_label('ner').score:.4f})"
            )
        return entities
