# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
spaCy es_core_news_md model loader implementation
"""

from typing import Optional

from ....config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    ModelConfig,
    StrEnum,
)
from ....base import ForgeModel


class ModelVariant(StrEnum):
    ES_CORE_NEWS_MD = "es_core_news_md"


class ModelLoader(ForgeModel):
    """spaCy es_core_news_md model loader implementation."""

    _VARIANTS = {
        ModelVariant.ES_CORE_NEWS_MD: ModelConfig(
            pretrained_model_name="es_core_news_md",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ES_CORE_NEWS_MD

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant."""
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.model = None
        self.sample_text = "El presidente del gobierno de España visitó la sede de las Naciones Unidas en Nueva York."

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="spaCy",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TOKEN_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, **kwargs):
        """Load the spaCy es_core_news_md model."""
        import spacy

        try:
            nlp = spacy.load(self.model_name)
        except OSError:
            spacy.cli.download(self.model_name)
            nlp = spacy.load(self.model_name)

        self.model = nlp
        return nlp

    def load_inputs(self, dtype_override=None):
        """Prepare sample input for the NER model."""
        return self.sample_text

    def decode_output(self, co_out):
        """Decode the model output for named entity recognition."""
        if self.model is None:
            self.load_model()

        doc = self.model(self.sample_text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]

        print(f"Context: {self.sample_text}")
        print(f"Entities: {entities}")
        return entities
