# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Flair Arabic Multi NER model loader implementation for token classification.

This loader wraps the megantosh/flair-arabic-multi-ner SequenceTagger model
for Arabic Named Entity Recognition using the Flair NLP framework.
"""

import torch.nn as nn
from flair.data import Sentence
from flair.models import SequenceTagger

from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel


class FlairNERWrapper(nn.Module):
    """Wrapper around Flair SequenceTagger for use as a torch.nn.Module."""

    def __init__(self, tagger):
        super().__init__()
        self.tagger = tagger

    def forward(self, sentence):
        """Run NER on input Flair Sentence.

        Args:
            sentence: A flair.data.Sentence object.

        Returns:
            flair.data.Sentence: The sentence annotated with NER predictions.
        """
        self.tagger.predict(sentence)
        return sentence


class ModelVariant(StrEnum):
    """Available Flair Arabic Multi NER model variants."""

    MEGANTOSH_FLAIR_ARABIC_MULTI_NER = "megantosh/flair-arabic-multi-ner"


class ModelLoader(ForgeModel):
    """Flair Arabic Multi NER model loader implementation."""

    _VARIANTS = {
        ModelVariant.MEGANTOSH_FLAIR_ARABIC_MULTI_NER: ModelConfig(
            pretrained_model_name="megantosh/flair-arabic-multi-ner",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MEGANTOSH_FLAIR_ARABIC_MULTI_NER

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.sample_text = (
            "عمرو عادلي أستاذ للاقتصاد السياسي بالجامعة الأمريكية بالقاهرة"
        )
        self.model = None

    @classmethod
    def _get_model_info(cls, variant_name=None):
        if variant_name is None:
            variant_name = "base"

        return ModelInfo(
            model="FlairArabicMultiNER",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TOKEN_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load Flair Arabic Multi NER SequenceTagger model."""
        tagger = SequenceTagger.load(self.model_name)
        model = FlairNERWrapper(tagger)
        model.eval()
        self.model = model
        return model

    def load_inputs(self, dtype_override=None):
        """Prepare sample input for Flair Arabic NER."""
        if self.model is None:
            self.load_model(dtype_override=dtype_override)

        sentence = Sentence(self.sample_text)
        return (sentence,)

    def decode_output(self, co_out):
        """Decode the model output for token classification."""
        entities = []
        for entity in co_out.get_spans("ner"):
            entities.append(f"{entity.text} ({entity.get_label('ner').value})")

        print(f"Context: {self.sample_text}")
        print(f"Entities: {entities}")
