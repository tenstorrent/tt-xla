# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Stanza model loader implementation for token classification (NER).

Stanza is Stanford NLP's Python NLP library built on PyTorch.
This loader wraps the Polish NER pipeline component.
"""

import stanza
from typing import Optional

from third_party.tt_forge_models.config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from third_party.tt_forge_models.base import ForgeModel


class ModelVariant(StrEnum):
    """Available Stanza model variants for token classification."""

    STANFORDNLP_STANZA_PL = "stanfordnlp/stanza-pl"


class ModelLoader(ForgeModel):
    """Stanza model loader implementation for token classification."""

    _VARIANTS = {
        ModelVariant.STANFORDNLP_STANZA_PL: ModelConfig(
            pretrained_model_name="stanfordnlp/stanza-pl",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.STANFORDNLP_STANZA_PL

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Stanza",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TOKEN_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the Stanza NER pipeline for Polish."""
        stanza.download("pl")
        nlp = stanza.Pipeline("pl", processors="tokenize,ner")
        self.pipeline = nlp

        # Extract the NER processor's underlying PyTorch model
        model = nlp.processors["ner"]._model
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        self.model = model
        return model

    def load_inputs(self, dtype_override=None):
        """Prepare sample input for Stanza NER token classification."""
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        sample_text = "Warszawa jest stolicą Polski."
        doc = self.pipeline(sample_text)
        return doc
