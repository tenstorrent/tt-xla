# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Stanza model loader implementation for token classification (NER).

Stanza is Stanford NLP's Python NLP library built on PyTorch.
This loader wraps the Vietnamese NER pipeline component.
"""

import torch.nn as nn
import stanza

from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....base import ForgeModel


class StanzaNERWrapper(nn.Module):
    """Wrapper around stanza NER pipeline for use as a torch.nn.Module."""

    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = pipeline
        ner_processor = pipeline.processors["ner"]
        self.ner_model = ner_processor._trainer.model

    def forward(self, text):
        """Run NER on input text through the stanza pipeline.

        Args:
            text: Input text string.

        Returns:
            stanza.Document: Annotated document with NER predictions.
        """
        return self.pipeline(text)


class ModelVariant(StrEnum):
    """Available Stanza model variants for token classification."""

    STANFORDNLP_STANZA_VI = "stanfordnlp/stanza-vi"


class ModelLoader(ForgeModel):
    """Stanza model loader implementation for token classification."""

    _VARIANTS = {
        ModelVariant.STANFORDNLP_STANZA_VI: ModelConfig(
            pretrained_model_name="stanfordnlp/stanza-vi",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.STANFORDNLP_STANZA_VI

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = self._variant_config.pretrained_model_name
        self.sample_text = (
            "Thủ tướng Nguyễn Xuân Phúc đã đến thăm thành phố Hồ Chí Minh hôm qua."
        )
        self.pipeline = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant_name=None):
        if variant_name is None:
            variant_name = "base"

        return ModelInfo(
            model="Stanza",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TOKEN_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load Stanza model for Vietnamese token classification."""
        stanza.download("vi", processors="tokenize,ner")
        self.pipeline = stanza.Pipeline("vi", processors="tokenize,ner")
        model = StanzaNERWrapper(self.pipeline)
        model.eval()
        self.model = model
        return model

    def load_inputs(self, dtype_override=None):
        """Prepare sample input for Stanza token classification."""
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        return (self.sample_text,)

    def decode_output(self, co_out):
        """Decode the model output for token classification."""
        entities = []
        for sent in co_out.sentences:
            for ent in sent.ents:
                entities.append(f"{ent.text} ({ent.type})")

        print(f"Context: {self.sample_text}")
        print(f"Entities: {entities}")
