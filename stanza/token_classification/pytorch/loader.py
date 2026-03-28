# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Stanza model loader implementation for token classification (NER).
"""

import stanza
import torch
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


class StanzaNERWrapper(torch.nn.Module):
    """Wrapper around stanza NER pipeline components as an nn.Module."""

    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = pipeline
        self.ner_tagger = pipeline.processors["ner"]._trainer.model

    def forward(self, text):
        """Run NER prediction through the stanza pipeline.

        Args:
            text: Input text string to process.

        Returns:
            stanza.Document: Annotated document with NER predictions.
        """
        doc = self.pipeline(text)
        return doc


class ModelVariant(StrEnum):
    """Available Stanza model variants for token classification."""

    STANZA_EN = "stanfordnlp/stanza-en"


class ModelLoader(ForgeModel):
    """Stanza model loader implementation for token classification (NER)."""

    _VARIANTS = {
        ModelVariant.STANZA_EN: ModelConfig(
            pretrained_model_name="stanfordnlp/stanza-en",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.STANZA_EN

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.sample_text = "My name is Sarah and I live in London."
        self.pipeline = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant=None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional variant name string. If None, uses DEFAULT_VARIANT.

        Returns:
            ModelInfo: Information about the model and variant
        """
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
        """Load Stanza NER model.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            Not used for Stanza models.

        Returns:
            torch.nn.Module: The Stanza NER wrapper model instance.
        """
        stanza.download("en")
        self.pipeline = stanza.Pipeline("en", processors="tokenize,ner", **kwargs)
        self.model = StanzaNERWrapper(self.pipeline)
        self.model.eval()
        return self.model

    def load_inputs(self, dtype_override=None):
        """Prepare sample input for Stanza NER.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            Not used for Stanza models.

        Returns:
            stanza.Document: Processed document with NER annotations.
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)
        doc = self.pipeline(self.sample_text)
        self.doc = doc
        return doc

    def decode_output(self, co_out):
        """Decode the model output for NER token classification.

        Args:
            co_out: Model output (stanza Document or raw output).
        """
        if hasattr(co_out, "sentences"):
            entities = []
            for sent in co_out.sentences:
                for ent in sent.ents:
                    entities.append((ent.text, ent.type))
        else:
            entities = co_out
        print(f"Context: {self.sample_text}")
        print(f"Entities: {entities}")
        return entities
