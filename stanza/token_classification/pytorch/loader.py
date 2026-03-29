# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Stanza model loader implementation for token classification.
"""

import torch
import torch.nn as nn
import stanza
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


class StanzaNERWrapper(nn.Module):
    """Wrapper around stanza NER pipeline for use as a torch.nn.Module."""

    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = pipeline
        # Extract the actual NER neural network model for parameter registration
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

    STANFORDNLP_STANZA_ES = "stanfordnlp/stanza-es"


class ModelLoader(ForgeModel):
    """Stanza model loader implementation for token classification."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.STANFORDNLP_STANZA_ES: ModelConfig(
            pretrained_model_name="stanfordnlp/stanza-es",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.STANFORDNLP_STANZA_ES

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        self.model_name = self._variant_config.pretrained_model_name
        self.sample_text = (
            "El presidente de México visitó la ciudad de Madrid ayer por la tarde."
        )
        self.pipeline = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant_name: Optional variant name string. If None, uses 'base'.

        Returns:
            ModelInfo: Information about the model and variant
        """
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
        """Load Stanza model for token classification.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The wrapped Stanza NER model instance.
        """
        stanza.download("es", processors="tokenize,ner")
        self.pipeline = stanza.Pipeline("es", processors="tokenize,ner")
        model = StanzaNERWrapper(self.pipeline)
        model.eval()
        self.model = model
        return model

    def load_inputs(self, dtype_override=None):
        """Prepare sample input for Stanza token classification.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model will use its default dtype (typically float32).

        Returns:
            tuple: Input text that can be fed to the model.
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        return (self.sample_text,)

    def decode_output(self, co_out):
        """Decode the model output for token classification.

        Args:
            co_out: Model output (stanza Document)
        """
        entities = []
        for sent in co_out.sentences:
            for ent in sent.ents:
                entities.append(f"{ent.text} ({ent.type})")

        print(f"Context: {self.sample_text}")
        print(f"Entities: {entities}")
