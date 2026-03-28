# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
NuNER Zero model loader implementation
"""

from typing import Optional

from ...config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    ModelConfig,
    StrEnum,
)
from ...base import ForgeModel


class ModelVariant(StrEnum):
    NUNER_ZERO = "NuNER_Zero"
    NUNER_ZERO_SPAN = "NuNER_Zero-span"


class ModelLoader(ForgeModel):
    """NuNER Zero model loader implementation."""

    _VARIANTS = {
        ModelVariant.NUNER_ZERO: ModelConfig(pretrained_model_name="numind/NuNerZero"),
        ModelVariant.NUNER_ZERO_SPAN: ModelConfig(
            pretrained_model_name="numind/NuNerZero_span"
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NUNER_ZERO

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant."""
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="NuNER",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TOKEN_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, **kwargs):
        """Load and return the NuNER Zero GLiNER model."""
        from gliner import GLiNER

        model_name = self._variant_config.pretrained_model_name
        model = GLiNER.from_pretrained(model_name, **kwargs)
        self.model = model
        return self.model.eval()

    def load_inputs(self, batch_size=1):
        """Load and return sample inputs for the NuNER Zero model.

        Returns a batch suitable for the GLiNER model forward pass.
        """
        text = (
            "At the annual technology summit, the keynote address was delivered by a senior member of the "
            "Association for Computing Machinery Special Interest Group on Algorithms and Computation Theory."
        )
        self.text = [text]
        labels = ["person", "organization", "event"]
        entity_types = list(dict.fromkeys(labels))

        (
            tokens,
            all_start_token_idx_to_text_idx,
            all_end_token_idx_to_text_idx,
        ) = self.model.prepare_inputs(
            texts=[text],
        )
        self.all_start_token_idx_to_text_idx = all_start_token_idx_to_text_idx
        self.all_end_token_idx_to_text_idx = all_end_token_idx_to_text_idx

        input_x = self.model.prepare_base_input(tokens)

        collator = self.model.data_collator_class(
            self.model.config,
            data_processor=self.model.data_processor,
            return_tokens=True,
            return_entities=True,
            return_id_to_classes=True,
            prepare_labels=False,
        )

        batch = collator(input_x, entity_types=entity_types)
        self.batch = batch
        return batch

    def post_processing(self, co_out):
        decoded = self.model.decoder.decode(
            self.batch["tokens"],
            self.batch["id_to_classes"],
            co_out,
            flat_ner=True,
            threshold=0.5,
            multi_label=False,
        )
        all_entities = []
        for i, spans in enumerate(decoded):
            start_token_idx_to_text_idx = self.all_start_token_idx_to_text_idx[i]
            end_token_idx_to_text_idx = self.all_end_token_idx_to_text_idx[i]
            entities = []
            for span in spans:
                start_text_idx = start_token_idx_to_text_idx[span.start]
                end_text_idx = end_token_idx_to_text_idx[span.end]
                ent_details = {
                    "start": start_text_idx,
                    "end": end_text_idx,
                    "text": self.text[i][start_text_idx:end_text_idx],
                    "label": span.entity_type,
                    "score": span.score,
                }
                entities.append(ent_details)
            all_entities.append(entities)
        return all_entities
