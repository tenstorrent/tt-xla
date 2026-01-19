# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GLiNER model loader implementation
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
from .src.model_utils import pre_processing, post_processing


class ModelVariant(StrEnum):
    GLINER_LARGEV2 = "urchade/gliner_largev2"
    GLINER_MULTI_V21 = "urchade/gliner_multi-v2.1"


class ModelLoader(ForgeModel):
    """GLiNER model loader implementation."""

    _VARIANTS = {
        ModelVariant.GLINER_LARGEV2: ModelConfig(
            pretrained_model_name=str(ModelVariant.GLINER_LARGEV2)
        ),
        ModelVariant.GLINER_MULTI_V21: ModelConfig(
            pretrained_model_name=str(ModelVariant.GLINER_MULTI_V21)
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GLINER_MULTI_V21

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant."""
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):

        if variant in [ModelVariant.GLINER_MULTI_V21]:
            group = ModelGroup.RED
        else:
            group = ModelGroup.GENERALITY

        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="gliner",
            variant=variant,
            group=group,
            task=ModelTask.NLP_TOKEN_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self):
        """Load and return the GLiNER model callable (batch_predict_entities)."""
        from gliner import GLiNER

        model_name = self._variant_config.pretrained_model_name
        model = GLiNER.from_pretrained(model_name)
        self.model = model
        return self.model.eval()

    def load_inputs(self, batch_size=1):
        """Load and return sample inputs for the GLiNER model with default settings.

        Returns a tuple (texts, labels) suitable for GLiNER.batch_predict_entities.
        """
        text = (
            "Cristiano Ronaldo dos Santos Aveiro (Portuguese pronunciation: [kɾiʃ'tjɐnu ʁɔ'naldu]; born 5 February 1985) "
            "is a Portuguese professional footballer who plays as a forward for and captains both Saudi Pro League club Al Nassr "
            "and the Portugal national team."
        )
        self.text = text
        labels = ["person", "award", "date", "competitions", "teams"]
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
        outputs = []
        decoded12 = self.model.decoder.decode(
            self.batch["tokens"],
            self.batch["id_to_classes"],
            co_out,
            flat_ner=True,
            threshold=0.5,
            multi_label=False,
        )
        outputs.extend(decoded12)
        all_entities = []
        for i, output in enumerate(outputs):
            start_token_idx_to_text_idx = self.all_start_token_idx_to_text_idx[i]
            end_token_idx_to_text_idx = self.all_end_token_idx_to_text_idx[i]
            entities = []
            for start_token_idx, end_token_idx, ent_type, ent_score in output:
                start_text_idx = start_token_idx_to_text_idx[start_token_idx]
                end_text_idx = end_token_idx_to_text_idx[end_token_idx]
                ent_details = {
                    "start": start_token_idx_to_text_idx[start_token_idx],
                    "end": end_token_idx_to_text_idx[end_token_idx],
                    "text": self.text[i][start_text_idx:end_text_idx],
                    "label": ent_type,
                    "score": ent_score,
                }
                entities.append(ent_details)

            all_entities.append(entities)
        return all_entities
