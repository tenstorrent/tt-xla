# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GLiNER2 model loader implementation
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
    GLINER2_MULTI_V1 = "Multi_v1"


class ModelLoader(ForgeModel):
    """GLiNER2 model loader implementation."""

    _VARIANTS = {
        ModelVariant.GLINER2_MULTI_V1: ModelConfig(
            pretrained_model_name="fastino/gliner2-multi-v1"
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GLINER2_MULTI_V1

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant."""
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = "base"

        return ModelInfo(
            model="GLiNER2",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TOKEN_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, **kwargs):
        """Load and return the GLiNER2 model."""
        from gliner2 import GLiNER2

        model_name = self._variant_config.pretrained_model_name
        model = GLiNER2.from_pretrained(model_name, **kwargs)
        self.model = model
        return self.model.eval()

    def load_inputs(self, batch_size=1):
        """Load and return sample inputs for the GLiNER2 model."""
        text = (
            "Cristiano Ronaldo dos Santos Aveiro (Portuguese pronunciation: "
            "[kɾiʃ'tjɐnu ʁɔ'naldu]; born 5 February 1985) is a Portuguese "
            "professional footballer who plays as a forward for and captains "
            "both Saudi Pro League club Al Nassr and the Portugal national team."
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
        """Decode model output into entity predictions."""
        outputs = []
        decoded = self.model.decoder.decode(
            self.batch["tokens"],
            self.batch["id_to_classes"],
            co_out,
            flat_ner=True,
            threshold=0.5,
            multi_label=False,
        )
        outputs.extend(decoded)
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
                    "text": self.text[start_text_idx:end_text_idx],
                    "label": ent_type,
                    "score": ent_score,
                }
                entities.append(ent_details)
            all_entities.append(entities)
        return all_entities
