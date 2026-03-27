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
        """Load and return sample inputs for the GLiNER2 model.

        Uses the GLiNER2 processor and collator to prepare a PreprocessedBatch
        containing input_ids, attention_mask, and schema metadata.
        """
        from gliner2.training.trainer import ExtractorCollator

        self.text = (
            "Cristiano Ronaldo dos Santos Aveiro (Portuguese pronunciation: "
            "[kɾiʃ'tjɐnu ʁɔ'naldu]; born 5 February 1985) is a Portuguese "
            "professional footballer who plays as a forward for and captains "
            "both Saudi Pro League club Al Nassr and the Portugal national team."
        )
        self.entity_types = ["person", "award", "date", "competitions", "teams"]

        schema = (
            self.model.create_schema().entities(self.entity_types).build_schema_dict()
        )

        self.model.processor.change_mode(is_training=False)
        collator = ExtractorCollator(self.model.processor, is_training=False)
        dataset = list(zip([self.text], [schema]))
        batch = collator(dataset)
        self.batch = batch
        return batch

    def post_processing(self, co_out):
        """Decode model output into entity predictions."""
        results = self.model.extract_entities(
            self.text, self.entity_types, threshold=0.5
        )
        print(f"Text: {self.text}")
        print(f"Entities: {results}")
        return results
