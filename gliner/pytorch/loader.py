# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GLiNER model loader implementation
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
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="gliner",
            variant=variant,
            group=ModelGroup.RED,
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
        if self._variant == ModelVariant.GLINER_LARGEV2:
            model = model.batch_predict_entities
        return model

    def load_inputs(self, batch_size=1):
        """Load and return sample inputs for the GLiNER model with default settings.

        Returns a tuple (texts, labels) suitable for GLiNER.batch_predict_entities.
        """
        if self._variant == ModelVariant.GLINER_MULTI_V21:
            text = """Cristiano Ronaldo dos Santos Aveiro was born 5 February 1985) is a Portuguese professional footballer."""
            labels = ["person", "award", "date", "competitions", "teams"]
            texts, raw_batch = pre_processing(self.model, [text], labels)
            self.text = text
            self.raw_batch = raw_batch
            return texts
        else:
            text = (
                "Cristiano Ronaldo dos Santos Aveiro (Portuguese pronunciation: [kɾiʃ'tjɐnu ʁɔ'naldu]; born 5 February 1985) "
                "is a Portuguese professional footballer who plays as a forward for and captains both Saudi Pro League club Al Nassr "
                "and the Portugal national team."
            )
            labels = ["person", "award", "date", "competitions", "teams"]
            texts = [text] * batch_size

            return (texts, labels)

    def post_processing(self, co_out):
        if self._variant == ModelVariant.GLINER_MULTI_V21:
            entities = post_processing(self.model, co_out, [self.text], self.raw_batch)
            for entity in entities:
                print(entity["text"], "=>", entity["label"])
