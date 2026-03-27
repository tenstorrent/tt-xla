# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GLiNER2 model loader implementation

GLiNER2 is a multi-task information extraction model built on a DeBERTa-v3
encoder. The gliner2 library's forward() is training-only (returns losses),
so we extract the encoder backbone and prepare standard transformer inputs
for inference testing.
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
        """Load the GLiNER2 model and return its DeBERTa encoder backbone."""
        from gliner2 import GLiNER2

        model_name = self._variant_config.pretrained_model_name
        model = GLiNER2.from_pretrained(model_name, **kwargs)
        self.model = model
        self.model.eval()
        # Return the encoder backbone for tensor-level inference testing
        return self.model.encoder

    def load_inputs(self, batch_size=1):
        """Prepare tokenized inputs for the DeBERTa encoder backbone."""
        self.text = (
            "Cristiano Ronaldo dos Santos Aveiro (Portuguese pronunciation: "
            "[kɾiʃ'tjɐnu ʁɔ'naldu]; born 5 February 1985) is a Portuguese "
            "professional footballer who plays as a forward for and captains "
            "both Saudi Pro League club Al Nassr and the Portugal national team."
        )

        tokenizer = self.model.processor.tokenizer
        inputs = tokenizer(
            self.text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return inputs

    def decode_output(self, co_out):
        """Run full GLiNER2 entity extraction and print results."""
        self.entity_types = ["person", "award", "date", "competitions", "teams"]
        results = self.model.extract_entities(
            self.text, self.entity_types, threshold=0.5
        )
        print(f"Text: {self.text}")
        print(f"Entities: {results}")
        return results
