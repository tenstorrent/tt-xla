# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
AIDO.RNA model loader implementation for embedding generation on RNA sequences.
"""
from typing import Optional

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available AIDO.RNA model variants."""

    AIDO_RNA_1B600M = "genbio-ai/AIDO.RNA-1.6B"


class ModelLoader(ForgeModel):
    """AIDO.RNA model loader for embedding generation on RNA sequences."""

    _VARIANTS = {
        ModelVariant.AIDO_RNA_1B600M: ModelConfig(
            pretrained_model_name="genbio-ai/AIDO.RNA-1.6B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.AIDO_RNA_1B600M

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._model_instance = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="AIDO.RNA",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from modelgenerator.tasks import Embed

        model = Embed.from_config({"model.backbone": "aido_rna_1b600m"}).eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        self._model_instance = model
        return model

    def load_inputs(self, dtype_override=None):
        if self._model_instance is None:
            self.load_model(dtype_override=dtype_override)

        # Sample RNA sequence
        rna_sequence = "ACGUACGUACGUACGU"

        transformed_batch = self._model_instance.transform(
            {"sequences": [rna_sequence]}
        )

        return transformed_batch
