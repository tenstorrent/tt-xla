# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
REVE Positions model loader for EEG electrode position feature extraction.
"""
from typing import Optional

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


class ModelVariant(StrEnum):
    """Available REVE Positions model variants."""

    REVE_POSITIONS = "brain-bzh/reve-positions"


class ModelLoader(ForgeModel):
    """REVE Positions model loader for EEG electrode position feature extraction."""

    _VARIANTS = {
        ModelVariant.REVE_POSITIONS: ModelConfig(
            pretrained_model_name="brain-bzh/reve-positions",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.REVE_POSITIONS

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="REVE Positions",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import AutoModel

        model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, **model_kwargs
        )
        model.eval()

        self.model = model

        return model

    def load_inputs(self, dtype_override=None):
        # Sample EEG electrode names from the 10-20 system
        electrode_names = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2"]
        return electrode_names

    def output_postprocess(self, output, inputs=None):
        if isinstance(output, (tuple, list)):
            return output[0]
        return output
