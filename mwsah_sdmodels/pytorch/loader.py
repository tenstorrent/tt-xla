# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MWSAH/sdmodels GGUF (MWSAH/sdmodels) model loader implementation.

HyperFlux Dedistilled is a 12B parameter text-to-image generation model in GGUF quantized format,
based on the FLUX transformer architecture.

Available variants:
- HYPERFLUX_DEDISTILLED_Q4_K_M: Q4_K_M quantized variant (~6.91 GB)
"""

from typing import Optional

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from .src.model_utils import load_sdmodels_gguf_pipe, sdmodels_preprocessing

REPO_ID = "MWSAH/sdmodels"
BASE_MODEL = "black-forest-labs/FLUX.1-dev"


class ModelVariant(StrEnum):
    """Available MWSAH/sdmodels GGUF model variants."""

    HYPERFLUX_DEDISTILLED_Q4_K_M = "hyperFluxDedistilled_hyper16Q4KM"


class ModelLoader(ForgeModel):
    """MWSAH/sdmodels GGUF model loader implementation."""

    _VARIANTS = {
        ModelVariant.HYPERFLUX_DEDISTILLED_Q4_K_M: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.HYPERFLUX_DEDISTILLED_Q4_K_M

    GGUF_FILE = "hyperFluxDedistilled_hyper16Q4KM.gguf"

    prompt = "An astronaut riding a green horse"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="MWSAH/sdmodels GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the HyperFlux Dedistilled transformer from GGUF checkpoint.

        Returns:
            torch.nn.Module: The FLUX transformer model instance.
        """
        if self.pipeline is None:
            self.pipeline = load_sdmodels_gguf_pipe(REPO_ID, self.GGUF_FILE, BASE_MODEL)

        if dtype_override is not None:
            self.pipeline.transformer = self.pipeline.transformer.to(dtype_override)

        return self.pipeline.transformer

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the model.

        Returns:
            dict: Input tensors for the FLUX transformer model.
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else None

        return sdmodels_preprocessing(self.pipeline, self.prompt, dtype=dtype)
