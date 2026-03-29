# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan2.2-TI2V-5B-Turbo GGUF (hum-ma/Wan2.2-TI2V-5B-Turbo-GGUF) model loader implementation.

Wan2.2-TI2V-5B-Turbo is a 5B parameter text+image-to-video generation model in GGUF
quantized format, based on the Wan diffusion architecture.

Available variants:
- WAN22_TI2V_5B_TURBO_Q4_0: Q4_0 quantized variant (~3.03 GB)
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
from .src.model_utils import load_wan_ti2v_gguf_pipe

REPO_ID = "hum-ma/Wan2.2-TI2V-5B-Turbo-GGUF"
BASE_MODEL = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"


class ModelVariant(StrEnum):
    """Available Wan2.2-TI2V-5B-Turbo GGUF model variants."""

    WAN22_TI2V_5B_TURBO_Q4_0 = "wan2.2_ti2v_5b_turbo_Q4_0"


class ModelLoader(ForgeModel):
    """Wan2.2-TI2V-5B-Turbo GGUF model loader implementation."""

    _VARIANTS = {
        ModelVariant.WAN22_TI2V_5B_TURBO_Q4_0: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.WAN22_TI2V_5B_TURBO_Q4_0

    GGUF_FILE = "Wan2_2-TI2V-5B-Turbo-Q4_0.gguf"

    prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Wan2.2-TI2V-5B-Turbo GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Wan TI2V transformer from GGUF checkpoint.

        Returns:
            torch.nn.Module: The Wan transformer model instance.
        """
        if self.pipeline is None:
            self.pipeline = load_wan_ti2v_gguf_pipe(REPO_ID, self.GGUF_FILE, BASE_MODEL)

        if dtype_override is not None:
            self.pipeline.transformer = self.pipeline.transformer.to(dtype_override)

        return self.pipeline.transformer

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the model.

        Returns:
            dict: Prompt dict for the Wan TI2V pipeline.
        """
        return {"prompt": self.prompt}
