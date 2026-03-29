# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SDXS-512-0.9-OpenVINO (rupeshs/sdxs-512-0.9-openvino) model loader implementation.

SDXS-512-0.9 is a distilled Stable Diffusion model optimized for fast single-step
text-to-image generation. This variant uses OpenVINO for optimized CPU inference
via the optimum-intel library.

Available variants:
- SDXS_512_0_9_OPENVINO: rupeshs/sdxs-512-0.9-openvino text-to-image generation
"""

from typing import Optional

from optimum.intel.openvino.modeling_diffusion import OVStableDiffusionPipeline

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


REPO_ID = "rupeshs/sdxs-512-0.9-openvino"


class ModelVariant(StrEnum):
    """Available SDXS-512-0.9-OpenVINO model variants."""

    SDXS_512_0_9_OPENVINO = "SDXS_512_0.9_OpenVINO"


class ModelLoader(ForgeModel):
    """SDXS-512-0.9-OpenVINO model loader implementation."""

    _VARIANTS = {
        ModelVariant.SDXS_512_0_9_OPENVINO: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.SDXS_512_0_9_OPENVINO

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="SDXS_512_0.9_OpenVINO",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.ONNX,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the SDXS-512-0.9-OpenVINO pipeline.

        Returns:
            OVStableDiffusionPipeline: The OpenVINO Stable Diffusion pipeline instance.
        """
        self.pipeline = OVStableDiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            ov_config={"CACHE_DIR": ""},
            **kwargs,
        )
        return self.pipeline

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for the SDXS model.

        Returns:
            list: A list of sample text prompts.
        """
        return ["A cute cat"] * batch_size
