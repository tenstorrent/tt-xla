# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BFS (Best Face Swap) LoRA model loader implementation.

Loads the Qwen-Image-Edit base pipeline and applies BFS LoRA weights
from Alissonerdx/BFS-Best-Face-Swap for high-fidelity face/head swapping.

Available variants:
- HEAD_V5_2511: Head swap v5 on Qwen-Image-Edit 2511 (recommended)
- HEAD_V3_2509: Head swap v3 on Qwen-Image-Edit 2509
- FACE_V1_2509: Face-only swap v1 on Qwen-Image-Edit 2509
"""

from typing import Any, Optional

import torch
from diffusers import QwenImageEditPipeline
from PIL import Image

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

LORA_REPO = "Alissonerdx/BFS-Best-Face-Swap"

# Base model repos per Qwen-Image-Edit version
_BASE_MODELS = {
    "2509": "Qwen/Qwen-Image-Edit-2509",
    "2511": "Qwen/Qwen-Image-Edit-2511",
}

# LoRA weight filenames and their corresponding base model version
_VARIANT_META = {
    "head_v5_2511": {
        "lora_file": "bfs_head_v5_2511_original.safetensors",
        "base_version": "2511",
    },
    "head_v3_2509": {
        "lora_file": "bfs_head_v3_qwen_image_edit_2509.safetensors",
        "base_version": "2509",
    },
    "face_v1_2509": {
        "lora_file": "bfs_face_v1_qwen_image_edit_2509.safetensors",
        "base_version": "2509",
    },
}


class ModelVariant(StrEnum):
    """Available BFS face swap LoRA variants."""

    HEAD_V5_2511 = "Head_V5_2511"
    HEAD_V3_2509 = "Head_V3_2509"
    FACE_V1_2509 = "Face_V1_2509"


_VARIANT_KEYS = {
    ModelVariant.HEAD_V5_2511: "head_v5_2511",
    ModelVariant.HEAD_V3_2509: "head_v3_2509",
    ModelVariant.FACE_V1_2509: "face_v1_2509",
}


class ModelLoader(ForgeModel):
    """BFS (Best Face Swap) LoRA model loader."""

    _VARIANTS = {
        ModelVariant.HEAD_V5_2511: ModelConfig(
            pretrained_model_name=_BASE_MODELS["2511"],
        ),
        ModelVariant.HEAD_V3_2509: ModelConfig(
            pretrained_model_name=_BASE_MODELS["2509"],
        ),
        ModelVariant.FACE_V1_2509: ModelConfig(
            pretrained_model_name=_BASE_MODELS["2509"],
        ),
    }
    DEFAULT_VARIANT = ModelVariant.HEAD_V5_2511

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[QwenImageEditPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="BFS_FACE_SWAP",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ):
        """Load the Qwen-Image-Edit pipeline with BFS LoRA weights applied.

        Returns:
            QwenImageEditPipeline with LoRA weights loaded.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        self.pipeline = QwenImageEditPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
        )

        meta = _VARIANT_META[_VARIANT_KEYS[self._variant]]
        self.pipeline.load_lora_weights(
            LORA_REPO,
            weight_name=meta["lora_file"],
        )

        return self.pipeline

    def load_inputs(self, **kwargs) -> Any:
        """Prepare sample inputs for face swap inference.

        Returns:
            dict with prompt, image, and control_image keys.
        """
        # Body/target image and face/source image (placeholders)
        body_image = Image.new("RGB", (512, 512), color=(200, 180, 160))
        face_image = Image.new("RGB", (512, 512), color=(160, 140, 120))

        prompt = (
            "head_swap: start with Picture 1 as the base image, keeping its "
            "lighting, environment, and background. Remove the head from "
            "Picture 1 completely and replace it with the head from Picture 2, "
            "strictly preserving the hair, eye color, and nose structure of "
            "Picture 2. High quality, sharp details, 4k"
        )

        return {
            "prompt": prompt,
            "image": body_image,
            "control_image": face_image,
        }
