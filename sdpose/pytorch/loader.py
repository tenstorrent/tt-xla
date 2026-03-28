# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SDPose model loader implementation for whole-body pose estimation.

Loads the UNet2DConditionModel backbone from Comfy-Org/SDPose, a repackaged
single-file version of SDPose-OOD (Stable Diffusion UNet fine-tuned for
whole-body keypoint detection).

Reference: https://huggingface.co/Comfy-Org/SDPose
Upstream: https://huggingface.co/teemosliang/SDPose-Wholebody
"""

from typing import Optional

import torch
from diffusers import UNet2DConditionModel
from huggingface_hub import hf_hub_download

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

REPO_ID = "Comfy-Org/SDPose"
UPSTREAM_REPO = "teemosliang/SDPose-Wholebody"

# UNet architecture constants (SD v2 backbone)
IN_CHANNELS = 4
CROSS_ATTENTION_DIM = 1024
SAMPLE_SIZE = 64  # 64x64 latent (512x512 pixel)
CLASS_EMBED_DIM = 4  # projection_class_embeddings_input_dim


class ModelVariant(StrEnum):
    """Available SDPose model variants."""

    WHOLEBODY = "Wholebody"


class ModelLoader(ForgeModel):
    """SDPose model loader for whole-body pose estimation UNet backbone."""

    _VARIANTS = {
        ModelVariant.WHOLEBODY: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.WHOLEBODY

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="SDPose",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_KEYPOINT_DET,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the SDPose UNet backbone.

        Returns:
            UNet2DConditionModel: The SD v2 UNet fine-tuned for pose estimation.
        """
        ckpt_path = hf_hub_download(
            repo_id=REPO_ID,
            filename="checkpoints/sdpose_wholebody_fp16.safetensors",
        )

        model = UNet2DConditionModel.from_single_file(
            ckpt_path,
            config=UPSTREAM_REPO,
            subfolder="unet",
            torch_dtype=dtype_override or torch.float32,
            **kwargs,
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Prepare synthetic inputs for the UNet backbone.

        Returns:
            dict: Input tensors (sample, timestep, encoder_hidden_states, class_labels).
        """
        dtype = dtype_override or torch.float32

        # Noisy latent input [B, C, H, W]
        sample = torch.randn(
            batch_size, IN_CHANNELS, SAMPLE_SIZE, SAMPLE_SIZE, dtype=dtype
        )

        # Diffusion timestep
        timestep = torch.tensor([1], dtype=dtype).expand(batch_size)

        # Text encoder hidden states (SD v2 uses OpenCLIP with dim=1024)
        encoder_hidden_states = torch.randn(
            batch_size, 77, CROSS_ATTENTION_DIM, dtype=dtype
        )

        # Class embedding projection input
        class_labels = torch.randn(batch_size, CLASS_EMBED_DIM, dtype=dtype)

        return {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "class_labels": class_labels,
        }
