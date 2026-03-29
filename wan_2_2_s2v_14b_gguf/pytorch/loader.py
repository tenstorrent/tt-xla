# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
QuantStack Wan 2.2 Sound-to-Video 14B GGUF model loader implementation.

Loads the GGUF-quantized Wan 2.2 S2V 14B denoising transformer from
QuantStack/Wan2.2-S2V-14B-GGUF. This model generates video from audio input.

Base model: Wan-AI/Wan2.2-S2V-14B
Format: GGUF (quantized for ComfyUI inference via ComfyUI-GGUF)

Available variants:
- WAN22_S2V_14B_Q4_0: Q4_0 quantization (12.8 GB)
"""

from typing import Any, Optional

import torch
from huggingface_hub import hf_hub_download  # type: ignore[import]

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

REPO_ID = "QuantStack/Wan2.2-S2V-14B-GGUF"

_GGUF_FILES = {
    "Q4_0": "Wan2.2-S2V-14B-Q4_0.gguf",
}


class ModelVariant(StrEnum):
    """Available Wan 2.2 S2V 14B GGUF model variants."""

    WAN22_S2V_14B_Q4_0 = "S2V_14B_Q4_0"


class ModelLoader(ForgeModel):
    """Wan 2.2 Sound-to-Video 14B GGUF model loader.

    Downloads the GGUF-quantized denoising transformer from HuggingFace.
    The GGUF file represents the UNet/transformer component of the Wan 2.2
    Sound-to-Video pipeline.
    """

    _VARIANTS = {
        ModelVariant.WAN22_S2V_14B_Q4_0: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.WAN22_S2V_14B_Q4_0

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._model_path = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="WAN_2_2_S2V_14B_GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Download and return the path to the GGUF model file.

        The GGUF format is not directly loadable into a PyTorch model via
        diffusers. The returned path can be used with ComfyUI-GGUF for
        inference.

        Returns:
            str: Local path to the downloaded GGUF model file.
        """
        gguf_filename = _GGUF_FILES["Q4_0"]
        self._model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=gguf_filename,
        )
        return self._model_path

    def load_inputs(self, **kwargs) -> Any:
        """Prepare dummy inputs for the sound-to-video model.

        Returns a dict with a dummy audio waveform tensor (5 seconds at 16kHz).
        """
        dtype = kwargs.get("dtype_override", torch.float32)
        sample_rate = 16000
        duration_sec = 5
        return {
            "audio": torch.randn(1, sample_rate * duration_sec, dtype=dtype),
        }
