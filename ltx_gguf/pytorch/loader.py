# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LTX-2.3 GGUF model loader for tt_forge_models.

QuantStack/LTX-2.3-GGUF is a quantized GGUF conversion of the Lightricks/LTX-2.3
video generation model (21B parameter DiT). Available in multiple quantization
levels (Q2_K through Q8_0).

Repository: https://huggingface.co/QuantStack/LTX-2.3-GGUF
"""

from typing import Any, Optional

import torch
from diffusers import GGUFQuantizationConfig, LTX2VideoTransformer3DModel
from huggingface_hub import hf_hub_download

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)

REPO_ID = "QuantStack/LTX-2.3-GGUF"


class ModelVariant(StrEnum):
    """Available LTX-2.3 GGUF quantization variants."""

    LTX_2_3_Q4_K_M = "2.3_Q4_K_M"
    LTX_2_3_Q8_0 = "2.3_Q8_0"


# GGUF filenames within the repository
_GGUF_FILES = {
    ModelVariant.LTX_2_3_Q4_K_M: "ltx-2.3-Q4_K_M.gguf",
    ModelVariant.LTX_2_3_Q8_0: "ltx-2.3-Q8_0.gguf",
}


class ModelLoader(ForgeModel):
    """Loader for LTX-2.3 GGUF quantized video transformer model.

    Loads the LTX2VideoTransformer3DModel from a single GGUF file using
    diffusers' GGUFQuantizationConfig.
    """

    _VARIANTS = {
        ModelVariant.LTX_2_3_Q4_K_M: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
        ModelVariant.LTX_2_3_Q8_0: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LTX_2_3_Q4_K_M

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer: Optional[LTX2VideoTransformer3DModel] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="LTX GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        gguf_filename = _GGUF_FILES[self._variant]
        gguf_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=gguf_filename,
        )

        self._transformer = LTX2VideoTransformer3DModel.from_single_file(
            gguf_path,
            quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
            torch_dtype=dtype,
        )
        self._transformer.eval()
        return self._transformer

    def load_inputs(self, dtype_override=None, **kwargs) -> Any:
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if self._transformer is None:
            self.load_model(dtype_override=dtype)

        batch_size = 1
        config = self._transformer.config

        latent_num_frames = 2
        latent_height = 2
        latent_width = 2
        video_seq_len = latent_num_frames * latent_height * latent_width
        frame_rate = 24.0

        hidden_states = torch.randn(
            batch_size, video_seq_len, config.in_channels, dtype=dtype
        )
        audio_hidden_states = torch.randn(
            batch_size, 2, config.audio_in_channels, dtype=dtype
        )

        caption_channels = config.caption_channels
        encoder_hidden_states = torch.randn(
            batch_size, 8, caption_channels, dtype=dtype
        )
        audio_encoder_hidden_states = torch.randn(
            batch_size, 8, caption_channels, dtype=dtype
        )

        timestep = torch.tensor([0.5], dtype=dtype).expand(batch_size)
        audio_timestep = torch.tensor([0.5], dtype=dtype).expand(batch_size)
        sigma = torch.tensor([0.5], dtype=dtype).expand(batch_size)
        audio_sigma = torch.tensor([0.5], dtype=dtype).expand(batch_size)

        return {
            "hidden_states": hidden_states,
            "audio_hidden_states": audio_hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "audio_encoder_hidden_states": audio_encoder_hidden_states,
            "timestep": timestep,
            "audio_timestep": audio_timestep,
            "sigma": sigma,
            "audio_sigma": audio_sigma,
            "num_frames": latent_num_frames,
            "height": latent_height,
            "width": latent_width,
            "fps": frame_rate,
            "audio_num_frames": 2,
            "return_dict": False,
        }

    def unpack_forward_output(self, output: Any) -> torch.Tensor:
        if isinstance(output, tuple):
            return output[0]
        return output
