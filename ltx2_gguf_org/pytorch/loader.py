# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
gguf-org/ltx2-gguf model loader for tt_forge_models.

gguf-org/ltx2-gguf is a quantized GGUF conversion of the Lightricks/LTX-2
video generation model (19B parameter DiT). Available in multiple quantization
levels including IQ4_NL, IQ4_XS, Q2_K, and Q2_K_S.

Repository: https://huggingface.co/gguf-org/ltx2-gguf
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

REPO_ID = "gguf-org/ltx2-gguf"


class ModelVariant(StrEnum):
    """Available LTX-2 GGUF quantization variants from gguf-org."""

    LTX2_19B_DEV_IQ4_NL = "19B_dev_IQ4_NL"
    LTX2_19B_DEV_Q2_K = "19B_dev_Q2_K"


# GGUF filenames within the repository
_GGUF_FILES = {
    ModelVariant.LTX2_19B_DEV_IQ4_NL: "ltx2-19b-dev-iq4_nl.gguf",
    ModelVariant.LTX2_19B_DEV_Q2_K: "ltx2-19b-dev-q2_k.gguf",
}


class ModelLoader(ForgeModel):
    """Loader for gguf-org LTX-2 GGUF quantized video transformer model.

    Loads the LTX2VideoTransformer3DModel from a single GGUF file using
    diffusers' GGUFQuantizationConfig.
    """

    _VARIANTS = {
        ModelVariant.LTX2_19B_DEV_IQ4_NL: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
        ModelVariant.LTX2_19B_DEV_Q2_K: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LTX2_19B_DEV_IQ4_NL

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer: Optional[LTX2VideoTransformer3DModel] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="LTX2 GGUF Org",
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

        config = self._transformer.config

        batch_size = 1
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
