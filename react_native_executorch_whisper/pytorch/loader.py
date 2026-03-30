# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
React Native ExecuTorch Whisper small speech recognition model loader implementation.

Note: The original model (software-mansion/react-native-executorch-whisper-small) is in
ExecuTorch .pte format intended for mobile inference. Since ExecuTorch format is not
compatible with PyTorch, this loader uses the base model (openai/whisper-small) via
WhisperForConditionalGeneration.
"""

import torch
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperConfig,
)
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
from ...tools.utils import get_file


class ModelVariant(StrEnum):
    """Available React Native ExecuTorch Whisper model variants."""

    WHISPER_SMALL = "Small"


class ModelLoader(ForgeModel):
    """React Native ExecuTorch Whisper small speech recognition model loader implementation."""

    _VARIANTS = {
        ModelVariant.WHISPER_SMALL: ModelConfig(
            pretrained_model_name="openai/whisper-small",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.WHISPER_SMALL

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="React_Native_ExecuTorch_Whisper",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"use_cache": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        self.processor = WhisperProcessor.from_pretrained(pretrained_model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        self.model.eval()
        if dtype_override is not None:
            self.model.to(dtype_override)
        return self.model

    def load_inputs(self, dtype_override=None):
        if self.model is None or self.processor is None:
            self.load_model()

        model_config = WhisperConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )

        weights_pth = get_file("test_files/pytorch/whisper/1272-128104-0000.pt")
        sample = torch.load(weights_pth, weights_only=False)
        sample_audio = sample["audio"]["array"]
        model_param = next(self.model.parameters())
        device, dtype = model_param.device, dtype_override or model_param.dtype

        sampling_rate = 16000
        processed = self.processor(
            sample_audio, return_tensors="pt", sampling_rate=sampling_rate
        )
        input_features = processed.input_features.to(device=device, dtype=dtype)

        decoder_input_ids = torch.full(
            (1, 2),
            model_config.decoder_start_token_id,
            dtype=torch.long,
            device=device,
        )
        return [input_features, decoder_input_ids]
