# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Tiny WhisperForConditionalGeneration LoRA model loader implementation for speech recognition.
"""

import torch
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    WhisperConfig,
)
from peft import PeftModel
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
    """Available Tiny WhisperForConditionalGeneration LoRA model variants."""

    TINY_WHISPER_LORA = "Tiny_Whisper_LoRA"


class ModelLoader(ForgeModel):
    """Tiny WhisperForConditionalGeneration LoRA model loader for speech recognition tasks."""

    _VARIANTS = {
        ModelVariant.TINY_WHISPER_LORA: ModelConfig(
            pretrained_model_name="peft-internal-testing/tiny_WhisperForConditionalGeneration-lora",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TINY_WHISPER_LORA

    BASE_MODEL_NAME = "openai/whisper-tiny"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Tiny_WhisperForConditionalGeneration_LoRA",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load a Tiny WhisperForConditionalGeneration model with LoRA adapter."""
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        self.processor = WhisperProcessor.from_pretrained(self.BASE_MODEL_NAME)

        base_model = WhisperForConditionalGeneration.from_pretrained(
            self.BASE_MODEL_NAME, use_cache=False, **model_kwargs
        )

        adapter_name = self._variant_config.pretrained_model_name
        model = PeftModel.from_pretrained(base_model, adapter_name)
        model = model.merge_and_unload()

        for param in model.parameters():
            param.requires_grad = False

        model.eval()
        self.model = model
        return model

    def load_inputs(self, dtype_override=None):
        """Generate sample inputs for the model."""
        if self.model is None or self.processor is None:
            self.load_model()

        model_config = WhisperConfig.from_pretrained(self.BASE_MODEL_NAME)

        # Load audio sample
        weights_pth = get_file("test_files/pytorch/whisper/1272-128104-0000.pt")
        sample = torch.load(weights_pth, weights_only=False)
        sample_audio = sample["audio"]["array"]
        model_param = next(self.model.parameters())
        device, dtype = model_param.device, dtype_override or model_param.dtype

        # Preprocess audio
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
