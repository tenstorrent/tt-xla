# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Parakeet TDT HF model loader implementation for automatic speech recognition.
"""

from typing import Optional

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Parakeet TDT HF model variants."""

    TDT_0_6B_V3_HF = "TDT 0.6B v3 HF"


class ModelLoader(ForgeModel):
    """Parakeet TDT HF model loader implementation for automatic speech recognition."""

    _VARIANTS = {
        ModelVariant.TDT_0_6B_V3_HF: ModelConfig(
            pretrained_model_name="bezzam/parakeet-tdt-0.6b-v3-hf",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TDT_0_6B_V3_HF

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._processor = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Parakeet_TDT_HF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        from transformers import AutoProcessor

        self._processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
        )

        return self._processor

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import AutoModel

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        self.model = AutoModel.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        self.model.eval()
        if dtype_override is not None:
            self.model.to(dtype_override)

        return self.model

    def load_inputs(self, dtype_override=None):
        import numpy as np
        import torch

        if self.model is None:
            self.load_model(dtype_override=dtype_override)

        if self._processor is None:
            self._load_processor()

        model_param = next(self.model.parameters())
        device = model_param.device
        dtype = dtype_override or model_param.dtype

        # Generate a synthetic 1-second audio waveform at 16kHz
        sampling_rate = 16000
        duration_seconds = 1
        audio_array = np.random.randn(sampling_rate * duration_seconds).astype(
            np.float32
        )

        inputs = self._processor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )

        # Cast inputs to match model dtype and device
        for key in inputs:
            if (
                isinstance(inputs[key], torch.Tensor)
                and inputs[key].is_floating_point()
            ):
                inputs[key] = inputs[key].to(device=device, dtype=dtype)
            elif isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(device=device)

        return inputs
