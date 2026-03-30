# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Higgs Audio V2 Tokenizer model loader implementation for audio tokenization tasks
"""
import torch
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


class ModelVariant(StrEnum):
    """Available Higgs Audio V2 Tokenizer model variants."""

    DEFAULT = "Default"


class ModelLoader(ForgeModel):
    """Higgs Audio V2 Tokenizer model loader implementation for audio feature extraction."""

    _VARIANTS = {
        ModelVariant.DEFAULT: ModelConfig(
            pretrained_model_name="bosonai/higgs-audio-v2-tokenizer",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEFAULT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="HiggsAudioV2Tokenizer",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Higgs Audio V2 Tokenizer model."""
        from huggingface_hub import hf_hub_download
        from boson_multimodal.tokenizer import HiggsAudioTokenizer

        pretrained_model_name = self._variant_config.pretrained_model_name

        # Download config and weights from HuggingFace Hub
        config_path = hf_hub_download(
            repo_id=pretrained_model_name, filename="config.json"
        )
        model_path = hf_hub_download(
            repo_id=pretrained_model_name, filename="model.pth"
        )

        model = HiggsAudioTokenizer.from_pretrained(config_path, model_path)

        if dtype_override is not None:
            model = model.to(dtype=dtype_override)

        model.eval()
        self.model = model
        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Higgs Audio V2 Tokenizer model."""
        # Generate a synthetic 1-second audio waveform at 24kHz (model's native sample rate)
        sample_rate = 24000
        duration_seconds = 1
        audio = torch.randn(1, 1, sample_rate * duration_seconds)

        if dtype_override is not None:
            audio = audio.to(dtype=dtype_override)

        return {"audio": audio}
