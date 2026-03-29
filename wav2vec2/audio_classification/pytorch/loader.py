# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Wav2Vec2 model loader implementation for audio classification.
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
    """Available Wav2Vec2 audio classification model variants."""

    LARGE_ROBUST_12_FT_EMOTION_MSP_DIM = "Large_Robust_12_FT_Emotion_MSP_Dim"
    DEEPFAKE_AUDIO_DETECTION = "Deepfake_Audio_Detection"


class ModelLoader(ForgeModel):
    """Wav2Vec2 model loader implementation for audio classification (PyTorch)."""

    _VARIANTS = {
        ModelVariant.LARGE_ROBUST_12_FT_EMOTION_MSP_DIM: ModelConfig(
            pretrained_model_name="audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
        ),
        ModelVariant.DEEPFAKE_AUDIO_DETECTION: ModelConfig(
            pretrained_model_name="mo-thecreator/Deepfake-audio-detection",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LARGE_ROBUST_12_FT_EMOTION_MSP_DIM

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Wav2Vec2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        from transformers import Wav2Vec2FeatureExtractor

        processor_kwargs = {}
        if dtype_override is not None:
            processor_kwargs["dtype"] = dtype_override

        self._processor = Wav2Vec2FeatureExtractor.from_pretrained(
            self._variant_config.pretrained_model_name, **processor_kwargs
        )

        return self._processor

    def load_model(self, *, dtype_override=None, **kwargs):
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        if self._variant == ModelVariant.DEEPFAKE_AUDIO_DETECTION:
            model = self._load_deepfake_model(**model_kwargs)
        else:
            model = self._load_emotion_model(**model_kwargs)

        model.eval()
        if dtype_override is not None:
            model.to(dtype_override)

        return model

    def _load_deepfake_model(self, **model_kwargs):
        from transformers import AutoModelForAudioClassification

        return AutoModelForAudioClassification.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )

    def _load_emotion_model(self, **model_kwargs):
        import torch
        import torch.nn as nn
        from transformers import Wav2Vec2Config
        from transformers.models.wav2vec2.modeling_wav2vec2 import (
            Wav2Vec2Model,
            Wav2Vec2PreTrainedModel,
        )

        class RegressionHead(nn.Module):
            """Classification head for emotion regression."""

            def __init__(self, config):
                super().__init__()
                self.dense = nn.Linear(config.hidden_size, config.hidden_size)
                self.dropout = nn.Dropout(config.final_dropout)
                self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

            def forward(self, features, **kwargs):
                x = features
                x = self.dropout(x)
                x = self.dense(x)
                x = torch.tanh(x)
                x = self.dropout(x)
                x = self.out_proj(x)
                return x

        class EmotionModel(Wav2Vec2PreTrainedModel):
            """Speech emotion classifier."""

            def __init__(self, config):
                super().__init__(config)
                self.config = config
                self.wav2vec2 = Wav2Vec2Model(config)
                self.classifier = RegressionHead(config)
                self.post_init()

            def forward(self, input_values):
                outputs = self.wav2vec2(input_values)
                hidden_states = outputs[0]
                hidden_states = torch.mean(hidden_states, dim=1)
                logits = self.classifier(hidden_states)
                return hidden_states, logits

        # Load config with workaround for vocab_size=None in upstream config
        import json
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(
            self._variant_config.pretrained_model_name, "config.json"
        )
        with open(config_path) as f:
            config_dict = json.load(f)
        config_dict["vocab_size"] = config_dict.get("vocab_size") or 1
        config_dict["num_labels"] = config_dict.get("num_labels") or 3
        config = Wav2Vec2Config(**config_dict)

        return EmotionModel.from_pretrained(
            self._variant_config.pretrained_model_name,
            config=config,
            **model_kwargs,
        )

    def load_inputs(self, dtype_override=None):
        import numpy as np

        if self._processor is None:
            self._load_processor(dtype_override=dtype_override)

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

        return inputs
