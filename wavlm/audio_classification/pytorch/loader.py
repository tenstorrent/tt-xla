# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
WavLM model loader implementation for audio classification (age/sex prediction).
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
    """Available WavLM audio classification model variants."""

    LARGE_AGE_SEX = "Large_Age_Sex"


class ModelLoader(ForgeModel):
    """WavLM model loader implementation for audio classification (age/sex prediction)."""

    _VARIANTS = {
        ModelVariant.LARGE_AGE_SEX: ModelConfig(
            pretrained_model_name="tiantiaf/wavlm-large-age-sex",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LARGE_AGE_SEX

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._feature_extractor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="WavLM",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_feature_extractor(self, dtype_override=None):
        from transformers import AutoFeatureExtractor

        feature_extractor_kwargs = {}
        if dtype_override is not None:
            feature_extractor_kwargs["dtype"] = dtype_override

        self._feature_extractor = AutoFeatureExtractor.from_pretrained(
            "microsoft/wavlm-large", **feature_extractor_kwargs
        )

        return self._feature_extractor

    def load_model(self, *, dtype_override=None, **kwargs):
        import torch
        import torch.nn as nn
        from transformers import WavLMConfig
        from transformers.models.wavlm.modeling_wavlm import (
            WavLMModel,
            WavLMPreTrainedModel,
        )

        class ModelHead(nn.Module):
            """Classification/regression head."""

            def __init__(self, config, num_labels):
                super().__init__()
                self.dense = nn.Linear(config.hidden_size, config.hidden_size)
                self.dropout = nn.Dropout(config.final_dropout)
                self.out_proj = nn.Linear(config.hidden_size, num_labels)

            def forward(self, features, **kwargs):
                x = features
                x = self.dropout(x)
                x = self.dense(x)
                x = torch.tanh(x)
                x = self.dropout(x)
                x = self.out_proj(x)
                return x

        class AgeSexModel(WavLMPreTrainedModel):
            """WavLM-based age and sex classifier."""

            def __init__(self, config):
                super().__init__(config)
                self.config = config
                self.wavlm = WavLMModel(config)
                self.age = ModelHead(config, 1)
                self.sex = ModelHead(config, 2)
                self.init_weights()

            def forward(self, input_values):
                outputs = self.wavlm(input_values)
                hidden_states = outputs[0]
                hidden_states = torch.mean(hidden_states, dim=1)
                logits_age = self.age(hidden_states)
                logits_sex = torch.softmax(self.sex(hidden_states), dim=1)
                return hidden_states, logits_age, logits_sex

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        # Load config
        import json
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(
            self._variant_config.pretrained_model_name, "config.json"
        )
        with open(config_path) as f:
            config_dict = json.load(f)
        config_dict["vocab_size"] = config_dict.get("vocab_size") or 1
        config = WavLMConfig(**config_dict)

        model = AgeSexModel.from_pretrained(
            self._variant_config.pretrained_model_name,
            config=config,
            **model_kwargs,
        )
        model.eval()
        if dtype_override is not None:
            model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        import numpy as np

        if self._feature_extractor is None:
            self._load_feature_extractor(dtype_override=dtype_override)

        # Generate a synthetic 1-second audio waveform at 16kHz
        sampling_rate = 16000
        duration_seconds = 1
        audio_array = np.random.randn(sampling_rate * duration_seconds).astype(
            np.float32
        )

        inputs = self._feature_extractor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )

        return inputs
