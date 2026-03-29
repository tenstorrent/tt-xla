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
        import torch.nn.functional as F
        from huggingface_hub import PyTorchModelHubMixin
        from transformers import WavLMConfig, WavLMModel

        class LoRALinear(nn.Module):
            """LoRA-adapted linear layer matching loralib parameter names."""

            def __init__(self, in_features, out_features, rank):
                super().__init__()
                self.weight = nn.Parameter(torch.empty(out_features, in_features))
                self.bias = nn.Parameter(torch.empty(out_features))
                self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
                self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

            def forward(self, x):
                return F.linear(x, self.weight, self.bias) + x @ self.lora_A.transpose(
                    0, 1
                ) @ self.lora_B.transpose(0, 1)

        class WavLMWrapper(nn.Module, PyTorchModelHubMixin):
            """WavLM-based age and sex classifier with LoRA fine-tuning."""

            def __init__(
                self,
                pretrain_model="wavlm_large",
                hidden_dim=256,
                output_class_num=2,
                lora_rank=16,
                finetune_method="lora",
                freeze_params=True,
                apply_reg=True,
                use_conv_output=True,
                num_dataset=4,
                apply_gradient_reversal=False,
            ):
                super().__init__()
                self.apply_reg = apply_reg

                # Load backbone config and create model without pretrained weights
                # (weights will be loaded from the saved state dict)
                backbone_config = WavLMConfig.from_pretrained("microsoft/wavlm-large")
                self.backbone_model = WavLMModel(backbone_config)
                num_layers = backbone_config.num_hidden_layers
                encoder_dim = backbone_config.hidden_size

                # Apply LoRA to feed-forward layers in the second half of encoder
                if finetune_method == "lora":
                    half = num_layers // 2
                    for i in range(half, num_layers):
                        layer = self.backbone_model.encoder.layers[i]
                        ff = layer.feed_forward
                        ff.intermediate_dense = LoRALinear(
                            backbone_config.hidden_size,
                            backbone_config.intermediate_size,
                            lora_rank,
                        )
                        ff.output_dense = LoRALinear(
                            backbone_config.intermediate_size,
                            backbone_config.hidden_size,
                            lora_rank,
                        )

                # Weighted layer sum across all hidden states
                self.weights = nn.Parameter(
                    torch.ones(num_layers + 1) / (num_layers + 1)
                )

                # Conv1d feature projection
                self.model_seq = nn.Sequential(
                    nn.Conv1d(encoder_dim, hidden_dim, 1),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Conv1d(hidden_dim, hidden_dim, 1),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Conv1d(hidden_dim, hidden_dim, 1),
                )

                # Age prediction head
                if apply_reg:
                    self.age_dist_layer = nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, 1),
                    )
                else:
                    self.age_dist_layer = nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, 7),
                    )

                # Sex classification head
                self.sex_layer = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, output_class_num),
                )

            def forward(self, input_values):
                outputs = self.backbone_model(input_values, output_hidden_states=True)
                hidden_states = torch.stack(outputs.hidden_states, dim=0)

                norm_weights = F.softmax(self.weights, dim=0)
                weighted = (hidden_states * norm_weights.view(-1, 1, 1, 1)).sum(dim=0)

                # Conv1d expects (batch, channels, time)
                features = weighted.transpose(1, 2)
                features = self.model_seq(features)
                features = features.mean(dim=2)

                age_pred = self.age_dist_layer(features)
                if self.apply_reg:
                    age_pred = torch.sigmoid(age_pred)

                sex_pred = self.sex_layer(features)

                return age_pred, sex_pred

        model = WavLMWrapper.from_pretrained(
            self._variant_config.pretrained_model_name,
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
