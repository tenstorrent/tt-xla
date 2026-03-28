# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Pyannote speaker segmentation MLX model loader implementation.

Loads the aufklarer/Pyannote-Segmentation-MLX model, an MLX-compatible
conversion of pyannote/segmentation-3.0 (PyanNet) for speaker segmentation
and voice activity detection.
"""

import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
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


class PyanNet(nn.Module):
    """PyanNet speaker segmentation model (PyTorch reconstruction).

    Architecture: SincNet encoder -> BiLSTM -> Linear classifier.
    Reconstructed from the MLX-converted safetensors weights.
    """

    def __init__(self, config):
        super().__init__()
        sincnet_cfg = config["sincnet"]
        lstm_cfg = config["lstm"]
        linear_cfg = config["linear"]

        # SincNet encoder (pre-computed sinc filters as regular convolutions)
        sincnet_layers = []
        in_channels = 1
        for n_filters, kernel_size, stride, pool_size in zip(
            sincnet_cfg["n_filters"],
            sincnet_cfg["kernel_sizes"],
            sincnet_cfg["strides"],
            sincnet_cfg["pool_sizes"],
        ):
            sincnet_layers.append(
                nn.Conv1d(in_channels, n_filters, kernel_size, stride=stride)
            )
            sincnet_layers.append(nn.LeakyReLU(0.2))
            sincnet_layers.append(nn.MaxPool1d(pool_size))
            in_channels = n_filters
        self.sincnet = nn.Sequential(*sincnet_layers)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=sincnet_cfg["n_filters"][-1],
            hidden_size=lstm_cfg["hidden_size"],
            num_layers=lstm_cfg["num_layers"],
            batch_first=True,
            bidirectional=lstm_cfg["bidirectional"],
        )

        # Linear classifier
        lstm_out_size = lstm_cfg["hidden_size"] * (
            2 if lstm_cfg["bidirectional"] else 1
        )
        linear_layers = []
        in_features = lstm_out_size
        for _ in range(linear_cfg["num_layers"] - 1):
            linear_layers.append(nn.Linear(in_features, linear_cfg["hidden_size"]))
            linear_layers.append(nn.LeakyReLU(0.2))
            in_features = linear_cfg["hidden_size"]
        linear_layers.append(nn.Linear(in_features, config["num_classes"]))
        self.linear = nn.Sequential(*linear_layers)

    def forward(self, waveform):
        # waveform: (batch, channels, samples) e.g. (1, 1, 160000)
        x = self.sincnet(waveform)
        x = x.transpose(1, 2)  # (batch, frames, features)
        x, _ = self.lstm(x)
        x = self.linear(x)
        return F.softmax(x, dim=-1)


class ModelVariant(StrEnum):
    """Available Pyannote segmentation MLX model variants."""

    SEGMENTATION_MLX = "Segmentation_MLX"


class ModelLoader(ForgeModel):
    """Pyannote speaker segmentation MLX model loader implementation."""

    _VARIANTS = {
        ModelVariant.SEGMENTATION_MLX: ModelConfig(
            pretrained_model_name="aufklarer/Pyannote-Segmentation-MLX",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SEGMENTATION_MLX

    def __init__(self, variant=None):
        super().__init__(variant)
        self._model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Pyannote",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the Pyannote segmentation MLX model.

        Downloads the config and safetensors weights from HuggingFace,
        reconstructs the PyanNet architecture in PyTorch, and loads the weights.
        """
        pretrained_name = self._variant_config.pretrained_model_name

        config_path = hf_hub_download(pretrained_name, "config.json")
        with open(config_path) as f:
            config = json.load(f)

        self._model = PyanNet(config)

        weights_path = hf_hub_download(pretrained_name, "model.safetensors")
        state_dict = load_file(weights_path)
        self._model.load_state_dict(state_dict, strict=False)

        self._model.eval()
        if dtype_override is not None:
            self._model.to(dtype_override)
        return self._model

    def load_inputs(self, dtype_override=None):
        """Load sample audio inputs for the segmentation model.

        Generates a 10-second mono audio waveform at 16kHz as expected
        by the model: shape (batch_size, num_channels, num_samples) = (1, 1, 160000).
        """
        dtype = dtype_override or torch.float32
        waveform = torch.randn(1, 1, 160000, dtype=dtype)
        return [waveform]
