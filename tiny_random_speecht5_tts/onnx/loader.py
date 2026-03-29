# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Tiny Random SpeechT5 OpenVINO model loader for text-to-speech tasks.

Loads the optimum-internal-testing/tiny-random-SpeechT5ForTextToSpeech-openvino
encoder model in OpenVINO IR format.
"""

import torch
from huggingface_hub import hf_hub_download
from openvino import Core
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

_REPO_ID = "optimum-internal-testing/tiny-random-SpeechT5ForTextToSpeech-openvino"
_ENCODER_XML = "openvino_encoder_model.xml"
_ENCODER_BIN = "openvino_encoder_model.bin"


class ModelVariant(StrEnum):
    """Available Tiny Random SpeechT5 TTS model variants."""

    ENCODER = "Encoder"


class ModelLoader(ForgeModel):
    """Tiny Random SpeechT5 OpenVINO model loader for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.ENCODER: ModelConfig(
            pretrained_model_name=_REPO_ID,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ENCODER

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant=None):
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="TinyRandomSpeechT5TTS",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.ONNX,
        )

    def load_model(self, **kwargs):
        """Load and return the SpeechT5 encoder in OpenVINO IR format.

        Returns:
            openvino.Model: The loaded OpenVINO encoder model.
        """
        xml_path = hf_hub_download(repo_id=_REPO_ID, filename=_ENCODER_XML)
        hf_hub_download(repo_id=_REPO_ID, filename=_ENCODER_BIN)
        core = Core()
        model = core.read_model(xml_path)
        return model

    def load_inputs(self, **kwargs):
        """Load and return sample inputs for the SpeechT5 encoder.

        Returns:
            dict: Input tensors for the encoder model.
        """
        input_ids = torch.randint(0, 100, (1, 10), dtype=torch.long)
        attention_mask = torch.ones(1, 10, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
