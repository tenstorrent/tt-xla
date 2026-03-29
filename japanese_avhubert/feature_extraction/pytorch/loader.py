# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Japanese AVHuBERT model loader implementation for audio-visual feature extraction.
"""

import torch
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
    """Available Japanese AVHuBERT model variants."""

    LARGE_NOISE_PT = "Large_noise_pt"


class ModelLoader(ForgeModel):
    """Japanese AVHuBERT model loader for audio-visual feature extraction."""

    _VARIANTS = {
        ModelVariant.LARGE_NOISE_PT: ModelConfig(
            pretrained_model_name="enactic/japanese-avhubert-large_noise_pt",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LARGE_NOISE_PT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Japanese_AVHuBERT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        from transformers import AutoFeatureExtractor

        processor_kwargs = {}
        if dtype_override is not None:
            processor_kwargs["dtype"] = dtype_override

        self._processor = AutoFeatureExtractor.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            **processor_kwargs,
        )

        return self._processor

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import AutoModel

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            **model_kwargs,
        )
        model.eval()
        if dtype_override is not None:
            model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        import numpy as np
        import tempfile
        import os
        import soundfile as sf

        if self._processor is None:
            self._load_processor(dtype_override=dtype_override)

        # Generate synthetic 1-second audio waveform at 16kHz
        sampling_rate = 16000
        audio_array = np.random.randn(sampling_rate).astype(np.float32)

        # Save to temporary wav file for the feature extractor
        audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(audio_file.name, audio_array, sampling_rate)
        audio_path = audio_file.name

        # Generate synthetic grayscale mouth ROI video frames (25 fps, 1 second, 96x96)
        num_frames = 25
        frame_size = 96
        video_frames = np.random.randint(
            0, 256, (num_frames, frame_size, frame_size), dtype=np.uint8
        )

        # Save to temporary video file
        video_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        video_path = video_file.name
        video_file.close()

        import cv2

        writer = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            25,
            (frame_size, frame_size),
            False,
        )
        for frame in video_frames:
            writer.write(frame)
        writer.release()

        try:
            inputs = self._processor(raw_audio=audio_path, raw_video=video_path)
        finally:
            os.unlink(audio_path)
            os.unlink(video_path)

        return inputs
