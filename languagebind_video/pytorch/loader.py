# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LanguageBind Video model loader implementation for zero-shot video classification.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

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


@dataclass
class LanguageBindVideoConfig(ModelConfig):
    """Configuration specific to LanguageBind Video models."""

    num_frames: int = 8


class ModelVariant(StrEnum):
    """Available LanguageBind Video model variants."""

    VIDEO_FT = "Video_FT"


class ModelLoader(ForgeModel):
    """LanguageBind Video model loader for zero-shot video classification tasks."""

    _VARIANTS = {
        ModelVariant.VIDEO_FT: LanguageBindVideoConfig(
            pretrained_model_name="LanguageBind/LanguageBind_Video_FT",
            num_frames=8,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VIDEO_FT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self.text_prompts = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="LanguageBind-Video",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_ZS_IMAGE_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"return_dict": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(
            pretrained_model_name,
            trust_remote_code=True,
            **model_kwargs,
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor()

        num_frames = self._variant_config.num_frames

        # Create synthetic video input: list of frames as numpy arrays
        video = [
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            for _ in range(num_frames)
        ]

        self.text_prompts = ["playing sports", "eating spaghetti", "go shopping"]

        inputs = self.processor(
            text=self.text_prompts,
            videos=[video],
            return_tensors="pt",
            padding=True,
        )

        # Replicate tensors for batch size
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        # Convert floating point inputs to dtype_override if specified
        if dtype_override is not None:
            for key in inputs:
                if torch.is_tensor(inputs[key]) and inputs[key].dtype.is_floating_point:
                    inputs[key] = inputs[key].to(dtype_override)

        return inputs

    def unpack_forward_output(self, fwd_output):
        if isinstance(fwd_output, tuple):
            tensors = []
            for item in fwd_output:
                if isinstance(item, torch.Tensor):
                    tensors.append(item.flatten())
                elif hasattr(item, "last_hidden_state"):
                    tensors.append(item.last_hidden_state.flatten())
            if tensors:
                return torch.cat(tensors, dim=0)
        return fwd_output
