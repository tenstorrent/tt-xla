# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MOSS-TTS Local Transformer model loader implementation for text-to-speech tasks
"""
import torch
from transformers import AutoModel, AutoProcessor
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
    """Available MOSS-TTS model variants."""

    LOCAL_TRANSFORMER = "LocalTransformer"


class ModelLoader(ForgeModel):
    """MOSS-TTS Local Transformer model loader implementation for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.LOCAL_TRANSFORMER: ModelConfig(
            pretrained_model_name="OpenMOSS-Team/MOSS-TTS-Local-Transformer",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LOCAL_TRANSFORMER

    sample_text = "Hello, my dog is cute."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="MOSS-TTS",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        processor_kwargs = {}
        if dtype_override is not None:
            processor_kwargs["torch_dtype"] = dtype_override

        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            **processor_kwargs,
        )

        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()
        self.model = model

        return model

    def load_inputs(self, dtype_override=None):
        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        messages = self.processor.build_user_message(text=self.sample_text)
        inputs = self.processor(messages, return_tensors="pt")

        return inputs
