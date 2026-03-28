# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
EvoQwen2.5-VL-Retriever model loader implementation for visual document retrieval tasks.
"""
import torch
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from typing import Optional
from PIL import Image

from ...base import ForgeModel
from ...config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from .src.model import Wrapper


class ModelVariant(StrEnum):
    """Available EvoQwen2.5-VL-Retriever model variants for visual document retrieval."""

    EVO_QWEN_2_5_VL_RETRIEVER_7B_V1 = "7B_v1"


class ModelLoader(ForgeModel):
    """EvoQwen2.5-VL-Retriever model loader for visual document retrieval tasks."""

    _VARIANTS = {
        ModelVariant.EVO_QWEN_2_5_VL_RETRIEVER_7B_V1: LLMModelConfig(
            pretrained_model_name="ApsaraStackMaaS/EvoQwen2.5-VL-Retriever-7B-v1",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.EVO_QWEN_2_5_VL_RETRIEVER_7B_V1

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="EvoQwen2.5-VL-Retriever",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TEXT_SIM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = ColQwen2_5_Processor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"low_cpu_mem_usage": True}

        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["torch_dtype"] = torch.float32
        model_kwargs |= kwargs

        model = ColQwen2_5.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()
        model = Wrapper(model)

        return model

    def load_inputs(self, dtype_override=None):
        if self.processor is None:
            self._load_processor()

        # Sample document image for retrieval
        image = Image.new("RGB", (224, 224), color=(128, 128, 128))

        inputs = self.processor.process_images([image])

        if dtype_override is not None:
            for key in inputs:
                if (
                    isinstance(inputs[key], torch.Tensor)
                    and inputs[key].is_floating_point()
                ):
                    inputs[key] = inputs[key].to(dtype_override)

        return inputs
