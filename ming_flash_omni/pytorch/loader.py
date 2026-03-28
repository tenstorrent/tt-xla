# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Ming-flash-omni 2.0 model loader implementation for multimodal conditional generation.
"""
import torch
from transformers import AutoModel, AutoProcessor
from typing import Optional

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
    """Available Ming-flash-omni model variants."""

    MING_FLASH_OMNI_2_0 = "flash-omni-2.0"


class ModelLoader(ForgeModel):
    """Ming-flash-omni 2.0 model loader implementation for multimodal conditional generation."""

    _VARIANTS = {
        ModelVariant.MING_FLASH_OMNI_2_0: LLMModelConfig(
            pretrained_model_name="inclusionAI/Ming-flash-omni-2.0",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MING_FLASH_OMNI_2_0

    messages = [
        {
            "role": "HUMAN",
            "content": [
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant."""
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant."""
        return ModelInfo(
            model="Ming-flash-omni 2.0",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load processor for the current variant."""
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Ming-flash-omni 2.0 model instance."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }

        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["torch_dtype"] = torch.float32
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.config.use_cache = False
        model.eval()
        model = Wrapper(model)

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Ming-flash-omni 2.0 model."""
        if self.processor is None:
            self._load_processor()

        text = self.processor.apply_chat_template(
            self.messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            return_tensors="pt",
        )

        return inputs
