# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
VideoChatFlash-Qwen model loader implementation for multimodal video/image conditional generation.
"""

from typing import Optional

from transformers import AutoModel, AutoTokenizer

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
from ...tools.utils import cast_input_to_type


class ModelVariant(StrEnum):
    """Available VideoChatFlash-Qwen model variants."""

    TINY = "tiny"


class ModelLoader(ForgeModel):
    """VideoChatFlash-Qwen model loader for multimodal video/image conditional generation."""

    _VARIANTS = {
        ModelVariant.TINY: ModelConfig(
            pretrained_model_name="optimum-intel-internal-testing/tiny-videochat-flash-qwen",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TINY

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize VideoChatFlash-Qwen model loader."""
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="VideoChatFlash-Qwen",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        model_name = self._variant_config.pretrained_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the VideoChatFlash-Qwen model instance."""
        model_name = self._variant_config.pretrained_model_name

        model_kwargs = {
            "trust_remote_code": True,
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(str(model_name), **model_kwargs)
        model.eval()

        if self.tokenizer is None:
            self._load_tokenizer()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for VideoChatFlash-Qwen."""
        if self.tokenizer is None:
            self._load_tokenizer()

        text_prompt = "<video>\nWhat is shown in this video?"

        inputs = self.tokenizer(text_prompt, return_tensors="pt")

        if dtype_override:
            inputs = {
                k: cast_input_to_type(v, dtype_override) for k, v in inputs.items()
            }

        return dict(inputs)
