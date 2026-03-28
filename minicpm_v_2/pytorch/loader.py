# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MiniCPM-V-2 model loader implementation for multimodal visual question answering.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from transformers.integrations.tensor_parallel import ALL_PARALLEL_STYLES
from PIL import Image
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
from ...tools.utils import get_file

# Fix parallel styles issue for torch 2.7.0+ compatibility
if ALL_PARALLEL_STYLES is None:
    import transformers.modeling_utils as mu

    mu.ALL_PARALLEL_STYLES = ["rowwise", "colwise", "headwise"]

# Monkey patch Resampler for compatibility with torch 2.7.0
original_getattr = nn.Module.__getattr__


def patched_getattr(self, name):
    if name == "_initialize_weights" and self.__class__.__name__ == "Resampler":

        def _initialize_weights(module_self):
            if hasattr(module_self, "_init_weights"):
                module_self._init_weights(module_self)

        return _initialize_weights
    return original_getattr(self, name)


nn.Module.__getattr__ = patched_getattr


class ModelVariant(StrEnum):
    """Available MiniCPM-V-2 model variants."""

    DEFAULT = "Default"


class ModelLoader(ForgeModel):
    """MiniCPM-V-2 model loader for multimodal visual question answering."""

    _VARIANTS = {
        ModelVariant.DEFAULT: ModelConfig(
            pretrained_model_name="openbmb/MiniCPM-V-2",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEFAULT

    sample_text = "What is shown in this image?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="MiniCPM-V-2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VISUAL_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the MiniCPM-V-2 model instance."""
        model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(model_name, **model_kwargs)
        model.eval()

        if self.tokenizer is None:
            self._load_tokenizer()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for MiniCPM-V-2."""
        if self.tokenizer is None:
            self._load_tokenizer()

        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(image_file).convert("RGB")

        question = self.sample_text
        msgs = [{"role": "user", "content": question}]

        return {
            "image": image,
            "msgs": msgs,
            "tokenizer": self.tokenizer,
        }

    def decode_output(self, outputs):
        if isinstance(outputs, str):
            return outputs

        if isinstance(outputs, (tuple, list)):
            return outputs[0] if isinstance(outputs[0], str) else str(outputs[0])

        if self.tokenizer is None:
            self._load_tokenizer()

        if torch.is_tensor(outputs):
            if outputs.dtype in [torch.long, torch.int]:
                return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            logits = outputs
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
            return self.tokenizer.decode(next_token_id)

        if hasattr(outputs, "logits"):
            next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
            return self.tokenizer.decode(next_token_id)

        return str(outputs)
