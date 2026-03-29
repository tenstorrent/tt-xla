# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MiniCPM-Llama3-V-2.5 model loader implementation for multimodal visual question answering.
"""

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
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


class ModelVariant(StrEnum):
    """Available MiniCPM-Llama3-V-2.5 model variants."""

    MINICPM_LLAMA3_V_2_5 = "MiniCPM-Llama3-V-2.5"


class ModelLoader(ForgeModel):
    """MiniCPM-Llama3-V-2.5 model loader for multimodal visual question answering."""

    _VARIANTS = {
        ModelVariant.MINICPM_LLAMA3_V_2_5: ModelConfig(
            pretrained_model_name="openbmb/MiniCPM-Llama3-V-2_5",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MINICPM_LLAMA3_V_2_5

    sample_text = "What is in the image?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize MiniCPM-Llama3-V-2.5 model loader."""
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="MiniCPM-Llama3-V-2.5",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VISUAL_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the MiniCPM-Llama3-V-2.5 model instance."""
        model_name = self._variant_config.pretrained_model_name
        model = AutoModel.from_pretrained(
            str(model_name),
            trust_remote_code=True,
            torch_dtype=torch.float16,
            **kwargs,
        )
        model.eval()

        if dtype_override:
            model = model.to(dtype_override)

        if self.tokenizer is None:
            self._load_tokenizer()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for MiniCPM-Llama3-V-2.5."""
        if self.tokenizer is None:
            self._load_tokenizer()

        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(image_file).convert("RGB")

        msgs = [{"role": "user", "content": self.sample_text}]

        return {"image": image, "msgs": msgs}

    def unpack_forward_output(self, fwd_output):
        if hasattr(fwd_output, "logits"):
            return fwd_output.logits
        if isinstance(fwd_output, tuple):
            return fwd_output[0]
        return fwd_output

    def decode_output(self, **kwargs):
        outputs = kwargs.get("outputs")
        if outputs is None:
            return None

        if self.tokenizer is None:
            self._load_tokenizer()

        if isinstance(outputs, str):
            return outputs

        if isinstance(outputs, torch.Tensor):
            if outputs.dtype in (torch.long, torch.int32, torch.int64):
                token_ids = outputs
            else:
                token_ids = outputs.argmax(dim=-1)
        else:
            token_ids = outputs

        return self.tokenizer.decode(token_ids[0], skip_special_tokens=True)
