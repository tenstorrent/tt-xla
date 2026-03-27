# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MiniCPM-Llama3-V-2_5 model loader implementation for multimodal visual question answering
"""

from typing import Optional

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)
from ...tools.utils import get_file


class ModelVariant(StrEnum):
    """Available MiniCPM-Llama3-V-2_5 model variants."""

    DEFAULT = "Default"


class ModelLoader(ForgeModel):
    """MiniCPM-Llama3-V-2_5 model loader for multimodal visual question answering."""

    _VARIANTS = {
        ModelVariant.DEFAULT: ModelConfig(
            pretrained_model_name="openbmb/MiniCPM-Llama3-V-2_5",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEFAULT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None
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

    def load_model(self, **kwargs):
        """Load and return the MiniCPM-Llama3-V-2_5 model instance."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.model = AutoModel.from_pretrained(
            pretrained_model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            **kwargs,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        self.model.eval()

        return self.model

    def load_inputs(self, **kwargs):
        """Load and return sample inputs for the MiniCPM-Llama3-V-2_5 model.

        Returns:
            dict: Input arguments containing image and messages for the model.
        """
        image_file = get_file(
            "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
        )
        image = Image.open(image_file).convert("RGB")

        question = "What is in the image?"
        msgs = [{"role": "user", "content": question}]

        return {
            "image": image,
            "msgs": msgs,
            "tokenizer": self.tokenizer,
            "sampling": False,
        }

    def decode_output(self, outputs, **kwargs):
        """Decode model outputs into human-readable text.

        Args:
            outputs: Raw model output (string from .chat() method)

        Returns:
            str: Decoded output text
        """
        if isinstance(outputs, str):
            return outputs

        if torch.is_tensor(outputs) and outputs.dtype in [torch.long, torch.int]:
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
        return self.tokenizer.decode(next_token_id)
