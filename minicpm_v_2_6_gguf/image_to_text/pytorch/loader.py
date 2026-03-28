# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MiniCPM-V-2_6 GGUF model loader implementation for image to text.
"""
from transformers import AutoModel, AutoTokenizer
from typing import Optional
from PIL import Image

from ....base import ForgeModel
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....tools.utils import get_file


class ModelVariant(StrEnum):
    """Available MiniCPM-V-2_6 GGUF model variants for image to text."""

    MINICPM_V_2_6_GGUF = "v2_6_gguf"


class ModelLoader(ForgeModel):
    """MiniCPM-V-2_6 GGUF model loader implementation for image to text tasks."""

    _VARIANTS = {
        ModelVariant.MINICPM_V_2_6_GGUF: LLMModelConfig(
            pretrained_model_name="openbmb/MiniCPM-V-2_6-gguf",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MINICPM_V_2_6_GGUF

    _GGUF_FILES = {
        ModelVariant.MINICPM_V_2_6_GGUF: "ggml-model-Q4_K_M.gguf",
    }

    _BASE_MODELS = {
        ModelVariant.MINICPM_V_2_6_GGUF: "openbmb/MiniCPM-V-2_6",
    }

    sample_image = "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="MiniCPM-V-2_6 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_IMAGE_TO_TEXT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @property
    def _gguf_file(self):
        """Get the GGUF filename for the current variant."""
        return self._GGUF_FILES[self._variant]

    @property
    def _base_model(self):
        """Get the base model name for tokenizer loading."""
        return self._BASE_MODELS[self._variant]

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._base_model, trust_remote_code=True
        )

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self._gguf_file

        model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs).eval()

        self.config = model.config
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self._base_model, trust_remote_code=True
            )

        image_file = get_file(self.sample_image)
        image = Image.open(image_file).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]

        return {"messages": messages, "image": image}

    def load_config(self):
        self.config = AutoModel.from_pretrained(
            self._variant_config.pretrained_model_name,
            gguf_file=self._gguf_file,
            trust_remote_code=True,
        ).config
        return self.config
