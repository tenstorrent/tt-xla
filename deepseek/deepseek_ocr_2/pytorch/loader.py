# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek OCR-2 model loader implementation for document OCR tasks.
"""
from transformers import AutoTokenizer, AutoModel
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
from ....tools.utils import get_file

# Reuse preprocessing utilities from DeepSeek-OCR
from ...deepseek_ocr.pytorch.src.model_utils import preprocess


class ModelVariant(StrEnum):
    """Available DeepSeek OCR-2 model variants."""

    DEEPSEEK_OCR_2 = "Ocr2"
    DEEPSEEK_OCR_2_UNSLOTH = "Ocr2-Unsloth"


class ModelLoader(ForgeModel):
    """DeepSeek OCR-2 model loader implementation for document OCR tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.DEEPSEEK_OCR_2: ModelConfig(
            pretrained_model_name="deepseek-ai/DeepSeek-OCR-2",
        ),
        ModelVariant.DEEPSEEK_OCR_2_UNSLOTH: ModelConfig(
            pretrained_model_name="unsloth/DeepSeek-OCR-2",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.DEEPSEEK_OCR_2

    # Shared configuration parameters
    sample_prompt = "<image>\n<|grounding|>Convert the document to markdown. "

    def __init__(
        self,
        variant: Optional[ModelVariant] = None,
    ):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="DeepSeek",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_DOC_OCR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        pretrained_model_name = self._variant_config.pretrained_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model = AutoModel.from_pretrained(
            pretrained_model_name,
            trust_remote_code=True,
            use_safetensors=True,
            **kwargs,
        )

        model.config.return_dict = False
        model.config.use_cache = False

        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        image_file = get_file("test_images/doc.png")

        inputs = preprocess(
            tokenizer=self.tokenizer,
            prompt=self.sample_prompt,
            image_file=image_file,
            base_size=1024,
            image_size=640,
            crop_mode=True,
        )

        if dtype_override is not None:
            for idx, (images_crop, images_ori) in enumerate(inputs["images"]):
                inputs["images"][idx] = (
                    images_crop.to(dtype_override),
                    images_ori.to(dtype_override),
                )

        return inputs
