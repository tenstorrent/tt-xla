# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Donut model loader implementation for document understanding tasks.
"""
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
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


class ModelVariant(StrEnum):
    """Available Donut model variants for document understanding tasks."""

    CORD_V2 = "cord_v2"


class ModelLoader(ForgeModel):
    """Donut model loader implementation for document understanding tasks."""

    _VARIANTS = {
        ModelVariant.CORD_V2: ModelConfig(
            pretrained_model_name="naver-clova-ix/donut-base-finetuned-cord-v2",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CORD_V2

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="donut",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_DOC_OCR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        kwargs = {}
        if dtype_override is not None:
            kwargs["torch_dtype"] = dtype_override

        self.processor = DonutProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, **kwargs
        )

        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = VisionEncoderDecoderModel.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        filepath = hf_hub_download(
            repo_id="hf-internal-testing/fixtures_docvqa",
            filename="nougat_paper.png",
            repo_type="dataset",
        )
        image = Image.open(filepath)
        pixel_values = self.processor(image, return_tensors="pt").pixel_values

        if dtype_override is not None:
            pixel_values = pixel_values.to(dtype_override)

        if batch_size > 1:
            pixel_values = pixel_values.repeat_interleave(batch_size, dim=0)

        return {"pixel_values": pixel_values}

    def unpack_forward_output(self, fwd_output):
        if hasattr(fwd_output, "logits"):
            return fwd_output.logits
        return fwd_output
