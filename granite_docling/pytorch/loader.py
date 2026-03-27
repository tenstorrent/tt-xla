# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Granite Docling model loader implementation for document understanding tasks.
"""
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
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
    """Available Granite Docling model variants for document understanding tasks."""

    GRANITE_DOCLING_258M = "granite_docling_258M"


class ModelLoader(ForgeModel):
    """Granite Docling model loader implementation for document understanding tasks."""

    _VARIANTS = {
        ModelVariant.GRANITE_DOCLING_258M: ModelConfig(
            pretrained_model_name="ibm-granite/granite-docling-258M",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GRANITE_DOCLING_258M

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Convert this page to docling."},
            ],
        },
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="granite_docling",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_DOC_OCR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForVision2Seq.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor()

        filepath = hf_hub_download(
            repo_id="hf-internal-testing/fixtures_docvqa",
            filename="nougat_paper.png",
            repo_type="dataset",
        )
        image = Image.open(filepath)

        prompt = self.processor.apply_chat_template(
            self.messages, add_generation_prompt=True
        )
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt")

        if dtype_override is not None:
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        if batch_size > 1:
            for key in inputs:
                if torch.is_tensor(inputs[key]):
                    inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def unpack_forward_output(self, fwd_output):
        if hasattr(fwd_output, "logits"):
            return fwd_output.logits
        return fwd_output
