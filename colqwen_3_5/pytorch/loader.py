# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ColQwen 3.5 model loader implementation for visual document retrieval.
"""
import torch
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
    """Available ColQwen 3.5 model variants."""

    COLQWEN_3_5_4_5B_V3 = "4.5B_v3"


class ModelLoader(ForgeModel):
    """ColQwen 3.5 model loader for visual document retrieval tasks."""

    _VARIANTS = {
        ModelVariant.COLQWEN_3_5_4_5B_V3: ModelConfig(
            pretrained_model_name="athrael-soju/colqwen3.5-4.5B-v3",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.COLQWEN_3_5_4_5B_V3

    sample_image = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rabbit.png"
    sample_query = "What is shown in this image?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="ColQwen 3.5",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TEXT_SIM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        from colpali_engine.models import ColQwen3_5Processor

        self.processor = ColQwen3_5Processor.from_pretrained(
            self._variant_config.pretrained_model_name,
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        from colpali_engine.models import ColQwen3_5

        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16
        model_kwargs |= kwargs

        model = ColQwen3_5.from_pretrained(
            pretrained_model_name,
            **model_kwargs,
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        from PIL import Image
        import requests

        if self.processor is None:
            self._load_processor()

        # Process a sample image as a document
        image = Image.open(requests.get(self.sample_image, stream=True).raw)
        batch_images = self.processor.process_images([image])

        if dtype_override is not None:
            for key in batch_images:
                if batch_images[key].is_floating_point():
                    batch_images[key] = batch_images[key].to(dtype_override)

        return batch_images

    def decode_output(self, outputs, inputs=None):
        if isinstance(outputs, torch.Tensor):
            return outputs.tolist()
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state.tolist()
        return outputs[0].tolist()
