# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ColQwen2 model loader implementation for visual document retrieval.
"""

import torch
from colpali_engine.models import ColQwen2, ColQwen2Processor
from typing import Optional
from PIL import Image

from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel
from ...tools.utils import get_file


class ModelVariant(StrEnum):
    """Available ColQwen2 model variants for visual document retrieval."""

    COLQWEN2_V1_0 = "vidore/colqwen2-v1.0"


class ModelLoader(ForgeModel):
    """ColQwen2 model loader implementation for visual document retrieval.

    ColQwen2 extends Qwen2-VL with a ColBERT late-interaction mechanism to produce
    multi-vector embeddings for both images and text queries, enabling efficient
    visual document retrieval.
    """

    _VARIANTS = {
        ModelVariant.COLQWEN2_V1_0: ModelConfig(
            pretrained_model_name="vidore/colqwen2-v1.0",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.COLQWEN2_V1_0

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="ColQwen2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TEXT_SIM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = ColQwen2Processor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = ColQwen2.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor()

        image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
        image_path = get_file(image_url)
        image = Image.open(image_path).convert("RGB")

        batch_images = self.processor.process_images([image])

        for key in batch_images:
            if torch.is_tensor(batch_images[key]):
                batch_images[key] = batch_images[key].repeat_interleave(
                    batch_size, dim=0
                )
                if dtype_override is not None and batch_images[key].is_floating_point():
                    batch_images[key] = batch_images[key].to(dtype_override)

        return batch_images

    def decode_output(self, outputs, inputs=None):
        if isinstance(outputs, (tuple, list)):
            return outputs[0]
        if isinstance(outputs, torch.Tensor):
            return outputs
        return outputs

    def unpack_forward_output(self, fwd_output):
        if isinstance(fwd_output, torch.Tensor):
            return fwd_output.flatten()
        if isinstance(fwd_output, (tuple, list)):
            tensors = [t.flatten() for t in fwd_output if isinstance(t, torch.Tensor)]
            if tensors:
                return torch.cat(tensors, dim=0)
        return fwd_output
