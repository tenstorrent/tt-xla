# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ColSmol model loader implementation for visual document retrieval.
"""

import torch
from PIL import Image
from colpali_engine.models import ColIdefics3, ColIdefics3Processor
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
from ....tools.utils import cast_input_to_type, get_file


class ModelVariant(StrEnum):
    """Available ColSmol model variants."""

    COL_SMOL_500M = "colSmol_500M"


class ModelLoader(ForgeModel):
    """ColSmol model loader for visual document retrieval."""

    _VARIANTS = {
        ModelVariant.COL_SMOL_500M: ModelConfig(
            pretrained_model_name="vidore/colSmol-500M",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.COL_SMOL_500M

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ColSmol model loader."""
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ColSmol",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TEXT_SIM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = ColIdefics3Processor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the ColSmol model instance."""
        model_name = self._variant_config.pretrained_model_name
        model = ColIdefics3.from_pretrained(
            str(model_name),
            torch_dtype=dtype_override or torch.bfloat16,
            attn_implementation="eager",
            **kwargs,
        )
        model.eval()

        if self.processor is None:
            self._load_processor()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for ColSmol image embedding."""
        if self.processor is None:
            self._load_processor()

        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(image_file)
        images = [image]

        batch_images = self.processor.process_images(images)

        if dtype_override:
            batch_images = {
                k: cast_input_to_type(v, dtype_override)
                for k, v in batch_images.items()
            }

        return batch_images
