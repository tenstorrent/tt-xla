# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ColPali model loader implementation for visual document retrieval.
"""
import torch
from colpali_engine.models import ColPali, ColPaliProcessor
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


class ModelVariant(StrEnum):
    """Available ColPali model variants."""

    V1_3 = "v1.3"


class ModelLoader(ForgeModel):
    """ColPali model loader for visual document retrieval."""

    _VARIANTS = {
        ModelVariant.V1_3: ModelConfig(
            pretrained_model_name="vidore/colpali-v1.3",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V1_3

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ColPali model loader."""
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ColPali",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TEXT_SIM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = ColPaliProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the ColPali model instance."""
        model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16
        model_kwargs |= kwargs

        model = ColPali.from_pretrained(model_name, **model_kwargs)
        model.eval()

        if self.processor is None:
            self._load_processor()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for ColPali."""
        if self.processor is None:
            self._load_processor()

        images = [Image.new("RGB", (448, 448), color="white")]
        batch_images = self.processor.process_images(images)

        for key in batch_images:
            if torch.is_tensor(batch_images[key]):
                batch_images[key] = batch_images[key].repeat_interleave(
                    batch_size, dim=0
                )

        if dtype_override is not None:
            for key in batch_images:
                if (
                    torch.is_tensor(batch_images[key])
                    and batch_images[key].is_floating_point()
                ):
                    batch_images[key] = batch_images[key].to(dtype_override)

        return batch_images

    def post_process(self, outputs):
        """Post-process ColPali outputs to compute document-query similarity scores."""
        if self.processor is None:
            self._load_processor()

        queries = ["What is the document about?"]
        batch_queries = self.processor.process_queries(queries)

        scores = self.processor.score_multi_vector(outputs, batch_queries)
        print(f"Document-query similarity scores: {scores}")
