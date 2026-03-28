# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ColQwen2.5 model loader implementation for visual document retrieval.
"""
import torch
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
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
    """Available ColQwen2.5 model variants."""

    V0_2 = "v0.2"


class ModelLoader(ForgeModel):
    """ColQwen2.5 model loader for visual document retrieval."""

    _VARIANTS = {
        ModelVariant.V0_2: ModelConfig(
            pretrained_model_name="vidore/colqwen2.5-v0.2",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V0_2

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ColQwen2.5 model loader."""
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ColQwen2.5",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TEXT_SIM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = ColQwen2_5_Processor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the ColQwen2.5 model instance."""
        model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16
        model_kwargs |= kwargs

        model = ColQwen2_5.from_pretrained(model_name, **model_kwargs)
        model.eval()

        if self.processor is None:
            self._load_processor()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for ColQwen2.5."""
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
        """Post-process ColQwen2.5 outputs to compute document-query similarity scores."""
        if self.processor is None:
            self._load_processor()

        queries = ["What is the document about?"]
        batch_queries = self.processor.process_queries(queries)

        scores = self.processor.score_multi_vector(outputs, batch_queries)
        print(f"Document-query similarity scores: {scores}")
