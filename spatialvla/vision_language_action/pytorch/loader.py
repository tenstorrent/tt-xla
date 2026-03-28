# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SpatialVLA model loader implementation for vision-language-action prediction.
"""

import torch
from transformers import AutoModel, AutoProcessor
from PIL import Image
from typing import Optional

from ....tools.utils import get_file
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


class SpatialVLAInferenceWrapper(torch.nn.Module):
    """Wraps SpatialVLAForConditionalGeneration to use predict_action for inference.

    SpatialVLA's forward() is for training; predict_action() is the inference entry point.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, **kwargs):
        """Run inference via predict_action. Returns action predictions."""
        return self.model.predict_action(**kwargs)


class ModelVariant(StrEnum):
    """Available SpatialVLA model variants."""

    SPATIALVLA_4B_224_PT = "4B_224_pt"


class ModelLoader(ForgeModel):
    """SpatialVLA model loader implementation for action prediction tasks."""

    _VARIANTS = {
        ModelVariant.SPATIALVLA_4B_224_PT: ModelConfig(
            pretrained_model_name="IPEC-COMMUNITY/spatialvla-4b-224-pt",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SPATIALVLA_4B_224_PT

    sample_prompt = "pick the cup"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="SpatialVLA",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_ACTION_PREDICTION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()

        if self.processor is None:
            self._load_processor()

        return SpatialVLAInferenceWrapper(model)

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor()

        image_file = get_file(
            "https://rail.eecs.berkeley.edu/datasets/bridge_release/raw/"
            "bridge_data_v2/datacol2_toykitchen7/drawer_pnp/01/"
            "2023-04-19_09-18-15/raw/traj_group0/traj0/images0/im_30.jpg"
        )
        image = Image.open(image_file).convert("RGB")

        inputs = self.processor(
            text=self.sample_prompt,
            images=image,
            return_tensors="pt",
        )

        if dtype_override is not None:
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        if batch_size > 1:
            for key, value in inputs.items():
                if torch.is_tensor(value) and value.dim() > 0:
                    inputs[key] = value.repeat_interleave(batch_size, dim=0)

        return inputs
