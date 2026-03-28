# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Nemotron ColEmbed VL model loader implementation for multimodal visual document retrieval.
"""

from PIL import Image
from transformers import AutoModel, AutoProcessor
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


class ModelVariant(StrEnum):
    """Available Nemotron ColEmbed VL model variants."""

    NEMOTRON_COLEMBED_VL_4B_V2 = "ColEmbed_VL_4B_V2"


class ModelLoader(ForgeModel):
    """Nemotron ColEmbed VL model loader for multimodal visual document retrieval."""

    _VARIANTS = {
        ModelVariant.NEMOTRON_COLEMBED_VL_4B_V2: ModelConfig(
            pretrained_model_name="nvidia/nemotron-colembed-vl-4b-v2",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NEMOTRON_COLEMBED_VL_4B_V2

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Nemotron ColEmbed VL",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TEXT_SIM,
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

        if self.processor is None:
            self._load_processor()

        model_kwargs = {"trust_remote_code": True, "attn_implementation": "eager"}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(
            pretrained_model_name,
            **model_kwargs,
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        if self.processor is None:
            self._load_processor()

        image_file = get_file(
            "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
        )
        image = Image.open(image_file)

        inputs = self.processor.process_images([[image]])

        return inputs

    def decode_output(self, outputs, inputs=None):
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        elif isinstance(outputs, (tuple, list)):
            return outputs[0]
        return outputs
