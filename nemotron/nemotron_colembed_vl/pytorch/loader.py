# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Nemotron ColEmbed VL model loader implementation for multimodal visual document retrieval.
"""

from PIL import Image
from transformers import AutoModel
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

    LLAMA_NEMOTRON_COLEMBED_VL_3B_V2 = "Llama_Nemotron_ColEmbed_VL_3B_V2"


class ModelLoader(ForgeModel):
    """Nemotron ColEmbed VL model loader for multimodal visual document retrieval."""

    _VARIANTS = {
        ModelVariant.LLAMA_NEMOTRON_COLEMBED_VL_3B_V2: ModelConfig(
            pretrained_model_name="nvidia/llama-nemotron-colembed-vl-3b-v2",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LLAMA_NEMOTRON_COLEMBED_VL_3B_V2

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

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

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

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
        image_file = get_file(
            "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
        )
        image = Image.open(image_file)

        return {
            "queries": ["Describe this image."],
            "images": [image],
        }

    def decode_output(self, outputs, inputs=None):
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        elif isinstance(outputs, (tuple, list)):
            return outputs[0]
        return outputs
