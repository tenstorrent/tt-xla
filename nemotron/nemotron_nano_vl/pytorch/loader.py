# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Nemotron Nano VL model loader implementation for multimodal visual question answering
"""
import torch
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor
from PIL import Image
from typing import Optional
from ....tools.utils import get_file, cast_input_to_type
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
    """Available Nemotron Nano VL model variants."""

    NEMOTRON_NANO_VL_8B_V1 = "Nano_VL_8B_V1"


class ModelLoader(ForgeModel):
    """Nemotron Nano VL model loader implementation for multimodal visual question answering tasks."""

    _VARIANTS = {
        ModelVariant.NEMOTRON_NANO_VL_8B_V1: ModelConfig(
            pretrained_model_name="nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NEMOTRON_NANO_VL_8B_V1

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.image_processor = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Nemotron",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VISUAL_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        return self.tokenizer

    def _load_image_processor(self):
        self.image_processor = AutoImageProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            device="cpu",
        )
        return self.image_processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()
        if self.image_processor is None:
            self._load_image_processor()

        model_kwargs = {"trust_remote_code": True, "_attn_implementation": "eager"}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()
        self.model = model

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer()
        if self.image_processor is None:
            self._load_image_processor()

        image_file = get_file(
            "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
        )
        image = Image.open(image_file)

        question = "<image>\nWhat is shown in this image?"
        image_features = self.image_processor(image)

        generation_config = {
            "max_new_tokens": 20,
            "do_sample": False,
        }

        return {
            "tokenizer": self.tokenizer,
            "question": question,
            "generation_config": generation_config,
            **image_features,
        }

    def decode_output(self, outputs, input_length=None):
        if isinstance(outputs, str):
            return outputs

        if self.tokenizer is None:
            self._load_tokenizer()

        if torch.is_tensor(outputs) and outputs.dtype in [torch.long, torch.int]:
            if input_length is not None:
                outputs = outputs[:, input_length:]
            return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        else:
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
            return self.tokenizer.decode(next_token_id)
