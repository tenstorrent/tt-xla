# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Llama 4 Scout model loader implementation for multimodal image-text-to-text generation
"""
import torch
from transformers import AutoProcessor, Llama4ForConditionalGeneration
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
    """Available Llama 4 Scout model variants."""

    LLAMA_4_SCOUT_17B_16E_INSTRUCT = "4_Scout_17B_16E_Instruct"


class ModelLoader(ForgeModel):
    """Llama 4 Scout model loader implementation for multimodal image-text-to-text tasks."""

    _VARIANTS = {
        ModelVariant.LLAMA_4_SCOUT_17B_16E_INSTRUCT: ModelConfig(
            pretrained_model_name="RedHatAI/Llama-4-Scout-17B-16E-Instruct",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LLAMA_4_SCOUT_17B_16E_INSTRUCT

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.processor = None
        self.tokenizer = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Llama",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load processor and tokenizer for the current variant.

        Returns:
            The loaded processor instance
        """
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        self.tokenizer = self.processor.tokenizer
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Llama 4 Scout model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Llama 4 Scout model instance for multimodal tasks.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor()

        model_kwargs = {"trust_remote_code": True, "_attn_implementation": "eager"}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = Llama4ForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        model.eval()
        self.model = model

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Llama 4 Scout model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input arguments that can be fed to the model.
        """
        if self.processor is None:
            self._load_processor()

        url = "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
        image_file = get_file(url)
        image = Image.open(image_file)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": url},
                    {"type": "text", "text": "What is this image about?"},
                ],
            },
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        device = self.model.device if self.model is not None else "cpu"
        inputs = self.processor(images=[image], text=prompt, return_tensors="pt").to(
            device
        )

        if dtype_override is not None:
            for key in inputs:
                if torch.is_tensor(inputs[key]):
                    inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        if batch_size > 1:
            for key in inputs:
                if torch.is_tensor(inputs[key]):
                    inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        arguments = {
            **inputs,
            "use_cache": False,
            "max_new_tokens": 20,
            "do_sample": False,
            "pad_token_id": (
                self.tokenizer.eos_token_id
                if self.tokenizer.eos_token_id is not None
                else self.tokenizer.pad_token_id
            ),
        }

        return arguments

    def decode_output(self, outputs, input_length=None):
        """Decode model outputs into human-readable text.

        Args:
            outputs: Model output from a forward pass or generated token IDs
            input_length: Optional length of input tokens to slice from output

        Returns:
            str: Decoded output text
        """
        if self.processor is None:
            self._load_processor()

        if torch.is_tensor(outputs) and outputs.dtype in [torch.long, torch.int]:
            if input_length is not None:
                outputs = outputs[:, input_length:]
            decoded_output = self.processor.batch_decode(
                outputs, skip_special_tokens=True
            )[0]
        else:
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
            decoded_output = self.tokenizer.decode(next_token_id)

        return decoded_output
