# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Llama 3.2 Vision model loader implementation for multimodal visual question answering
"""
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
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
    """Available Llama 3.2 Vision model variants."""

    LLAMA_3_2_11B_VISION = "llama_3_2_11b_vision"
    LLAMA_3_2_11B_VISION_INSTRUCT = "llama_3_2_11b_vision_instruct"


class ModelLoader(ForgeModel):
    """Llama 3.2 Vision model loader implementation for multimodal visual question answering tasks."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.LLAMA_3_2_11B_VISION: ModelConfig(
            pretrained_model_name="meta-llama/Llama-3.2-11B-Vision",
        ),
        ModelVariant.LLAMA_3_2_11B_VISION_INSTRUCT: ModelConfig(
            pretrained_model_name="meta-llama/Llama-3.2-11B-Vision-Instruct",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.LLAMA_3_2_11B_VISION_INSTRUCT

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
        self.messages = None

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
            model="llama-3.2-vision",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.MM_VISUAL_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load processor and tokenizer for the current variant.

        Returns:
            The loaded processor instance
        """
        # Initialize processor
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        self.tokenizer = self.processor.tokenizer
        return self.processor

    def load_model(self, dtype_override=None):
        """Load and return the Llama 3.2 Vision model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype.

        Returns:
            torch.nn.Module: The Llama 3.2 Vision model instance for multimodal tasks.
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Ensure processor is loaded
        if self.processor is None:
            self._load_processor()

        # Load pre-trained model from HuggingFace
        model_kwargs = {"trust_remote_code": True, "_attn_implementation": "eager"}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        model.eval()
        self.model = model

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Llama 3.2 Vision model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input arguments that can be fed to the model.
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Ensure processor is initialized
        if self.processor is None:
            self._load_processor()

        # Load image from URL
        image_file = get_file(
            "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
        )
        image = Image.open(image_file)

        if pretrained_model_name == "meta-llama/Llama-3.2-11B-Vision":
            # Base model: Use raw prompt
            prompt = "<|image|><|begin_of_text|> What is this image about?"

        elif pretrained_model_name == "meta-llama/Llama-3.2-11B-Vision-Instruct":
            # Set up messages
            self.messages = [
                {"role": "user", "content": "<|image_1|>\nWhat is this image about?"},
            ]

            # Apply chat template
            prompt = self.tokenizer.apply_chat_template(
                self.messages, tokenize=False, add_generation_prompt=True
            )

        # Process inputs
        device = self.model.device if self.model is not None else "cpu"
        # Use keyword arguments: processor expects images first, then text
        inputs = self.processor(images=[image], text=prompt, return_tensors="pt").to(
            device
        )

        # Convert dtype if specified (only for floating point tensors)
        # Integer tensors like input_ids, attention_mask should remain as integers
        if dtype_override is not None:
            for key in inputs:
                if torch.is_tensor(inputs[key]):
                    inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        # Add batch dimension
        if batch_size > 1:
            for key in inputs:
                if torch.is_tensor(inputs[key]):
                    inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        # Return arguments dict
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
        """Helper method to decode model outputs into human-readable text.

        Args:
            outputs: Model output from a forward pass or generated token IDs
            input_length: Optional length of input tokens to slice from output

        Returns:
            str: Decoded output text
        """
        if self.processor is None:
            self._load_processor()

        # Check if outputs are token IDs (from generation) or logits
        if torch.is_tensor(outputs) and outputs.dtype in [torch.long, torch.int]:
            # Token IDs
            if input_length is not None:
                outputs = outputs[:, input_length:]
            decoded_output = self.processor.batch_decode(
                outputs, skip_special_tokens=True
            )[0]
        else:
            # Logits - get next token
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
            decoded_output = self.tokenizer.decode(next_token_id)

        return decoded_output
