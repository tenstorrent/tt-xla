# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Phi-4 Reasoning Vision model loader implementation for multimodal visual question answering
"""
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
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


class ModelVariant(StrEnum):
    """Available Phi-4 Reasoning Vision model variants."""

    PHI_4_REASONING_VISION_15B = "Reasoning_Vision_15B"


class ModelLoader(ForgeModel):
    """Phi-4 Reasoning Vision model loader implementation for multimodal visual question answering tasks."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.PHI_4_REASONING_VISION_15B: ModelConfig(
            pretrained_model_name="microsoft/Phi-4-reasoning-vision-15B",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.PHI_4_REASONING_VISION_15B

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
        """Get model information for dashboard and metrics reporting."""
        return ModelInfo(
            model="Phi-4",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VISUAL_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load processor and tokenizer for the current variant."""
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        self.tokenizer = self.processor.tokenizer
        return self.processor

    def load_model(self, **kwargs):
        """Load and return the Phi-4 Reasoning Vision model instance."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor()

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name,
            trust_remote_code=True,
            _attn_implementation="eager",
            **kwargs,
        )

        model.eval()
        self.model = model
        return model

    def load_inputs(self, batch_size=1):
        """Load and return sample inputs for the Phi-4 Reasoning Vision model."""
        if self.processor is None:
            self._load_processor()

        # Load image from URL
        image_file = get_file(
            "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
        )
        image = Image.open(image_file)

        # Set up messages with image placeholder
        self.messages = [
            {"role": "user", "content": "<|image_1|>\nWhat is this image about?"},
        ]

        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            self.messages, tokenize=False, add_generation_prompt=True
        )

        # Process inputs with image and text
        inputs = self.processor(prompt, [image], return_tensors="pt").to(
            self.model.device
        )

        # Add batch dimension
        if batch_size > 1:
            for key in inputs:
                if torch.is_tensor(inputs[key]):
                    inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        # Return arguments dict
        arguments = {
            **inputs,
            "use_cache": False,
        }

        return arguments

    def decode_output(self, outputs, input_length=None):
        """Helper method to decode model outputs into human-readable text."""
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
