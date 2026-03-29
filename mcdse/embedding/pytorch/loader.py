# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MCDSE-2B model loader implementation for multimodal document embedding.
"""
import torch.nn.functional as F
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
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
from .src.utils import last_token_pool


class ModelVariant(StrEnum):
    """Available MCDSE model variants."""

    MCDSE_2B_V1 = "2B_v1"


class ModelLoader(ForgeModel):
    """MCDSE model loader for multimodal document embedding tasks."""

    _VARIANTS = {
        ModelVariant.MCDSE_2B_V1: ModelConfig(
            pretrained_model_name="marco/mcdse-2b-v1",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MCDSE_2B_V1

    # Sample query for testing
    sample_query = "What is shown in this image?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="MCDSE",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TEXT_SIM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
        )
        self.processor.tokenizer.padding_side = "left"
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the MCDSE model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The MCDSE model instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor()

        model_kwargs = {"output_hidden_states": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            pretrained_model_name,
            **model_kwargs,
        )
        model.padding_side = "left"
        model.eval()

        return model

    def _format_query(self, query):
        """Format a text query with a dummy image placeholder."""
        dummy_image = Image.new("RGB", (56, 56))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": dummy_image},
                    {"type": "text", "text": f"Query: {query}"},
                ],
            }
        ]
        return messages, [dummy_image]

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs using a text query with dummy image.

        Args:
            dtype_override: Optional torch.dtype override.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.processor is None:
            self._load_processor()

        messages, images = self._format_query("What is the document about?")

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=images,
            padding=True,
            return_tensors="pt",
        )

        if dtype_override is not None and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        return inputs

    def decode_output(self, outputs, inputs=None):
        """Decode model output to normalized embeddings.

        Args:
            outputs: Model output with hidden_states.
            inputs: Optional input tensors for attention mask.

        Returns:
            list: Normalized embedding vectors.
        """
        if inputs is None:
            inputs = self.load_inputs()

        # Extract last hidden state from hidden_states
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states[-1]
        elif hasattr(outputs, "last_hidden_state"):
            hidden_states = outputs.last_hidden_state
        else:
            hidden_states = outputs[0]

        # Pool embeddings using last token
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            embeddings = last_token_pool(hidden_states, attention_mask)
        else:
            embeddings = hidden_states[:, -1]

        # Normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.tolist()
