# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MiniCPM-o-2_6 model loader implementation for multimodal inference
"""

from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_utils import ALL_PARALLEL_STYLES
from PIL import Image
import requests
from io import BytesIO

from tt_forge_models.config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from tt_forge_models.base import ForgeModel


# Fix parallel styles issue for torch 2.7.0+ compatibility - works fine in torch 2.3.1
if ALL_PARALLEL_STYLES is None:
    import transformers.modeling_utils as mu

    mu.ALL_PARALLEL_STYLES = ["rowwise", "colwise", "headwise"]

# Monkey patch Resampler for compatibility - Fixes: Resampler doesn't have _initialize_weights method in torch 2.7.0
original_getattr = nn.Module.__getattr__


def patched_getattr(self, name):
    if name == "_initialize_weights" and self.__class__.__name__ == "Resampler":

        def _initialize_weights(module_self):
            if hasattr(module_self, "_init_weights"):
                module_self._init_weights(module_self)

        return _initialize_weights
    return original_getattr(self, name)


nn.Module.__getattr__ = patched_getattr


@dataclass
class MiniCPMConfig(ModelConfig):
    """Configuration specific to MiniCPM-o-2_6 models"""

    pretrained_model_name: str = "openbmb/MiniCPM-o-2_6"
    max_new_tokens: int = 256
    temperature: float = 0.5
    sample_image_url: Optional[str] = "https://picsum.photos/512/512"


class ModelVariant(StrEnum):
    """Available MiniCPM-o-2_6 model variants."""

    DEFAULT = "default"


# Model variants configuration
_VARIANTS = {
    ModelVariant.DEFAULT: MiniCPMConfig(
        pretrained_model_name="openbmb/MiniCPM-o-2_6",
        max_new_tokens=256,
        temperature=0.5,
    ),
}


class ModelLoader(ForgeModel):
    """MiniCPM-o-2_6 model loader implementation for multimodal inference."""

    _VARIANTS = _VARIANTS
    DEFAULT_VARIANT = ModelVariant.DEFAULT

    def __init__(self, variant=None):
        """Initialize MiniCPM model loader."""
        super().__init__(variant)
        self.model = None
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[StrEnum] = None) -> ModelInfo:
        """Get model information for MiniCPM-o-2_6."""
        return ModelInfo(
            name="MiniCPM-o-2_6",
            group=ModelGroup.RED,
            task=ModelTask.IMAGE_TO_TEXT,
            source=ModelSource.HUGGINGFACE,
            framework=Framework.PYTORCH,
        )

    def load_model(self, **kwargs):
        """Load and return the MiniCPM-o-2_6 model instance.

        Args:
            **kwargs: Additional model-specific arguments.

        Returns:
            torch.nn.Module: The loaded model instance
        """
        config = self._variant_config

        # Load model and tokenizer
        self.model = AutoModel.from_pretrained(
            config.pretrained_model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            **kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.pretrained_model_name, trust_remote_code=True
        )

        # Set model to eval mode
        self.model.eval()

        return self.model

    def load_inputs(self, **kwargs) -> Dict[str, Any]:
        """Load and return sample inputs for the model.

        Args:
            **kwargs: Additional input-specific arguments.

        Returns:
            Dict: Sample inputs containing text and image
        """
        config = self._variant_config

        # Load sample image
        if config.sample_image_url:
            try:
                response = requests.get(config.sample_image_url)
                image = Image.open(BytesIO(response.content))
            except Exception as e:
                image = Image.new("RGB", (512, 512), color="white")
        else:
            image = Image.new("RGB", (512, 512), color="white")

        # Sample text input
        question = "Describe this image in detail."

        # Return inputs in the format expected by predict method
        return {"question": question, "image": image}

    def predict(
        self, inputs: Optional[Dict[str, Any]] = None, **kwargs
    ) -> Dict[str, Any]:
        """Run inference on the loaded model.

        Args:
            inputs: Optional dictionary containing 'question' and 'image' keys.
                   If None, uses sample inputs from load_inputs().
            **kwargs: Additional inference arguments.

        Returns:
            Dict containing inference results
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load_model() first.")

        # Get inputs
        if inputs is None:
            inputs = self.load_inputs()

        question = inputs.get("question", "Describe this image.")
        image = inputs.get("image")

        if image is None:
            raise ValueError("Image input is required for MiniCPM inference")

        # Prepare input messages
        msgs = [{"role": "user", "content": [question, image]}]

        # Run inference
        with torch.no_grad():
            config = self._variant_config
            result = self.model.chat(
                msgs=msgs,
                tokenizer=self.tokenizer,
                sampling=True,
                temperature=config.temperature,
                max_new_tokens=config.max_new_tokens,
                **kwargs
            )

        return {
            "question": question,
            "response": result,
            "model": "MiniCPM-o-2_6",
            "temperature": config.temperature,
            "max_new_tokens": config.max_new_tokens,
        }

    @classmethod
    def decode_output(cls, outputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Decode model outputs into human-readable format.

        Args:
            outputs: Raw model outputs from predict()
            **kwargs: Additional decoding arguments

        Returns:
            Dict: Decoded outputs with human-readable results
        """
        decoded = {
            "question": outputs.get("question", ""),
            "answer": outputs.get("response", ""),
            "model_info": {
                "name": outputs.get("model", "MiniCPM-o-2_6"),
                "temperature": outputs.get("temperature", 0.5),
                "max_tokens": outputs.get("max_new_tokens", 256),
            },
            "confidence": None,  # MiniCPM doesn't provide confidence scores
            "processing_time": None,  # Could be added with timing
        }
        return decoded

    def post_processing(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Post-process model outputs.

        Args:
            outputs: Raw model outputs

        Returns:
            Dict: Post-processed outputs
        """
        if "response" in outputs:
            outputs["response"] = outputs["response"].strip()

        # Add metadata
        outputs["post_processed"] = True
        outputs["timestamp"] = torch.tensor([]).new_zeros(0).device

        return outputs

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)
