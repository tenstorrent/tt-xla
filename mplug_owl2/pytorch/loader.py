# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MPLUG-Owl2 model loader for multimodal text generation (image + text → text).
"""
from typing import Optional, Dict, Any

import torch
from PIL import Image
from transformers import AutoTokenizer
from transformers.models.clip.image_processing_clip import CLIPImageProcessor

from ...tools.utils import get_file
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

from .src.modeling_mplug_owl2 import (
    MPLUGOwl2LlamaForCausalLM,
)
from .src.conversation import conv_templates
from .src.model_utils import (
    process_images,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
    tokenizer_image_token,
    load_weights,
)
from .src.configuration_mplug_owl2 import MPLUGOwl2Config


class ModelVariant(StrEnum):
    """Available MPLUG-Owl2 model variants."""

    LLAMA2_7B = "llama2_7b"


class ModelLoader(ForgeModel):
    """MPLUG-Owl2 model loader for multimodal text generation.

    Provides utilities to load the pretrained model and construct sample inputs.
    """

    _VARIANTS = {
        ModelVariant.LLAMA2_7B: ModelConfig(
            pretrained_model_name="MAGAer13/mplug-owl2-llama2-7b",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LLAMA2_7B

    # Default text prompt and image used for sample inputs
    default_query = "Describe the image."
    default_image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.tokenizer = None
        self.image_processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant.
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="mplug_owl2",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.MM_CAUSAL_LM,
            source=ModelSource.GITHUB,
            framework=Framework.TORCH,
        )

    def _load_processors(self):
        """Load tokenizer and image processor for the current variant."""

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            use_fast=False,
            trust_remote_code=True,
        )
        # Image processor
        self.image_processor = CLIPImageProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )

    def load_model(self, dtype_override=None):
        """Load and return the MPLUG-Owl2 model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The MPLUG-Owl2 model instance for multimodal text generation.
        """

        model_kwargs: Dict[str, Any] = {
            "return_dict": False,
            "use_cache": False,
            "dtype": dtype_override if dtype_override is not None else torch.float32,
        }

        # create model
        config = get_file("test_files/pytorch/mplug_owl2/config.json")
        model_config = MPLUGOwl2Config.from_pretrained(config, **model_kwargs)
        model = MPLUGOwl2LlamaForCausalLM(model_config)

        # load weights
        model = load_weights(model)
        model.eval()
        return model

    def load_inputs(
        self, dtype_override=None, batch_size: int = 1, query: Optional[str] = None
    ):
        """Load and return sample inputs with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override input dtype where applicable.
            batch_size: Batch size for the returned inputs.
            query: Optional user query to include after the image token. Defaults to a generic prompt.

        Returns:
            Dict[str, torch.Tensor]: Dictionary with keys:
                - "input_ids": tokenized prompt with image token(s), shape (B, T)
                - "images": processed image tensor, shape (B, C, H, W)
        """
        if self.tokenizer is None or self.image_processor is None:
            self._load_processors()

        # Image
        image_file = get_file(self.default_image_url)
        image = Image.open(image_file).convert("RGB")

        image_tensor = process_images([image], self.image_processor)  # (B, C, H, W)
        if batch_size > 1:
            image_tensor = image_tensor.repeat_interleave(batch_size, dim=0)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            image_tensor = image_tensor.to(dtype_override)

        # Prompt construction via conversation template
        user_query = query if query is not None else self.default_query
        conv = conv_templates["mplug_owl2"].copy()
        inp = DEFAULT_IMAGE_TOKEN + user_query
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(
            0
        )  # (1, T)
        if batch_size > 1:
            input_ids = input_ids.repeat_interleave(batch_size, dim=0)

        return {"input_ids": input_ids, "images": image_tensor}
