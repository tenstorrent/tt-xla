# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Fuyu model loader implementation for multimodal AI tasks.
"""

import os
import torch
import torch.nn as nn
from PIL import Image
from transformers import (
    AutoTokenizer,
    FuyuConfig,
    FuyuForCausalLM,
    FuyuImageProcessor,
    FuyuProcessor,
)
from typing import Optional

from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel
from ...tools.utils import get_file


def generate_fuyu_embedding(model, input_ids, image_patches, image_patches_indices):
    """Generate embeddings for fuyu model combining text and image inputs."""
    inputs_embeds = model.model.language_model.get_input_embeddings()(input_ids)
    patch_embeddings = model.model.vision_embed_tokens(
        image_patches.to(model.model.vision_embed_tokens.weight.dtype)
    )
    inputs_embeds = model.model.gather_continuous_embeddings(
        word_embeddings=inputs_embeds,
        continuous_embeddings=patch_embeddings,
        image_patch_input_indices=image_patches_indices,
    )
    return inputs_embeds


class FuyuModelWrapper(nn.Module):
    """Wrapper for Fuyu model to handle embeddings properly."""

    def __init__(self, model):
        super().__init__()
        self.fuyu_model = model
        self.fuyu_config = model.config

    def forward(self, inputs_embeds):
        output_attentions = self.fuyu_config.output_attentions
        use_cache = self.fuyu_config.use_cache

        # retrieve input_ids and inputs_embeds
        batch_size, seq_length, _ = inputs_embeds.shape
        seq_length_with_past = seq_length
        past_key_values_length = 0

        device = inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length,
            seq_length + past_key_values_length,
            dtype=torch.long,
            device=device,
        )
        position_ids = position_ids.unsqueeze(0)

        # PersimmonForCausalLM
        output_hidden_states = (
            self.fuyu_model.model.language_model.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.fuyu_model.model.language_model(
            input_ids=None,
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        return outputs


class ModelVariant(StrEnum):
    """Available Fuyu model variants."""

    FUYU_8B = "adept/fuyu-8b"


class ModelLoader(ForgeModel):
    """Fuyu model loader implementation for multimodal tasks."""

    _VARIANTS = {
        ModelVariant.FUYU_8B: ModelConfig(
            pretrained_model_name="adept/fuyu-8b",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.FUYU_8B

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional variant to get info for. If None, uses default.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="fuyu",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional variant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        # Configuration parameters
        self.text_prompt = "Generate a coco-style caption.\n"
        self.image_url = (
            "https://huggingface.co/adept-hf-collab/fuyu-8b/resolve/main/bus.png"
        )
        self.tokenizer = None
        self.processor = None
        self.model = None

    def load_model(self, dtype_override=None):
        """Load a Fuyu model from Hugging Face."""

        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Configure the model with reduced layers for testing
        config = FuyuConfig.from_pretrained(pretrained_model_name)
        config_dict = config.to_dict()
        config_dict["use_cache"] = False
        config_dict["text_config"]["num_hidden_layers"] = 1
        config = FuyuConfig(**config_dict)

        # Initialize tokenizer and image processor
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, **tokenizer_kwargs
        )
        image_processor = FuyuImageProcessor()
        self.processor = FuyuProcessor(
            image_processor=image_processor, tokenizer=self.tokenizer
        )

        # Load pre-trained model from HuggingFace
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        fuyu_model = FuyuForCausalLM.from_pretrained(
            pretrained_model_name, config=config, **model_kwargs
        )

        # Store the raw model for embedding generation
        self.model = fuyu_model

        # Wrap the model
        framework_model = FuyuModelWrapper(fuyu_model)

        return framework_model

    def load_inputs(self, dtype_override=None):
        """Generate sample inputs for Fuyu model."""

        # Ensure model and processor are initialized
        if self.model is None or self.processor is None:
            self.load_model(dtype_override=dtype_override)

        # Download and load the image
        input_image = get_file(self.image_url)
        image_pil = Image.open(str(input_image))

        # Process text and image inputs
        model_inputs = self.processor(
            text=self.text_prompt, images=[image_pil], device="cpu", return_tensor="pt"
        )

        # Generate embeddings
        inputs_embeds = generate_fuyu_embedding(
            self.model,
            model_inputs["input_ids"],
            model_inputs["image_patches"],
            model_inputs["image_patches_indices"],
        )
        inputs_embeds = inputs_embeds.clone().detach()

        # Clean up the downloaded image
        if os.path.exists("bus.png"):
            os.remove("bus.png")

        # Return as list for consistency with other loaders
        return inputs_embeds
