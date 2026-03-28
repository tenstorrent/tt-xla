# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Mistral Pixtral model loader implementation
"""


import torch
from transformers import LlavaForConditionalGeneration
from typing import Optional
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....base import ForgeModel


class ModelVariant(StrEnum):
    """Available Pixtral model variants."""

    PIXTRAL_12B_COMMUNITY = "12B_Community"
    PIXTRAL_12B_EXPERIMENTAL = "12B_Experimental"
    PIXTRAL_12B_2409_BNB_4BIT = "12B_2409_Bnb_4bit"


class ModelLoader(ForgeModel):
    """Pixtral model loader implementation."""

    _VARIANTS = {
        ModelVariant.PIXTRAL_12B_COMMUNITY: ModelConfig(
            pretrained_model_name="mistral-community/pixtral-12b",
        ),
        ModelVariant.PIXTRAL_12B_EXPERIMENTAL: ModelConfig(
            pretrained_model_name="mistral-experimental/pixtral-12b",
        ),
        ModelVariant.PIXTRAL_12B_2409_BNB_4BIT: ModelConfig(
            pretrained_model_name="unsloth/Pixtral-12B-2409-bnb-4bit",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.PIXTRAL_12B_COMMUNITY

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

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

        if variant in (
            ModelVariant.PIXTRAL_12B_EXPERIMENTAL,
            ModelVariant.PIXTRAL_12B_2409_BNB_4BIT,
        ):
            group = ModelGroup.VULCAN
        else:
            group = ModelGroup.RED

        return ModelInfo(
            model="Pixtral",
            variant=variant,
            group=group,
            task=ModelTask.MM_VISUAL_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Mistral Pixtral model instance with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Mistral Pixtral model instance.

        """
        model_name = self._variant_config.pretrained_model_name

        # Load pre-trained model from HuggingFace
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        # Quantized variants need device_map="cpu" for CPU-based loading
        if self._variant in (ModelVariant.PIXTRAL_12B_2409_BNB_4BIT,):
            model_kwargs["device_map"] = "cpu"

        model = LlavaForConditionalGeneration.from_pretrained(
            model_name, **model_kwargs
        )
        self.model = model
        self.config = model.config
        return model

    def load_inputs(self, batch_size=1):
        """Load and return sample inputs for the Mistral Pixtral model with default settings.

        Args:
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors that can be fed to the model.
        """

        # https://github.com/tenstorrent/tt-torch/issues/904
        inputs = {
            "input_ids": torch.tensor(
                [[1, 3, 12483, 1593, 11386, 10, 51883, 3226, 1063, 10, 4]],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.long
            ),
        }

        # Use repeat_interleave to expand batch dimension
        inputs = {
            "input_ids": inputs["input_ids"].repeat_interleave(batch_size, dim=0),
            "attention_mask": inputs["attention_mask"].repeat_interleave(
                batch_size, dim=0
            ),
        }

        return inputs

    def get_mesh_config(self, num_devices: int):
        mesh_shape = (1, num_devices)
        return mesh_shape, ("batch", "model")

    @staticmethod
    def _get_language_model(model):
        """Get the language_model sub-module, handling nested model wrapping."""
        if hasattr(model, "language_model"):
            return model.language_model
        if hasattr(model, "model") and hasattr(model.model, "language_model"):
            return model.model.language_model
        raise AttributeError("Cannot find language_model on the model")

    @staticmethod
    def _get_vision_tower(model):
        """Get the vision_tower sub-module, handling nested model wrapping."""
        if hasattr(model, "vision_tower"):
            return model.vision_tower
        if hasattr(model, "model") and hasattr(model.model, "vision_tower"):
            return model.model.vision_tower
        raise AttributeError("Cannot find vision_tower on the model")

    def load_shard_spec(self, model):
        shard_specs = {}
        language_model = self._get_language_model(model)
        for layer in language_model.layers:
            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")

        vision_tower = self._get_vision_tower(model)
        for layer in vision_tower.transformer.layers:
            shard_specs[layer.feed_forward.up_proj.weight] = ("model", "batch")
            shard_specs[layer.feed_forward.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.feed_forward.down_proj.weight] = ("batch", "model")

            shard_specs[layer.attention.q_proj.weight] = ("model", "batch")
            shard_specs[layer.attention.k_proj.weight] = ("model", "batch")
            shard_specs[layer.attention.v_proj.weight] = ("model", "batch")
            shard_specs[layer.attention.o_proj.weight] = ("batch", "model")

        return shard_specs
