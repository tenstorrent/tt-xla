# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
OpenVLA-OFT model loader implementation for action prediction.
"""
import torch
from transformers import AutoProcessor
from PIL import Image
from typing import Optional
from datasets import load_dataset

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
from ...openvla.pytorch.src.modeling_prismatic import OpenVLAForActionPrediction


class ModelVariant(StrEnum):
    """Available OpenVLA-OFT model variants"""

    OPENVLA_OFT_FINETUNED_LIBERO_10 = "Finetuned_Libero_10"
    OPENVLA_OFT_FINETUNED_LIBERO_GOAL = "Finetuned_Libero_Goal"
    OPENVLA_OFT_FINETUNED_LIBERO_OBJECT = "Finetuned_Libero_Object"
    OPENVLA_OFT_FINETUNED_LIBERO_SPATIAL = "Finetuned_Libero_Spatial"
    OPENVLA_OFT_FINETUNED_LIBERO_SPATIAL_OBJECT_GOAL_10 = (
        "openvla_oft_finetuned_libero_spatial_object_goal_10"
    )


class ModelLoader(ForgeModel):
    """OpenVLA-OFT model loader implementation for action prediction tasks."""

    _VARIANTS = {
        ModelVariant.OPENVLA_OFT_FINETUNED_LIBERO_10: ModelConfig(
            pretrained_model_name="moojink/openvla-7b-oft-finetuned-libero-10",
        ),
        ModelVariant.OPENVLA_OFT_FINETUNED_LIBERO_GOAL: ModelConfig(
            pretrained_model_name="moojink/openvla-7b-oft-finetuned-libero-goal",
        ),
        ModelVariant.OPENVLA_OFT_FINETUNED_LIBERO_OBJECT: ModelConfig(
            pretrained_model_name="moojink/openvla-7b-oft-finetuned-libero-object",
        ),
        ModelVariant.OPENVLA_OFT_FINETUNED_LIBERO_SPATIAL: ModelConfig(
            pretrained_model_name="moojink/openvla-7b-oft-finetuned-libero-spatial",
        ),
        ModelVariant.OPENVLA_OFT_FINETUNED_LIBERO_SPATIAL_OBJECT_GOAL_10: ModelConfig(
            pretrained_model_name="moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.OPENVLA_OFT_FINETUNED_LIBERO_10

    # Shared configuration parameters
    sample_image_url = "https://rail.eecs.berkeley.edu/datasets/bridge_release/raw/bridge_data_v2/datacol2_toykitchen7/drawer_pnp/01/2023-04-19_09-18-15/raw/traj_group0/traj0/images0/im_30.jpg"
    sample_prompt = "In: What action should the robot take to open the drawer?\nOut:"

    system_prompt = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    )
    sample_prompt_1 = f"{system_prompt} USER: What action should the robot take to open the drawer? ASSISTANT:"

    def __init__(
        self,
        variant: Optional[ModelVariant] = None,
    ):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        # All OpenVLA-OFT variants are customer-requested (RED)
        return ModelInfo(
            model="OpenVLA-OFT",
            variant=variant,
            group=ModelGroup.RED
            if variant == ModelVariant.OPENVLA_OFT_FINETUNED_LIBERO_10
            else ModelGroup.GENERALITY,
            task=ModelTask.MM_ACTION_PREDICTION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load processor for the current variant.

        Returns:
            The loaded processor instance
        """
        # Load the processor from HuggingFace
        pretrained_model_name = self._variant_config.pretrained_model_name
        self.processor = AutoProcessor.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )

        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the OpenVLA-OFT model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The OpenVLA-OFT model instance for action prediction.
        """

        repo_id = self._variant_config.pretrained_model_name
        model = OpenVLAForActionPrediction.from_pretrained(
            repo_id,
            trust_remote_code=True,
            **kwargs,
        )

        model.config.return_dict = False
        model.config.use_cache = False

        # Apply dtype override if specified
        if dtype_override is None:
            dtype_override = torch.float32

        model = model.to(dtype_override)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the OpenVLA-OFT model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        # Ensure processor is initialized
        if self.processor is None:
            self._load_processor()

        # Load image from HuggingFace dataset
        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"].convert("RGB")

        # Choose the prompt based on variant
        sample_prompt = (
            self.sample_prompt
            if "openvla-7b" in self._variant_config.pretrained_model_name
            else self.sample_prompt_1
        )

        # Process the image and prompt
        inputs = self.processor(sample_prompt, image)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        return inputs
