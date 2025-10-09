# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
OpenVLA model loader implementation for action prediction.
"""
import os
import shutil
import torch
from transformers import AutoProcessor
from PIL import Image
from typing import Optional

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
from ...tools.utils import get_file
from .src.modeling_prismatic import OpenVLAForActionPrediction
from huggingface_hub import snapshot_download


class ModelVariant(StrEnum):
    """Available OpenVLA model variants."""

    OPENVLA_7B = "openvla_7b"
    OPENVLA_V01_7B = "openvla_v01_7b"


class ModelLoader(ForgeModel):
    """OpenVLA model loader implementation for action prediction tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.OPENVLA_7B: ModelConfig(
            pretrained_model_name="openvla/openvla-7b",
        ),
        ModelVariant.OPENVLA_V01_7B: ModelConfig(
            pretrained_model_name="openvla/openvla-v01-7b",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.OPENVLA_7B

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

        if variant == "openvla_7b":
            group = ModelGroup.RED
        else:
            group = ModelGroup.GENERALITY

        return ModelInfo(
            model="openvla",
            variant=variant,
            group=group,
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

    def load_model(self, dtype_override=None):
        """Load and return the OpenVLA model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The OpenVLA model instance for action prediction.
        """

        repo_id = self._variant_config.pretrained_model_name

        # Create a local folder name from the model name
        model_path = repo_id.split("/")[-1].replace("-", "_") + "_weights"

        # create folder if it doesn't exist
        os.makedirs(model_path, exist_ok=True)

        # Download only the essential files.
        # Note: Weight files are not fetched from the S3 bucket since they are >14 GB,
        #       and can be directly retrieved from HuggingFace instead.
        snapshot_download(
            repo_id=repo_id,
            local_dir=model_path,
            local_dir_use_symlinks=False,
            allow_patterns=[
                "*.safetensors",
                "config.json",
                "model.safetensors.index.json",
            ],
        )

        # Load Model
        model = OpenVLAForActionPrediction.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
        )

        # Configure model settings
        model.config.return_dict = False
        model.config.use_cache = False

        # Apply dtype override if specified
        if dtype_override is None:
            dtype_override = torch.float32

        model = model.to(dtype_override)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the OpenVLA model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        # Ensure processor is initialized
        if self.processor is None:
            self._load_processor()

        # Load the sample image
        image_file = get_file(self.sample_image_url)
        image = Image.open(image_file).convert("RGB")

        # Choose the prompt based on variant
        sample_prompt = (
            self.sample_prompt
            if self._variant_config.pretrained_model_name == "openvla/openvla-7b"
            else self.sample_prompt_1
        )

        # Process the image and prompt
        inputs = self.processor(sample_prompt, image)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        return inputs
