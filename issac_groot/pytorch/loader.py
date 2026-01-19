# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Isaac GR00T model loader implementation for robotic policy inference.
"""

from typing import Optional, Dict, Any
import torch
import numpy as np

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
from .src.model import Gr00tPolicyModule
from .src.utils import LeRobotSingleDataset, load_data_config


class ModelVariant(StrEnum):
    """Available Isaac GR00T model variants."""

    GROOT_N1_5_3B = "GR00T_n1.5-3b"


class ModelLoader(ForgeModel):
    """Isaac GR00T model loader for robotic policy inference."""

    _VARIANTS = {
        ModelVariant.GROOT_N1_5_3B: ModelConfig(
            pretrained_model_name="nvidia/GR00T-N1.5-3B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GROOT_N1_5_3B
    DEFAULT_DATA_CONFIG = "fourier_gr1_arms_only"
    DEFAULT_EMBODIMENT = "gr1"
    DEFAULT_DENOISING_STEPS = 4

    def __init__(
        self,
        variant: Optional[ModelVariant] = None,
        data_config: Optional[str] = None,
        embodiment_tag: Optional[str] = None,
        denoising_steps: Optional[int] = None,
    ):
        """
        Initialize Isaac GR00T model loader.

        Args:
            variant: Model variant to load (default: GROOT_N1_5_3B)
            data_config: Data configuration name (default: fourier_gr1_arms_only)
            embodiment_tag: Embodiment tag (default: gr1)
            denoising_steps: Number of denoising steps (default: 4)
        """
        super().__init__(variant)
        self.data_config = data_config or self.DEFAULT_DATA_CONFIG
        self.embodiment_tag = embodiment_tag or self.DEFAULT_EMBODIMENT
        self.denoising_steps = denoising_steps or self.DEFAULT_DENOISING_STEPS

        # Lazy initialization
        self._modality_config = None
        self._modality_transform = None
        self._model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="isaac_groot",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.MM_ACTION_PREDICTION,
            source=ModelSource.GITHUB,
            framework=Framework.TORCH,
        )

    def _load_data_config(self):
        """Load data configuration."""
        if self._modality_config is None:
            data_config_obj = load_data_config(self.data_config)
            self._modality_config = data_config_obj.modality_config()
            self._modality_transform = data_config_obj.transform()

    def load_model(self, dtype_override=None):
        """
        Load and return the Isaac GR00T policy model.

        Args:
            dtype_override: Optional dtype to cast model to (not recommended for GR00T)

        Returns:
            Gr00tPolicyModule instance (nn.Module)
        """
        self._load_data_config()

        pretrained_model_name = self._variant_config.pretrained_model_name

        # Create policy module
        policy = Gr00tPolicyModule(
            model_path=pretrained_model_name,
            embodiment_tag=self.embodiment_tag,
            modality_config=self._modality_config,
            modality_transform=self._modality_transform,
            denoising_steps=self.denoising_steps,
        )
        policy.eval()

        if dtype_override:
            print(
                f"Warning: GR00T uses mixed precision (BFloat16/Float32). dtype_override may cause issues."
            )

        # Store model reference for preprocessing in load_inputs
        self._model = policy

        return policy

    def load_inputs(self, dtype_override=None):
        """
        Load and return preprocessed input observations for Isaac GR00T.

        Uses the demo dataset (robot_sim.PickNPlace) and loads the first sample (index 0).
        Model must be loaded before calling this method.

        Args:
            dtype_override: Optional dtype to cast inputs to (not recommended for GR00T)

        Returns:
            Dictionary containing preprocessed observation tensors ready for model inference
        """
        # Ensure model is loaded
        if not hasattr(self, "_model") or self._model is None:
            raise RuntimeError(
                "Model must be loaded before loading inputs. "
                "Call load_model() first."
            )

        self._load_data_config()

        # Load dataset (no dataset_path needed - all files loaded via get_file)
        dataset = LeRobotSingleDataset(
            modality_configs=self._modality_config,
            embodiment_tag=self.embodiment_tag,
            video_backend="decord",
            video_backend_kwargs=None,
            transforms=None,
        )

        # Get first sample (index 0)
        observations = dataset[0]

        if dtype_override:
            print(
                f"Warning: dtype_override may not work well with mixed-type observations (videos, states, etc.)."
            )

        # Apply preprocessing
        observations = self._model.preprocess(observations)

        return observations

    def postprocess(self, raw_action_tensor: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Postprocess raw action tensor to action dictionary.

        Args:
            raw_action_tensor: Raw action tensor from model forward (with return_raw=True)

        Returns:
            Dictionary containing unnormalized actions (e.g., 'action.left_arm', 'action.right_arm', etc.)
        """
        if self._model is None:
            raise RuntimeError(
                "Model must be loaded before postprocessing. Call load_model() first."
            )

        # Always use is_batch=True since we always work with batched inputs
        return self._model.postprocess(raw_action_tensor, is_batch=True)
