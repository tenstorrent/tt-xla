# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import tree
from einops import rearrange
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError
from PIL import Image
from pydantic import Field, PrivateAttr, model_validator
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    PretrainedConfig,
    PreTrainedModel,
    ProcessorMixin,
)
from transformers.data.data_collator import DataCollatorMixin
from transformers.feature_extraction_utils import BatchFeature

from ....tools.utils import get_file
from .utils import (
    EMBODIMENT_TAG_MAPPING,
    ComposedModalityTransform,
    DatasetMetadata,
    Eagle2_5_VLConfig,
    Eagle2_5_VLForConditionalGeneration,
    EmbodimentTag,
    FlowmatchingActionHead,
    FlowmatchingActionHeadConfig,
    InvertibleModalityTransform,
    ModalityConfig,
)

from transformers.image_processing_utils import BatchFeature, get_patch_output_size
from transformers.image_processing_utils_fast import (
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
    group_images_by_shape,
    reorder_images,
)
from transformers.image_utils import IMAGENET_STANDARD_MEAN  # 0.5, 0.5, 0.5
from transformers.image_utils import IMAGENET_STANDARD_STD  # 0.5, 0.5, 0.5
from transformers.image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
    get_image_size,
    make_flat_list_of_images,
    validate_kwargs,
)
from transformers.image_utils import ImageInput
from transformers.video_utils import VideoInput
from transformers.processing_utils import Unpack
from transformers.utils import (
    TensorType,
    add_start_docstrings,
    is_torch_available,
    is_torchvision_v2_available,
)

if is_torch_available():
    import torch
if is_torchvision_v2_available():
    from torchvision.transforms.v2 import functional as F
    from transformers.image_utils import pil_torch_interpolation_mapping
else:
    from torchvision.transforms import functional as F

import base64
import math
import os
import re
import time
import warnings
from functools import lru_cache
from io import BytesIO
from typing import Any, List, Literal, Optional, Union

import requests
import torch
import torchvision
from packaging import version
from PIL import Image
from torchvision import io
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.video_utils import VideoInput
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import logging


class EagleBackbone(nn.Module):
    def __init__(
        self,
        tune_llm: bool = False,
        tune_visual: bool = False,
        select_layer: int = -1,
        reproject_vision: bool = False,
        use_flash_attention: bool = False,
        load_bf16: bool = False,
        eagle_path: str | None = None,
        project_to_dim: int = 1536,
    ):
        """
        Args:
            tune_llm: whether to tune the LLM model (default: True)
            tune_visual: whether to tune the visual model (default: False)
        """
        super().__init__()
        assert (
            not reproject_vision
        ), "Reproject vision is not implemented here, set to False"

        # Load config from JSON directly (no AutoConfig)
        # Path to local eagle model config files (relative to this file)
        config_path = get_file("test_files/pytorch/Issac_groot/config.json")
        with open(config_path, "r") as f:
            config_dict = json.load(f)

        # Create config object explicitly
        config = Eagle2_5_VLConfig(**config_dict)

        # Disable flash attention if not available or not requested
        config._attn_implementation = "eager"

        # Instantiate model directly (no AutoModel)
        self.eagle_model = Eagle2_5_VLForConditionalGeneration(config)

        if project_to_dim is not None:
            self.eagle_linear = torch.nn.Linear(2048, project_to_dim)
        else:
            self.eagle_linear = torch.nn.Identity()

        # needed since we don't use these layers. Also saves compute
        while len(self.eagle_model.language_model.model.layers) > select_layer:
            self.eagle_model.language_model.model.layers.pop(-1)

        self.select_layer = select_layer
        self.set_trainable_parameters(tune_llm, tune_visual)

    def set_trainable_parameters(self, tune_llm: bool, tune_visual: bool):
        self.tune_llm = tune_llm
        self.tune_visual = tune_visual
        for p in self.parameters():
            p.requires_grad = True
        if not tune_llm:
            self.eagle_model.language_model.requires_grad_(False)
        if not tune_visual:
            self.eagle_model.vision_model.requires_grad_(False)
            self.eagle_model.mlp1.requires_grad_(False)
        print(f"Tune backbone llm: {self.tune_llm}")
        print(f"Tune backbone visual: {self.tune_visual}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_llm and not tune_visual:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Backbone trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No backbone trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if self.eagle_model.language_model and not self.tune_llm:
                self.eagle_model.language_model.eval()
            if self.eagle_model.vision_model and not self.tune_visual:
                self.eagle_model.vision_model.eval()

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def forward_eagle(self, vl_input: BatchFeature) -> BatchFeature:
        eagle_prefix = "eagle_"
        eagle_input = {
            k.removeprefix(eagle_prefix): v
            for k, v in vl_input.items()
            if k.startswith(eagle_prefix)
        }
        del eagle_input["image_sizes"]

        eagle_output = self.eagle_model(
            **eagle_input, output_hidden_states=True, return_dict=True
        )
        eagle_features = eagle_output.hidden_states[self.select_layer]

        eagle_features = self.eagle_linear(eagle_features)
        return eagle_features, eagle_input["attention_mask"]

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()

        eagle_embeds, eagle_mask = self.forward_eagle(vl_input)

        # YL (TODO HACK): to resolve DDP issue when tune_visual=True
        # Ensure all trainable parameters in vision_model are used in the forward pass for DDP compatibility
        if self.training and self.tune_visual:
            dummy_term = torch.tensor(
                0.0,
                device=eagle_embeds.device,
                dtype=eagle_embeds.dtype,
                requires_grad=True,
            )
            for param in self.eagle_model.vision_model.parameters():
                if param.requires_grad:
                    dummy_term = dummy_term + 0.0 * param.sum()
            eagle_embeds = eagle_embeds + dummy_term

        return BatchFeature(
            data={
                "backbone_features": eagle_embeds,
                "backbone_attention_mask": eagle_mask,
            }
        )  # [B, T2, hidden_size]


BACKBONE_FEATURE_KEY = "backbone_features"
ACTION_KEY = "action_pred"
LOSS_KEY = "loss"
ERROR_MSG = "Error: unexpected input/output"
N_COLOR_CHANNELS = 3


@dataclass
class GR00T_N1_5_Config(PretrainedConfig):
    model_type = "gr00t_n1_5"
    backbone_cfg: dict = field(init=False, metadata={"help": "Backbone configuration."})

    action_head_cfg: dict = field(
        init=False, metadata={"help": "Action head configuration."}
    )

    action_horizon: int = field(init=False, metadata={"help": "Action horizon."})

    action_dim: int = field(init=False, metadata={"help": "Action dimension."})
    compute_dtype: str = field(default="float32", metadata={"help": "Compute dtype."})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class GR00T_N1_5(PreTrainedModel):
    supports_gradient_checkpointing = True
    config_class = GR00T_N1_5_Config
    """
    we expect the backbone output to have a key 'backbone_features' with shape (batch_size, n, hidden_size)
    here n is variable and can be e.g. time, 1 or user specified
    we expect the action head output to have a key 'action_pred' with shape (batch_size, time, action_dim) during inference time
    we expect these to have type BatchFeature, and they can of course have many other user specified keys too
    """

    def __init__(
        self,
        config: GR00T_N1_5_Config,
        local_model_path: str,
    ):
        assert isinstance(config.backbone_cfg, dict)
        assert isinstance(config.action_head_cfg, dict)

        super().__init__(config)
        self.local_model_path = local_model_path

        self.backbone = EagleBackbone(**config.backbone_cfg)
        action_head_cfg = FlowmatchingActionHeadConfig(**config.action_head_cfg)
        self.action_head = FlowmatchingActionHead(action_head_cfg)

        self.action_horizon = config.action_horizon
        self.action_dim = config.action_dim
        self.compute_dtype = config.compute_dtype

    def validate_inputs(self, inputs):
        # NOTE -- this should be handled internally by the model
        # however, doing that will likely be breaking changes -- so we'll need to do it after the deadline

        detected_error = False
        error_msg = ERROR_MSG
        if "action" in inputs:
            action = inputs["action"]
            type_ok = isinstance(action, torch.Tensor)
            shape_ok = (
                len(action.shape) == 3
                and action.shape[1] == self.action_horizon
                and action.shape[2] == self.action_dim
            )
            if not type_ok:
                error_msg += f"\n{action.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{action.shape=}"
                detected_error = True

        if "video" in inputs:
            video = inputs["video"]
            type_ok = isinstance(video, np.ndarray)
            dtype_ok = video.dtype == np.uint8
            shape_ok = len(video.shape) == 6 and video.shape[3] == N_COLOR_CHANNELS
            if not type_ok:
                error_msg += f"\n{type(video)=}"
                detected_error = True
            if not dtype_ok:
                error_msg += f"\n{video.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{video.shape=}"
                detected_error = True

        if detected_error:
            raise ValueError(error_msg)

    def validate_data(self, action_head_outputs, backbone_outputs, is_training):
        fail_backbone = (
            not isinstance(backbone_outputs, BatchFeature)
            or BACKBONE_FEATURE_KEY not in backbone_outputs
        )

        if fail_backbone:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(backbone_outputs, BatchFeature)=}"
            error_msg += f"\n{BACKBONE_FEATURE_KEY in backbone_outputs=}"
            error_msg += f"\n{backbone_outputs[BACKBONE_FEATURE_KEY].shape=}"
            raise ValueError(error_msg)

        fail_action_head = (not isinstance(action_head_outputs, BatchFeature)) or not (
            (
                LOSS_KEY in action_head_outputs and is_training
            )  # there might not be an action prediction during training
            or (
                ACTION_KEY in action_head_outputs
                and action_head_outputs[ACTION_KEY].shape[1] == self.action_horizon
                and action_head_outputs[ACTION_KEY].shape[2] == self.action_dim
            )
        )

        if fail_action_head:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(action_head_outputs, BatchFeature)=}"
            error_msg += f"\n{LOSS_KEY in action_head_outputs=}"
            error_msg += f"\n{action_head_outputs[ACTION_KEY].shape=}"
            error_msg += f"\n{self.action_horizon=}"
            error_msg += f"\n{self.action_dim=}"
            raise ValueError(error_msg)

    def forward(
        self,
        inputs: dict,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        # Because the behavior of backbones remains the same for training and inference, we can use `forward` for backbones.
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head(backbone_outputs, action_inputs)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        return action_head_outputs

    def prepare_input(self, inputs) -> Tuple[BatchFeature, BatchFeature]:
        self.validate_inputs(inputs)
        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        # Convert any numpy arrays to tensors before using BatchFeature.to()
        def convert_numpy_to_tensor(x):
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x)
            return x

        # Process backbone inputs: convert numpy arrays, then use BatchFeature.to() for device transfer
        if isinstance(backbone_inputs, BatchFeature):
            # First convert any numpy arrays to tensors
            backbone_dict = {
                k: convert_numpy_to_tensor(v) for k, v in backbone_inputs.items()
            }
            backbone_inputs = BatchFeature(backbone_dict)
            # Then use .to() to move to device (keeps original dtypes)
            backbone_inputs = backbone_inputs.to(self.device)
        else:
            # For non-BatchFeature, use tree.map_structure
            def to_device(x):
                if isinstance(x, np.ndarray):
                    x = torch.from_numpy(x)
                if isinstance(x, torch.Tensor):
                    return x.to(self.device)
                return x

            backbone_inputs = tree.map_structure(to_device, backbone_inputs)

        # Process action inputs: convert numpy arrays, then use BatchFeature.to() and cast floating point to action_head dtype
        if isinstance(action_inputs, BatchFeature):
            # First convert any numpy arrays to tensors
            action_dict = {
                k: convert_numpy_to_tensor(v) for k, v in action_inputs.items()
            }
            action_inputs = BatchFeature(action_dict)
            # Move to device first
            action_inputs = action_inputs.to(self.device)
            # Then cast floating point tensors to action_head dtype
            action_dict = {}
            for k, v in action_inputs.items():
                if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                    action_dict[k] = v.to(dtype=self.action_head.dtype)
                else:
                    action_dict[k] = v
            action_inputs = BatchFeature(action_dict)
        else:
            # For non-BatchFeature, use tree.map_structure
            def to_device_with_dtype(x):
                if isinstance(x, np.ndarray):
                    x = torch.from_numpy(x)
                if isinstance(x, torch.Tensor):
                    if torch.is_floating_point(x):
                        return x.to(self.device, dtype=self.action_head.dtype)
                    else:
                        return x.to(self.device)
                return x

            action_inputs = tree.map_structure(to_device_with_dtype, action_inputs)

        return backbone_inputs, action_inputs

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        tune_visual = kwargs.pop("tune_visual", True)
        tune_llm = kwargs.pop("tune_llm", False)
        tune_projector = kwargs.pop("tune_projector", True)
        tune_diffusion_model = kwargs.pop("tune_diffusion_model", True)

        print(f"Loading pretrained dual brain from {pretrained_model_name_or_path}")
        print(f"Tune backbone vision tower: {tune_visual}")
        print(f"Tune backbone LLM: {tune_llm}")
        print(f"Tune action head projector: {tune_projector}")
        print(f"Tune action head DiT: {tune_diffusion_model}")

        # get the current model path being downloaded
        try:
            local_model_path = snapshot_download(
                pretrained_model_name_or_path, repo_type="model"
            )
        except (HFValidationError, RepositoryNotFoundError):
            print(
                f"Model not found or avail in the huggingface hub. Loading from local path: {pretrained_model_name_or_path}"
            )
            local_model_path = pretrained_model_name_or_path

        pretrained_model = super().from_pretrained(
            local_model_path, local_model_path=local_model_path, **kwargs
        )

        pretrained_model.backbone.set_trainable_parameters(
            tune_visual=tune_visual, tune_llm=tune_llm
        )
        pretrained_model.action_head.set_trainable_parameters(
            tune_projector=tune_projector, tune_diffusion_model=tune_diffusion_model
        )
        return pretrained_model


AutoConfig.register("gr00t_n1_5", GR00T_N1_5_Config)
AutoModel.register(GR00T_N1_5_Config, GR00T_N1_5)


COMPUTE_DTYPE = torch.bfloat16


class BasePolicy(ABC):
    @abstractmethod
    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Abstract method to get the action for a given state.

        Args:
            observations: The observations from the environment.

        Returns:
            The action to take in the environment in dictionary format.
        """
        raise NotImplementedError

    @abstractmethod
    def get_modality_config(self) -> Dict[str, ModalityConfig]:
        """
        Return the modality config of the policy.
        """
        raise NotImplementedError


class Gr00tPolicy(BasePolicy):
    """
    A wrapper for Gr00t model checkpoints that handles loading the model, applying transforms,
    making predictions, and unapplying transforms. This loads some custom configs, stats
    and metadata related to the model checkpoints used
    in the Gr00t model.
    """

    def __init__(
        self,
        model_path: str,
        embodiment_tag: Union[str, EmbodimentTag],
        modality_config: Dict[str, ModalityConfig],
        modality_transform: ComposedModalityTransform,
        denoising_steps: Optional[int] = None,
        device: Union[int, str] = "cpu",
    ):
        """
        Initialize the Gr00tPolicy.

        Args:
            model_path (str): Path to the model checkpoint directory or the huggingface hub id.
            modality_config (Dict[str, ModalityConfig]): The modality config for the model.
            modality_transform (ComposedModalityTransform): The modality transform for the model.
            embodiment_tag (Union[str, EmbodimentTag]): The embodiment tag for the model.
            denoising_steps: Number of denoising steps to use for the action head.
            device (Union[int, str]): Device to run the model on.
        """
        try:
            # NOTE(YL) this returns the local path to the model which is normally
            # saved in ~/.cache/huggingface/hub/
            model_path = snapshot_download(model_path, repo_type="model")
            # HFValidationError, RepositoryNotFoundError
        except (HFValidationError, RepositoryNotFoundError):
            print(
                f"Model not found or avail in the huggingface hub. Loading from local path: {model_path}"
            )

        self._modality_config = modality_config
        self._modality_transform = modality_transform
        self._modality_transform.eval()  # set this to eval mode
        self.model_path = Path(model_path)
        self.device = device

        # Convert string embodiment tag to EmbodimentTag enum if needed
        if isinstance(embodiment_tag, str):
            self.embodiment_tag = EmbodimentTag(embodiment_tag)
        else:
            self.embodiment_tag = embodiment_tag

        # Load model
        self._load_model(model_path)
        # Load transforms
        self._load_metadata(self.model_path / "experiment_cfg")
        # Load horizons
        self._load_horizons()

        if denoising_steps is not None:
            if hasattr(self.model, "action_head") and hasattr(
                self.model.action_head, "num_inference_timesteps"
            ):
                self.model.action_head.num_inference_timesteps = denoising_steps
                print(f"Set action denoising steps to {denoising_steps}")

    def apply_transforms(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply transforms to the observation.

        Args:
            obs (Dict[str, Any]): The observation to transform.

        Returns:
            Dict[str, Any]: The transformed observation.
        """
        # Ensure correct dimensions before applying transforms
        return self._modality_transform(obs)

    def unapply_transforms(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Unapply transforms to the action.

        Args:
            action (Dict[str, Any]): The action to unapply transforms to.

        Returns:
            Dict[str, Any]: The untransformed action.
        """
        return self._modality_transform.unapply(action)

    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a prediction with the model.
        Args:
            obs (Dict[str, Any]): The observation to make a prediction for.

        e.g. obs = {
            "video.<>": np.ndarray,  # (T, H, W, C)
            "state.<>": np.ndarray, # (T, D)
            "annotation.<>": np.ndarray, # (T, )
        }

        or with batched input:
        e.g. obs = {
            "video.<>": np.ndarray,, # (B, T, H, W, C)
            "state.<>": np.ndarray, # (B, T, D)
            "annotation.<>": np.ndarray, # (B, T, )
        }

        Returns:
            Dict[str, Any]: The predicted action.
        """
        # Create a copy to avoid mutating input
        obs_copy = observations.copy()

        is_batch = self._check_state_is_batched(obs_copy)
        if not is_batch:
            obs_copy = unsqueeze_dict_values(obs_copy)

        # Convert to numpy arrays
        for k, v in obs_copy.items():
            if not isinstance(v, np.ndarray):
                obs_copy[k] = np.array(v)

        normalized_input = self.apply_transforms(obs_copy)
        normalized_action = self._get_action_from_normalized_input(normalized_input)
        unnormalized_action = self._get_unnormalized_action(normalized_action)

        if not is_batch:
            unnormalized_action = squeeze_dict_values(unnormalized_action)
        return unnormalized_action

    def _get_action_from_normalized_input(
        self, normalized_input: Dict[str, Any]
    ) -> torch.Tensor:
        # Set up autocast context if needed
        # Note: Using no_grad instead of inference_mode to avoid conflicts with outer contexts
        with torch.no_grad():
            model_pred = self.model(normalized_input)

        normalized_action = model_pred["action_pred"].float()
        return normalized_action

    def _get_unnormalized_action(
        self, normalized_action: torch.Tensor
    ) -> Dict[str, Any]:
        return self.unapply_transforms({"action": normalized_action.cpu()})

    def get_modality_config(self) -> Dict[str, ModalityConfig]:
        """
        Get the modality config for the model, overrides the base class method
        """
        return self._modality_config

    @property
    def modality_config(self) -> Dict[str, ModalityConfig]:
        return self._modality_config

    @property
    def modality_transform(self) -> ComposedModalityTransform:
        return self._modality_transform

    @property
    def video_delta_indices(self) -> np.ndarray:
        """Get the video delta indices."""
        return self._video_delta_indices

    @property
    def state_delta_indices(self) -> np.ndarray | None:
        """Get the state delta indices."""
        return self._state_delta_indices

    @property
    def denoising_steps(self) -> int:
        """Get the number of denoising steps."""
        return self.model.action_head.num_inference_timesteps

    @denoising_steps.setter
    def denoising_steps(self, value: int):
        """Set the number of denoising steps."""
        self.model.action_head.num_inference_timesteps = value

    def _check_state_is_batched(self, obs: Dict[str, Any]) -> bool:
        for k, v in obs.items():
            if "state" in k and len(v.shape) < 3:  # (B, Time, Dim)
                return False
        return True

    def _load_model(self, model_path):
        model = GR00T_N1_5.from_pretrained(model_path, torch_dtype=COMPUTE_DTYPE)
        model.eval()  # Set model to eval mode

        # Update action_horizon to match modality config
        # Get the expected action horizon from the modality config
        expected_action_horizon = len(self._modality_config["action"].delta_indices)

        if expected_action_horizon != model.action_head.config.action_horizon:
            print(
                f"Policy: Recreating action head with action_horizon {expected_action_horizon} (was {model.action_head.config.action_horizon})"
            )

            # Update the action head config
            new_action_head_config = model.action_head.config
            new_action_head_config.action_horizon = (
                expected_action_horizon  # Create new action head with updated config
            )
            new_action_head = FlowmatchingActionHead(new_action_head_config)

            # Copy the weights from the old action head to the new one
            new_action_head.load_state_dict(
                model.action_head.state_dict(), strict=False
            )

            # Replace the action head
            model.action_head = new_action_head

            # Update model config AND the action_head_cfg dictionary that gets saved
            model.config.action_horizon = expected_action_horizon
            model.action_horizon = expected_action_horizon
            model.config.action_head_cfg["action_horizon"] = expected_action_horizon

        model.to(device=self.device)  # type: ignore

        self.model = model

    def _load_metadata(self, exp_cfg_dir: Path):
        """Load the transforms for the model."""
        # Load metadata for normalization stats
        metadata_path = exp_cfg_dir / "metadata.json"
        with open(metadata_path, "r") as f:
            metadatas = json.load(f)

        # Get metadata for the specific embodiment
        metadata_dict = metadatas.get(self.embodiment_tag.value)
        if metadata_dict is None:
            raise ValueError(
                f"No metadata found for embodiment tag: {self.embodiment_tag.value}",
                f"make sure the metadata.json file is present at {metadata_path}",
            )

        metadata = DatasetMetadata.model_validate(metadata_dict)

        self._modality_transform.set_metadata(metadata)
        self.metadata = metadata

    def _load_horizons(self):
        """Load the horizons needed for the model."""
        # Get modality configs
        # Video horizons
        self._video_delta_indices = np.array(
            self._modality_config["video"].delta_indices
        )
        self._assert_delta_indices(self._video_delta_indices)
        self._video_horizon = len(self._video_delta_indices)
        # State horizons (if used)
        if "state" in self._modality_config:
            self._state_delta_indices = np.array(
                self._modality_config["state"].delta_indices
            )
            self._assert_delta_indices(self._state_delta_indices)
            self._state_horizon = len(self._state_delta_indices)
        else:
            self._state_horizon = None
            self._state_delta_indices = None

    def _assert_delta_indices(self, delta_indices: np.ndarray):
        """Assert that the delta indices are valid."""
        # All delta indices should be non-positive because there's no way to get the future observations
        assert np.all(delta_indices <= 0), f"{delta_indices=}"
        # The last delta index should be 0 because it doesn't make sense to not use the latest observation
        assert delta_indices[-1] == 0, f"{delta_indices=}"
        if len(delta_indices) > 1:
            # The step is consistent
            assert np.all(
                np.diff(delta_indices) == delta_indices[1] - delta_indices[0]
            ), f"{delta_indices=}"
            # And the step is positive
            assert (delta_indices[1] - delta_indices[0]) > 0, f"{delta_indices=}"


def unsqueeze_dict_values(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unsqueeze the values of a dictionary.
    This converts the data to be batched of size 1.
    """
    unsqueezed_data = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            unsqueezed_data[k] = np.expand_dims(v, axis=0)
        elif isinstance(v, list):
            unsqueezed_data[k] = np.expand_dims(np.array(v), axis=0)  # Fixed
        elif isinstance(v, torch.Tensor):
            unsqueezed_data[k] = v.unsqueeze(0)
        else:
            unsqueezed_data[k] = v
    return unsqueezed_data


def squeeze_dict_values(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Squeeze the values of a dictionary. This removes the batch dimension.
    """
    squeezed_data = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            squeezed_data[k] = np.squeeze(v, axis=0)  # Fixed: only remove batch dim
        elif isinstance(v, torch.Tensor):
            squeezed_data[k] = v.squeeze(0)  # Fixed: only remove batch dim
        else:
            squeezed_data[k] = v
    return squeezed_data


def crop(
    img: torch.Tensor, left: int, top: int, right: int, bottom: int
) -> torch.Tensor:
    """Crop the given numpy array.

    Args:
        img (torch.Tensor): Image to be cropped. Format should be (C, H, W).
        left (int): The left coordinate of the crop box.
        top (int): The top coordinate of the crop box.
        right (int): The right coordinate of the crop box.
        bottom (int): The bottom coordinate of the crop box.

    Returns:
        torch.Tensor: Cropped image.
    """
    if not isinstance(img, torch.Tensor):
        raise TypeError("img should be torch.Tensor. Got {}".format(type(img)))

    if img.ndim not in [2, 3]:
        raise ValueError("Image should have 2 or 3 dimensions. Got {}".format(img.ndim))

    img_height = img.shape[1]
    img_width = img.shape[2]
    if top < 0 or left < 0 or bottom > img_height or right > img_width:
        raise ValueError("Crop coordinates out of bounds")

    if top >= bottom or left >= right:
        raise ValueError("Invalid crop coordinates")

    return img[:, top:bottom, left:right]


class Eagle2_5_VLFastImageProcessorKwargs(DefaultFastImageProcessorKwargs):
    max_dynamic_tiles: Optional[int]
    min_dynamic_tiles: Optional[int]
    use_thumbnail: Optional[bool]
    pad_during_tiling: Optional[bool]
    do_pad: Optional[bool]


@add_start_docstrings(
    "Constructs a fast ConvNeXT image processor. Based on [`SiglipImageProcessor`] with incorporation of processing each video frame.",
    # BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
    """
        image_grid_pinpoints (`List[List[int]]`, *optional*):
            A list of possible resolutions to use for processing high resolution images. The best resolution is selected
            based on the original size of the image. Can be overridden by `image_grid_pinpoints` in the `preprocess`
            method. Not used for processing videos.
        do_pad (`bool`, *optional*):
            Whether to pad the image. If `True`, will pad the patch dimension of the images in the batch to the largest
            number of patches in the batch. Padding will be applied to the bottom and right with zeros.
    """,
)
class Eagle2_5_VLImageProcessorFast(BaseImageProcessorFast):
    resample = PILImageResampling.BICUBIC
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"height": 448, "width": 448}
    default_to_square = False
    crop_size = None
    do_resize = True
    do_center_crop = None
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    do_pad = True
    max_dynamic_tiles = 12
    min_dynamic_tiles = 1
    use_thumbnail = True
    pad_during_tiling = False
    valid_kwargs = Eagle2_5_VLFastImageProcessorKwargs
    model_input_names = ["pixel_values_videos"]

    def __init__(self, **kwargs: Unpack[Eagle2_5_VLFastImageProcessorKwargs]):
        super().__init__(**kwargs)

    @add_start_docstrings(
        # BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS,
        """
            max_dynamic_tiles (`int`, *optional*):
                The maximum number of dynamic tiles to use for processing high resolution images.
            min_dynamic_tiles (`int`, *optional*):
                The minimum number of dynamic tiles to use for processing high resolution images.
            use_thumbnail (`bool`, *optional*):
                Whether to use a thumbnail for processing high resolution images.
            pad_during_tiling (`bool`, *optional*):
                Whether to pad the image during tiling.
            do_pad (`bool`, *optional*):
                    Whether to pad the image. If `True`, will pad the patch dimension of the images in the batch to the largest
                    number of patches in the batch. Padding will be applied to the bottom and right with zeros.
        """,
    )

    # NOTE(YL): we will overload the preprocess method to add the image_flags
    # def preprocess(
    #     self, images: ImageInput, **kwargs: Unpack[Eagle2_5_VLFastImageProcessorKwargs]
    # ) -> BatchFeature:
    #     return super().preprocess(images, **kwargs)

    def _prepare_images_structure(
        self,
        images: ImageInput,
    ) -> ImageInput:
        """
        Prepare the images structure for processing.

        Args:
            images (`ImageInput`):
                The input images to process.

        Returns:
            `ImageInput`: The images with a valid nesting.
        """
        return make_flat_list_of_images(images)

    def _prepare_videos_structure(self, videos: VideoInput) -> VideoInput:
        return self._prepare_images_structure(videos)

    def _prepare_input_videos(
        self,
        videos: VideoInput,
        do_convert_rgb: Optional[bool] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        device: Optional["torch.device"] = None,
    ) -> list["torch.Tensor"]:
        """
        Prepare the input images for processing.
        """
        videos = self._prepare_videos_structure(videos)
        process_video_fn = partial(
            self._process_image,
            do_convert_rgb=do_convert_rgb,
            input_data_format=input_data_format,
            device=device,
        )
        # todo: yoni - check if we can parallelize this efficiently
        processed_videos = []
        for video in videos:
            processed_videos.append(process_video_fn(video))

        return processed_videos

    def _resize_for_patching(
        self,
        image: "torch.Tensor",
        target_resolution: tuple,
        interpolation: "F.InterpolationMode",
        input_data_format: ChannelDimension,
    ) -> "torch.Tensor":
        """
        Resizes an image to a target resolution while maintaining aspect ratio.

        Args:
            image ("torch.Tensor"):
                The input image.
            target_resolution (tuple):
                The target resolution (height, width) of the image.
            interpolation (`InterpolationMode`):
                Resampling filter to use if resizing the image.
            input_data_format (`ChannelDimension` or `str`):
                The channel dimension format of the input image.

        Returns:
            "torch.Tensor": The resized and padded image.
        """
        new_height, new_width = get_patch_output_size(
            image, target_resolution, input_data_format
        )

        # Resize the image
        resized_image = F.resize(
            image, (new_height, new_width), interpolation=interpolation
        )

        return resized_image

    def find_closest_aspect_ratio(
        self, aspect_ratio, target_ratios, width, height, image_size
    ):
        """
        previous version mainly foucs on ratio.
        We also consider area ratio here.
        """
        best_factor = float("-inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            # ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            # area_ratio = (ratio[0] * ratio[1] * image_size * image_size) / area
            """
            new area > 60% of original image area is enough.
            """
            factor_based_on_area_n_ratio = min(
                (ratio[0] * ratio[1] * image_size * image_size) / area, 0.6
            ) * min(
                target_aspect_ratio / aspect_ratio, aspect_ratio / target_aspect_ratio
            )

            if factor_based_on_area_n_ratio > best_factor:
                best_factor = factor_based_on_area_n_ratio
                best_ratio = ratio

        return best_ratio

    def _pad_for_patching(
        self,
        image: "torch.Tensor",
        target_resolution: tuple,
        input_data_format: ChannelDimension,
    ) -> "torch.Tensor":
        """
        Pad an image to a target resolution while maintaining aspect ratio.
        """
        target_height, target_width = target_resolution
        new_height, new_width = get_patch_output_size(
            image, target_resolution, input_data_format
        )

        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2

        padded_image = F.pad(image, padding=[paste_x, paste_y, paste_x, paste_y])

        return padded_image

    def _get_image_patches(
        self,
        image: "torch.Tensor",
        min_num: int,
        max_num: int,
        size: tuple,
        tile_size: int,
        use_thumbnail: bool,
        interpolation: "F.InterpolationMode",
        pad_during_tiling: bool,
    ) -> List["torch.Tensor"]:
        image_size = get_image_size(image, channel_dim=ChannelDimension.FIRST)
        orig_height, orig_width = image_size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, tile_size
        )

        # calculate the target width and height
        target_width = tile_size * target_aspect_ratio[0]
        target_height = tile_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
        if pad_during_tiling:
            resized_image = self._resize_for_patching(
                image,
                (target_height, target_width),
                interpolation=interpolation,
                input_data_format=ChannelDimension.FIRST,
            )
            padded_image = self._pad_for_patching(
                resized_image,
                (target_height, target_width),
                input_data_format=ChannelDimension.FIRST,
            )
            image_used_to_split = padded_image
        else:
            image_used_to_split = F.resize(
                image, (target_height, target_width), interpolation=interpolation
            )

        processed_tiles = []
        for i in range(blocks):
            box = (
                (i % (target_width // tile_size)) * tile_size,
                (i // (target_width // tile_size)) * tile_size,
                ((i % (target_width // tile_size)) + 1) * tile_size,
                ((i // (target_width // tile_size)) + 1) * tile_size,
            )
            # split the image
            split_img = crop(image_used_to_split, box[0], box[1], box[2], box[3])
            processed_tiles.append(split_img)
        assert len(processed_tiles) == blocks

        if use_thumbnail and len(processed_tiles) != 1:
            thumbnail_img = F.resize(
                image, (tile_size, tile_size), interpolation=interpolation
            )
            processed_tiles.append(thumbnail_img)

        return processed_tiles

    def _pad_for_batching(
        self,
        pixel_values: List["torch.Tensor"],
    ) -> List["torch.Tensor"]:
        """
        Pads images on the `num_of_patches` dimension with zeros to form a batch of same number of patches.

        Args:
            pixel_values (`List[torch.Tensor]`):
                An array of pixel values of each images of shape (`batch_size`, `num_patches`, `image_in_3D`)

        Returns:
            List[`torch.Tensor`]: The padded images.
        """
        max_patch = max(len(x) for x in pixel_values)
        pixel_values = [
            torch.nn.functional.pad(
                image, pad=[0, 0, 0, 0, 0, 0, 0, max_patch - image.shape[0]]
            )
            for image in pixel_values
        ]

        return pixel_values

    def _preprocess(
        self,
        images: List["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        max_dynamic_tiles: int,
        min_dynamic_tiles: int,
        use_thumbnail: bool,
        pad_during_tiling: bool,
        interpolation: Optional["F.InterpolationMode"],
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: Optional[Union[float, List[float]]],
        image_std: Optional[Union[float, List[float]]],
        do_pad: bool,
        return_tensors: Optional[Union[str, TensorType]],
    ) -> BatchFeature:
        processed_images = []
        image_sizes = []
        # Determine the size tuple
        if size and size.height and size.width:
            size_tuple = (size.height, size.width)
        else:
            size_tuple = (size.shortest_edge, size.shortest_edge)

        # Determine the patch size
        if crop_size and crop_size.height:
            tile_size = crop_size.height
        elif size and size.height:
            tile_size = size.height
        else:
            tile_size = size.shortest_edge

        for image in images:
            image_patches = self._get_image_patches(
                image,
                min_num=min_dynamic_tiles,
                max_num=max_dynamic_tiles,
                size=size_tuple,
                tile_size=tile_size,
                use_thumbnail=use_thumbnail,
                interpolation=interpolation,
                pad_during_tiling=pad_during_tiling,
            )

            # Group images by size for batched processing
            processed_image_patches_grouped = {}
            grouped_image_patches, grouped_image_patches_index = group_images_by_shape(
                image_patches
            )

            for shape, stacked_image_patches in grouped_image_patches.items():
                if do_resize:
                    stacked_image_patches = self.resize(
                        image=stacked_image_patches,
                        size=size,
                        interpolation=interpolation,
                    )
                if do_center_crop:
                    stacked_image_patches = self.center_crop(
                        stacked_image_patches, crop_size
                    )
                # Fused rescale and normalize
                stacked_image_patches = self.rescale_and_normalize(
                    stacked_image_patches,
                    do_rescale,
                    rescale_factor,
                    do_normalize,
                    image_mean,
                    image_std,
                )
                processed_image_patches_grouped[shape] = stacked_image_patches
            processed_image_patches = reorder_images(
                processed_image_patches_grouped, grouped_image_patches_index
            )
            processed_image_patches = (
                torch.stack(processed_image_patches, dim=0)
                if return_tensors
                else processed_image_patches
            )
            processed_images.append(processed_image_patches)
            image_sizes.append(get_image_size(image, ChannelDimension.FIRST))

        if do_pad:
            processed_images = self._pad_for_batching(processed_images)

        # processed_images = torch.stack(processed_images, dim=0) if return_tensors else processed_images
        processed_images = (
            torch.cat(processed_images, dim=0) if return_tensors else processed_images
        )
        return BatchFeature(
            data={"pixel_values": processed_images, "image_sizes": image_sizes},
            tensor_type=return_tensors,
        )

    def preprocess(
        self,
        images: ImageInput,
        videos: VideoInput = None,
        **kwargs: Unpack[Eagle2_5_VLFastImageProcessorKwargs],
    ) -> BatchFeature:
        validate_kwargs(
            captured_kwargs=kwargs.keys(),
            valid_processor_keys=self.valid_kwargs.__annotations__.keys(),
        )
        # Set default kwargs from self. This ensures that if a kwarg is not provided
        # by the user, it gets its default value from the instance, or is set to None.
        for kwarg_name in self.valid_kwargs.__annotations__:
            kwargs.setdefault(kwarg_name, getattr(self, kwarg_name, None))

        # Extract parameters that are only used for preparing the input images
        do_convert_rgb = kwargs.pop("do_convert_rgb")
        input_data_format = kwargs.pop("input_data_format")
        device = kwargs.pop("device")
        # Prepare input images
        if images is not None:
            images = self._prepare_input_images(
                images=images,
                do_convert_rgb=do_convert_rgb,
                input_data_format=input_data_format,
                device=device,
            )

        if videos is not None:
            videos = self._prepare_input_images(
                images=videos,
                do_convert_rgb=do_convert_rgb,
                input_data_format=input_data_format,
                device=device,
            )

        # Update kwargs that need further processing before being validated
        kwargs = self._further_process_kwargs(**kwargs)

        # Validate kwargs
        self._validate_preprocess_kwargs(**kwargs)

        # torch resize uses interpolation instead of resample
        resample = kwargs.pop("resample")
        kwargs["interpolation"] = (
            pil_torch_interpolation_mapping[resample]
            if isinstance(resample, (PILImageResampling, int))
            else resample
        )

        # Pop kwargs that are not needed in _preprocess
        kwargs.pop("default_to_square")
        kwargs.pop("data_format")
        if images is not None:
            return self._preprocess(images, **kwargs)
        elif videos is not None:
            return self._preprocess(videos, **kwargs)


FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 256


def adjust_by_factor(
    number: int, factor: int, method: Literal["round", "ceil", "floor"] = "round"
) -> int:
    """Adjusts 'number' to the nearest, ceiling, or floor multiple of 'factor'."""
    op = {"round": round, "ceil": math.ceil, "floor": math.floor}[method]
    return op(number / factor) * factor


def to_rgb(pil_image: Image.Image) -> Image.Image:
    if pil_image.mode == "RGBA":
        white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
        white_background.paste(
            pil_image, mask=pil_image.split()[3]
        )  # Use alpha channel as mask
        return white_background
    else:
        return pil_image.convert("RGB")


def fetch_image(ele: dict[str, str | Image.Image]) -> Image.Image:
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        response = requests.get(image, stream=True)
        image_obj = Image.open(BytesIO(response.content))
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            image_obj = Image.open(BytesIO(data))
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(
            f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}"
        )
    image = to_rgb(image_obj)
    if "scale_factor" in ele:
        scale_factor = ele["scale_factor"]
        image = image.resize(
            (image.width * scale_factor, image.height * scale_factor), Image.BILINEAR
        )
    return image


def smart_nframes(
    ele: dict,
    total_frames: int,
    video_fps: int | float,
) -> int:
    assert not (
        "fps" in ele and "nframes" in ele
    ), "Only accept either `fps` or `nframes`"
    if "nframes" in ele:
        nframes = adjust_by_factor(ele["nframes"], FRAME_FACTOR, method="round")
    else:
        fps = ele.get("fps", FPS)
        min_frames = adjust_by_factor(
            ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR, method="ceil"
        )
        max_frames = adjust_by_factor(
            ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)),
            FRAME_FACTOR,
            method="floor",
        )
        nframes = total_frames / video_fps * fps
        if nframes > total_frames:
            logger.warning(
                f"smart_nframes: nframes[{nframes}] > total_frames[{total_frames}]"
            )
        nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
        nframes = adjust_by_factor(nframes, FRAME_FACTOR, method="floor")
    if not (FRAME_FACTOR <= nframes and nframes <= total_frames):
        raise ValueError(
            f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}."
        )
    return nframes


def _read_video_torchvision(
    ele: dict,
) -> (torch.Tensor, float, list):
    """read video using torchvision.io.read_video and return also per-frame timestamps"""
    video_path = ele["video"]
    if version.parse(torchvision.__version__) < version.parse("0.19.0"):
        if "http://" in video_path or "https://" in video_path:
            warnings.warn(
                "torchvision < 0.19.0 does not support http/https video path, please upgrade to 0.19.0."
            )
        if "file://" in video_path:
            video_path = video_path[7:]
    st = time.time()
    video, audio, info = io.read_video(
        video_path,
        start_pts=ele.get("video_start", 0.0),
        end_pts=ele.get("video_end", None),
        pts_unit="sec",
        output_format="TCHW",
    )
    total_frames, video_fps = video.size(0), info["video_fps"]
    logger.info(
        f"torchvision:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s"
    )
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    # Calculate frame indices and corresponding timestamps (based on video start time)
    idx = torch.linspace(0, total_frames - 1, nframes).round().long()
    start_time = ele.get("video_start", 0.0)
    timestamps = (start_time + idx.to(torch.float32) / video_fps).tolist()
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    video = video[idx]
    return video, sample_fps, timestamps


def is_decord_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("decord") is not None


def _read_video_decord(
    ele: dict,
) -> (torch.Tensor, float, list):
    """read video using decord.VideoReader and return also per-frame timestamps"""
    import decord

    video_path = ele["video"]
    st = time.time()
    vr = decord.VideoReader(video_path)
    if "video_start" in ele or "video_end" in ele:
        raise NotImplementedError(
            "not support start_pts and end_pts in decord for now."
        )
    total_frames, video_fps = len(vr), vr.get_avg_fps()
    logger.info(
        f"decord:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s"
    )
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
    start_time = ele.get("video_start", 0.0)  # TODO:
    timestamps = [start_time + i / video_fps for i in idx]
    video = vr.get_batch(idx).asnumpy()
    video = torch.tensor(video).permute(0, 3, 1, 2)  # Convert to TCHW format
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    return video, sample_fps, timestamps


VIDEO_READER_BACKENDS = {
    "decord": _read_video_decord,
    "torchvision": _read_video_torchvision,
}


@lru_cache(maxsize=1)
def get_video_reader_backend() -> str:
    if is_decord_available():
        video_reader_backend = "decord"
    else:
        video_reader_backend = "torchvision"
    return video_reader_backend


def fetch_video(
    ele: dict, return_video_sample_fps: bool = False
) -> torch.Tensor | list[Image.Image]:
    if isinstance(ele["video"], str):
        video_reader_backend = get_video_reader_backend()
        try:
            video, sample_fps, timestamps = VIDEO_READER_BACKENDS[video_reader_backend](
                ele
            )
        except Exception as e:
            logger.warning(
                f"video_reader_backend {video_reader_backend} error, use torchvision as default, msg: {e}"
            )
            video, sample_fps, timestamps = VIDEO_READER_BACKENDS["torchvision"](ele)

        nframes, _, height, width = video.shape

        if return_video_sample_fps:
            return video, sample_fps, timestamps
        return video
    else:
        assert isinstance(ele["video"], (list, tuple))
        process_info = ele.copy()
        process_info.pop("type", None)
        process_info.pop("video", None)
        images = [
            fetch_image({"image": video_element, **process_info})
            for video_element in ele["video"]
        ]
        nframes = adjust_by_factor(len(images), FRAME_FACTOR, method="ceil")
        if len(images) < nframes:
            images.extend([images[-1]] * (nframes - len(images)))

        timestamps = [-1 for i in range(nframes)]  # not sure about this
        if return_video_sample_fps:
            return images, process_info.pop("fps", 2.0), timestamps
        return images


class Eagle2_5_VLProcessorKwargs(ProcessingKwargs, total=False):
    # see processing_utils.ProcessingKwargs documentation for usage.
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "images_kwargs": {},
        "videos_kwargs": {"max_dynamic_tiles": 1},
    }


class Eagle2_5_VLProcessor(ProcessorMixin):

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = [
        "chat_template",
        "num_image_tokens",
        "vision_feature_select_strategy",
        "image_token",
        "video_token",
        "images_kwargs",
        "videos_kwargs",
        "text_kwargs",
    ]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        vision_feature_select_strategy=None,
        chat_template=None,
        image_token="<IMG_CONTEXT>",
        video_token="<IMG_CONTEXT>",
        tokens_per_tile=256,
        image_placeholder="image",
        video_placeholder="video",
        image_start_token="<img>",
        image_end_token="</img>",
        **kwargs,
    ):
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.image_token = (
            tokenizer.image_token if hasattr(tokenizer, "image_token") else image_token
        )
        self.video_token = (
            tokenizer.video_token if hasattr(tokenizer, "video_token") else video_token
        )
        self.image_token_id = (
            tokenizer.image_token_id
            if getattr(tokenizer, "image_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.image_token)
        )
        self.video_token_id = (
            tokenizer.video_token_id
            if getattr(tokenizer, "video_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.video_token)
        )
        self.image_placeholder = image_placeholder
        self.video_placeholder = video_placeholder
        self.tokens_per_tile = tokens_per_tile
        self.image_start_token = image_start_token
        self.image_end_token = image_end_token
        if "auto_map" in kwargs:
            self.auto_map = kwargs["auto_map"]
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def replace_media_placeholder(
        self, text, image_list, video_list, timestamps_list, fps_list, **output_kwargs
    ):

        num_of_images_in_this_sample = 0
        num_of_videos_in_this_sample = 0
        # Regular expression pattern to match formats like <image-1> or <video-2>
        pattern = re.compile(
            rf"<({self.image_placeholder}|{self.video_placeholder})-(\d+)>"
        )
        unified_frame_list = []

        video_min_dynamic_tiles = output_kwargs["videos_kwargs"].get(
            "min_dynamic_tiles", self.image_processor.min_dynamic_tiles
        )
        video_max_dynamic_tiles = output_kwargs["videos_kwargs"].get(
            "max_dynamic_tiles", self.image_processor.max_dynamic_tiles
        )
        video_use_thumbnail = output_kwargs["videos_kwargs"].get(
            "use_thumbnail", self.image_processor.use_thumbnail
        )

        tile_size = self.image_processor.size.get("height", 448)

        # Function to replace tags in a single text
        def replace_in_text(text):
            # repl callback function for each match replacement operation
            def repl(match):
                nonlocal unified_frame_list
                nonlocal num_of_images_in_this_sample
                nonlocal num_of_videos_in_this_sample
                media_type = match.group(1)  # 'image' or 'video'
                idx_in_list = int(match.group(2)) - 1  # Convert to list index (0-based)
                # Select the corresponding path based on media type
                idx_mapper = {
                    0: "first",
                    1: "second",
                    2: "third",
                    3: "fourth",
                    4: "fifth",
                    5: "sixth",
                    6: "seventh",
                    7: "eighth",
                    8: "ninth",
                    9: "tenth",
                }
                if media_type == "image":
                    image_inputs = self.image_processor(
                        images=[image_list[idx_in_list]],
                        videos=None,
                        **output_kwargs["images_kwargs"],
                    )
                    num_all_tiles = image_inputs["pixel_values"].shape[0]
                    special_placeholder = f"<image {idx_in_list+1}>{self.image_start_token}{self.image_token * num_all_tiles * self.tokens_per_tile}{self.image_end_token}"
                    unified_frame_list.append(image_inputs)
                    num_of_images_in_this_sample += 1

                elif media_type == "video":
                    video_inputs = self.image_processor(
                        images=None,
                        videos=[video_list[idx_in_list]],
                        **output_kwargs["videos_kwargs"],
                    )
                    num_all_tiles = video_inputs["pixel_values"].shape[0]
                    image_sizes = video_inputs["image_sizes"]
                    if timestamps_list is not None and -1 not in timestamps_list:
                        frame_timestamps = timestamps_list[idx_in_list]
                    else:
                        frame_timestamps = None
                    sampled_fps = (
                        fps_list[idx_in_list] if fps_list is not None else None
                    )

                    num_of_tiles_each_frame = [
                        self.get_number_tiles_based_on_image_size(
                            image_size,
                            video_min_dynamic_tiles,
                            video_max_dynamic_tiles,
                            video_use_thumbnail,
                            tile_size,
                        )
                        for image_size in image_sizes
                    ]
                    assert (
                        sum(num_of_tiles_each_frame) == num_all_tiles
                    ), f"The number of tiles in each frame is not equal to the total number of tiles: {sum(num_of_tiles_each_frame)} != {num_all_tiles}"

                    if frame_timestamps is not None:
                        assert len(frame_timestamps) == len(
                            num_of_tiles_each_frame
                        ), f"The number of timestamps is not equal to the number of frames: {len(frame_timestamps)} != {len(num_of_tiles_each_frame)}"
                        special_placeholder = [
                            f"Frame {i+1} sample at {frame_timestamps[i]:.2f}s: {self.image_start_token}{self.image_token * num_of_tiles * self.tokens_per_tile}{self.image_end_token}"
                            for i, num_of_tiles in enumerate(num_of_tiles_each_frame)
                        ]
                    else:
                        special_placeholder = [
                            f"Frame {i+1}: {self.image_start_token}{self.image_token * num_of_tiles * self.tokens_per_tile}{self.image_end_token}"
                            for i, num_of_tiles in enumerate(num_of_tiles_each_frame)
                        ]

                    if sampled_fps is not None:
                        special_placeholder = (
                            f"The {idx_mapper[idx_in_list]} video sampled with {sampled_fps:.2f} fps: "
                            + "".join(special_placeholder)
                        )
                    else:
                        special_placeholder = (
                            f"The {idx_mapper[idx_in_list]} video: "
                            + "".join(special_placeholder)
                        )
                    unified_frame_list.append(video_inputs)
                    num_of_videos_in_this_sample += 1
                else:
                    raise ValueError(f"Unknown media type: {media_type}")
                return special_placeholder

            return pattern.sub(repl, text)

        text = replace_in_text(text)
        if len(unified_frame_list) > 0:
            pixel_values = torch.cat(
                [frame["pixel_values"] for frame in unified_frame_list]
            )
            image_sizes = torch.cat(
                [frame["image_sizes"] for frame in unified_frame_list]
            )
        else:
            pixel_values = None
            image_sizes = None
        return (
            text,
            pixel_values,
            image_sizes,
            num_of_images_in_this_sample,
            num_of_videos_in_this_sample,
        )

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[
            TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]
        ] = None,
        audio=None,
        videos: VideoInput = None,
        **kwargs: Unpack[Eagle2_5_VLProcessorKwargs],
    ) -> BatchFeature:

        output_kwargs = self._merge_kwargs(
            Eagle2_5_VLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if isinstance(text, str):
            text_list = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise ValueError(
                "Invalid input text. Please provide a string, or a list of strings"
            )
        elif isinstance(text, list) and isinstance(text[0], str):
            text_list = text

        if images is None:
            images = []
        if videos is None:
            videos = []

        pixel_values_list = []
        image_sizes_list = []
        new_sample_list = []
        image_start_idx = 0
        video_start_idx = 0
        timestamps_batch = output_kwargs["videos_kwargs"].pop("timestamps", None)
        fps_batch = output_kwargs["videos_kwargs"].pop("fps", None)
        for sample in text_list:
            timestamps_list = (
                timestamps_batch[video_start_idx:]
                if timestamps_batch is not None
                else None
            )
            fps_list = fps_batch[video_start_idx:] if fps_batch is not None else None
            (
                sample,
                pixel_values,
                image_sizes,
                num_of_images_in_this_sample,
                num_of_videos_in_this_sample,
            ) = self.replace_media_placeholder(
                sample,
                images[image_start_idx:],
                videos[video_start_idx:],
                timestamps_list,
                fps_list,
                **output_kwargs,
            )
            new_sample_list.append(sample)
            if pixel_values is not None:
                pixel_values_list.append(pixel_values)
                image_sizes_list.append(image_sizes)
            image_start_idx += num_of_images_in_this_sample
            video_start_idx += num_of_videos_in_this_sample

        if len(pixel_values_list) > 0:
            image_inputs = {
                "pixel_values": torch.cat(pixel_values_list),
                "image_sizes": torch.cat(image_sizes_list),
            }
        else:
            image_inputs = {}
        video_inputs = {}
        text_inputs = self.tokenizer(new_sample_list, **output_kwargs["text_kwargs"])
        return BatchFeature(data={**text_inputs, **image_inputs, **video_inputs})

    def get_number_tiles_based_on_image_size(
        self,
        image_size: tuple,
        min_num: int,
        max_num: int,
        use_thumbnail: bool,
        tile_size: int,
    ) -> int:
        """
        Get the number of tiles based on the image size.
        """
        orig_height, orig_width = image_size
        aspect_ratio = orig_width / orig_height
        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self.image_processor.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, tile_size
        )
        tiles_num = target_aspect_ratio[0] * target_aspect_ratio[1]
        if use_thumbnail and tiles_num > 1:
            tiles_num += 1
        return tiles_num

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.batch_decode with CLIP->Llama
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->Llama
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.model_input_names
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

    # override to save video-config in a separate config file
    def save_pretrained(self, save_directory, **kwargs):
        if os.path.isfile(save_directory):
            raise ValueError(
                f"Provided path ({save_directory}) should be a directory, not a file"
            )
        os.makedirs(save_directory, exist_ok=True)

        outputs = super().save_pretrained(save_directory, **kwargs)
        return outputs

    # override to load video-config from a separate config file
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        processor = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        # if return_unused_kwargs a tuple is returned where the second element is 'unused_kwargs'
        if isinstance(processor, tuple):
            processor = processor[0]
        return processor

    # Copy from https://github.com/QwenLM/Qwen2.5-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py
    def process_vision_info(
        self,
        conversations: list[dict] | list[list[dict]],
        return_video_kwargs: bool = False,
    ) -> tuple[
        list[Image.Image] | None,
        list[torch.Tensor | list[Image.Image]] | None,
        Optional[dict],
    ]:

        vision_infos = self.extract_vision_info(conversations)
        ## Read images or videos
        image_inputs = []
        video_inputs = []
        video_sample_fps_list = []
        video_timestamps_list = []
        for vision_info in vision_infos:
            if "image" in vision_info or "image_url" in vision_info:
                image_inputs.append(fetch_image(vision_info))
            elif "video" in vision_info:
                video_input, video_sample_fps, video_timestamps = fetch_video(
                    vision_info, return_video_sample_fps=True
                )
                video_sample_fps_list.append(video_sample_fps)
                video_inputs.append(video_input)
                video_timestamps_list.append(video_timestamps)
            else:
                raise ValueError("image, image_url or video should in content.")
        if len(image_inputs) == 0:
            image_inputs = None
        if len(video_inputs) == 0:
            video_inputs = None
        if return_video_kwargs:
            return (
                image_inputs,
                video_inputs,
                {"fps": video_sample_fps_list, "timestamps": video_timestamps_list},
            )
        return image_inputs, video_inputs

    def extract_vision_info(
        self, conversations: list[dict] | list[list[dict]]
    ) -> list[dict]:
        vision_infos = []
        if isinstance(conversations[0], dict):
            conversations = [conversations]
        for conversation in conversations:
            for message in conversation:
                if isinstance(message["content"], list):
                    for ele in message["content"]:
                        if (
                            "image" in ele
                            or "image_url" in ele
                            or "video" in ele
                            or ele["type"] in ("image", "image_url", "video")
                        ):
                            vision_infos.append(ele)
        return vision_infos

    def py_apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False
    ):
        """
        Renders a chat conversation using a custom template with verification of tokens.

        The purpose is to check for the existence of tokens like "<image-1>" or "<video-1>"
        in the message text and skip adding them if they already exist.

        Args:
            messages (list): A list of message dictionaries. Each message should contain:
                - 'role': The role of the speaker (e.g., 'system', 'user', 'assistant').
                - 'content': Either a string or a list of content blocks. In the list each block may contain:
                      * 'type': The type of content, such as 'image' or 'video'.
                      * 'text': The actual text if present.
                      * Other keys such as 'image', 'image_url', or 'video'.
            add_generation_prompt (bool): If True, appends "<|im_start|>assistant" at the end of the rendered string.
            tokenize (bool): If True, tokenize the rendered string.
        Returns:
            str: The final rendered chat string according to the specified template.
        """
        assert not tokenize, "tokenize is not supported yet"
        result = ""
        image_count = 0
        video_count = 0

        message_text = ""
        for idx, message in enumerate(messages):
            if message.get("role") != "user":
                continue
            # If content is a string, simply output it.
            content = message.get("content")
            if isinstance(content, str):
                message_text += content
            elif isinstance(content, list):
                # Process each content item.
                for item in content:
                    # If the block is a dictionary and contains text, add it to message_text.
                    if isinstance(item, dict) and "text" in item:
                        message_text += item["text"]
                    # If an item is already a string in the list, add it directly.
                    elif isinstance(item, str):
                        message_text += item

        for idx, message in enumerate(messages):
            # If the first message is not from the system, prepend a default system message.
            if idx == 0 and message.get("role") != "system":
                result += "<|im_start|>system\n"
                result += "You are a helpful assistant.\n"
                result += "<|im_end|>\n"

            # Start the current message block with its role.
            result += f"<|im_start|>{message.get('role', '')}\n"
            content = message.get("content")

            # If content is a string, simply output it.
            if isinstance(content, str):
                result += content
                result += "<|im_end|>\n"
            else:
                # Process each content item.
                for item in content:
                    # Check if the item is an image (explicitly by type or by key presence).
                    if isinstance(item, dict) and (
                        item.get("type") == "image"
                        or "image" in item
                        or "image_url" in item
                    ):
                        image_count += 1
                        candidate_token = f"<image-{image_count}>"
                        # Only add the token if it is not already present in the collected text.
                        if candidate_token not in message_text:
                            result += candidate_token
                    # Check if the item is a video.
                    elif isinstance(item, dict) and (
                        item.get("type") == "video" or "video" in item
                    ):
                        video_count += 1
                        candidate_token = f"<video-{video_count}>"
                        # Only add the token if it is not already present.
                        if candidate_token not in message_text:
                            result += candidate_token
                    # If the item contains text, add it.
                    elif isinstance(item, dict) and "text" in item:
                        result += item["text"]
                    # If the item is a string (and not handled already), add it.
                    elif isinstance(item, str):
                        result += item
                result += "<|im_end|>\n"

        # Optionally add assistant generation prompt at the end.
        if add_generation_prompt:
            result += "<|im_start|>assistant\n"

        return result

    @classmethod
    def from_args_and_dict(cls, args, processor_dict: dict[str, Any], **kwargs):
        """
        Instantiates a type of [`~processing_utils.ProcessingMixin`] from a Python dictionary of parameters.

        Args:
            processor_dict (`Dict[str, Any]`):
                Dictionary that will be used to instantiate the processor object. Such a dictionary can be
                retrieved from a pretrained checkpoint by leveraging the
                [`~processing_utils.ProcessingMixin.to_dict`] method.
            kwargs (`Dict[str, Any]`):
                Additional parameters from which to initialize the processor object.

        Returns:
            [`~processing_utils.ProcessingMixin`]: The processor object instantiated from those
            parameters.
        """
        processor_dict = processor_dict.copy()
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

        # We have to pop up some unused (but specific) kwargs and then validate that it doesn't contain unused kwargs
        # If we don't pop, some specific kwargs will raise a warning
        if "processor_class" in processor_dict:
            del processor_dict["processor_class"]

        # if "auto_map" in processor_dict:
        #    del processor_dict["auto_map"]

        unused_kwargs = cls.validate_init_kwargs(
            processor_config=processor_dict, valid_kwargs=cls.valid_kwargs
        )
        processor = cls(*args, **processor_dict)

        # Update processor with kwargs if needed
        for key in set(kwargs.keys()):
            if hasattr(processor, key):
                setattr(processor, key, kwargs.pop(key))

        kwargs.update(unused_kwargs)
        logger.info(f"Processor {processor}")
        if return_unused_kwargs:
            return processor, kwargs
        else:
            return processor


def formalize_language(language: str) -> str:
    """
    1. Force lowercase
    2. Remove all punctuations
    """
    language = language.lower()
    language = re.sub(r"[^\w\s]", "", language)
    return language


def build_eagle_processor() -> ProcessorMixin:
    """
    Build Eagle processor directly without AutoProcessor.
    Loads tokenizer and image processor from configs manually - only necessary files.
    """
    from transformers import Qwen2TokenizerFast

    # Compute base path for eagle model files
    # Path to local eagle model files (relative to this file)

    # Load tokenizer with specific file paths (not directory)
    vocab_file = get_file("test_files/pytorch/Issac_groot/vocab.json")
    merges_file = get_file("test_files/pytorch/Issac_groot/merges.txt")
    tokenizer_config_file = get_file(
        "test_files/pytorch/Issac_groot/tokenizer_config.json"
    )

    tokenizer = Qwen2TokenizerFast(vocab_file=vocab_file, merges_file=merges_file)
    tokenizer.padding_side = "left"

    # Load and apply tokenizer config
    with open(tokenizer_config_file, "r") as f:
        tokenizer_config = json.load(f)
        if "model_max_length" in tokenizer_config:
            tokenizer.model_max_length = tokenizer_config["model_max_length"]

    # Add custom attributes needed by Eagle processor
    tokenizer.image_token = "<IMG_CONTEXT>"
    tokenizer.video_token = "<IMG_CONTEXT>"

    # Load chat template if exists
    chat_template = None
    chat_template_file = get_file("test_files/pytorch/Issac_groot/chat_template.json")
    if os.path.exists(chat_template_file):
        with open(chat_template_file, "r") as f:
            chat_data = json.load(f)
            chat_template = chat_data.get("chat_template")
            if chat_template:
                tokenizer.chat_template = chat_template

    # Load processor config
    processor_config_file = get_file(
        "test_files/pytorch/Issac_groot/processor_config.json"
    )
    with open(processor_config_file, "r") as f:
        processor_config = json.load(f)

    # Load image processor config
    image_processor_config_file = get_file(
        "test_files/pytorch/Issac_groot/preprocessor_config.json"
    )
    with open(image_processor_config_file, "r") as f:
        image_processor_config = json.load(f)

    # Create image processor directly
    image_processor = Eagle2_5_VLImageProcessorFast(**image_processor_config)

    # Create Eagle processor directly
    eagle_processor = Eagle2_5_VLProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        chat_template=chat_template,
        image_token=processor_config.get("image_token", "<IMG_CONTEXT>"),
        video_token=processor_config.get("video_token", "<IMG_CONTEXT>"),
        tokens_per_tile=processor_config.get("tokens_per_tile", 256),
        image_placeholder=processor_config.get("image_placeholder", "image"),
        video_placeholder=processor_config.get("video_placeholder", "video"),
        image_start_token=processor_config.get("image_start_token", "<img>"),
        image_end_token=processor_config.get("image_end_token", "</img>"),
    )

    return eagle_processor


def collate(features: List[dict], eagle_processor) -> dict:
    batch = {}
    keys = features[0].keys()

    for key in keys:
        values = [elem[key] for elem in features]

        if key == "eagle_content":
            text_list = []
            image_inputs = []
            for v in values:
                curr_text_list = v["text_list"]
                curr_image_inputs = v["image_inputs"]
                text_list += curr_text_list
                image_inputs += curr_image_inputs
            eagle_inputs = eagle_processor(
                text=text_list, images=image_inputs, return_tensors="pt", padding=True
            )
            for k, v in eagle_inputs.items():
                k = "eagle_" + k
                batch[k] = v
        elif key in ("pixel_values", "image_grid_thw", "attention_mask", "input_ids"):
            # Concat in existing batch dimension.
            batch[key] = torch.cat(values)
        else:
            # state, state_mask, action and action_mask.
            # Stack to form the batch dimension.
            batch[key] = torch.from_numpy(np.stack(values))
    return batch


class DefaultDataCollator(DataCollatorMixin):
    def __init__(self):
        super().__init__()
        self.eagle_processor = build_eagle_processor()

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        return collate(features, self.eagle_processor)


class GR00TTransform(InvertibleModalityTransform):

    # -- We inherit from ModalityTransform, so we keep apply_to as well --
    apply_to: list[str] = Field(
        default_factory=list,
        description="Not used in this transform, kept for compatibility.",
    )
    training: bool = Field(
        default=True, description="Whether to apply the transform in training mode."
    )
    formalize_language: bool = Field(
        default=False, description="Formalize language if True."
    )
    embodiment_tag_mapping: dict[str, int] = Field(
        description="The projector index of each embodiment tag.",
        default=EMBODIMENT_TAG_MAPPING,
    )
    language_dropout_prob: float = Field(
        default=0.0,
        description="Dropout probability for language.",
    )

    # Private attributes to keep track of shapes/dimensions across apply/unapply
    _language_key: Optional[list[str]] = PrivateAttr(default=None)

    eagle_processor: ProcessorMixin = Field(default=None)

    # XEmbDiT arguments
    default_instruction: str = Field(default="Perform the default behavior.")
    max_state_dim: int
    max_action_dim: int
    state_horizon: int
    action_horizon: int

    max_length: int = 512
    embodiment_tag: EmbodimentTag | None = None

    @model_validator(mode="after")
    def build_processor(self):
        """Build the eagle processor if not already set."""
        if self.eagle_processor is None:
            self.eagle_processor = build_eagle_processor()
        return self

    def set_metadata(self, dataset_metadata: DatasetMetadata):
        """Set the metadata for the transform."""
        super().set_metadata(dataset_metadata)
        self.embodiment_tag = dataset_metadata.embodiment_tag

    def get_embodiment_tag(self) -> int:
        """Get the embodiment tag from the data."""
        assert (
            self.embodiment_tag is not None
        ), "Embodiment tag not set. Please call set_metadata first."
        return self.embodiment_tag_mapping[self.embodiment_tag.value]

    def check_keys_and_batch_size(self, data):
        grouped_keys = {}
        for key in data.keys():
            if "annotation" in key:
                modality = "language"
            else:
                try:
                    modality, _ = key.split(".")
                except:  # noqa: E722
                    modality = "others"  # will contain the video, state, and action
            if modality not in grouped_keys:
                grouped_keys[modality] = []
            grouped_keys[modality].append(key)

        # Check if data is already preprocessed (has eagle_* keys instead of video key)
        is_preprocessed = "video" not in data and any(
            key.startswith("eagle_") for key in data.keys()
        )

        if is_preprocessed:
            # Data is already preprocessed, use _was_batched or infer from eagle keys
            if "_was_batched" in data:
                # Handle numpy array or boolean
                was_batched = data["_was_batched"]
                if hasattr(was_batched, "item"):
                    was_batched = was_batched.item()
                is_batched = bool(was_batched)
            else:
                # Infer from eagle_attention_mask or eagle_pixel_values shape
                if "eagle_attention_mask" in data:
                    mask = data["eagle_attention_mask"]
                    is_batched = mask.ndim > 1 and mask.shape[0] > 1
                elif "eagle_pixel_values" in data:
                    pixels = data["eagle_pixel_values"]
                    is_batched = pixels.ndim > 3 and pixels.shape[0] > 1
                else:
                    # Default to not batched if we can't determine
                    is_batched = False
            batch_size = data.get(
                "eagle_attention_mask", data.get("eagle_pixel_values", None)
            )
            if batch_size is not None:
                batch_size = batch_size.shape[0] if is_batched else 1
            else:
                batch_size = 1
        else:
            # Use video key to determine batch size.
            if "video" not in data:
                # No video key and not preprocessed - might be an error, but try to infer
                # Check if we have state or other keys to infer batch size
                if "state" in data:
                    state = data["state"]
                    is_batched = state.ndim > 1 and state.shape[0] > 1
                    batch_size = state.shape[0] if is_batched else 1
                else:
                    # Default to not batched
                    is_batched = False
                    batch_size = 1
            else:
                video_ndim = data["video"].ndim
                if video_ndim == 5:  # Interpret as [T, V, H, W, C]
                    is_batched = False
                    batch_size = 1
                elif video_ndim == 6:  # Interpret as [B, T, V, H, W, C]
                    is_batched = True
                    batch_size = data["video"].shape[0]
                else:
                    raise ValueError(
                        f"Unsupported video number of dimensions: {video_ndim}"
                    )

        # Handle language
        if "language" in grouped_keys:
            language_keys = grouped_keys["language"]
            assert len(language_keys) == 1, f"{language_keys=}"
            self._language_key = language_keys[0]
        return is_batched, batch_size

    def _apply_vlm_processing(self, batch: dict) -> BatchFeature:
        """
        Args:
            batch:
                video: [V, T, C, H, W]
        Returns: required input with the format `BatchFeature`
        """
        # TODO(YL, FH): check if this is correct
        images = batch["images"]  # [V, T, C, H, W]
        images.shape[0]

        np_images = rearrange(images, "v t c h w -> (t v) c h w")
        text_content = []

        # handle language
        lang = batch["language"]
        if isinstance(lang, list):
            lang = lang[0]
        text_content.append({"type": "text", "text": lang})

        eagle_images = [Image.fromarray(np.transpose(v, (1, 2, 0))) for v in np_images]
        eagle_image = [{"type": "image", "image": img} for img in eagle_images]
        eagle_conversation = [
            {
                "role": "user",
                "content": eagle_image + text_content,
            }
        ]

        text_list = [
            self.eagle_processor.apply_chat_template(
                eagle_conversation, tokenize=False, add_generation_prompt=True
            )
        ]
        image_inputs, video_inputs = self.eagle_processor.process_vision_info(
            eagle_conversation
        )
        eagle_content = {
            "image_inputs": image_inputs,
            "video_inputs": video_inputs,
            "text_list": text_list,
        }
        inputs = {}
        inputs["eagle_content"] = eagle_content
        return inputs

    def _prepare_video(self, data: dict):
        """Process, stack, and pad images from data['video']."""
        ## TODO(YL, FH): check if this is correct
        images = rearrange(
            data["video"],
            "t v h w c -> v t c h w",
        )
        return images

    def _prepare_language(self, data: dict):
        """Tokenize data['language'] (or default_instruction if missing)."""
        if self._language_key is not None:
            raw_language = data[self._language_key]
            if isinstance(raw_language, list):
                raw_language = raw_language[0]

            # Language dropout
            if self.training and self.language_dropout_prob > 1e-9:
                if random.random() < self.language_dropout_prob:
                    raw_language = self.default_instruction
        else:
            raw_language = self.default_instruction
        return raw_language

    def _prepare_state(self, data: dict):
        """
        Gathers final state from data['state'], then pads to max_state_dim.
        Return (state, state_mask, n_state_tokens).
        """
        if "state" not in data:
            state = np.zeros((self.state_horizon, self.max_state_dim))
            state_mask = np.zeros((self.state_horizon, self.max_state_dim), dtype=bool)
            n_state_tokens = self.state_horizon
            return state, state_mask, n_state_tokens

        state = data["state"]
        assert (
            state.shape[0] == self.state_horizon
        ), f"{state.shape=}, {self.state_horizon=}"

        n_state_dims = state.shape[-1]

        # Instead of asserting, just take the first max_state_dim dimensions if needed
        if n_state_dims > self.max_state_dim:
            state = state[:, : self.max_state_dim]
            n_state_dims = self.max_state_dim
        else:
            # Pad up to max_state_dim if smaller
            state = np.pad(
                state, ((0, 0), (0, self.max_state_dim - n_state_dims)), "constant"
            )

        # Create mask for real state dims
        state_mask = np.zeros_like(state).astype(bool)
        state_mask[:, :n_state_dims] = True

        # We only have 1 "proprio" token to represent the entire state
        n_state_tokens = state.shape[0]
        return state, state_mask, n_state_tokens

    def _prepare_action(self, data: dict):
        """
        Pad to max_action_dim, return masks.
        """
        if "action" not in data:
            actions = np.zeros((self.action_horizon, self.max_action_dim))
            actions_mask = np.zeros(
                (self.action_horizon, self.max_action_dim), dtype=bool
            )
            n_action_tokens = self.action_horizon
            return actions, actions_mask, n_action_tokens

        actions = data["action"]
        assert (
            actions.shape[0] == self.action_horizon
        ), f"{actions.shape=}, {self.action_horizon=}"

        n_action_tokens = actions.shape[0]  # T
        n_action_dims = actions.shape[1]

        assert (
            n_action_dims <= self.max_action_dim
        ), f"Action dim {n_action_dims} exceeds max allowed {self.max_action_dim}."

        # Pad the channel dimension
        actions = np.pad(
            actions, ((0, 0), (0, self.max_action_dim - n_action_dims)), "constant"
        )

        # Create mask: [T, max_action_dim]
        actions_mask = np.zeros((n_action_tokens, self.max_action_dim), dtype=bool)
        actions_mask[:, :n_action_dims] = True

        return actions, actions_mask, n_action_tokens

    def apply_single(self, data: dict) -> dict:
        # Check if data is already preprocessed (has eagle_* keys instead of video key)
        is_preprocessed = "video" not in data and any(
            key.startswith("eagle_") for key in data.keys()
        )

        if is_preprocessed:
            # Data is already preprocessed, return as-is (or with minimal processing)
            # The eagle_* keys are already in the correct format
            return data

        transformed_data = {}

        # 1) Prepare video and language with vlm processing.
        images = self._prepare_video(data)
        images = images.astype(np.uint8)
        language = self._prepare_language(data)
        batch_data = {"images": images, "language": language}
        vlm_outputs = self._apply_vlm_processing(batch_data)

        # 2) Prepare state
        state, state_mask, _ = self._prepare_state(data)
        transformed_data["state"] = state
        transformed_data["state_mask"] = state_mask

        if self.training:
            # 3) Prepare actions
            transformed_data["segmentation_target"] = np.zeros((2,))
            transformed_data["segmentation_target_mask"] = np.zeros((1,))
            transformed_data["has_real_action"] = np.ones((), dtype=bool)
            actions, actions_mask, _ = self._prepare_action(data)
            transformed_data["action"] = actions
            transformed_data["action_mask"] = actions_mask

        for k, v in vlm_outputs.items():
            assert (
                k not in transformed_data
            ), f"Key {k} already exists in transformed_data."
            transformed_data[k] = v

        transformed_data["embodiment_id"] = self.get_embodiment_tag()

        if self.training:
            action_and_mask_keys = ["action", "action_mask"]
            assert all(
                transformed_data[key].shape == transformed_data["action"].shape
                for key in action_and_mask_keys
            ), f"Shape mismatch: {[(key, transformed_data[key].shape) for key in action_and_mask_keys]}"

        return transformed_data

    def apply_batch(self, data: dict, batch_size: int) -> dict:
        # Split on batch dimension.
        data_split = [
            tree.map_structure(lambda x: x[i], data) for i in range(batch_size)
        ]
        # Process each element.
        data_split_processed = [self.apply_single(elem) for elem in data_split]
        return collate(data_split_processed, self.eagle_processor)

    def apply(self, data: dict) -> dict:
        is_batched, batch_size = self.check_keys_and_batch_size(data)
        if is_batched:
            return self.apply_batch(data, batch_size)
        else:
            return self.apply_single(data)

    def unapply(self, data: dict) -> dict:
        # Leave as is so that ConcatTransform can split the values
        return data

    def __call__(self, data: dict) -> dict:
        return self.apply(data)


class Gr00tPolicyModule(nn.Module):
    def __init__(
        self,
        model_path: str,
        embodiment_tag: Union[str, EmbodimentTag],
        modality_config: Dict[str, ModalityConfig],
        modality_transform: ComposedModalityTransform,
        denoising_steps: Optional[int] = None,
        device: Union[int, str] = "cpu",
    ):
        """
        Initialize the Gr00tPolicyModule.

        Args:
            model_path: Path to the model checkpoint directory or huggingface hub id
            embodiment_tag: The embodiment tag for the model
            modality_config: The modality config for the model
            modality_transform: The modality transform for the model
            denoising_steps: Number of denoising steps for diffusion inference
            device: Device to run the model on (cpu, cuda, or device index)
        """
        super().__init__()

        # Create the underlying policy
        self.policy = Gr00tPolicy(
            model_path=model_path,
            embodiment_tag=embodiment_tag,
            modality_config=modality_config,
            modality_transform=modality_transform,
            denoising_steps=denoising_steps,
            device=device,
        )

        # Register the model as a submodule so parameters are tracked
        self.model = self.policy.model

        # Store config
        self._device = device
        self._embodiment_tag = embodiment_tag

    def preprocess(self, observations: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Preprocess raw observations into normalized tensors.

        Args:
            observations: Raw observations dict with video, state, etc.

        Returns:
            Dictionary of normalized tensors ready for the model
        """
        # Create a copy to avoid mutating input
        obs_copy = observations.copy()

        # Handle batching
        is_batch = self.policy._check_state_is_batched(obs_copy)
        if not is_batch:
            obs_copy = unsqueeze_dict_values(obs_copy)

        # Convert to numpy arrays
        for k, v in obs_copy.items():
            if not isinstance(v, np.ndarray):
                obs_copy[k] = np.array(v)

        # Apply transforms
        normalized_input = self.policy.apply_transforms(obs_copy)

        # Store whether input was batched for postprocessing
        normalized_input["_was_batched"] = is_batch

        return normalized_input

    def postprocess(
        self, action: Dict[str, Any], was_batched: bool = True
    ) -> Dict[str, Any]:
        """
        Postprocess normalized action back to original format.

        Args:
            action: Normalized action from the model
            was_batched: Whether the original input was batched

        Returns:
            Unnormalized action in original format
        """
        unnormalized_action = self.policy._get_unnormalized_action(action)

        if not was_batched:
            unnormalized_action = squeeze_dict_values(unnormalized_action)

        return unnormalized_action

    def forward_from_normalized(
        self, normalized_input: Dict[str, torch.Tensor], return_raw: bool = False
    ) -> Union[Dict[str, Any], torch.Tensor]:
        """
        Forward pass from already normalized/preprocessed inputs.

        Args:
            normalized_input: Pre-processed input tensors
            return_raw: If True, return raw action tensor. If False, return dict.

        Returns:
            Action dictionary or raw action tensor
        """
        # Extract batching info if available
        was_batched = normalized_input.pop("_was_batched", True)

        # Get normalized action from model
        normalized_action = self.policy._get_action_from_normalized_input(
            normalized_input
        )

        if return_raw:
            return normalized_action

        # Postprocess to unnormalized action
        unnormalized_action = self.postprocess(normalized_action, was_batched)

        return unnormalized_action

    def forward(
        self,
        observations: Optional[Dict[str, Any]] = None,
        preprocessed: bool = False,
        return_raw: bool = False,
        **kwargs,
    ) -> Union[Dict[str, Any], torch.Tensor]:
        """
        Forward pass through the policy.

        Args:
            observations: Either raw observations or preprocessed normalized inputs.
                         If None, will be constructed from kwargs.
            preprocessed: If True, observations are already normalized
            return_raw: If True, return raw action tensor. If False, return dict.
            **kwargs: Additional keyword arguments that will be merged into observations
                     if observations is None.

        Returns:
            Action dictionary or raw action tensor

        Example:
            >>> # Raw observations
            >>> action = policy_module(observations)
            >>>
            >>> # Preprocessed observations
            >>> normalized = policy_module.preprocess(observations)
            >>> action = policy_module(normalized, preprocessed=True)
            >>>
            >>> # Get raw action tensor (no postprocessing)
            >>> raw_action = policy_module(observations, return_raw=True)
        """
        # Handle case where observations are passed as keyword arguments
        if observations is None:
            observations = kwargs
        elif kwargs:
            # Merge kwargs into observations if both are provided
            observations = {**observations, **kwargs}

        if preprocessed:
            return self.forward_from_normalized(observations, return_raw=return_raw)
        else:
            # Preprocess
            normalized_input = self.preprocess(observations)
            # Forward
            return self.forward_from_normalized(normalized_input, return_raw=return_raw)

    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convenience method matching the original policy API.

        Args:
            observations: Raw observations

        Returns:
            Action dictionary
        """
        return self.policy.get_action(observations)

    @property
    def modality_config(self) -> Dict[str, ModalityConfig]:
        """Get the modality configuration."""
        return self.policy.modality_config

    @property
    def embodiment_tag(self) -> EmbodimentTag:
        """Get the embodiment tag."""
        return self.policy.embodiment_tag

    @property
    def denoising_steps(self) -> int:
        """Get the number of denoising steps."""
        return self.policy.denoising_steps

    def eval(self):
        """Set the model to evaluation mode."""
        super().eval()
        self.policy._modality_transform.eval()
        return self

    def train(self, mode: bool = True):
        """Set the model to training mode."""
        super().train(mode)
        if mode:
            self.policy._modality_transform.train()
        else:
            self.policy._modality_transform.eval()
        return self
