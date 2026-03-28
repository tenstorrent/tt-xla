# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
X-VLA model loader implementation for vision-language-action prediction.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


def _setup_policies_namespace() -> None:
    """Register lerobot.policies in sys.modules so subpackage imports work when this loader
    is imported outside the normal lerobot package context (e.g. via tt-forge-models dynamic
    import). Without this, 'from lerobot.policies.xvla...' can fail with import errors.
    """
    spec = importlib.util.find_spec("lerobot")
    if spec is None or spec.origin is None:
        return
    policies_path = Path(spec.origin).resolve().parent / "policies"
    if not policies_path.exists():
        return
    if "lerobot.policies" in sys.modules:
        return
    policies_module = types.ModuleType("lerobot.policies")
    policies_module.__path__ = [str(policies_path)]
    sys.modules["lerobot.policies"] = policies_module


class XVLAInferenceWrapper(torch.nn.Module):
    """Wraps XVLAPolicy to use predict_action_chunk (inference) instead of forward (training).

    XVLAPolicy.forward() computes training loss; for inference we use predict_action_chunk.
    See: https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/xvla/modeling_xvla.py
    """

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, batch: dict) -> torch.Tensor:
        """Run inference via predict_action_chunk. Returns action tensor (B, n_steps, action_dim)."""
        return self.policy.predict_action_chunk(batch)


class ModelVariant(StrEnum):
    """Available X-VLA model variants."""

    XVLA_BASE = "xvla_base"


class ModelLoader(ForgeModel):
    """X-VLA model loader implementation for vision-language-action prediction tasks."""

    _VARIANTS = {
        ModelVariant.XVLA_BASE: ModelConfig(
            pretrained_model_name="lerobot/xvla-base",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.XVLA_BASE

    sample_task = "pick the red block"
    robot_type = ""

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.preprocess = None
        self.config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="X-VLA",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_ACTION_PREDICTION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processors(self, device: torch.device):
        _setup_policies_namespace()
        import lerobot.policies.xvla.processor_xvla  # noqa: F401
        from lerobot.processor import PolicyProcessorPipeline

        self.preprocess = PolicyProcessorPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            config_filename="policy_preprocessor.json",
            overrides={"device_processor": {"device": str(device)}},
        )

    def load_model(self, *, dtype_override=None, device: str = "cpu", **kwargs):
        _setup_policies_namespace()
        from lerobot.policies.xvla.modeling_xvla import XVLAPolicy

        model = XVLAPolicy.from_pretrained(
            self._variant_config.pretrained_model_name, **kwargs
        )
        model.to(device)
        model = model.to(dtype=torch.float32)
        model.eval()
        self.config = model.config
        if self.preprocess is None:
            self._load_processors(torch.device(device))
        return XVLAInferenceWrapper(model)

    def load_inputs(
        self, dtype_override=None, batch_size: int = 1, device: str = "cpu"
    ):
        _setup_policies_namespace()
        from lerobot.policies.xvla.configuration_xvla import XVLAConfig
        from lerobot.policies.utils import prepare_observation_for_inference

        if self.config is None:
            self.config = XVLAConfig.from_pretrained(
                self._variant_config.pretrained_model_name
            )

        if self.preprocess is None:
            self._load_processors(torch.device(device))

        dummy_observation = build_dummy_observation(self.config.input_features or {})
        obs_frame = prepare_observation_for_inference(
            observation=dummy_observation,
            device=torch.device(device),
            task=self.sample_task,
            robot_type=self.robot_type,
        )

        inputs = self.preprocess(obs_frame)

        if batch_size > 1:
            for key, value in inputs.items():
                if torch.is_tensor(value) and value.dim() > 0:
                    inputs[key] = value.repeat_interleave(batch_size, dim=0)

        return {"batch": inputs}

    def unpack_forward_output(self, fwd_output):
        """predict_action_chunk returns action tensor (B, n_steps, action_dim) directly."""
        return fwd_output


def build_dummy_observation(input_features: dict) -> dict[str, np.ndarray]:
    from lerobot.configs.types import FeatureType

    observation: dict[str, np.ndarray] = {}
    for key, feature in input_features.items():
        if not key.startswith("observation."):
            continue
        if feature.type == FeatureType.VISUAL:
            channels, height, width = feature.shape
            observation[key] = np.zeros((height, width, channels), dtype=np.uint8)
        elif feature.type in (FeatureType.STATE, FeatureType.ENV):
            observation[key] = np.zeros(feature.shape, dtype=np.float32)
    return observation
