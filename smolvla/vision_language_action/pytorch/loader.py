# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SmolVLA model loader implementation for action prediction.
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
    import). Without this, 'from lerobot.policies.smolvla...' can fail with import errors.
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


class SmolVLAInferenceWrapper(torch.nn.Module):
    """Wraps SmolVLAPolicy to use predict_action_chunk (inference) instead of forward (training).

    SmolVLAPolicy.forward() computes training loss; for inference we use predict_action_chunk.
    See: https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/smolvla/modeling_smolvla.py
    """

    def __init__(self, policy: SmolVLAPolicy):
        super().__init__()
        self.policy = policy

    def forward(self, batch: dict) -> torch.Tensor:
        """Run inference via predict_action_chunk. Returns action tensor (B, n_steps, action_dim)."""
        return self.policy.predict_action_chunk(batch)


class ModelVariant(StrEnum):
    """Available SmolVLA model variants."""

    SMOLVLA_BASE = "smolvla_base"
    SMOLVLA_LIBERO = "smolvla_libero"


class ModelLoader(ForgeModel):
    """SmolVLA model loader implementation for action prediction tasks."""

    _VARIANTS = {
        ModelVariant.SMOLVLA_BASE: ModelConfig(
            pretrained_model_name="lerobot/smolvla_base",
        ),
        ModelVariant.SMOLVLA_LIBERO: ModelConfig(
            pretrained_model_name="HuggingFaceVLA/smolvla_libero",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SMOLVLA_BASE

    sample_task = "pick the red block"
    robot_type = ""

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.preprocess = None
        self.config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        group = (
            ModelGroup.VULCAN
            if variant == ModelVariant.SMOLVLA_LIBERO
            else ModelGroup.RED
        )
        return ModelInfo(
            model="SmolVLA",
            variant=variant,
            group=group,
            task=ModelTask.MM_ACTION_PREDICTION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processors(self, device: torch.device):
        _setup_policies_namespace()
        import lerobot.policies.smolvla.processor_smolvla  # noqa: F401
        from lerobot.processor import PolicyProcessorPipeline

        self.preprocess = PolicyProcessorPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            config_filename="policy_preprocessor.json",
            overrides={"device_processor": {"device": str(device)}},
        )

    def load_model(self, *, dtype_override=None, device: str = "cpu", **kwargs):
        _setup_policies_namespace()
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        # SmolVLA: always use float32. torch.compile/inductor on CPU has dtype consistency
        # issues with bfloat16 (mat1/mat2 mismatch). Pretrained weights load as bfloat16;
        # preprocess produces float32 - explicit float32 ensures model and inputs match.
        model = SmolVLAPolicy.from_pretrained(
            self._variant_config.pretrained_model_name, **kwargs
        )
        model.to(device)
        model = model.to(dtype=torch.float32)
        model.eval()
        self.config = model.config
        if self.preprocess is None:
            self._load_processors(torch.device(device))
        # Wrap so model(**inputs) runs predict_action_chunk (inference) not forward (training).
        return SmolVLAInferenceWrapper(model)

    def load_inputs(
        self, dtype_override=None, batch_size: int = 1, device: str = "cpu"
    ):
        _setup_policies_namespace()
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.utils import prepare_observation_for_inference

        if self.config is None:
            self.config = SmolVLAConfig.from_pretrained(
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
