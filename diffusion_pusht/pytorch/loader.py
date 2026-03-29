# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Diffusion Policy PushT model loader for tt_forge_models.

Diffusion Policy uses a denoising U-Net conditioned on environment keypoints
and agent state to predict action sequences for robotic control tasks.
The model iteratively denoises through the U-Net to produce actions.

Reference: https://huggingface.co/lerobot/diffusion_pusht_keypoints
"""

from typing import Optional

import torch
from lerobot.policies.diffusion import DiffusionPolicy

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Diffusion Policy PushT model variants."""

    KEYPOINTS = "keypoints"


class ModelLoader(ForgeModel):
    """Diffusion Policy PushT model loader.

    Loads the Diffusion Policy model trained on the PushT keypoints
    environment for robotic action prediction. The model predicts
    2D agent actions from environment state and agent position.
    """

    _VARIANTS = {
        ModelVariant.KEYPOINTS: ModelConfig(
            pretrained_model_name="lerobot/diffusion_pusht_keypoints",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.KEYPOINTS

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.policy = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="DiffusionPushT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_ACTION_PREDICTION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        dtype = dtype_override if dtype_override is not None else torch.float32

        self.policy = DiffusionPolicy.from_pretrained(
            self._variant_config.pretrained_model_name,
        )
        self.policy.eval()

        return self.policy

    def load_inputs(self, dtype_override=None, **kwargs):
        dtype = dtype_override if dtype_override is not None else torch.float32

        batch_size = 1
        n_obs_steps = 2

        # observation.environment_state: 16-dim keypoints of the T-block
        environment_state = torch.randn((batch_size, n_obs_steps, 16), dtype=dtype)
        # observation.state: 2-dim agent position (x, y)
        state = torch.randn((batch_size, n_obs_steps, 2), dtype=dtype)

        return {
            "observation.environment_state": environment_state,
            "observation.state": state,
        }

    def unpack_forward_output(self, output):
        if isinstance(output, dict) and "action" in output:
            return output["action"]
        elif isinstance(output, torch.Tensor):
            return output
        elif isinstance(output, tuple):
            return output[0]
        return output
