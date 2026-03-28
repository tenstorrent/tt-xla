# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ACT (Action Chunking Transformer) model loader implementation for action prediction.
"""

from __future__ import annotations

from typing import Optional

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


class ACTInferenceWrapper(torch.nn.Module):
    """Wraps ACTPolicy to use select_action (inference) instead of forward (training).

    ACTPolicy.forward() computes training loss; for inference we need select_action
    which returns action chunks.
    """

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, batch: dict) -> torch.Tensor:
        """Run inference via select_action. Returns action tensor."""
        return self.policy.select_action(batch)


class ModelVariant(StrEnum):
    """Available ACT model variants."""

    ALOHA_SIM_TRANSFER_CUBE_HUMAN = "aloha_sim_transfer_cube_human"


class ModelLoader(ForgeModel):
    """ACT model loader implementation for action prediction tasks."""

    _VARIANTS = {
        ModelVariant.ALOHA_SIM_TRANSFER_CUBE_HUMAN: ModelConfig(
            pretrained_model_name="lerobot/act_aloha_sim_transfer_cube_human",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ALOHA_SIM_TRANSFER_CUBE_HUMAN

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.config = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="ACT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_ACTION_PREDICTION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, device: str = "cpu", **kwargs):
        from lerobot.policies.act.modeling_act import ACTPolicy

        model = ACTPolicy.from_pretrained(
            self._variant_config.pretrained_model_name, **kwargs
        )
        model.to(device)
        if dtype_override is not None:
            model = model.to(dtype=dtype_override)
        model.eval()
        self.config = model.config
        return ACTInferenceWrapper(model)

    def load_inputs(
        self, dtype_override=None, batch_size: int = 1, device: str = "cpu"
    ):
        from lerobot.configs.types import FeatureType

        if self.config is None:
            from lerobot.policies.act.configuration_act import ACTConfig

            self.config = ACTConfig.from_pretrained(
                self._variant_config.pretrained_model_name
            )

        observation = {}
        for key, feature in (self.config.input_features or {}).items():
            if not key.startswith("observation."):
                continue
            if feature.type == FeatureType.VISUAL:
                channels, height, width = feature.shape
                observation[key] = torch.rand(batch_size, channels, height, width)
            elif feature.type in (FeatureType.STATE, FeatureType.ENV):
                observation[key] = torch.rand(batch_size, *feature.shape)

        if dtype_override is not None:
            for key in observation:
                if observation[key].dtype.is_floating_point:
                    observation[key] = observation[key].to(dtype_override)

        return {"batch": observation}
