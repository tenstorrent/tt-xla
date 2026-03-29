# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Pi-0 FAST model loader implementation for action prediction tasks
"""
import torch
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


class ModelVariant(StrEnum):
    """Available Pi-0 FAST model variants."""

    LIBERO_V044 = "pi0fast_libero_v044"


class ModelLoader(ForgeModel):
    """Pi-0 FAST model loader implementation for action prediction task."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.LIBERO_V044: ModelConfig(
            pretrained_model_name="lerobot/pi0fast-libero-v044",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.LIBERO_V044

    def __init__(
        self,
        variant: Optional[ModelVariant] = None,
    ):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="pi_0_fast",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_ACTION_PREDICTION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the Pi-0 FAST model instance with default settings.

        Returns:
            torch.nn.Module: The Pi-0 FAST Policy instance.
        """

        from .src.model import get_custom_pi0fast_policy

        self.pretrained_model_name = self._variant_config.pretrained_model_name
        self.pi_0_fast = get_custom_pi0fast_policy(self.pretrained_model_name)
        self.pi_0_fast.eval()
        return self.pi_0_fast

    def load_inputs(self, dtype_override=None, episode_index=0):
        """
        Load and preprocess inputs for action sampling.
        Returns images, image masks, language tokens, language masks, and state.
        """
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from .src.model import preprocess_for_sampling

        self.preprocess, self.postprocess_fn = make_pre_post_processors(
            self.pi_0_fast.config,
            self.pretrained_model_name,
            preprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        dataset = LeRobotDataset("lerobot/libero")
        frame_index = dataset.meta.episodes["dataset_from_index"][episode_index]
        frame = dict(dataset[frame_index])
        batch = self.preprocess(frame)
        (
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
        ) = self.pi_0_fast.preprocess_for_sampling(batch)

        return images, img_masks, lang_tokens, lang_masks, state

    def postprocess(self, pred_action):
        """Apply postprocessing to predicted actions."""
        return self.postprocess_fn(pred_action)
