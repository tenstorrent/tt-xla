# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
NoobAI XL (Laxhar/noobai-XL-1.0) model loader implementation.

NoobAI XL is a Stable Diffusion XL text-to-image model fine-tuned on Danbooru
and E621 datasets, based on Illustrious XL.

Available variants:
- NOOBAI_XL_1_0: Laxhar/noobai-XL-1.0 text-to-image generation
"""

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
from .src.model_utils import load_pipe


REPO_ID = "Laxhar/noobai-XL-1.0"


class ModelVariant(StrEnum):
    """Available NoobAI XL model variants."""

    NOOBAI_XL_1_0 = "noobai-XL-1.0"


class ModelLoader(ForgeModel):
    """NoobAI XL model loader implementation."""

    _VARIANTS = {
        ModelVariant.NOOBAI_XL_1_0: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.NOOBAI_XL_1_0

    # Shared configuration parameters
    prompt = "masterpiece, best quality, 1girl, solo, standing, outdoors, blue sky"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="NoobAI XL",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the NoobAI XL pipeline.

        Returns:
            DiffusionPipeline: The NoobAI XL pipeline instance.
        """
        self.pipeline = load_pipe(self._variant_config.pretrained_model_name)

        if dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype_override)

        return self.pipeline

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the NoobAI XL model.

        Returns:
            list: Input tensors that can be fed to the model:
                - latent_model_input (torch.Tensor): Latent input for the UNet
                - timestep (torch.Tensor): Timestep tensor
                - prompt_embeds (torch.Tensor): Encoded prompt embeddings
                - added_cond_kwargs (dict): Additional conditioning inputs
        """
        from ...stable_diffusion_xl.pytorch.src.model_utils import (
            stable_diffusion_preprocessing_xl,
        )

        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        (
            latent_model_input,
            timesteps,
            prompt_embeds,
            timestep_cond,
            added_cond_kwargs,
            add_time_ids,
        ) = stable_diffusion_preprocessing_xl(self.pipeline, self.prompt)

        if dtype_override:
            latent_model_input = latent_model_input.to(dtype_override)
            timesteps = timesteps.to(dtype_override)
            prompt_embeds = prompt_embeds.to(dtype_override)

        return [latent_model_input, timesteps, prompt_embeds, added_cond_kwargs]
