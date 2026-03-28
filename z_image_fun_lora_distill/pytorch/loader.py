# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Z-Image-Fun-Lora-Distill model loader implementation.

Distill LoRA adapter for the Z-Image diffusion transformer that distills both
inference steps and classifier-free guidance (CFG) for fast text-to-image
generation. Trained from scratch (not based on Z-Image-Turbo weights).

Available variants:
- DISTILL_8_STEPS: alibaba-pai/Z-Image-Fun-Lora-Distill 8-step distilled LoRA
"""

from typing import Optional, Dict, Any

import torch
from diffusers import DiffusionPipeline

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


BASE_MODEL_ID = "alibaba-pai/Z-Image"


class ModelVariant(StrEnum):
    """Available Z-Image-Fun-Lora-Distill model variants."""

    DISTILL_8_STEPS = "Distill_8_Steps"


class ModelLoader(ForgeModel):
    """Z-Image-Fun-Lora-Distill model loader implementation."""

    _VARIANTS = {
        ModelVariant.DISTILL_8_STEPS: ModelConfig(
            pretrained_model_name="alibaba-pai/Z-Image-Fun-Lora-Distill",
        ),
    }
    DEFAULT_VARIANT = ModelVariant.DISTILL_8_STEPS

    DEFAULT_PROMPT = "A serene mountain landscape at sunrise, photorealistic, 8k"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[DiffusionPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Z-Image-Fun-Lora-Distill",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(
        self,
        dtype_override: Optional[torch.dtype] = None,
    ) -> DiffusionPipeline:
        """Load Z-Image base pipeline and fuse distill LoRA weights.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            DiffusionPipeline: Pipeline with distill LoRA weights fused.
        """
        pipe_dtype = dtype_override if dtype_override is not None else torch.float32

        self.pipeline = DiffusionPipeline.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=pipe_dtype,
            trust_remote_code=True,
        )

        # Load and fuse distill LoRA weights
        adapter_id = self._variant_config.pretrained_model_name
        self.pipeline.load_lora_weights(
            adapter_id,
            weight_name="Z-Image-Fun-Lora-Distill-8-Steps-2603.safetensors",
        )
        self.pipeline.fuse_lora(lora_scale=0.8)

        self.pipeline.to("cpu")

        return self.pipeline

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ):
        """Load and return the Z-Image pipeline with distill LoRA.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            DiffusionPipeline: The Z-Image pipeline with distill LoRA fused.
        """
        if self.pipeline is None:
            return self._load_pipeline(dtype_override=dtype_override)

        if dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype=dtype_override)

        return self.pipeline

    def load_inputs(self, prompt: Optional[str] = None) -> Dict[str, Any]:
        """Load and return sample inputs for the Z-Image model.

        Args:
            prompt: Optional text prompt. Defaults to DEFAULT_PROMPT.

        Returns:
            dict: Input kwargs for the pipeline.
        """
        prompt_value = prompt if prompt is not None else self.DEFAULT_PROMPT
        return {
            "prompt": prompt_value,
            "guidance_scale": 1.0,
            "num_inference_steps": 8,
        }
