# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
calcuis/qwen-image-gguf model loader implementation for text-to-image generation.
"""

from typing import Optional, Dict, Any

import torch
from diffusers import (
    DiffusionPipeline,
    GGUFQuantizationConfig,
    QwenImageTransformer2DModel,
)

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


class ModelVariant(StrEnum):
    """Available Qwen-Image GGUF model variants."""

    QWEN_IMAGE_Q4_K_M = "qwen-image-q4_k_m"


class ModelLoader(ForgeModel):
    """calcuis/qwen-image-gguf model loader implementation."""

    _VARIANTS = {
        ModelVariant.QWEN_IMAGE_Q4_K_M: ModelConfig(
            pretrained_model_name="calcuis/qwen-image-gguf",
        ),
    }

    _GGUF_FILES = {
        ModelVariant.QWEN_IMAGE_Q4_K_M: "qwen-image-q4_k_m.gguf",
    }

    _PIPELINE_REPO = "callgg/qi-decoder"

    DEFAULT_VARIANT = ModelVariant.QWEN_IMAGE_Q4_K_M

    DEFAULT_PROMPT = "a pig holding a sign that says hello world"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[DiffusionPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Qwen-Image-GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(
        self,
        dtype_override: Optional[torch.dtype] = None,
        device_map: Optional[str] = None,
        low_cpu_mem_usage: bool = True,
        extra_pipe_kwargs: Optional[Dict[str, Any]] = None,
    ) -> DiffusionPipeline:
        if extra_pipe_kwargs is None:
            extra_pipe_kwargs = {}

        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16

        gguf_file = self._GGUF_FILES[self.variant]
        gguf_path = (
            f"https://huggingface.co/{self._variant_config.pretrained_model_name}"
            f"/blob/main/{gguf_file}"
        )

        transformer = QwenImageTransformer2DModel.from_single_file(
            gguf_path,
            quantization_config=GGUFQuantizationConfig(compute_dtype=compute_dtype),
            torch_dtype=compute_dtype,
        )

        pipe_kwargs = {
            "transformer": transformer,
            "torch_dtype": compute_dtype,
            "device_map": device_map,
            "low_cpu_mem_usage": low_cpu_mem_usage,
        }
        pipe_kwargs.update(extra_pipe_kwargs)

        self.pipeline = DiffusionPipeline.from_pretrained(
            self._PIPELINE_REPO,
            **pipe_kwargs,
        )

        return self.pipeline

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        device_map: Optional[str] = None,
        low_cpu_mem_usage: bool = True,
        extra_pipe_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Load and return the Qwen-Image GGUF text-to-image pipeline.
        """
        if self.pipeline is None:
            return self._load_pipeline(
                dtype_override=dtype_override,
                device_map=device_map,
                low_cpu_mem_usage=low_cpu_mem_usage,
                extra_pipe_kwargs=extra_pipe_kwargs,
            )

        if dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype=dtype_override)

        return self.pipeline

    def load_inputs(self, prompt: Optional[str] = None) -> Dict[str, Any]:
        prompt_value = prompt if prompt is not None else self.DEFAULT_PROMPT
        return {"prompt": prompt_value}
