# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
dots.mocr model loader implementation for multimodal document OCR tasks.
"""
import os

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM
from typing import Optional

from ...base import ForgeModel
from ...config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from .src.model import Wrapper


class ModelVariant(StrEnum):
    """Available dots.mocr model variants for multimodal document OCR tasks."""

    DOTS_MOCR = "Mocr"


class ModelLoader(ForgeModel):
    """dots.mocr model loader implementation for multimodal document OCR tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.DOTS_MOCR: LLMModelConfig(
            pretrained_model_name="rednote-hilab/dots.mocr",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.DOTS_MOCR

    # Sample image URL for testing
    sample_image_url = (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    )
    sample_prompt = "Convert the document to markdown."

    # Vision processing parameters
    min_pixels = 56 * 56
    max_pixels = 13 * 28 * 1280

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="dots.mocr",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_DOC_OCR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load processor for the current variant.

        The dots.mocr custom processor does not pass video_processor to its
        parent Qwen2_5_VLProcessor, which causes a TypeError on newer
        transformers versions. We work around this by loading the processor
        components individually and constructing a Qwen2_5_VLProcessor.

        Returns:
            The loaded processor instance
        """
        from transformers import (
            AutoTokenizer,
            AutoImageProcessor,
            Qwen2_5_VLProcessor,
        )
        from transformers.models.qwen2_vl.video_processing_qwen2_vl import (
            Qwen2VLVideoProcessor,
        )

        model_path = self._get_local_model_path()

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        image_processor = AutoImageProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        video_processor = Qwen2VLVideoProcessor()

        self.processor = Qwen2_5_VLProcessor(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
            chat_template=tokenizer.chat_template,
        )

        return self.processor

    def _get_local_model_path(self):
        """Download model to a local path without dots in the name.

        The dot in 'dots.mocr' causes Python import issues with HuggingFace's
        dynamic module loading, so we download to a local directory instead.

        Returns:
            str: Local path to the downloaded model weights.
        """
        repo_id = self._variant_config.pretrained_model_name
        model_path = "DotsMOCR_weights"
        os.makedirs(model_path, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            local_dir=model_path,
            local_dir_use_symlinks=False,
        )
        return model_path

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the dots.mocr model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Wrapped dots.mocr model instance for document OCR tasks.
        """
        model_path = self._get_local_model_path()

        model_kwargs = {
            "low_cpu_mem_usage": True,
            "use_cache": False,
            "trust_remote_code": True,
        }

        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override
        else:
            model_kwargs["dtype"] = torch.float32
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        model.float()
        # The vision tower's forward defaults bf16=True which casts pixel
        # values to bfloat16, causing a dtype mismatch with float32 conv
        # weights. Patch the forward to disable this cast.
        orig_vt_forward = model.vision_tower.forward

        def _vt_forward_no_bf16(hidden_states, grid_thw, bf16=False):
            return orig_vt_forward(hidden_states, grid_thw, bf16=bf16)

        model.vision_tower.forward = _vt_forward_no_bf16
        model.eval()
        model = Wrapper(model)

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the dots.mocr model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
                           If specified, converts pixel_values to the specified dtype.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        from PIL import Image
        import requests

        if self.processor is None:
            self._load_processor()

        image = Image.open(
            requests.get(self.sample_image_url, stream=True).raw
        ).convert("RGB")

        # Build the text prompt manually since the dots.mocr chat template
        # does not support multimodal content lists.
        # The Qwen2.5VL processor replaces <|image_pad|> with the actual
        # vision token sequence during processing. The dots.mocr model uses
        # <|imgpad|> (token id 151665) as its image token, but we need
        # <|image_pad|> in the text for the processor to expand it into
        # the correct number of vision tokens. We then remap the token ids
        # after processing.
        text = (
            "<|user|><|vision_start|><|image_pad|><|vision_end|>"
            + self.sample_prompt
            + "<|endofuser|><|assistant|>"
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )

        # Remap <|image_pad|> (151655) to <|imgpad|> (151665) which is
        # the image_token_id the dots.mocr model uses for its img_mask
        QWEN_IMAGE_PAD_ID = 151655
        DOTS_IMGPAD_ID = 151665
        inputs["input_ids"] = inputs["input_ids"].masked_fill(
            inputs["input_ids"] == QWEN_IMAGE_PAD_ID, DOTS_IMGPAD_ID
        )

        if dtype_override is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        return inputs
