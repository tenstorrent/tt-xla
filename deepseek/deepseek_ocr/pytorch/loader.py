# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek OCR model loader implementation for document OCR tasks.
"""
import os
from transformers import AutoTokenizer
from typing import Optional

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....tools.utils import get_file
from .src.modeling_deepseekocr import DeepseekOCRForCausalLM
from .src.model_utils import preprocess
from huggingface_hub import snapshot_download


class ModelVariant(StrEnum):
    """Available DeepSeek OCR model variants."""

    DEEPSEEK_OCR = "deepseek_ocr"


class ModelLoader(ForgeModel):
    """DeepSeek OCR model loader implementation for document OCR tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.DEEPSEEK_OCR: ModelConfig(
            pretrained_model_name="deepseek-ai/DeepSeek-OCR",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.DEEPSEEK_OCR

    # Shared configuration parameters
    sample_prompt = "<image>\n<|grounding|>Convert the document to markdown. "

    def __init__(
        self,
        variant: Optional[ModelVariant] = None,
    ):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="deepseek_ocr",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_DOC_OCR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        """Load tokenizer for the current variant.

        Returns:
            The loaded tokenizer instance
        """
        # Load the tokenizer from HuggingFace
        pretrained_model_name = self._variant_config.pretrained_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )

        return self.tokenizer

    def load_model(self, dtype_override=None):
        """Load and return the DeepSeek OCR model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The DeepSeek OCR model instance for document OCR.
        """

        repo_id = self._variant_config.pretrained_model_name

        # Create a local folder name from the model name
        model_path = repo_id.split("/")[-1].replace("-", "_") + "_weights"

        # create folder if it doesn't exist
        os.makedirs(model_path, exist_ok=True)

        # Download only the essential files.
        snapshot_download(
            repo_id=repo_id,
            local_dir=model_path,
            local_dir_use_symlinks=False,
            allow_patterns=[
                "*.safetensors",
                "config.json",
                "model.safetensors.index.json",
            ],
        )

        # Load Model
        model = DeepseekOCRForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
        )

        # Configure model settings
        model.config.return_dict = False
        model.config.use_cache = False

        # Apply dtype override if specified
        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the DeepSeek OCR model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer()

        # Load the sample image
        image_file = get_file("test_images/doc.png")

        # Process the image and prompt using the preprocess function
        inputs = preprocess(
            tokenizer=self.tokenizer,
            prompt=self.sample_prompt,
            image_file=image_file,
            base_size=1024,
            image_size=640,
            crop_mode=True,
        )

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            # Convert image tensors to the requested dtype
            for idx, (images_crop, images_ori) in enumerate(inputs["images"]):
                inputs["images"][idx] = (
                    images_crop.to(dtype_override),
                    images_ori.to(dtype_override),
                )

        return inputs
