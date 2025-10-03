# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""VisionTextDualEncoder model loader implementation for image-text tasks."""

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


class ModelVariant(StrEnum):
    """Available VisionTextDualEncoder model variants for image-text tasks."""

    BASE = "base"


class ModelLoader(ForgeModel):
    """VisionTextDualEncoder model loader implementation for image-text tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="base",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self._cached_model = None
        self._vision_model_path = None
        self._text_model_path = None

    @classmethod
    def _get_model_info(cls, variant_name=None):
        """Get model information.

        Args:
            variant_name: Optional variant name string. If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant_name is None:
            variant_name = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="vision_text_dual_encoder",
            variant=variant_name,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.JAX,
        )

    def _get_model_paths(self):
        """Get the vision and text model paths based on the variant."""
        if self._variant == ModelVariant.BASE:
            self._vision_model_path = "google/vit-base-patch16-224"
            self._text_model_path = "google-bert/bert-base-uncased"
        else:
            raise ValueError("Unknown variant: " + str(self._variant))

    def load_model(self, dtype_override=None):
        """Load and return VisionTextDualEncoder model from Hugging Face.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.
                            If not provided, the model will use its default dtype.

        Returns:
            The loaded model instance.
        """
        if self._cached_model is None:
            self._get_model_paths()

            # Load the model with dtype override if specified
            model_kwargs = {}
            if dtype_override is not None:
                model_kwargs["dtype"] = dtype_override

            from transformers import FlaxVisionTextDualEncoderModel

            self._cached_model = (
                FlaxVisionTextDualEncoderModel.from_vision_text_pretrained(
                    self._vision_model_path, self._text_model_path, **model_kwargs
                )
            )

        return self._cached_model

    def load_inputs(self, dtype_override=None):
        """Load inputs for the VisionTextDualEncoder model.

        Args:
            dtype_override: Optional dtype to override the inputs' default dtype.
                            If not provided, the inputs will use its default dtype.

        Returns:
            The loaded inputs.
        """

        from transformers import (
            AutoImageProcessor,
            AutoTokenizer,
            VisionTextDualEncoderProcessor,
            ViTConfig,
        )
        from datasets import load_dataset
        import numpy as np

        # Load model configuration
        model_config = ViTConfig.from_pretrained(self._vision_model_path)
        image_size = model_config.image_size

        # Load cats dataset and get a cat image
        dataset = load_dataset("huggingface/cats-image")["test"]
        cat_image = dataset[0]["image"]

        # Resize image to match model's expected size
        cat_image = cat_image.resize((image_size, image_size))
        cat_image = np.array(cat_image)

        # Load tokenizer and image processor
        tokenizer = AutoTokenizer.from_pretrained(self._text_model_path)
        image_processor = AutoImageProcessor.from_pretrained(self._vision_model_path)
        processor = VisionTextDualEncoderProcessor(image_processor, tokenizer)

        # Process inputs
        inputs = processor(
            text="A cute cat image", images=cat_image, return_tensors="jax"
        )

        # Apply dtype override if specified
        if dtype_override is not None:
            for key, value in inputs.items():
                if hasattr(value, "astype"):
                    inputs[key] = value.astype(dtype_override)

        return inputs
