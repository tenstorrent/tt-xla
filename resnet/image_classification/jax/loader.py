# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
ResNet model loader implementation for image classification.
"""

from typing import Optional, List, Tuple, Union, Dict
import jax

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
from ....tools.jax_utils import cast_hf_model_to_type


class ModelVariant(StrEnum):
    """Available ResNet model variants."""

    RESNET_18 = "resnet-18"
    RESNET_26 = "resnet-26"
    RESNET_34 = "resnet-34"
    RESNET_50 = "resnet-50"
    RESNET_101 = "resnet-101"
    RESNET_152 = "resnet-152"


class ModelLoader(ForgeModel):
    """ResNet model loader implementation for image classification."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.RESNET_18: ModelConfig(
            pretrained_model_name="microsoft/resnet-18",
        ),
        ModelVariant.RESNET_26: ModelConfig(
            pretrained_model_name="microsoft/resnet-26",
        ),
        ModelVariant.RESNET_34: ModelConfig(
            pretrained_model_name="microsoft/resnet-34",
        ),
        ModelVariant.RESNET_50: ModelConfig(
            pretrained_model_name="microsoft/resnet-50",
        ),
        ModelVariant.RESNET_101: ModelConfig(
            pretrained_model_name="microsoft/resnet-101",
        ),
        ModelVariant.RESNET_152: ModelConfig(
            pretrained_model_name="microsoft/resnet-152",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.RESNET_50

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

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
            model="resnet_v1.5",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.JAX,
        )

    @staticmethod
    def _get_renaming_patterns(variant: ModelVariant) -> List[Tuple[str, str]]:
        """Get renaming patterns for converting PyTorch weights to JAX."""
        PATTERNS = [
            (r"convolution.weight", r"convolution.kernel"),
            (r"normalization.running_mean", r"normalization.mean"),
            (r"normalization.running_var", r"normalization.var"),
            (r"normalization.weight", r"normalization.scale"),
            (r"classifier\.(\d+).weight", r"classifier.\1.kernel"),
        ]

        if variant in (ModelVariant.RESNET_18, ModelVariant.RESNET_34):
            PATTERNS.append((r"layer\.(\d+)\.", r"layer.layer_\1."))

        return PATTERNS

    @staticmethod
    def _get_banned_subkeys() -> List[str]:
        """Get banned subkeys that should be excluded from weight conversion."""
        return ["num_batches_tracked"]

    @staticmethod
    def _download_weights(
        model_variant: ModelVariant,
    ) -> Union[Dict[str, jax.Array], Dict[str, "torch.Tensor"]]:
        """Download weights for the specified model variant."""
        import torch
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open

        filename = "model.safetensors"
        if model_variant == ModelVariant.RESNET_101:
            filename = "pytorch_model.bin"

        hf_path = f"microsoft/{model_variant}"
        ckpt_path = hf_hub_download(repo_id=hf_path, filename=filename)

        if filename == "model.safetensors":
            with safe_open(ckpt_path, framework="flax", device="cpu") as f:
                return {key: f.get_tensor(key) for key in f.keys()}
        else:  # filename == "pytorch_model.bin"
            return torch.load(ckpt_path, map_location="cpu")

    def _torch_statedict_to_pytree(
        self,
        state_dict: Union[Dict[str, jax.Array], Dict[str, "torch.Tensor"]],
        patterns: List[Tuple[str, str]],
        banned_subkeys: List[str],
    ) -> Dict[str, jax.Array]:
        """Convert PyTorch state dict to JAX pytree with renaming patterns."""
        import re
        import torch

        def should_include_key(key: str) -> bool:
            return not any(banned in key for banned in banned_subkeys)

        def rename_key(key: str) -> str:
            for pattern, replacement in patterns:
                key = re.sub(pattern, replacement, key)
            return key

        jax_state_dict = {}
        for key, value in state_dict.items():
            if should_include_key(key):
                new_key = rename_key(key)
                if isinstance(value, torch.Tensor):
                    jax_state_dict[new_key] = jax.numpy.array(value.numpy())
                else:
                    jax_state_dict[new_key] = value

        return jax_state_dict

    def load_model(self, dtype_override=None):
        """Load and return the ResNet model instance for this instance's variant.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.

        Returns:
            model: The loaded model instance
        """
        from transformers import FlaxResNetForImageClassification, ResNetConfig

        hf_path = f"microsoft/{self._variant}"

        # Try to load directly first (some variants might have Flax checkpoints)
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override

        try:
            model = FlaxResNetForImageClassification.from_pretrained(
                hf_path, **model_kwargs
            )
        except Exception:
            # If direct loading fails, try loading with PyTorch weights conversion
            try:
                model = FlaxResNetForImageClassification.from_pretrained(
                    hf_path, from_pt=True, **model_kwargs
                )
            except Exception:
                # If that also fails, try manual weight conversion approach
                model_config = ResNetConfig.from_pretrained(hf_path)
                model = FlaxResNetForImageClassification(model_config)

                state_dict = self._download_weights(self._variant)

                variables = self._torch_statedict_to_pytree(
                    state_dict,
                    patterns=self._get_renaming_patterns(self._variant),
                    banned_subkeys=self._get_banned_subkeys(),
                )

                model.params = variables

        # Cast the model to the dtype_override if provided
        if dtype_override is not None:
            model = cast_hf_model_to_type(model, dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the ResNet model with this instance's variant settings.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.

        Returns:
            inputs: Input tensors that can be fed to the model.
        """
        from transformers import AutoImageProcessor
        from datasets import load_dataset

        # Load the image processor
        processor = AutoImageProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )

        # Load a sample image from open-source cats-image dataset
        dataset = load_dataset("huggingface/cats-image", split="test")
        sample = dataset[0]

        # Process the image using the processor
        inputs = processor(images=sample["image"], return_tensors="jax")

        if dtype_override is not None:
            # Apply dtype override to all tensors in the inputs
            for key, value in inputs.items():
                if hasattr(value, "astype"):
                    inputs[key] = value.astype(dtype_override)

        return inputs

    def wrapper_model(self, f):
        """Wrapper for model forward method that extracts the appropriate output.

        ResNet models return a tuple where the first element contains the logits.
        This wrapper extracts the logits from the tuple output.

        Args:
            f: The model forward function to wrap

        Returns:
            Wrapped function that extracts logits
        """

        def model(args, kwargs):
            out = f(*args, **kwargs)
            # ResNet returns a tuple, extract first element then get logits
            out = out[0]
            return out.logits

        return model
