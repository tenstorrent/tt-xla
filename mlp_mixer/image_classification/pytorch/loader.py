# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MLP-Mixer model loader implementation
"""

import timm
from typing import Optional
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from PIL import Image
from dataclasses import dataclass

from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....base import ForgeModel
from ....tools.utils import get_file, print_compiled_model_results
from .src.model import MLPMixer
import torch


@dataclass
class MLPMixerConfig(ModelConfig):
    """Configuration specific to MLP Mixer models"""

    source: ModelSource
    weights_available: bool = True
    use_21k_labels: bool = False


class ModelVariant(StrEnum):
    """Available MLP Mixer model variants."""

    # TIMM variants
    MIXER_B16_224 = "mixer_b16_224"
    MIXER_B16_224_IN21K = "mixer_b16_224_in21k"
    MIXER_B16_224_MIIL = "mixer_b16_224_miil"
    MIXER_B16_224_MIIL_IN21K = "mixer_b16_224_miil_in21k"
    MIXER_B32_224 = "mixer_b32_224"
    MIXER_L16_224 = "mixer_l16_224"
    MIXER_L16_224_IN21K = "mixer_l16_224_in21k"
    MIXER_L32_224 = "mixer_l32_224"
    MIXER_S16_224 = "mixer_s16_224"
    MIXER_S32_224 = "mixer_s32_224"
    MIXER_B16_224_GOOG_IN21K = "mixer_b16_224.goog_in21k"

    # GitHub variants
    MIXER_GITHUB = "mixer_github"


class ModelLoader(ForgeModel):
    """MLP Mixer model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        # TIMM variants
        ModelVariant.MIXER_B16_224: MLPMixerConfig(
            pretrained_model_name="mixer_b16_224",
            source=ModelSource.TIMM,
            weights_available=True,
            use_21k_labels=False,
        ),
        ModelVariant.MIXER_B16_224_IN21K: MLPMixerConfig(
            pretrained_model_name="mixer_b16_224_in21k",
            source=ModelSource.TIMM,
            weights_available=True,
            use_21k_labels=True,
        ),
        ModelVariant.MIXER_B16_224_MIIL: MLPMixerConfig(
            pretrained_model_name="mixer_b16_224_miil",
            source=ModelSource.TIMM,
            weights_available=True,
            use_21k_labels=False,
        ),
        ModelVariant.MIXER_B16_224_MIIL_IN21K: MLPMixerConfig(
            pretrained_model_name="mixer_b16_224_miil_in21k",
            source=ModelSource.TIMM,
            weights_available=True,
            use_21k_labels=True,
        ),
        ModelVariant.MIXER_B32_224: MLPMixerConfig(
            pretrained_model_name="mixer_b32_224",
            source=ModelSource.TIMM,
            weights_available=False,
            use_21k_labels=False,
        ),
        ModelVariant.MIXER_L16_224: MLPMixerConfig(
            pretrained_model_name="mixer_l16_224",
            source=ModelSource.TIMM,
            weights_available=True,
            use_21k_labels=False,
        ),
        ModelVariant.MIXER_L16_224_IN21K: MLPMixerConfig(
            pretrained_model_name="mixer_l16_224_in21k",
            source=ModelSource.TIMM,
            weights_available=True,
            use_21k_labels=True,
        ),
        ModelVariant.MIXER_L32_224: MLPMixerConfig(
            pretrained_model_name="mixer_l32_224",
            source=ModelSource.TIMM,
            weights_available=False,
            use_21k_labels=False,
        ),
        ModelVariant.MIXER_S16_224: MLPMixerConfig(
            pretrained_model_name="mixer_s16_224",
            source=ModelSource.TIMM,
            weights_available=False,
            use_21k_labels=False,
        ),
        ModelVariant.MIXER_S32_224: MLPMixerConfig(
            pretrained_model_name="mixer_s32_224",
            source=ModelSource.TIMM,
            weights_available=False,
            use_21k_labels=False,
        ),
        ModelVariant.MIXER_B16_224_GOOG_IN21K: MLPMixerConfig(
            pretrained_model_name="mixer_b16_224.goog_in21k",
            source=ModelSource.TIMM,
            weights_available=True,
            use_21k_labels=True,
        ),
        # GitHub variants
        ModelVariant.MIXER_GITHUB: MLPMixerConfig(
            pretrained_model_name="mixer_github",
            source=ModelSource.GITHUB,
            weights_available=False,
            use_21k_labels=False,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.MIXER_S32_224

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        # Get source from variant config
        source = cls._VARIANTS[variant].source

        return ModelInfo(
            model="mlp_mixer",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_CLS,
            source=source,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load pretrained MLP Mixer model for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The MLP Mixer model instance.
        """
        # Get the pretrained model name from the instance's variant config
        model_name = self._variant_config.pretrained_model_name
        source = self._variant_config.source
        weights_available = self._variant_config.weights_available

        if source == ModelSource.GITHUB:
            # Load model using GitHub source implementation
            model = MLPMixer(
                image_size=256,
                channels=3,
                patch_size=16,
                dim=512,
                depth=12,
                num_classes=1000,
            )
        else:
            # Load model using timm with appropriate pretrained weights setting
            model = timm.create_model(model_name, pretrained=weights_available)

        model.eval()

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        # Store model for use in load_inputs (to avoid reloading)
        self._cached_model = model

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Prepare sample input for MLP Mixer model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            torch.Tensor: Preprocessed input tensor suitable for MLP Mixer.
        """
        source = self._variant_config.source
        use_21k_labels = self._variant_config.use_21k_labels

        if source == ModelSource.GITHUB:
            inputs = torch.randn(1, 3, 256, 256)
        else:
            if use_21k_labels:
                # Use different image for 21K variants
                image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
            else:
                # Use standard image for 1K variants
                image_url = "https://images.rawpixel.com/image_1300/cHJpdmF0ZS9sci9pbWFnZXMvd2Vic2l0ZS8yMDIyLTA1L3BkMTA2LTA0Ny1jaGltXzEuanBn.jpg"

            # Get the Image
            image_file = get_file(image_url)
            image = Image.open(str(image_file)).convert("RGB")

            # Use cached model if available, otherwise load it
            if hasattr(self, "_cached_model") and self._cached_model is not None:
                model_for_config = self._cached_model
            else:
                model_for_config = self.load_model(dtype_override)

            # Preprocess image using model's data config
            data_config = resolve_data_config({}, model=model_for_config)
            timm_transforms = create_transform(**data_config)
            inputs = timm_transforms(image).unsqueeze(0)

        # Replicate tensors for batch size
        inputs = inputs.repeat_interleave(batch_size, dim=0)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs

    def print_cls_results(self, compiled_model_out):
        """Print classification results.

        Args:
            compiled_model_out: Output from the compiled model
        """
        # Determine label set based on variant
        use_21k_labels = self._variant_config.use_21k_labels
        print_compiled_model_results(
            compiled_model_out, use_1k_labels=not use_21k_labels
        )
