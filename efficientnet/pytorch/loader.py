# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
EfficientNet model loader implementation
"""

import torch
from typing import Optional
from dataclasses import dataclass

from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel
import torchvision.models as models
from torchvision.models._api import WeightsEnum
from ...tools.utils import (
    get_state_dict,
    VisionPreprocessor,
    VisionPostprocessor,
)
import timm


@dataclass
class EfficientNetConfig(ModelConfig):
    """Configuration specific to EfficientNet models"""

    model_function: str = None
    weights_class: str = None
    use_1k_labels: bool = False
    source: ModelSource = ModelSource.TORCHVISION


class ModelVariant(StrEnum):
    """Available EfficientNet model variants."""

    # Torchvision variants
    B0 = "efficientnet_b0"
    B1 = "efficientnet_b1"
    B2 = "efficientnet_b2"
    B3 = "efficientnet_b3"
    B4 = "efficientnet_b4"
    B5 = "efficientnet_b5"
    B6 = "efficientnet_b6"
    B7 = "efficientnet_b7"

    # TIMM variants (values are identifiers for reporting; pretrained model names live in config)
    TIMM_EFFICIENTNET_B0 = "timm_efficientnet_b0"
    TIMM_EFFICIENTNET_B4 = "timm_efficientnet_b4"
    HF_TIMM_EFFICIENTNET_B0_RA_IN1K = "hf_hub_timm_efficientnet_b0_ra_in1k"
    HF_TIMM_EFFICIENTNET_B4_RA2_IN1K = "hf_hub_timm_efficientnet_b4_ra2_in1k"
    HF_TIMM_EFFICIENTNET_B5_IN12K_FT_IN1K = "hf_hub_timm_efficientnet_b5_in12k_ft_in1k"
    HF_TIMM_TF_EFFICIENTNET_B0_AA_IN1K = "hf_hub_timm_tf_efficientnet_b0_aa_in1k"
    HF_TIMM_EFFICIENTNETV2_RW_S_RA2_IN1K = "hf_hub_timm_efficientnetv2_rw_s_ra2_in1k"
    HF_TIMM_TF_EFFICIENTNETV2_S_IN21K = "hf_hub_timm_tf_efficientnetv2_s_in21k"


class ModelLoader(ForgeModel):
    """EfficientNet model loader implementation."""

    # Static dataclass instances for each variant
    B0_CONFIG = EfficientNetConfig(
        pretrained_model_name="efficientnet_b0",
        model_function="efficientnet_b0",
        weights_class="EfficientNet_B0_Weights",
        source=ModelSource.TORCHVISION,
        use_1k_labels=True,
    )

    B1_CONFIG = EfficientNetConfig(
        pretrained_model_name="efficientnet_b1",
        model_function="efficientnet_b1",
        weights_class="EfficientNet_B1_Weights",
        source=ModelSource.TORCHVISION,
        use_1k_labels=True,
    )

    B2_CONFIG = EfficientNetConfig(
        pretrained_model_name="efficientnet_b2",
        model_function="efficientnet_b2",
        weights_class="EfficientNet_B2_Weights",
        source=ModelSource.TORCHVISION,
        use_1k_labels=True,
    )

    B3_CONFIG = EfficientNetConfig(
        pretrained_model_name="efficientnet_b3",
        model_function="efficientnet_b3",
        weights_class="EfficientNet_B3_Weights",
        source=ModelSource.TORCHVISION,
        use_1k_labels=True,
    )

    B4_CONFIG = EfficientNetConfig(
        pretrained_model_name="efficientnet_b4",
        model_function="efficientnet_b4",
        weights_class="EfficientNet_B4_Weights",
        source=ModelSource.TORCHVISION,
        use_1k_labels=True,
    )

    B5_CONFIG = EfficientNetConfig(
        pretrained_model_name="efficientnet_b5",
        model_function="efficientnet_b5",
        weights_class="EfficientNet_B5_Weights",
        source=ModelSource.TORCHVISION,
        use_1k_labels=True,
    )

    B6_CONFIG = EfficientNetConfig(
        pretrained_model_name="efficientnet_b6",
        model_function="efficientnet_b6",
        weights_class="EfficientNet_B6_Weights",
        source=ModelSource.TORCHVISION,
        use_1k_labels=True,
    )

    B7_CONFIG = EfficientNetConfig(
        pretrained_model_name="efficientnet_b7",
        model_function="efficientnet_b7",
        weights_class="EfficientNet_B7_Weights",
        source=ModelSource.TORCHVISION,
        use_1k_labels=True,
    )

    # TIMM config instances
    TIMM_EFFICIENTNET_B0_CONFIG = EfficientNetConfig(
        pretrained_model_name="efficientnet_b0",
        source=ModelSource.TIMM,
        use_1k_labels=True,
    )
    TIMM_EFFICIENTNET_B4_CONFIG = EfficientNetConfig(
        pretrained_model_name="efficientnet_b4",
        source=ModelSource.TIMM,
        use_1k_labels=True,
    )
    HF_TIMM_EFFICIENTNET_B0_RA_IN1K_CONFIG = EfficientNetConfig(
        pretrained_model_name="hf_hub:timm/efficientnet_b0.ra_in1k",
        source=ModelSource.TIMM,
        use_1k_labels=True,
    )
    HF_TIMM_EFFICIENTNET_B4_RA2_IN1K_CONFIG = EfficientNetConfig(
        pretrained_model_name="hf_hub:timm/efficientnet_b4.ra2_in1k",
        source=ModelSource.TIMM,
        use_1k_labels=True,
    )
    HF_TIMM_EFFICIENTNET_B5_IN12K_FT_IN1K_CONFIG = EfficientNetConfig(
        pretrained_model_name="hf_hub:timm/efficientnet_b5.in12k_ft_in1k",
        source=ModelSource.TIMM,
        use_1k_labels=True,
    )
    HF_TIMM_TF_EFFICIENTNET_B0_AA_IN1K_CONFIG = EfficientNetConfig(
        pretrained_model_name="hf_hub:timm/tf_efficientnet_b0.aa_in1k",
        source=ModelSource.TIMM,
        use_1k_labels=True,
    )
    HF_TIMM_EFFICIENTNETV2_RW_S_RA2_IN1K_CONFIG = EfficientNetConfig(
        pretrained_model_name="hf_hub:timm/efficientnetv2_rw_s.ra2_in1k",
        source=ModelSource.TIMM,
        use_1k_labels=True,
    )
    HF_TIMM_TF_EFFICIENTNETV2_S_IN21K_CONFIG = EfficientNetConfig(
        pretrained_model_name="hf_hub:timm/tf_efficientnetv2_s.in21k",
        source=ModelSource.TIMM,
        use_1k_labels=False,
    )

    # Dictionary using the static dataclass instances (for compatibility with existing tests)
    _VARIANTS = {
        # Torchvision variants
        ModelVariant.B0: B0_CONFIG,
        ModelVariant.B1: B1_CONFIG,
        ModelVariant.B2: B2_CONFIG,
        ModelVariant.B3: B3_CONFIG,
        ModelVariant.B4: B4_CONFIG,
        ModelVariant.B5: B5_CONFIG,
        ModelVariant.B6: B6_CONFIG,
        ModelVariant.B7: B7_CONFIG,
        # TIMM variants
        ModelVariant.TIMM_EFFICIENTNET_B0: TIMM_EFFICIENTNET_B0_CONFIG,
        ModelVariant.TIMM_EFFICIENTNET_B4: TIMM_EFFICIENTNET_B4_CONFIG,
        ModelVariant.HF_TIMM_EFFICIENTNET_B0_RA_IN1K: HF_TIMM_EFFICIENTNET_B0_RA_IN1K_CONFIG,
        ModelVariant.HF_TIMM_EFFICIENTNET_B4_RA2_IN1K: HF_TIMM_EFFICIENTNET_B4_RA2_IN1K_CONFIG,
        ModelVariant.HF_TIMM_EFFICIENTNET_B5_IN12K_FT_IN1K: HF_TIMM_EFFICIENTNET_B5_IN12K_FT_IN1K_CONFIG,
        ModelVariant.HF_TIMM_TF_EFFICIENTNET_B0_AA_IN1K: HF_TIMM_TF_EFFICIENTNET_B0_AA_IN1K_CONFIG,
        ModelVariant.HF_TIMM_EFFICIENTNETV2_RW_S_RA2_IN1K: HF_TIMM_EFFICIENTNETV2_RW_S_RA2_IN1K_CONFIG,
        ModelVariant.HF_TIMM_TF_EFFICIENTNETV2_S_IN21K: HF_TIMM_TF_EFFICIENTNETV2_S_IN21K_CONFIG,
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.B0

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self._cached_model = None
        self.model = None
        self._preprocessor = None
        self._postprocessor = None

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
        source = cls._VARIANTS[variant].source
        return ModelInfo(
            model="efficientnet",
            variant=variant,
            group=(
                ModelGroup.RED if variant == ModelVariant.B0 else ModelGroup.GENERALITY
            ),
            task=ModelTask.CV_IMAGE_CLS,
            source=source,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the EfficientNet model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The EfficientNet model instance.
        """
        source = self._variant_config.source

        if source == ModelSource.TORCHVISION:
            # Setup state dict function
            WeightsEnum.get_state_dict = get_state_dict

            # Get model function and weights class from dataclass config
            model_fn = getattr(models, self._variant_config.model_function)
            weights_class = getattr(models, self._variant_config.weights_class)

            # Load model with appropriate weights
            model = model_fn(weights=weights_class.IMAGENET1K_V1)
        else:
            # Load using timm
            model_name = self._variant_config.pretrained_model_name
            model = timm.create_model(model_name, pretrained=True)

        model.eval()
        self._cached_model = model
        self.model = model

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def input_preprocess(self, dtype_override=None, batch_size=1, image=None):
        """Preprocess input image(s) and return model-ready input tensor.

        Args:
            dtype_override: Optional torch.dtype override (default: float32).
            batch_size: Batch size (ignored if image is a list).
            image: PIL Image, URL string, tensor, list of images/URLs, or None (uses default image).

        Returns:
            torch.Tensor: Preprocessed input tensor.
        """
        if self._preprocessor is None:
            model_name = self._variant_config.pretrained_model_name
            source = self._variant_config.source

            # For TORCHVISION, use standard ImageNet preprocessing
            if source == ModelSource.TORCHVISION:

                def weight_class_name_fn(name: str) -> str:
                    # Handle efficientnet_b0 -> EfficientNet_B0_Weights
                    # Split by underscore and capitalize each part, then join with underscore
                    parts = name.split("_")
                    # Capitalize first letter of each part and join with underscore
                    capitalized_parts = [p.capitalize() for p in parts]
                    return "_".join(capitalized_parts) + "_Weights"

                self._preprocessor = VisionPreprocessor(
                    model_source=source,
                    model_name=model_name,
                    weight_class_name_fn=weight_class_name_fn,
                )
            else:
                self._preprocessor = VisionPreprocessor(
                    model_source=source,
                    model_name=model_name,
                )

            if hasattr(self, "model") and self.model is not None:
                self._preprocessor.set_cached_model(self.model)

        model_for_config = None
        if self._variant_config.source == ModelSource.TIMM:
            if hasattr(self, "model") and self.model is not None:
                model_for_config = self.model

        return self._preprocessor.preprocess(
            image=image,
            dtype_override=dtype_override,
            batch_size=batch_size,
            model_for_config=model_for_config,
        )

    def load_inputs(self, dtype_override=None, batch_size=1, image=None):
        """Load and return sample inputs (backward compatibility wrapper for input_preprocess).

        Args:
            dtype_override: Optional torch.dtype override.
            batch_size: Batch size (default: 1).
            image: Optional input image.

        Returns:
            torch.Tensor: Preprocessed input tensor.
        """
        return self.input_preprocess(
            image=image,
            dtype_override=dtype_override,
            batch_size=batch_size,
        )

    def output_postprocess(
        self,
        output=None,
        co_out=None,
        framework_model=None,
        compiled_model=None,
        inputs=None,
        dtype_override=None,
    ):
        """Post-process model outputs.

        Args:
            output: Model output tensor (returns dict if provided).
            co_out: Compiled model outputs (legacy, prints results).
            framework_model: Original framework model (legacy).
            compiled_model: Compiled model (legacy).
            inputs: Input images (legacy).
            dtype_override: Optional dtype override (legacy).

        Returns:
            dict or None: Prediction dict if output provided, else None (prints results).
        """
        if self._postprocessor is None:
            model_name = self._variant_config.pretrained_model_name
            source = self._variant_config.source
            use_1k_labels = self._variant_config.use_1k_labels

            self._postprocessor = VisionPostprocessor(
                model_source=source,
                model_name=model_name,
                model_instance=self.model,
                use_1k_labels=use_1k_labels,
            )

        # New usage: return dict from output tensor
        if output is not None:
            return self._postprocessor.postprocess(output, top_k=1, return_dict=True)

        # Legacy usage: print results (backward compatibility)
        self._postprocessor.print_results(
            co_out=co_out,
            framework_model=framework_model,
            compiled_model=compiled_model,
            inputs=inputs,
            dtype_override=dtype_override,
        )
        return None
