# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
VGG model loader implementation
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
from ...tools.utils import VisionPreprocessor, VisionPostprocessor
from datasets import load_dataset


@dataclass
class VGGConfig(ModelConfig):
    source: ModelSource
    model_function: Optional[str] = None  # for torchvision
    weights_class: Optional[str] = None  # for torchvision


class ModelVariant(StrEnum):
    """Available VGG model variants."""

    # OSMR (pytorchcv) image classification variants
    VGG11 = "11"
    VGG13 = "13"
    VGG16 = "16"
    VGG19 = "19"
    VGG19_BN_OSMR = "Bn_Vgg19"
    VGG19_BNB_OSMR = "Bn_Vgg19b"

    # TorchHub variant
    VGG19_BN = "19_Bn"

    # TIMM variant
    TIMM_VGG19_BN = "Timm_Vgg19_Bn"

    # Torchvision variants
    TV_VGG11 = "Torchvision_Vgg11"
    TV_VGG11_BN = "Torchvision_Vgg11_Bn"
    TV_VGG13 = "Torchvision_Vgg13"
    TV_VGG13_BN = "Torchvision_Vgg13_Bn"
    TV_VGG16 = "Torchvision_Vgg16"
    TV_VGG16_BN = "Torchvision_Vgg16_Bn"
    TV_VGG19 = "Torchvision_Vgg19"
    TV_VGG19_BN = "Torchvision_Vgg19_Bn"

    # HuggingFace vgg-pytorch
    HF_VGG19 = "HF_Vgg19"


class ModelLoader(ForgeModel):
    """VGG model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        # OSMR variants
        ModelVariant.VGG11: VGGConfig(
            pretrained_model_name="vgg11", source=ModelSource.OSMR
        ),
        ModelVariant.VGG13: VGGConfig(
            pretrained_model_name="vgg13", source=ModelSource.OSMR
        ),
        ModelVariant.VGG16: VGGConfig(
            pretrained_model_name="vgg16", source=ModelSource.OSMR
        ),
        ModelVariant.VGG19: VGGConfig(
            pretrained_model_name="vgg19", source=ModelSource.OSMR
        ),
        ModelVariant.VGG19_BN_OSMR: VGGConfig(
            pretrained_model_name="bn_vgg19", source=ModelSource.OSMR
        ),
        ModelVariant.VGG19_BNB_OSMR: VGGConfig(
            pretrained_model_name="bn_vgg19b", source=ModelSource.OSMR
        ),
        # TorchHub
        ModelVariant.VGG19_BN: VGGConfig(
            pretrained_model_name="vgg19_bn", source=ModelSource.TORCH_HUB
        ),
        # TIMM
        ModelVariant.TIMM_VGG19_BN: VGGConfig(
            pretrained_model_name="vgg19_bn", source=ModelSource.TIMM
        ),
        # Torchvision
        ModelVariant.TV_VGG11: VGGConfig(
            pretrained_model_name="vgg11",
            source=ModelSource.TORCHVISION,
            model_function="vgg11",
            weights_class="VGG11_Weights",
        ),
        ModelVariant.TV_VGG11_BN: VGGConfig(
            pretrained_model_name="vgg11_bn",
            source=ModelSource.TORCHVISION,
            model_function="vgg11_bn",
            weights_class="VGG11_BN_Weights",
        ),
        ModelVariant.TV_VGG13: VGGConfig(
            pretrained_model_name="vgg13",
            source=ModelSource.TORCHVISION,
            model_function="vgg13",
            weights_class="VGG13_Weights",
        ),
        ModelVariant.TV_VGG13_BN: VGGConfig(
            pretrained_model_name="vgg13_bn",
            source=ModelSource.TORCHVISION,
            model_function="vgg13_bn",
            weights_class="VGG13_BN_Weights",
        ),
        ModelVariant.TV_VGG16: VGGConfig(
            pretrained_model_name="vgg16",
            source=ModelSource.TORCHVISION,
            model_function="vgg16",
            weights_class="VGG16_Weights",
        ),
        ModelVariant.TV_VGG16_BN: VGGConfig(
            pretrained_model_name="vgg16_bn",
            source=ModelSource.TORCHVISION,
            model_function="vgg16_bn",
            weights_class="VGG16_BN_Weights",
        ),
        ModelVariant.TV_VGG19: VGGConfig(
            pretrained_model_name="vgg19",
            source=ModelSource.TORCHVISION,
            model_function="vgg19",
            weights_class="VGG19_Weights",
        ),
        ModelVariant.TV_VGG19_BN: VGGConfig(
            pretrained_model_name="vgg19_bn",
            source=ModelSource.TORCHVISION,
            model_function="vgg19_bn",
            weights_class="VGG19_BN_Weights",
        ),
        # HuggingFace
        ModelVariant.HF_VGG19: VGGConfig(
            pretrained_model_name="vgg19", source=ModelSource.HUGGING_FACE
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.TV_VGG19_BN

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
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
            model="VGG",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_CLS,
            source=source,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the VGG model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The VGG model instance.
        """

        import timm
        from pytorchcv.model_provider import get_model as ptcv_get_model
        from vgg_pytorch import VGG as HFVGG
        from torchvision import models as tv_models

        # Get the pretrained model name from the instance's variant config
        cfg = self._variant_config
        model_name = cfg.pretrained_model_name
        source = cfg.source

        if source == ModelSource.TORCH_HUB:
            model = torch.hub.load(
                "pytorch/vision:v0.10.0", model_name, pretrained=True
            )
        elif source == ModelSource.TIMM:
            model = timm.create_model(model_name, pretrained=True)
        elif source == ModelSource.OSMR:
            model = ptcv_get_model(model_name, pretrained=True)
        elif source == ModelSource.HUGGING_FACE:
            model = HFVGG.from_pretrained(model_name, **kwargs)
        elif source == ModelSource.TORCHVISION:
            weights = getattr(tv_models, self._variant_config.weights_class).DEFAULT
            model = getattr(tv_models, self._variant_config.model_function)(
                weights=weights
            )
        else:
            raise ValueError(f"Unsupported source: {source}")

        model.eval()

        # Store model for potential use in input preprocessing and postprocessing
        self.model = model

        # Update preprocessor with cached model (for TIMM models)
        if self._preprocessor is not None:
            self._preprocessor.set_cached_model(model)

        # Update postprocessor with model instance (for HuggingFace models)
        if self._postprocessor is not None:
            self._postprocessor.set_model_instance(model)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def input_preprocess(self, dtype_override=None, batch_size=1, image=None):
        """Preprocess input image(s) and return model-ready input tensor.

        Args:
            dtype_override: Optional torch.dtype override (default: float32).
            batch_size: Batch size (ignored if image is a list).
            image: PIL Image, URL string, tensor, list of images/URLs, or None (uses default COCO image).

        Returns:
            torch.Tensor: Preprocessed input tensor.
        """
        if self._preprocessor is None:
            model_name = self._variant_config.pretrained_model_name
            source = self._variant_config.source

            # Handle different sources
            if source == ModelSource.TIMM:
                preprocessor_source = ModelSource.TIMM
                preprocessor_model_name = model_name
            elif source == ModelSource.TORCHVISION:
                preprocessor_source = ModelSource.TORCHVISION
                preprocessor_model_name = model_name
                # For torchvision, use the weights_class from config
                weights_class = self._variant_config.weights_class

                def weight_class_name_fn(name: str) -> str:
                    return weights_class

            elif source == ModelSource.TORCH_HUB:
                # Torch Hub models use torchvision-style preprocessing
                preprocessor_source = ModelSource.TORCHVISION
                # For torch hub vgg19_bn, use VGG19_BN_Weights
                preprocessor_model_name = "vgg19_bn"

                def weight_class_name_fn(name: str) -> str:
                    return "VGG19_BN_Weights"

            elif source == ModelSource.HUGGING_FACE:
                # vgg_pytorch doesn't have a preprocessor on HuggingFace Hub
                # Use custom preprocessing with standard ImageNet transforms
                preprocessor_source = ModelSource.CUSTOM
                from torchvision import transforms

                def custom_preprocess_fn(img):
                    preprocess = transforms.Compose(
                        [
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(
                                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                            ),
                        ]
                    )
                    return preprocess(img)

            elif source == ModelSource.OSMR:
                # OSMR models use standard ImageNet preprocessing
                preprocessor_source = ModelSource.CUSTOM
                from torchvision import transforms

                def custom_preprocess_fn(img):
                    preprocess = transforms.Compose(
                        [
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(
                                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                            ),
                        ]
                    )
                    return preprocess(img)

            else:
                raise ValueError(f"Unsupported source for preprocessing: {source}")

            # Create preprocessor
            if source in [ModelSource.TORCHVISION, ModelSource.TORCH_HUB]:
                self._preprocessor = VisionPreprocessor(
                    model_source=preprocessor_source,
                    model_name=preprocessor_model_name,
                    weight_class_name_fn=weight_class_name_fn,
                )
            elif source in [ModelSource.OSMR, ModelSource.HUGGING_FACE]:
                self._preprocessor = VisionPreprocessor(
                    model_source=preprocessor_source,
                    model_name=model_name,
                    custom_preprocess_fn=custom_preprocess_fn,
                )
            else:
                self._preprocessor = VisionPreprocessor(
                    model_source=preprocessor_source,
                    model_name=preprocessor_model_name,
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
        """Load and return sample inputs for the model.

        Args:
            dtype_override: Optional torch.dtype override.
            batch_size: Batch size (default: 1).
            image: Optional input image. If None, loads from HuggingFace datasets.

        Returns:
            torch.Tensor: Preprocessed input tensor.
        """
        if image is None:
            dataset = load_dataset("huggingface/cats-image", split="test")
            image = dataset[0]["image"]
        return self.input_preprocess(
            image=image,
            dtype_override=dtype_override,
            batch_size=batch_size,
        )

    def output_postprocess(self, output):
        """Post-process model outputs.

        Args:
            output: Model output tensor.

        Returns:
            dict: Prediction dictionary with top predictions.
        """
        if self._postprocessor is None:
            model_name = self._variant_config.pretrained_model_name
            source = self._variant_config.source

            # Map sources to postprocessor sources
            if source == ModelSource.TIMM:
                postprocessor_source = ModelSource.TIMM
                postprocessor_model_name = model_name
            elif source in [ModelSource.TORCHVISION, ModelSource.TORCH_HUB]:
                # Both use ImageNet labels like torchvision
                postprocessor_source = ModelSource.TORCHVISION
                postprocessor_model_name = model_name
            elif source == ModelSource.HUGGING_FACE:
                # vgg_pytorch models use ImageNet labels like torchvision
                postprocessor_source = ModelSource.TORCHVISION
                postprocessor_model_name = (
                    "vgg19_bn"  # Use a standard torchvision name for labels
                )
            elif source == ModelSource.OSMR:
                # OSMR models use ImageNet labels like torchvision
                postprocessor_source = ModelSource.TORCHVISION
                postprocessor_model_name = (
                    "vgg19_bn"  # Use a standard torchvision name for labels
                )
            else:
                raise ValueError(f"Unsupported source for postprocessing: {source}")

            self._postprocessor = VisionPostprocessor(
                model_source=postprocessor_source,
                model_name=postprocessor_model_name,
                model_instance=self.model,
            )

        return self._postprocessor.postprocess(output, top_k=1, return_dict=True)
