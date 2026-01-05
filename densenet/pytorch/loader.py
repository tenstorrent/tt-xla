# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Densenet model loader implementation
"""

from typing import Optional
from dataclasses import dataclass
from torchvision import models
import torch
import os

from ...tools.utils import VisionPreprocessor, VisionPostprocessor

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

# Import xray-specific dependencies only when needed
try:
    import torchxrayvision as xrv
    import skimage
    import torchvision
    from .src.utils import op_norm

    XRAY_AVAILABLE = True
except ImportError:
    XRAY_AVAILABLE = False


@dataclass
class DenseNetConfig(ModelConfig):
    """Configuration specific to DenseNet models"""

    source: ModelSource


class ModelVariant(StrEnum):
    """Available DenseNet model variants."""

    # Torchvision variants
    DENSENET121 = "densenet121"
    DENSENET161 = "densenet161"
    DENSENET169 = "densenet169"
    DENSENET201 = "densenet201"

    # X-ray variants
    DENSENET121_XRAY = "densenet121_xray"


class ModelLoader(ForgeModel):
    """DenseNet model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        # Torchvision variants
        ModelVariant.DENSENET121: DenseNetConfig(
            pretrained_model_name="densenet121",
            source=ModelSource.TORCHVISION,
        ),
        ModelVariant.DENSENET161: DenseNetConfig(
            pretrained_model_name="densenet161",
            source=ModelSource.TORCHVISION,
        ),
        ModelVariant.DENSENET169: DenseNetConfig(
            pretrained_model_name="densenet169",
            source=ModelSource.TORCHVISION,
        ),
        ModelVariant.DENSENET201: DenseNetConfig(
            pretrained_model_name="densenet201",
            source=ModelSource.TORCHVISION,
        ),
        # X-ray variants
        ModelVariant.DENSENET121_XRAY: DenseNetConfig(
            pretrained_model_name="densenet121-res224-all",
            source=ModelSource.TORCH_XRAY_VISION,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.DENSENET121

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

        # Get source from variant config
        source = cls._VARIANTS[variant].source

        return ModelInfo(
            model="densenet",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_CLS,
            source=source,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load pretrained DenseNet model for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The DenseNet model instance.
        """
        # Get the pretrained model name and source from the instance's variant config
        model_name = self._variant_config.pretrained_model_name
        source = self._variant_config.source

        if source == ModelSource.TORCH_XRAY_VISION:
            if not XRAY_AVAILABLE:
                raise ImportError(
                    "torchxrayvision is required for X-ray models. Install it with: pip install torchxrayvision"
                )
            # Load X-ray model using torchxrayvision
            os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
            model = xrv.models.get_model(model_name)
        elif source == ModelSource.TORCHVISION:
            # Load model from torchvision
            # Get the weights class name (e.g., "densenet121" -> "DenseNet121_Weights")
            weight_class_name = model_name.replace("densenet", "DenseNet") + "_Weights"

            # Get the weights class and model function
            weights = getattr(models, weight_class_name).DEFAULT
            model_func = getattr(models, model_name)
            model = model_func(weights=weights)
        else:
            raise ValueError(f"Unsupported DenseNet source: {source}")

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
        source = self._variant_config.source

        # Handle X-ray variant separately (requires special preprocessing)
        if source == ModelSource.TORCH_XRAY_VISION:
            if not XRAY_AVAILABLE:
                raise ImportError(
                    "torchxrayvision is required for X-ray models. Install it with: pip install torchxrayvision"
                )
            from ...tools.utils import get_file

            # Use X-ray specific preprocessing
            img_path = get_file(
                "https://huggingface.co/spaces/torchxrayvision/torchxrayvision-classifier/resolve/main/16747_3_1.jpg"
            )
            img = skimage.io.imread(str(img_path))
            img = xrv.datasets.normalize(img, 255)
            # Check that images are 2D arrays
            if len(img.shape) > 2:
                img = img[:, :, 0]
            if len(img.shape) < 2:
                raise ValueError("error, dimension lower than 2 for image")
            # Add color channel
            img = img[None, :, :]
            transform = torchvision.transforms.Compose(
                [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)]
            )
            img = transform(img)
            inputs = torch.from_numpy(img).unsqueeze(0)

            # Replicate tensors for batch size
            inputs = inputs.repeat_interleave(batch_size, dim=0)

            # Only convert dtype if explicitly requested
            if dtype_override is not None:
                inputs = inputs.to(dtype_override)

            return inputs

        # Standard torchvision preprocessing using VisionPreprocessor
        if self._preprocessor is None:
            model_name = self._variant_config.pretrained_model_name

            def weight_class_name_fn(name: str) -> str:
                return name.replace("densenet", "DenseNet") + "_Weights"

            self._preprocessor = VisionPreprocessor(
                model_source=source,
                model_name=model_name,
                high_res_size=None,
                weight_class_name_fn=(
                    weight_class_name_fn if source == ModelSource.TORCHVISION else None
                ),
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
        source = self._variant_config.source

        # Handle X-ray variant separately (requires special postprocessing)
        if source == ModelSource.TORCH_XRAY_VISION:
            if output is not None:
                # For X-ray, return raw output or process with op_norm
                if hasattr(self, "model") and self.model is not None:
                    op_threshs = self.model.op_threshs
                    op_norm(output[0].to(torch.float32), op_threshs.to(torch.float32))
                return output
            elif co_out is not None:
                # Legacy usage
                if hasattr(self, "model") and self.model is not None:
                    op_threshs = self.model.op_threshs
                    op_norm(co_out[0].to(torch.float32), op_threshs.to(torch.float32))
                return None

        # Standard postprocessing using VisionPostprocessor
        if self._postprocessor is None:
            model_name = self._variant_config.pretrained_model_name

            self._postprocessor = VisionPostprocessor(
                model_source=source,
                model_name=model_name,
                model_instance=self.model,
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

    def post_process(self, co_out):
        """
        Post-processes the compiled model output based on the model's source (backward compatibility).

        Args:
            co_out : Output from the compiled model.
        """
        self.output_postprocess(co_out=co_out)
