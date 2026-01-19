# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
UNet model loader implementation with multiple sources (OSMR, TorchHub, SMP).
"""
import numpy as np
import torch
from typing import Optional, Callable
from dataclasses import dataclass
from PIL import Image
from torchvision import transforms

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
from ....tools.utils import get_file, VisionPreprocessor


@dataclass
class UnetConfig(ModelConfig):
    source: ModelSource
    hub_repo: Optional[str] = None
    hub_model: Optional[str] = None
    smp_encoder_name: Optional[str] = None


class ModelVariant(StrEnum):
    """Available UNet model variants."""

    OSMR_CITYSCAPES = "unet_cityscapes"
    SMP_UNET_RESNET101 = "smp_unet_resnet101"
    TORCHHUB_BRAIN_UNET = "torchhub_brain_unet"
    CARVANA_UNET = "carvana_unet"
    CARVANA_UNET_480x640 = "carvana_unet_480x640"


class ModelLoader(ForgeModel):
    """UNet model loader implementation supporting multiple sources."""

    _VARIANTS = {
        ModelVariant.OSMR_CITYSCAPES: UnetConfig(
            pretrained_model_name="unet_cityscapes",
            source=ModelSource.OSMR,
        ),
        ModelVariant.SMP_UNET_RESNET101: UnetConfig(
            pretrained_model_name="unet",
            source=ModelSource.TORCH_HUB,  # Match original test property even though loaded via SMP
            smp_encoder_name="resnet101",
        ),
        ModelVariant.TORCHHUB_BRAIN_UNET: UnetConfig(
            pretrained_model_name="unet",
            source=ModelSource.TORCH_HUB,
            hub_repo="mateuszbuda/brain-segmentation-pytorch",
            hub_model="unet",
        ),
        ModelVariant.CARVANA_UNET: UnetConfig(
            pretrained_model_name="carvana_unet",
            source=ModelSource.CUSTOM,
        ),
        ModelVariant.CARVANA_UNET_480x640: UnetConfig(
            pretrained_model_name="carvana_unet_480x640",
            source=ModelSource.CUSTOM,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.OSMR_CITYSCAPES

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        source = cls._VARIANTS[variant].source

        return ModelInfo(
            model="unet",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_SEG,
            source=source,
            framework=Framework.TORCH,
        )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None
        self._preprocessor = None

    def load_model(self, dtype_override=None):
        cfg = self._variant_config
        source = cfg.source

        if source == ModelSource.OSMR:
            from pytorchcv.model_provider import get_model as ptcv_get_model

            model = ptcv_get_model(cfg.pretrained_model_name, pretrained=False)

        elif source == ModelSource.TORCH_HUB and cfg.hub_repo is not None:
            model = torch.hub.load(
                cfg.hub_repo,
                cfg.hub_model,
                in_channels=3,
                out_channels=1,
                init_features=32,
                pretrained=True,
            )
            model.eval()

        elif source == ModelSource.TORCH_HUB and cfg.smp_encoder_name is not None:
            import segmentation_models_pytorch as smp

            model = smp.Unet(
                encoder_name=cfg.smp_encoder_name,
                encoder_weights="imagenet",
                in_channels=3,
                classes=1,
            )
            model.eval()

        else:
            from .src.unet import UNET

            model = UNET(in_channels=3, out_channels=1)

        model.eval()

        self.model = model

        if self._preprocessor is not None:
            self._preprocessor.set_cached_model(model)

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def _create_custom_preprocess_fn(self) -> Callable:
        """Create a custom preprocessing function based on the variant."""
        cfg = self._variant_config
        source = cfg.source

        if source == ModelSource.OSMR:

            def preprocess_fn(image: Image.Image) -> torch.Tensor:
                return torch.randn(1, 3, 224, 224)

        elif source == ModelSource.TORCH_HUB and cfg.hub_repo is not None:

            def preprocess_fn(image: Image.Image) -> torch.Tensor:
                from skimage.transform import resize
                from skimage.exposure import rescale_intensity

                img_array = np.array(image).astype(np.float32)

                if len(img_array.shape) == 2:
                    img_array = np.stack([img_array, img_array, img_array], axis=-1)
                elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
                    img_array = img_array[:, :, :3]

                target_size = 256
                if (
                    img_array.shape[0] != target_size
                    or img_array.shape[1] != target_size
                ):
                    img_array = resize(
                        img_array,
                        (target_size, target_size),
                        anti_aliasing=True,
                        preserve_range=True,
                    )

                for c in range(3):
                    channel = img_array[:, :, c]
                    p10 = np.percentile(channel, 10)
                    p99 = np.percentile(channel, 99)
                    channel = rescale_intensity(
                        channel, in_range=(p10, p99), out_range=(0, 1)
                    )
                    m = np.mean(channel)
                    s = np.std(channel)
                    if s > 0:
                        channel = (channel - m) / s
                    img_array[:, :, c] = channel

                img_tensor = torch.from_numpy(
                    img_array.transpose(2, 0, 1).astype(np.float32)
                )
                return img_tensor

        elif source == ModelSource.TORCH_HUB and cfg.smp_encoder_name is not None:
            import segmentation_models_pytorch as smp

            params = smp.encoders.get_preprocessing_params(cfg.smp_encoder_name)
            std = torch.tensor(params["std"]).view(1, 3, 1, 1)
            mean = torch.tensor(params["mean"]).view(1, 3, 1, 1)

            def preprocess_fn(image: Image.Image) -> torch.Tensor:
                img = image.convert("RGB")
                img_tensor = transforms.ToTensor()(img).unsqueeze(0)

                _, _, h, w = img_tensor.shape
                output_stride = 32
                new_h = ((h - 1) // output_stride + 1) * output_stride
                new_w = ((w - 1) // output_stride + 1) * output_stride

                if h != new_h or w != new_w:
                    pad_h = new_h - h
                    pad_w = new_w - w
                    img_tensor = torch.nn.functional.pad(
                        img_tensor, (0, pad_w, 0, pad_h), mode="constant", value=0
                    )

                return (img_tensor - mean) / std

        elif self._variant == ModelVariant.CARVANA_UNET_480x640:

            def preprocess_fn(image: Image.Image) -> torch.Tensor:
                return torch.rand(1, 3, 480, 640)

        else:

            def preprocess_fn(image: Image.Image) -> torch.Tensor:
                return torch.rand(1, 3, 224, 224)

        return preprocess_fn

    def _get_default_image_for_variant(self) -> Optional[str]:
        """Get default image URL for variants that need it."""
        cfg = self._variant_config
        source = cfg.source

        if source == ModelSource.TORCH_HUB and cfg.hub_repo is not None:
            return "https://github.com/mateuszbuda/brain-segmentation-pytorch/raw/master/assets/TCGA_CS_4944.png"
        elif source == ModelSource.TORCH_HUB and cfg.smp_encoder_name is not None:
            return "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
        else:
            return None

    def input_preprocess(self, dtype_override=None, batch_size=1, image=None):
        """Preprocess input image(s) and return model-ready input tensor.

        Args:
            dtype_override: Optional torch.dtype override (default: float32).
            batch_size: Batch size (ignored if image is a list).
            image: PIL Image, URL string, tensor, list of images/URLs, or None (uses default for variant).

        Returns:
            torch.Tensor: Preprocessed input tensor.
        """
        if self._preprocessor is None:
            cfg = self._variant_config
            source = cfg.source

            default_image_url = self._get_default_image_for_variant()
            custom_preprocess_fn = self._create_custom_preprocess_fn()

            self._preprocessor = VisionPreprocessor(
                model_source=ModelSource.CUSTOM,
                model_name=cfg.pretrained_model_name,
                default_image_url=default_image_url,
                custom_preprocess_fn=custom_preprocess_fn,
            )

            if hasattr(self, "model") and self.model is not None:
                self._preprocessor.set_cached_model(self.model)

        if image is None and self._get_default_image_for_variant() is None:
            image = Image.new("RGB", (224, 224), color=(128, 128, 128))

        return self._preprocessor.preprocess(
            image=image,
            dtype_override=dtype_override,
            batch_size=batch_size,
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

    def _extract_output_tensor(self, output) -> Optional[torch.Tensor]:
        """Extract and normalize output to always be a torch.Tensor.

        Args:
            output: Model output (torch.Tensor, list/tuple of tensors, or other).

        Returns:
            torch.Tensor if output contains a valid tensor, None otherwise.
        """
        if isinstance(output, torch.Tensor):
            return output
        elif isinstance(output, (list, tuple)) and len(output) > 0:
            if isinstance(output[0], torch.Tensor):
                return output[0]
        return None

    def output_postprocess(
        self,
        output=None,
        co_out=None,
        framework_model=None,
        compiled_model=None,
        inputs=None,
        dtype_override=None,
        threshold=0.5,
        apply_lcc=True,
    ):
        """Post-process model outputs for segmentation tasks.

        For brain segmentation UNet (TORCHHUB_BRAIN_UNET variant), applies:
        1. Thresholding (default 0.5) to convert probabilities to binary mask
        2. Largest connected component filtering (if medpy available and apply_lcc=True)

        Args:
            output: Model output tensor (returns dict with processed mask if provided).
            co_out: Compiled model outputs (legacy, prints results).
            framework_model: Original framework model (legacy).
            compiled_model: Compiled model (legacy).
            inputs: Input images (legacy).
            dtype_override: Optional dtype override (legacy).
            threshold: Threshold for binary mask conversion (default: 0.5).
            apply_lcc: Whether to apply largest connected component filtering (default: True).

        Returns:
            dict or None: For brain segmentation, returns dict with:
                        - "output": raw output tensor (torch.Tensor) - original model output
                        - "output_numpy": raw output as numpy array
                        - "output_shape": original output shape
                        - "output_dtype": original output dtype
                        - "mask": binary mask after thresholding and LCC (if applicable) - numpy array
                        - "mask_shape": mask shape
                        - "threshold": threshold value used
                        - "lcc_applied": whether LCC was applied
                        If output is None, returns None (for backward compatibility).
        """
        cfg = self._variant_config
        source = cfg.source

        if (
            output is not None
            and source == ModelSource.TORCH_HUB
            and cfg.hub_repo is not None
        ):
            # Extract output tensor, normalizing to always be a torch.Tensor
            output_tensor = self._extract_output_tensor(output)
            if output_tensor is None:
                return None

            output_np = output_tensor.detach().cpu().numpy()
            binary_mask = (output_np > threshold).astype(np.float32)

            lcc_actually_applied = False
            if apply_lcc:
                try:
                    from medpy.filter.binary import largest_connected_component

                    if len(binary_mask.shape) == 4:
                        for b in range(binary_mask.shape[0]):
                            for c in range(binary_mask.shape[1]):
                                mask_2d = binary_mask[b, c]
                                if np.any(mask_2d):
                                    mask_2d_int = np.round(mask_2d).astype(int)
                                    binary_mask[b, c] = largest_connected_component(
                                        mask_2d_int
                                    ).astype(np.float32)
                                    lcc_actually_applied = True
                    elif len(binary_mask.shape) == 3:
                        if binary_mask.shape[0] == 1:
                            mask_2d = binary_mask[0]
                            if np.any(mask_2d):
                                mask_2d_int = np.round(mask_2d).astype(int)
                                binary_mask[0] = largest_connected_component(
                                    mask_2d_int
                                ).astype(np.float32)
                                lcc_actually_applied = True
                        elif binary_mask.shape[0] <= 3:
                            for c in range(binary_mask.shape[0]):
                                mask_2d = binary_mask[c]
                                if np.any(mask_2d):
                                    mask_2d_int = np.round(mask_2d).astype(int)
                                    binary_mask[c] = largest_connected_component(
                                        mask_2d_int
                                    ).astype(np.float32)
                                    lcc_actually_applied = True
                        else:
                            for b in range(binary_mask.shape[0]):
                                mask_2d = binary_mask[b]
                                if np.any(mask_2d):
                                    mask_2d_int = np.round(mask_2d).astype(int)
                                    binary_mask[b] = largest_connected_component(
                                        mask_2d_int
                                    ).astype(np.float32)
                                    lcc_actually_applied = True
                    elif len(binary_mask.shape) == 2:
                        if np.any(binary_mask):
                            mask_2d_int = np.round(binary_mask).astype(int)
                            binary_mask = largest_connected_component(
                                mask_2d_int
                            ).astype(np.float32)
                            lcc_actually_applied = True
                except ImportError:
                    pass

            return {
                "output": output_tensor,
                "output_numpy": output_np,
                "output_shape": list(output_tensor.shape),
                "output_dtype": str(output_tensor.dtype),
                "mask": binary_mask,
                "mask_shape": list(binary_mask.shape),
                "threshold": threshold,
                "lcc_applied": lcc_actually_applied,
            }

        if output is not None:
            # Extract output tensor, normalizing to always be a torch.Tensor
            output_tensor = self._extract_output_tensor(output)
            if output_tensor is None:
                return None

            return {
                "output": output_tensor,
                "output_numpy": output_tensor.detach().cpu().numpy(),
                "output_shape": list(output_tensor.shape),
                "output_dtype": str(output_tensor.dtype),
            }

        return None
