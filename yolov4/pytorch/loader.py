# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
YOLOv4 model loader implementation
"""
import torch
from datasets import load_dataset
from torchvision import transforms
from typing import Optional
import os
from PIL import Image
from ...tools.utils import get_file, VisionPreprocessor

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
from .src.yolov4 import Yolov4
from .src.post_processing import (
    gen_yolov4_boxes_confs,
    get_region_boxes,
    post_processing,
    plot_boxes_cv2,
)


class ModelVariant(StrEnum):
    """Available YOLOv4 model variants."""

    BASE = "base"


class ModelLoader(ForgeModel):
    """YOLOv4 model loader implementation."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="",  # Not used
        )
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.model = None
        self._preprocessor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """

        return ModelInfo(
            model="yolov4",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.CUSTOM,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the YOLOv4 model instance with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The YOLOv4 model instance.
        """
        weights_pth = get_file("test_files/pytorch/yolov4/yolov4.pth")

        # Load weights checkpoint
        state_dict = torch.load(weights_pth, map_location="cpu")

        model = Yolov4()

        # Align keys and load weights
        new_state_dict = dict(zip(model.state_dict().keys(), state_dict.values()))
        model.load_state_dict(new_state_dict)
        model.eval()

        # Store model for potential use in input preprocessing
        self.model = model

        # Update preprocessor with cached model if it exists
        if self._preprocessor is not None:
            self._preprocessor.set_cached_model(model)

        # Apply dtype override if needed
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def input_preprocess(self, dtype_override=None, batch_size=1, image=None):
        """Preprocess input image(s) and return model-ready input tensor.

        Args:
            dtype_override: Optional torch.dtype override (default: float32).
            batch_size: Batch size (ignored if image is a list).
            image: PIL Image, URL string, tensor, list of images/URLs, or None (uses default dataset image).

        Returns:
            torch.Tensor: Preprocessed input tensor.
        """
        if self._preprocessor is None:
            # YOLOv4 uses custom preprocessing: resize to 480x640 and ToTensor
            def custom_preprocess_fn(img: Image.Image) -> torch.Tensor:
                transform = transforms.Compose(
                    [
                        transforms.Resize((480, 640)),
                        transforms.ToTensor(),
                    ]
                )
                return transform(img)

            self._preprocessor = VisionPreprocessor(
                model_source=ModelSource.CUSTOM,
                model_name="yolov4",
                custom_preprocess_fn=custom_preprocess_fn,
            )

            if hasattr(self, "model") and self.model is not None:
                self._preprocessor.set_cached_model(self.model)

        # If image is None, use huggingface cats-image dataset (backward compatibility)
        if image is None:
            dataset = load_dataset("huggingface/cats-image", split="test[:1]")
            image = dataset[0]["image"]

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

    def output_postprocess(self, co_out):
        """Post-process YOLOv4 model outputs for object detection.

        Args:
            co_out: Model output tensor from YOLOv4 forward pass.

        Returns:
            Post-processed detection results path.
        """
        y1, y2, y3 = gen_yolov4_boxes_confs(co_out)
        output = get_region_boxes([y1, y2, y3])
        results = post_processing(0.3, 0.4, output)
        coco_names_path = get_file(
            "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names"
        )
        with open(coco_names_path, "r") as f:
            class_names = [line.strip() for line in f.readlines()]

        # Print detected boxes info
        print("Detected boxes:")
        for box in results[0]:
            if len(box) >= 6:
                *coords, score, class_id = box[:6]  # in case there are more than 6
                x1, y1, x2, y2 = coords
                class_name = class_names[int(class_id)]
                print(
                    f"Class: {class_name}, Score: {score:.2f}, Box: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]"
                )

        # Load dataset
        dataset = load_dataset("huggingface/cats-image", split="test").with_format(
            "np"
        )  # get the image as an numpy array

        img_cv = dataset[0]["image"]
        output_dir = "yolov4_predictions"
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"yolov4_predicted.jpg"
        output_path = os.path.join(output_dir, output_filename)
        plot_boxes_cv2(img_cv, results[0], output_path, class_names)

        return output_path

    def unpack_forward_output(self, fwd_output):
        """Unpack forward pass output to extract a differentiable tensor.

        The YOLOv4 model returns (x4, x5, x6) where each tensor represents
        detection outputs at different scales:
        - x4: Small object detections [batch, 255, H1, W1]
        - x5: Medium object detections [batch, 255, H2, W2]
        - x6: Large object detections [batch, 255, H3, W3]

        For training, we flatten and concatenate all outputs to create a single
        tensor that allows gradients to flow through the entire network.

        Args:
            fwd_output: Output from the model's forward pass (tuple of tensors)

        Returns:
            torch.Tensor: Concatenated flattened outputs for backward pass
        """
        if isinstance(fwd_output, tuple):
            # Flatten each tensor and concatenate along dim=1
            flattened = [t.flatten(start_dim=1) for t in fwd_output]
            return torch.cat(flattened, dim=1)
        return fwd_output
