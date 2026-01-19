# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
RCNN model loader implementation for object detection
"""
import cv2
import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision
import torchvision.transforms as transforms
from typing import Optional, List, Generator, Tuple
import numpy as np

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...tools.utils import get_file

# Default image for testing
DEFAULT_IMAGE_PATH = "forge/test/models/files/samples/images/car.jpg"


class ModelVariant(StrEnum):
    """Available RCNN model variants."""

    ALEXNET = "alexnet"


class RCNNModel(nn.Module):
    """RCNN model that uses AlexNet as backbone with modified classifier."""

    def __init__(self, num_classes=2):
        super(RCNNModel, self).__init__()

        # Load AlexNet model
        self.backbone = torchvision.models.alexnet(pretrained=True)

        # Get number of features from the last classifier layer
        num_features = self.backbone.classifier[6].in_features

        # Create class specific linear SVMs [Refer Section 2 in paper]
        svm_layer = nn.Linear(num_features, num_classes)

        # Initialize the SVM layer
        init.normal_(svm_layer.weight, mean=0, std=0.01)
        init.constant_(svm_layer.bias, 0)

        # Replace AlexNet's ImageNet specific 1000-way classification layer
        # with a (N + 1)-way classification layer (N object classes + 1 background)
        self.backbone.classifier[6] = svm_layer

    def forward(self, x):
        return self.backbone(x)


class ModelLoader(ForgeModel):
    """RCNN model loader implementation."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.ALEXNET: ModelConfig(
            pretrained_model_name="torchvision/alexnet",
        )
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.ALEXNET

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
        return ModelInfo(
            model="rcnn",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.TORCHVISION,
            framework=Framework.TORCH,
        )

    def __init__(self, variant: Optional[ModelVariant] = None, num_classes: int = 2):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
            num_classes: Number of classes for classification (default: 2)
        """
        super().__init__(variant)
        self.num_classes = num_classes

    def load_model(self, dtype_override=None):
        """Load and return the RCNN model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype.

        Returns:
            torch.nn.Module: The RCNN model instance.
        """
        model = RCNNModel(num_classes=self.num_classes)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype=dtype_override)

        return model

    def load_inputs(
        self,
        batch_size=1,
        dtype_override=None,
        image_path=None,
        use_selective_search=False,
    ):
        """Generate sample inputs for the RCNN model.

        Args:
            batch_size: Number of samples in the batch
            dtype_override: Optional torch.dtype to override input dtype
            image_path: Optional path to input image. If None, uses default test image.
            use_selective_search: Whether to use selective search for region proposals

        Returns:
            List of input tensors matching the expected RCNN input format
        """
        if image_path is None:
            image_path = DEFAULT_IMAGE_PATH

        # Load image
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise FileNotFoundError(f"Could not load image from {image_path}")
        except Exception:
            # Create a dummy image if file not found
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Define transforms
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((227, 227)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        if use_selective_search:
            # Use selective search for region proposals (as in the original test)
            try:
                gs = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
                gs.setBaseImage(img)
                gs.switchToSelectiveSearchFast()
                rects = gs.process()
                rects[:, 2] += rects[:, 0]
                rects[:, 3] += rects[:, 1]

                # Take the first few rectangles for batch processing
                selected_rects = rects[:batch_size]
                inputs = []

                for rect in selected_rects:
                    xmin, ymin, xmax, ymax = rect
                    rect_img = img[ymin:ymax, xmin:xmax]
                    rect_transform = transform(rect_img)
                    inputs.append(rect_transform.unsqueeze(0))

                # Stack all inputs
                batch_input = torch.cat(inputs, dim=0)

            except Exception:
                # Fallback to whole image if selective search fails
                batch_input = torch.stack([transform(img) for _ in range(batch_size)])
        else:
            # Use whole image (simpler approach)
            batch_input = torch.stack([transform(img) for _ in range(batch_size)])

        if dtype_override is not None:
            batch_input = batch_input.to(dtype=dtype_override)

        return [batch_input]

    def generate_region_proposals_iterator(
        self, image_path=None, dtype_override=None
    ) -> Generator[Tuple[int, List[torch.Tensor]], None, None]:
        """Generate region proposals using selective search and yield them one by one.

        This method replicates the exact behavior of the original test file,
        yielding individual processed regions that can be used in a loop.

        Args:
            image_path: Path to the input image
            dtype_override: Optional torch.dtype to override input dtype

        Yields:
            Tuple of (index, [input_tensor]) for each region proposal
        """
        if image_path is None:
            image_path = DEFAULT_IMAGE_PATH

        try:
            img = cv2.imread(image_path)
            if img is None:
                raise FileNotFoundError(f"Could not load image from {image_path}")
        except Exception:
            # Create a dummy image if file not found
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Define transforms
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((227, 227)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        try:
            # Selective search - exactly as in the original test
            gs = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
            gs.setBaseImage(img)
            gs.switchToSelectiveSearchFast()
            rects = gs.process()
            rects[:, 2] += rects[:, 0]
            rects[:, 3] += rects[:, 1]

            print("Suggested number of proposals: %d" % len(rects))

            # Process each rectangle exactly as in the original test
            for idx, rect in enumerate(rects):
                xmin, ymin, xmax, ymax = rect
                rect_img = img[ymin:ymax, xmin:xmax]

                rect_transform = transform(rect_img)

                inputs = [rect_transform.unsqueeze(0)]

                if dtype_override is not None:
                    inputs = [inp.to(dtype_override) for inp in inputs]

                yield idx, inputs

        except Exception as e:
            print(f"Error in selective search: {e}")
            # Fallback: yield a single processed whole image
            whole_img_transform = transform(img)
            inputs = [whole_img_transform.unsqueeze(0)]

            if dtype_override is not None:
                inputs = [inp.to(dtype_override) for inp in inputs]

            yield 0, inputs

    def generate_region_proposals(self, image_path=None, max_proposals=10):
        """Generate region proposals using selective search.

        Args:
            image_path: Path to the input image
            max_proposals: Maximum number of proposals to return

        Returns:
            List of region proposals as [xmin, ymin, xmax, ymax] coordinates
        """
        if image_path is None:
            image_path = DEFAULT_IMAGE_PATH

        try:
            img = cv2.imread(image_path)
            if img is None:
                raise FileNotFoundError(f"Could not load image from {image_path}")

            # Selective search for region proposals
            gs = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
            gs.setBaseImage(img)
            gs.switchToSelectiveSearchFast()
            rects = gs.process()

            # Convert to [xmin, ymin, xmax, ymax] format
            rects[:, 2] += rects[:, 0]
            rects[:, 3] += rects[:, 1]

            # Return limited number of proposals
            return rects[:max_proposals].tolist()

        except Exception as e:
            print(f"Error in selective search: {e}")
            # Return dummy proposals if selective search fails
            return [[50, 50, 200, 200] for _ in range(max_proposals)]

    def preprocess_region(self, image, rect):
        """Preprocess a specific region of an image.

        Args:
            image: Input image (numpy array or PIL Image)
            rect: Region coordinates [xmin, ymin, xmax, ymax]

        Returns:
            torch.Tensor: Preprocessed region tensor
        """
        xmin, ymin, xmax, ymax = rect

        # Extract region from image
        if isinstance(image, np.ndarray):
            rect_img = image[ymin:ymax, xmin:xmax]
        else:
            rect_img = image

        # Apply transforms
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((227, 227)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        return transform(rect_img).unsqueeze(0)
