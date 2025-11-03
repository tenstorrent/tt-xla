# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Panoptic FPN model loader implementation based on CPU inference patterns
"""

from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import torch
import numpy as np
import importlib.util

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


@dataclass
class PanopticFPNConfig(ModelConfig):
    """Configuration specific to Panoptic FPN models"""

    config_file: str
    backbone: str
    num_classes: int
    image_size: tuple


class ModelVariant(StrEnum):
    """Available Panoptic FPN model variants."""

    # COCO variants
    RESNET_50_1X_COCO = "resnet50_1x_coco"
    RESNET_50_3X_COCO = "resnet50_3x_coco"
    RESNET_101_3X_COCO = "resnet101_3x_coco"


class ModelLoader(ForgeModel):
    """Panoptic FPN model loader implementation with tensor-based API (no image preprocessing)."""

    _VARIANTS = {
        ModelVariant.RESNET_50_1X_COCO: PanopticFPNConfig(
            pretrained_model_name="panoptic_fpn_R_50_1x",
            config_file="COCO-PanopticSegmentation/panoptic_fpn_R_50_1x.yaml",
            backbone="resnet50",
            num_classes=80,  # COCO instance classes
            image_size=(640, 640),
        ),
        ModelVariant.RESNET_50_3X_COCO: PanopticFPNConfig(
            pretrained_model_name="panoptic_fpn_R_50_3x",
            config_file="COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml",
            backbone="resnet50",
            num_classes=80,
            image_size=(640, 640),
        ),
        ModelVariant.RESNET_101_3X_COCO: PanopticFPNConfig(
            pretrained_model_name="panoptic_fpn_R_101_3x",
            config_file="COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml",
            backbone="resnet101",
            num_classes=80,
            image_size=(640, 640),
        ),
    }

    DEFAULT_VARIANT = ModelVariant.RESNET_50_1X_COCO

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.predictor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[StrEnum] = None) -> ModelInfo:
        """Get model information for the specified variant."""
        variant = variant or cls.DEFAULT_VARIANT

        return ModelInfo(
            model="panoptic_fpn",
            variant=variant,
            framework=Framework.TORCH,
            task=ModelTask.CV_PANOPTIC_SEG,
            source=ModelSource.DETECTRON2,
            group=ModelGroup.RED,
        )

    def _setup_cfg(
        self, device: str = "cpu", dtype_override: Optional[torch.dtype] = None
    ) -> "CfgNode":
        """Setup detectron2 configuration based on the CPU inference script.

        Args:
            device: Device to run inference on (default: cpu)
            dtype_override: Override model dtype (currently not used by detectron2)

        Returns:
            CfgNode: Configured detectron2 config object
        """
        # Import the model.py implementation
        try:
            import sys
            import os

            model_path = os.path.join(os.path.dirname(__file__), "model.py")
            spec = importlib.util.spec_from_file_location("panoptic_model", model_path)
            panoptic_model = importlib.util.module_from_spec(spec)
            sys.modules["panoptic_model"] = panoptic_model
            spec.loader.exec_module(panoptic_model)
        except ImportError as e:
            raise ImportError(f"Failed to import model.py: {e}")

        config = self._variant_config

        # Use the get_cfg() function from model.py
        cfg = panoptic_model.get_cfg()

        # Load the appropriate config file
        try:
            config_file = panoptic_model.get_config_file(config.config_file)
            cfg.merge_from_file(config_file)
        except Exception:
            # If config file not found, use default configuration
            cfg.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"
            cfg.MODEL.WEIGHTS = ""

        # Set device
        cfg.MODEL.DEVICE = device

        # Set confidence thresholds
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

        # Setup metadata for COCO panoptic dataset
        metadata = panoptic_model.MetadataCatalog.get("coco_2017_val_panoptic")
        metadata.thing_classes = (
            [
                "person",
                "bicycle",
                "car",
                "motorcycle",
                "airplane",
                "bus",
                "train",
                "truck",
                "boat",
            ]
            + [
                "traffic light",
                "fire hydrant",
                "stop sign",
                "parking meter",
                "bench",
                "bird",
                "cat",
                "dog",
                "horse",
                "sheep",
                "cow",
                "elephant",
                "bear",
                "zebra",
                "giraffe",
            ]
            + [
                "backpack",
                "umbrella",
                "handbag",
                "tie",
                "suitcase",
                "frisbee",
                "skis",
                "snowboard",
                "sports ball",
                "kite",
                "baseball bat",
                "baseball glove",
                "skateboard",
                "surfboard",
                "tennis racket",
            ]
            + [
                "bottle",
                "wine glass",
                "cup",
                "fork",
                "knife",
                "spoon",
                "bowl",
                "banana",
                "apple",
                "sandwich",
                "orange",
                "broccoli",
                "carrot",
                "hot dog",
                "pizza",
                "donut",
                "cake",
            ]
            + [
                "chair",
                "couch",
                "potted plant",
                "bed",
                "dining table",
                "toilet",
                "tv",
                "laptop",
                "mouse",
                "remote",
                "keyboard",
                "cell phone",
                "microwave",
                "oven",
                "toaster",
                "sink",
                "refrigerator",
                "book",
                "clock",
                "vase",
                "scissors",
                "teddy bear",
                "hair drier",
                "toothbrush",
            ]
        )
        metadata.stuff_classes = (
            [
                "banner",
                "blanket",
                "bridge",
                "cardboard",
                "counter",
                "curtain",
                "door-stuff",
                "floor-wood",
                "flower",
                "fruit",
                "gravel",
                "house",
                "light",
                "mirror-stuff",
                "net",
                "pillow",
                "platform",
            ]
            + [
                "playingfield",
                "railroad",
                "river",
                "road",
                "roof",
                "sand",
                "sea",
                "shelf",
                "snow",
                "stairs",
                "tent",
                "towel",
                "wall-brick",
                "wall-stone",
                "wall-tile",
                "wall-wood",
                "water-other",
                "window-blind",
                "window-other",
            ]
            + [
                "tree-merged",
                "fence",
                "ceiling",
                "sky-other",
                "cabinet",
                "table",
                "floor-other",
                "pavement",
                "mountain",
                "grass",
                "dirt",
                "paper",
                "food-other",
                "building-other",
                "rock",
                "wall-other",
                "rug",
            ]
        )

        return cfg

    def load_model(self, **kwargs) -> torch.nn.Module:
        """Load and return the Panoptic FPN model instance.

        Args:
            **kwargs: Additional model-specific arguments.
                     - dtype_override: Override model dtype (e.g., torch.bfloat16)
                     - device: Device to load model on (default: cpu)
                     - force_cpu: Force CPU usage even if CUDA is available

        Returns:
            torch.nn.Module: The Panoptic FPN model instance
        """
        dtype_override = kwargs.get("dtype_override", torch.float32)
        device = kwargs.get("device", "cpu")
        force_cpu = kwargs.get("force_cpu", True)

        # Force CPU usage (following CPU inference script pattern)
        if force_cpu:
            torch.cuda.is_available = lambda: False
            device = "cpu"

        # Setup configuration
        cfg = self._setup_cfg(device=device, dtype_override=dtype_override)

        # Create predictor using the model.py implementation
        try:
            # Import the model.py implementation
            import sys
            import os

            model_path = os.path.join(os.path.dirname(__file__), "model.py")
            spec = importlib.util.spec_from_file_location("panoptic_model", model_path)
            panoptic_model = importlib.util.module_from_spec(spec)
            sys.modules["panoptic_model"] = panoptic_model
            spec.loader.exec_module(panoptic_model)

            self.predictor = panoptic_model.DefaultPredictor(cfg)
            model = self.predictor.model
        except Exception as e:
            raise RuntimeError(f"Failed to create Panoptic FPN model: {str(e)}") from e

        # Apply dtype override if specified and different from float32
        if dtype_override != torch.float32:
            model = model.to(dtype=dtype_override)

        model.eval()
        return model

    def load_inputs(self, **kwargs) -> List[torch.Tensor]:
        """Load and return sample inputs for the model as tensors (C, H, W format).

        Args:
            **kwargs: Additional input-specific arguments.
                     - dtype_override: Override input dtype (e.g., torch.bfloat16)
                     - batch_size: Batch size for inputs (default: 1)
                     - image_size: Override image size (default: from config)

        Returns:
            List[torch.Tensor]: Sample inputs as tensors (C, H, W format)
        """
        config = self._variant_config
        dtype_override = kwargs.get("dtype_override", torch.float32)
        batch_size = kwargs.get("batch_size", 1)
        image_size = kwargs.get("image_size", config.image_size)

        inputs = []
        for i in range(batch_size):
            # Create random tensor in (C, H, W) format expected by the model
            tensor_input = torch.randn(
                3, image_size[0], image_size[1], dtype=dtype_override
            )
            inputs.append(tensor_input)

        return inputs

    def predict(self, inputs: List[torch.Tensor]) -> List[Dict[str, Any]]:
        """Run inference using tensors directly (bypassing image preprocessing).

        Args:
            inputs: List of input tensors in (C, H, W) format

        Returns:
            List of prediction dictionaries
        """
        if self.predictor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        results = []
        for tensor_input in inputs:
            if isinstance(tensor_input, torch.Tensor):
                # Run inference directly on the model with tensor input
                predictions = self._predict_tensor(tensor_input)
                results.append(predictions)
            else:
                raise ValueError("Input must be a torch.Tensor")

        return results

    def _predict_tensor(self, tensor_input: torch.Tensor) -> Dict[str, Any]:
        """Run inference on a single tensor input.

        Args:
            tensor_input: Input tensor in (C, H, W) format

        Returns:
            Prediction dictionary
        """
        with torch.no_grad():
            # Ensure tensor is on the correct device
            tensor_input = tensor_input.to(self.predictor.cfg.MODEL.DEVICE)

            # Create input dict in the format expected by detectron2 models
            height, width = tensor_input.shape[1], tensor_input.shape[2]
            inputs = {"image": tensor_input, "height": height, "width": width}

            # Run inference directly on the model
            predictions = self.predictor.model([inputs])[0]

            return predictions

    @classmethod
    def decode_output(cls, outputs: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Decode Panoptic FPN model outputs into human-readable format.

        Args:
            outputs: List of model output dictionaries from predict()
            **kwargs: Additional decoding arguments

        Returns:
            Decoded outputs with panoptic and semantic segmentation results
        """
        if not isinstance(outputs, list):
            outputs = [outputs]

        decoded_results = {"num_predictions": len(outputs), "predictions": []}

        for i, output in enumerate(outputs):
            decoded_pred = {"prediction_id": i}

            if isinstance(output, dict):
                for key, value in output.items():
                    if (
                        key == "panoptic_seg"
                        and isinstance(value, (tuple, list))
                        and len(value) == 2
                    ):
                        seg_map, segments_info = value
                        decoded_pred["panoptic_seg_shape"] = (
                            tuple(seg_map.shape) if hasattr(seg_map, "shape") else None
                        )
                        decoded_pred["num_segments"] = (
                            len(segments_info) if segments_info else 0
                        )
                        decoded_pred["segment_ids"] = (
                            [seg.get("id", -1) for seg in segments_info]
                            if segments_info
                            else []
                        )
                    elif key == "sem_seg" and hasattr(value, "shape"):
                        decoded_pred["sem_seg_shape"] = tuple(value.shape)
                        decoded_pred["num_semantic_classes"] = (
                            int(torch.unique(value).numel())
                            if isinstance(value, torch.Tensor)
                            else None
                        )
                    elif key == "instances" and hasattr(value, "__len__"):
                        decoded_pred["num_instances"] = len(value)
                        if hasattr(value, "pred_classes"):
                            decoded_pred[
                                "instance_classes"
                            ] = value.pred_classes.tolist()
                        if hasattr(value, "pred_boxes"):
                            decoded_pred["num_boxes"] = len(value.pred_boxes)
                    else:
                        decoded_pred[key] = (
                            str(type(value))
                            if not isinstance(value, (int, float, str, bool))
                            else value
                        )

            decoded_results["predictions"].append(decoded_pred)

        return decoded_results

    def post_processing(self, outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Post-process model outputs and print results."""
        decoded = self.decode_output(outputs)

        print("Panoptic FPN Inference Results:")
        print(f"  Number of predictions: {decoded['num_predictions']}")

        for pred in decoded["predictions"]:
            pred_id = pred["prediction_id"]
            print(f"  Prediction {pred_id}:")
            for key, value in pred.items():
                if key != "prediction_id":
                    print(f"    {key}: {value}")

        return decoded
