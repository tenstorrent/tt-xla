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

# Lazy imports - for test being collected

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
from ...tools.utils import get_file


@dataclass
class PanopticFPNConfig(ModelConfig):
    """Configuration specific to Panoptic FPN models"""

    config_file: str
    backbone: str
    num_classes: int
    image_size: tuple
    checkpoint_url: Optional[str] = None
    sample_image_url: Optional[str] = None


class ModelVariant(StrEnum):
    """Available Panoptic FPN model variants."""

    # COCO variants
    RESNET_50_1X_COCO = "resnet50_1x_coco"
    RESNET_50_3X_COCO = "resnet50_3x_coco"
    RESNET_101_3X_COCO = "resnet101_3x_coco"


# COCO class definitions for metadata
THING_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
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

STUFF_CLASSES = [
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
    "tree-merged",
    "fence-merged",
    "ceiling-merged",
    "sky-other-merged",
    "cabinet-merged",
    "table-merged",
    "floor-other-merged",
    "pavement-merged",
    "mountain-merged",
    "grass-merged",
    "dirt-merged",
    "paper-merged",
    "food-other-merged",
    "building-other-merged",
    "rock-merged",
    "wall-other-merged",
    "rug-merged",
]


class ModelLoader(ForgeModel):
    """Panoptic FPN model loader implementation with tensor-based API (no image preprocessing)."""

    _VARIANTS = {
        ModelVariant.RESNET_50_1X_COCO: PanopticFPNConfig(
            pretrained_model_name="panoptic_fpn_R_50_1x",
            config_file="COCO-PanopticSegmentation/panoptic_fpn_R_50_1x.yaml",
            backbone="resnet50",
            num_classes=80,  # COCO instance classes
            image_size=(640, 640),
            checkpoint_url="https://dl.fbaipublicfiles.com/detectron2/COCO-PanopticSegmentation/panoptic_fpn_R_50_1x/139514544/model_final_dbfeb4.pkl",
            sample_image_url="http://images.cocodataset.org/val2017/000000439715.jpg",
        ),
        ModelVariant.RESNET_50_3X_COCO: PanopticFPNConfig(
            pretrained_model_name="panoptic_fpn_R_50_3x",
            config_file="COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml",
            backbone="resnet50",
            num_classes=80,
            image_size=(640, 640),
            checkpoint_url="https://dl.fbaipublicfiles.com/detectron2/COCO-PanopticSegmentation/panoptic_fpn_R_50_3x/139514569/model_final_c10459.pkl",
            sample_image_url="http://images.cocodataset.org/val2017/000000439715.jpg",
        ),
        ModelVariant.RESNET_101_3X_COCO: PanopticFPNConfig(
            pretrained_model_name="panoptic_fpn_R_101_3x",
            config_file="COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml",
            backbone="resnet101",
            num_classes=80,
            image_size=(640, 640),
            checkpoint_url="https://dl.fbaipublicfiles.com/detectron2/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x/139514519/model_final_cafdb1.pkl",
            sample_image_url="http://images.cocodataset.org/val2017/000000439715.jpg",
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

    def load_model(self, **kwargs):
        """Load and return the Panoptic FPN model with DefaultPredictor.

        Returns:
            DefaultPredictor: The predictor instance ready for inference
        """
        # Lazy import - only import when actually loading the model
        from .src.model import (
            get_cfg,
            DefaultPredictor,
            get_config_file,
            get_checkpoint_url,
        )

        # Get configuration
        cfg = get_cfg()
        config = self._variant_config

        # Load config file
        config_file = get_config_file(config.config_file)
        cfg.merge_from_file(config_file)

        # Download and cache checkpoint weights using get_file utility
        # This prevents hanging on network requests during DefaultPredictor initialization
        checkpoint_url = get_checkpoint_url(config.config_file)
        checkpoint_path = get_file(checkpoint_url)

        # Set model weights to local cached file path instead of URL
        # yacs configs only accept builtin types, so store path as string
        cfg.MODEL.WEIGHTS = str(checkpoint_path)

        # Set device
        cfg.MODEL.DEVICE = "cpu"

        # Add confidence thresholds
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

        # Create predictor (handles model construction + weight loading)
        self.predictor = DefaultPredictor(cfg)

        # Keep reference to the underlying torch.nn.Module for tester expectations
        self._model = self.predictor.model

        # Setup metadata for downstream decoding/support tooling
        self._setup_metadata(cfg)

        # Tests expect a torch.nn.Module instance
        return self._model

    def _setup_metadata(self, cfg):
        """Setup metadata for COCO panoptic dataset like in panoptic_seg.py."""
        # Lazy import - only import when actually setting up metadata
        from .src.model import MetadataCatalog

        dataset_names = list(cfg.DATASETS.TRAIN) + list(cfg.DATASETS.TEST)
        for name in dataset_names:
            if not name:
                continue
            metadata = MetadataCatalog.get(name)
            if not hasattr(metadata, "thing_classes"):
                metadata.thing_classes = THING_CLASSES
            if not hasattr(metadata, "stuff_classes"):
                metadata.stuff_classes = STUFF_CLASSES

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

        batched_inputs: List[Dict[str, Any]] = []
        for _ in range(batch_size):
            # Create random tensor in (C, H, W) format expected by the model
            tensor_input = torch.rand(
                3, image_size[0], image_size[1], dtype=dtype_override
            )
            batched_inputs.append(
                {
                    "image": tensor_input,
                    "height": image_size[0],
                    "width": image_size[1],
                }
            )

        # Detectron2 models expect a single argument (list of dicts)
        return (batched_inputs,)

    def predict(self, image_path: Optional[str] = None) -> Dict[str, Any]:
        """Run inference on an image using the loaded predictor.

        Args:
            image_path: Optional path to image file. If None, uses sample image URL.

        Returns:
            Dict containing panoptic segmentation results
        """
        # Lazy imports - only import when actually running prediction
        import cv2
        import urllib.request

        if self.predictor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Load image
        if image_path is None:
            # Use sample image from config
            config = self._variant_config
            if config.sample_image_url:
                image_path = config.sample_image_url

        if image_path.startswith("http"):
            # Download image from URL
            print(f"Downloading image from: {image_path}")
            with urllib.request.urlopen(image_path) as response:
                image_data = np.asarray(bytearray(response.read()), dtype="uint8")
                im = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        else:
            # Load from local file
            im = cv2.imread(image_path)

        if im is None:
            raise ValueError(f"Could not load image from {image_path}")

        # Run inference
        outputs = self.predictor(im)

        # Print results like in panoptic_seg.py
        self._print_inference_results(outputs)

        return outputs

    def _print_inference_results(self, outputs):
        """Print inference results like in panoptic_seg.py."""
        inst = outputs.get("instances")
        if inst is not None:
            print("instances:", len(inst))
            try:
                print("classes:", inst.pred_classes.tolist())
            except Exception:
                pass

        sem = outputs.get("sem_seg")
        if sem is not None:
            print(
                "sem_shape:",
                tuple(sem.shape),
                "unique_on_argmax:",
                int(sem.argmax(0).unique().numel()),
            )

        ps = outputs.get("panoptic_seg")
        if ps is not None:
            pan, segs = ps
            print(
                "panoptic_unique_ids:",
                int(torch.unique(pan).numel()),
                "segments_len:",
                0 if segs is None else len(segs),
            )

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
