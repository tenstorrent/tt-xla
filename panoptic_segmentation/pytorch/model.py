# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import copy
import numpy as np
import torch
import os, cv2
import pkg_resources
import types
from typing import Optional
from collections import UserDict
import logging
from fvcore.common.config import CfgNode as _CfgNode


class PathManager:
    """Simplified PathManager for URL handling"""

    @staticmethod
    def get_local_path(path):
        """Download or return local path for given URL/path"""
        if path.startswith(("http://", "https://")):
            import urllib.request
            import tempfile
            import os

            try:
                # Create a temporary file
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=os.path.basename(path)
                ) as tmp_file:
                    tmp_path = tmp_file.name

                # Download the file
                urllib.request.urlretrieve(path, tmp_path)
                return tmp_path
            except Exception as e:
                print(f"Warning: Failed to download {path}, using as-is: {e}")
                return path
        return path

    @staticmethod
    def open(filename, mode="r"):
        """Open file, handling URLs"""
        if filename.startswith(("http://", "https://")):
            import urllib.request

            return urllib.request.urlopen(filename)
        return open(filename, mode)

    @staticmethod
    def isfile(filename):
        """Check if file exists, handling URLs"""
        if filename.startswith(("http://", "https://")):
            import urllib.request

            try:
                urllib.request.urlopen(filename)
                return True
            except:
                return False
        return os.path.isfile(filename)


def log_first_n(logging_level, message, *, n=10, name=None):
    """Simplified log_first_n function"""
    print(message)


class Metadata(types.SimpleNamespace):
    """
    A class that supports simple attribute setter/getter.
    It is intended for storing metadata of a dataset and make it accessible globally.

    Examples:
    ::
        # somewhere when you load the data:
        MetadataCatalog.get("mydataset").thing_classes = ["person", "dog"]

        # somewhere when you print statistics or visualize:
        classes = MetadataCatalog.get("mydataset").thing_classes
    """

    # the name of the dataset
    # set default to N/A so that `self.name` in the errors will not trigger getattr again
    name: str = "N/A"

    _RENAMED = {
        "class_names": "thing_classes",
        "dataset_id_to_contiguous_id": "thing_dataset_id_to_contiguous_id",
        "stuff_class_names": "stuff_classes",
    }

    def __getattr__(self, key):
        if key in self._RENAMED:
            log_first_n(
                logging.WARNING,
                "Metadata '{}' was renamed to '{}'!".format(key, self._RENAMED[key]),
                n=10,
            )
            return getattr(self, self._RENAMED[key])

        # "name" exists in every metadata
        if len(self.__dict__) > 1:
            raise AttributeError(
                "Attribute '{}' does not exist in the metadata of dataset '{}'. Available "
                "keys are {}.".format(key, self.name, str(self.__dict__.keys()))
            )
        else:
            raise AttributeError(
                f"Attribute '{key}' does not exist in the metadata of dataset '{self.name}': "
                "metadata is empty."
            )

    def __setattr__(self, key, val):
        if key in self._RENAMED:
            log_first_n(
                logging.WARNING,
                "Metadata '{}' was renamed to '{}'!".format(key, self._RENAMED[key]),
                n=10,
            )
            setattr(self, self._RENAMED[key], val)

        # Ensure that metadata of the same name stays consistent
        try:
            oldval = getattr(self, key)
            assert oldval == val, (
                "Attribute '{}' in the metadata of '{}' cannot be set "
                "to a different value!\n{} != {}".format(key, self.name, oldval, val)
            )
        except AttributeError:
            super().__setattr__(key, val)

    def as_dict(self):
        """
        Returns all the metadata as a dict.
        Note that modifications to the returned dict will not reflect on the Metadata object.
        """
        return copy.copy(self.__dict__)

    def set(self, **kwargs):
        """
        Set multiple metadata with kwargs.
        """
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self

    def get(self, key, default=None):
        """
        Access an attribute and return its value if exists.
        Otherwise return default.
        """
        try:
            return getattr(self, key)
        except AttributeError:
            return default


class _MetadataCatalog(UserDict):
    """
    MetadataCatalog is a global dictionary that provides access to
    :class:`Metadata` of a given dataset.

    The metadata associated with a certain name is a singleton: once created, the
    metadata will stay alive and will be returned by future calls to ``get(name)``.

    It's like global variables, so don't abuse it.
    It's meant for storing knowledge that's constant and shared across the execution
    of the program, e.g.: the class names in COCO.
    """

    def get(self, name):
        """
        Args:
            name (str): name of a dataset (e.g. coco_2014_train).

        Returns:
            Metadata: The :class:`Metadata` instance associated with this name,
            or create an empty one if none is available.
        """
        assert len(name)
        r = super().get(name, None)
        if r is None:
            r = self[name] = Metadata(name=name)
        return r

    def list(self):
        """
        List all registered metadata.

        Returns:
            list[str]: keys (names of datasets) of all registered metadata
        """
        return list(self.keys())

    def remove(self, name):
        """
        Alias of ``pop``.
        """
        self.pop(name)

    def __str__(self):
        return "MetadataCatalog(registered metadata: {})".format(", ".join(self.keys()))

    __repr__ = __str__


MetadataCatalog = _MetadataCatalog()
MetadataCatalog.__doc__ = (
    (_MetadataCatalog.__doc__ or "")
    + """
    .. automethod:: detectron2.data.catalog.MetadataCatalog.get
"""
)


class CfgNode(_CfgNode):
    @classmethod
    def _open_cfg(cls, filename):
        return PathManager.open(filename, "r")

    def merge_from_file(self, cfg_filename: str, allow_unsafe: bool = True) -> None:

        assert PathManager.isfile(
            cfg_filename
        ), f"Config file '{cfg_filename}' does not exist!"
        loaded_cfg = self.load_yaml_with_base(cfg_filename, allow_unsafe=allow_unsafe)
        loaded_cfg = type(self)(loaded_cfg)

        from detectron2.config.defaults import _C

        logger = logging.getLogger(__name__)

        loaded_ver = loaded_cfg.get("VERSION", None)
        if loaded_ver is None:
            from .compat import guess_version

            loaded_ver = guess_version(loaded_cfg, cfg_filename)
        assert (
            loaded_ver <= self.VERSION
        ), "Cannot merge a v{} config into a v{} config.".format(
            loaded_ver, self.VERSION
        )

        if loaded_ver == self.VERSION:
            self.merge_from_other_cfg(loaded_cfg)
        else:
            from .compat import upgrade_config, downgrade_config

            logger.warning(
                "Loading an old v{} config file '{}' by automatically upgrading to v{}. "
                "See docs/CHANGELOG.md for instructions to update your files.".format(
                    loaded_ver, cfg_filename, self.VERSION
                )
            )
            old_self = downgrade_config(self, to_version=loaded_ver)
            old_self.merge_from_other_cfg(loaded_cfg)
            new_config = upgrade_config(old_self)
            self.clear()
            self.update(new_config)


_C = CfgNode()
_C.VERSION = 2
_C.MODEL = CfgNode()
_C.MODEL.LOAD_PROPOSALS = False
_C.MODEL.MASK_ON = False
_C.MODEL.KEYPOINT_ON = False
_C.MODEL.DEVICE = "cuda"
_C.MODEL.META_ARCHITECTURE = "GeneralizedRCfgNodeN"
_C.MODEL.WEIGHTS = ""
_C.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675]
_C.MODEL.PIXEL_STD = [1.0, 1.0, 1.0]
_C.INPUT = CfgNode()
_C.INPUT.MIN_SIZE_TRAIN = (800,)
_C.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
_C.INPUT.MAX_SIZE_TRAIN = 1333
_C.INPUT.MIN_SIZE_TEST = 800
_C.INPUT.MAX_SIZE_TEST = 1333
_C.INPUT.RANDOM_FLIP = "horizontal"
_C.INPUT.CROP = CfgNode({"ENABLED": False})
_C.INPUT.CROP.TYPE = "relative_range"
_C.INPUT.CROP.SIZE = [0.9, 0.9]
_C.INPUT.FORMAT = "BGR"
_C.INPUT.MASK_FORMAT = "polygon"
_C.DATASETS = CfgNode()
_C.DATASETS.TRAIN = ()
_C.DATASETS.PROPOSAL_FILES_TRAIN = ()
_C.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN = 2000
_C.DATASETS.TEST = ()
_C.DATASETS.PROPOSAL_FILES_TEST = ()
_C.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST = 1000
_C.DATALOADER = CfgNode()
_C.DATALOADER.NUM_WORKERS = 4
_C.DATALOADER.ASPECT_RATIO_GROUPING = True
_C.DATALOADER.SAMPLER_TRAIN = "TrainingSampler"
_C.DATALOADER.REPEAT_THRESHOLD = 0.0
_C.DATALOADER.REPEAT_SQRT = True
_C.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True
_C.MODEL.BACKBONE = CfgNode()
_C.MODEL.BACKBONE.NAME = "build_resnet_backbone"
_C.MODEL.BACKBONE.FREEZE_AT = 2
_C.MODEL.FPN = CfgNode()
_C.MODEL.FPN.IN_FEATURES = []
_C.MODEL.FPN.OUT_CHANNELS = 256
_C.MODEL.FPN.NORM = ""
_C.MODEL.FPN.FUSE_TYPE = "sum"
_C.MODEL.PROPOSAL_GENERATOR = CfgNode()
_C.MODEL.PROPOSAL_GENERATOR.NAME = "RPN"
_C.MODEL.PROPOSAL_GENERATOR.MIN_SIZE = 0
_C.MODEL.ANCHOR_GENERATOR = CfgNode()
_C.MODEL.ANCHOR_GENERATOR.NAME = "DefaultAnchorGenerator"
_C.MODEL.ANCHOR_GENERATOR.SIZES = [[32, 64, 128, 256, 512]]
_C.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]
_C.MODEL.ANCHOR_GENERATOR.ANGLES = [[-90, 0, 90]]
_C.MODEL.ANCHOR_GENERATOR.OFFSET = 0.0
_C.MODEL.RPN = CfgNode()
_C.MODEL.RPN.HEAD_NAME = "StandardRPNHead"
_C.MODEL.RPN.IN_FEATURES = ["res4"]
_C.MODEL.RPN.BOUNDARY_THRESH = -1
_C.MODEL.RPN.IOU_THRESHOLDS = [0.3, 0.7]
_C.MODEL.RPN.IOU_LABELS = [0, -1, 1]
_C.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256
_C.MODEL.RPN.POSITIVE_FRACTION = 0.5
_C.MODEL.RPN.BBOX_REG_LOSS_TYPE = "smooth_l1"
_C.MODEL.RPN.BBOX_REG_LOSS_WEIGHT = 1.0
_C.MODEL.RPN.BBOX_REG_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
_C.MODEL.RPN.SMOOTH_L1_BETA = 0.0
_C.MODEL.RPN.LOSS_WEIGHT = 1.0
_C.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 12000
_C.MODEL.RPN.PRE_NMS_TOPK_TEST = 6000
_C.MODEL.RPN.POST_NMS_TOPK_TRAIN = 2000
_C.MODEL.RPN.POST_NMS_TOPK_TEST = 1000
_C.MODEL.RPN.NMS_THRESH = 0.7
_C.MODEL.RPN.CONV_DIMS = [-1]
_C.MODEL.ROI_HEADS = CfgNode()
_C.MODEL.ROI_HEADS.NAME = "Res5ROIHeads"
_C.MODEL.ROI_HEADS.NUM_CLASSES = 80
_C.MODEL.ROI_HEADS.IN_FEATURES = ["res4"]
_C.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.5]
_C.MODEL.ROI_HEADS.IOU_LABELS = [0, 1]
_C.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
_C.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25
_C.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
_C.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
_C.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT = True
_C.MODEL.ROI_BOX_HEAD = CfgNode()
_C.MODEL.ROI_BOX_HEAD.NAME = ""
_C.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE = "smooth_l1"
_C.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT = 1.0
_C.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS = (10.0, 10.0, 5.0, 5.0)
_C.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA = 0.0
_C.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 14
_C.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO = 0
_C.MODEL.ROI_BOX_HEAD.POOLER_TYPE = "ROIAlignV2"
_C.MODEL.ROI_BOX_HEAD.NUM_FC = 0
_C.MODEL.ROI_BOX_HEAD.FC_DIM = 1024
_C.MODEL.ROI_BOX_HEAD.NUM_CONV = 0
_C.MODEL.ROI_BOX_HEAD.CONV_DIM = 256
_C.MODEL.ROI_BOX_HEAD.NORM = ""
_C.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = False
_C.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES = False
_C.MODEL.ROI_BOX_HEAD.USE_FED_LOSS = False
_C.MODEL.ROI_BOX_HEAD.USE_SIGMOID_CE = False
_C.MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT_POWER = 0.5
_C.MODEL.ROI_BOX_HEAD.FED_LOSS_NUM_CLASSES = 50
_C.MODEL.ROI_BOX_CASCADE_HEAD = CfgNode()
_C.MODEL.ROI_BOX_CASCADE_HEAD.BBOX_REG_WEIGHTS = (
    (10.0, 10.0, 5.0, 5.0),
    (20.0, 20.0, 10.0, 10.0),
    (30.0, 30.0, 15.0, 15.0),
)
_C.MODEL.ROI_BOX_CASCADE_HEAD.IOUS = (0.5, 0.6, 0.7)
_C.MODEL.ROI_MASK_HEAD = CfgNode()
_C.MODEL.ROI_MASK_HEAD.NAME = "MaskRCfgNodeNConvUpsampleHead"
_C.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 14
_C.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO = 0
_C.MODEL.ROI_MASK_HEAD.NUM_CONV = 0
_C.MODEL.ROI_MASK_HEAD.CONV_DIM = 256
_C.MODEL.ROI_MASK_HEAD.NORM = ""
_C.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK = False
_C.MODEL.ROI_MASK_HEAD.POOLER_TYPE = "ROIAlignV2"
_C.MODEL.ROI_KEYPOINT_HEAD = CfgNode()
_C.MODEL.ROI_KEYPOINT_HEAD.NAME = "KRCfgNodeNConvDeconvUpsampleHead"
_C.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION = 14
_C.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO = 0
_C.MODEL.ROI_KEYPOINT_HEAD.CONV_DIMS = tuple(512 for _ in range(8))
_C.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 17
_C.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE = 1
_C.MODEL.ROI_KEYPOINT_HEAD.NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS = True
_C.MODEL.ROI_KEYPOINT_HEAD.LOSS_WEIGHT = 1.0
_C.MODEL.ROI_KEYPOINT_HEAD.POOLER_TYPE = "ROIAlignV2"
_C.MODEL.SEM_SEG_HEAD = CfgNode()
_C.MODEL.SEM_SEG_HEAD.NAME = "SemSegFPNHead"
_C.MODEL.SEM_SEG_HEAD.IN_FEATURES = ["p2", "p3", "p4", "p5"]
_C.MODEL.SEM_SEG_HEAD.IGNORE_VALUE = 255
_C.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 54
_C.MODEL.SEM_SEG_HEAD.CONVS_DIM = 128
_C.MODEL.SEM_SEG_HEAD.COMMON_STRIDE = 4
_C.MODEL.SEM_SEG_HEAD.NORM = "GN"
_C.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT = 1.0

_C.MODEL.PANOPTIC_FPN = CfgNode()
_C.MODEL.PANOPTIC_FPN.INSTANCE_LOSS_WEIGHT = 1.0

_C.MODEL.PANOPTIC_FPN.COMBINE = CfgNode({"ENABLED": True})
_C.MODEL.PANOPTIC_FPN.COMBINE.OVERLAP_THRESH = 0.5
_C.MODEL.PANOPTIC_FPN.COMBINE.STUFF_AREA_LIMIT = 4096
_C.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5
_C.MODEL.RETINANET = CfgNode()

_C.MODEL.RETINANET.NUM_CLASSES = 80

_C.MODEL.RETINANET.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
_C.MODEL.RETINANET.NUM_CONVS = 4
_C.MODEL.RETINANET.IOU_THRESHOLDS = [0.4, 0.5]
_C.MODEL.RETINANET.IOU_LABELS = [0, -1, 1]
_C.MODEL.RETINANET.PRIOR_PROB = 0.01
_C.MODEL.RETINANET.SCORE_THRESH_TEST = 0.05
_C.MODEL.RETINANET.TOPK_CANDIDATES_TEST = 1000
_C.MODEL.RETINANET.NMS_THRESH_TEST = 0.5

_C.MODEL.RETINANET.BBOX_REG_WEIGHTS = (1.0, 1.0, 1.0, 1.0)

_C.MODEL.RETINANET.FOCAL_LOSS_GAMMA = 2.0
_C.MODEL.RETINANET.FOCAL_LOSS_ALPHA = 0.25
_C.MODEL.RETINANET.SMOOTH_L1_LOSS_BETA = 0.1
_C.MODEL.RETINANET.BBOX_REG_LOSS_TYPE = "smooth_l1"

_C.MODEL.RETINANET.NORM = ""
_C.MODEL.RESNETS = CfgNode()

_C.MODEL.RESNETS.DEPTH = 50
_C.MODEL.RESNETS.OUT_FEATURES = ["res4"]

_C.MODEL.RESNETS.NUM_GROUPS = 1

_C.MODEL.RESNETS.NORM = "FrozenBN"

_C.MODEL.RESNETS.WIDTH_PER_GROUP = 64

_C.MODEL.RESNETS.STRIDE_IN_1X1 = True

_C.MODEL.RESNETS.RES5_DILATION = 1

_C.MODEL.RESNETS.RES2_OUT_CHANNELS = 256
_C.MODEL.RESNETS.STEM_OUT_CHANNELS = 64

_C.MODEL.RESNETS.DEFORM_ON_PER_STAGE = [False, False, False, False]
_C.MODEL.RESNETS.DEFORM_MODULATED = False
_C.MODEL.RESNETS.DEFORM_NUM_GROUPS = 1
_C.SOLVER = CfgNode()

# Options: WarmupMultiStepLR, WarmupCosineLR.
# See detectron2/solver/build.py for definition.
_C.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"

_C.SOLVER.MAX_ITER = 40000

_C.SOLVER.BASE_LR = 0.001
# The end lr, only used by WarmupCosineLR
_C.SOLVER.BASE_LR_END = 0.0

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.NESTEROV = False

_C.SOLVER.WEIGHT_DECAY = 0.0001
# The weight decay that's applied to parameters of normalization layers
# (typically the affine transformation)
_C.SOLVER.WEIGHT_DECAY_NORM = 0.0

_C.SOLVER.GAMMA = 0.1
# The iteration number to decrease learning rate by GAMMA.
_C.SOLVER.STEPS = (30000,)
# Number of decays in WarmupStepWithFixedGammaLR schedule
_C.SOLVER.NUM_DECAYS = 3

_C.SOLVER.WARMUP_FACTOR = 1.0 / 1000
_C.SOLVER.WARMUP_ITERS = 1000
_C.SOLVER.WARMUP_METHOD = "linear"
# Whether to rescale the interval for the learning schedule after warmup
_C.SOLVER.RESCALE_INTERVAL = False

# Save a checkpoint after every this number of iterations
_C.SOLVER.CHECKPOINT_PERIOD = 5000
_C.SOLVER.IMS_PER_BATCH = 16
_C.SOLVER.REFERENCE_WORLD_SIZE = 0
_C.SOLVER.BIAS_LR_FACTOR = 1.0
_C.SOLVER.WEIGHT_DECAY_BIAS = None  # None means following WEIGHT_DECAY

# Gradient clipping
_C.SOLVER.CLIP_GRADIENTS = CfgNode({"ENABLED": False})
_C.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"
# Maximum absolute value used for clipping gradients
_C.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
# Floating point number p for L-p norm to be used with the "norm"
# gradient clipping type; for L-inf, please specify .inf
_C.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0
_C.SOLVER.AMP = CfgNode({"ENABLED": False})
_C.TEST = CfgNode()
_C.TEST.EXPECTED_RESULTS = []
# The period (in terms of steps) to evaluate the model during training.
# Set to 0 to disable.
_C.TEST.EVAL_PERIOD = 0
_C.TEST.KEYPOINT_OKS_SIGMAS = []
_C.TEST.DETECTIONS_PER_IMAGE = 100

_C.TEST.AUG = CfgNode({"ENABLED": False})
_C.TEST.AUG.MIN_SIZES = (400, 500, 600, 700, 800, 900, 1000, 1100, 1200)
_C.TEST.AUG.MAX_SIZE = 4000
_C.TEST.AUG.FLIP = True

_C.TEST.PRECISE_BN = CfgNode({"ENABLED": False})
_C.TEST.PRECISE_BN.NUM_ITER = 200
_C.OUTPUT_DIR = "./output"
_C.SEED = -1
_C.CUDNN_BENCHMARK = False
_C.FLOAT32_PRECISION = ""
_C.VIS_PERIOD = 0
_C.GLOBAL = CfgNode()
_C.GLOBAL.HACK = 1.0


def get_cfg() -> CfgNode:

    return _C.clone()


def get_config_file(config_path):
    cfg_file = pkg_resources.resource_filename(
        "detectron2.model_zoo", os.path.join("configs", config_path)
    )
    if not os.path.exists(cfg_file):
        raise RuntimeError("{} not available in Model Zoo!".format(config_path))
    return cfg_file


class _ModelZooUrls:
    S3_PREFIX = "https://dl.fbaipublicfiles.com/detectron2/"

    # format: {config_path.yaml} -> model_id/model_final_{commit}.pkl
    CONFIG_PATH_TO_URL_SUFFIX = {
        "COCO-PanopticSegmentation/panoptic_fpn_R_50_1x": "139514544/model_final_dbfeb4.pkl",
        "COCO-PanopticSegmentation/panoptic_fpn_R_50_3x": "139514569/model_final_c10459.pkl",
        "COCO-PanopticSegmentation/panoptic_fpn_R_101_3x": "139514519/model_final_cafdb1.pkl",
    }

    @staticmethod
    def query(config_path: str) -> Optional[str]:
        name = config_path.replace(".yaml", "").replace(".py", "")
        if name in _ModelZooUrls.CONFIG_PATH_TO_URL_SUFFIX:
            suffix = _ModelZooUrls.CONFIG_PATH_TO_URL_SUFFIX[name]
            return _ModelZooUrls.S3_PREFIX + name + "/" + suffix
        return None


def get_checkpoint_url(config_path):
    url = _ModelZooUrls.query(config_path)
    if url is None:
        raise RuntimeError(
            "Pretrained model for {} is not available!".format(config_path)
        )
    return url


def build_model(cfg):
    """
    Simplified build_model function - builds the panoptic FPN model
    """
    from detectron2.modeling.meta_arch.panoptic_fpn import PanopticFPN

    model = PanopticFPN(cfg)
    model.to(torch.device(cfg.MODEL.DEVICE))
    return model


class DetectionCheckpointer:
    """
    Simplified checkpoint loader
    """

    def __init__(self, model):
        self.model = model

    def load(self, path):
        if path:
            from urllib.parse import urlparse

            parsed_url = urlparse(path)
            path = parsed_url._replace(query="").geturl()  # remove query from filename
            path = PathManager.get_local_path(path)

        if path.endswith(".pkl"):
            import pickle

            with open(path, "rb") as f:
                data = pickle.load(f, encoding="latin1")
            if "model" in data and "__author__" in data:
                # file is in Detectron2 model zoo format
                state_dict = data["model"]
            else:
                # assume file is from Caffe2 / Detectron1 model zoo
                if "blobs" in data:
                    # Detection models have "blobs", but ImageNet models don't
                    data = data["blobs"]
                state_dict = {
                    k: v for k, v in data.items() if not k.endswith("_momentum")
                }
        else:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint

        # Convert numpy arrays to tensors if needed
        for k, v in state_dict.items():
            if isinstance(v, np.ndarray):
                state_dict[k] = torch.from_numpy(v)

        self.model.load_state_dict(state_dict, strict=False)


class ResizeShortestEdge:
    """
    Simplified resize transform
    """

    def __init__(self, short_edge_length, max_size):
        self.short_edge_length = short_edge_length
        self.max_size = max_size

    def get_transform(self, image):
        h, w = image.shape[:2]
        if isinstance(self.short_edge_length, (list, tuple)):
            size = self.short_edge_length[0]  # Use first value for simplicity
        else:
            size = self.short_edge_length

        scale = size / min(h, w)
        if h * scale > self.max_size or w * scale > self.max_size:
            scale = self.max_size / max(h, w)

        new_h, new_w = int(h * scale), int(w * scale)
        return _ResizeTransform(h, w, new_h, new_w)


class _ResizeTransform:
    def __init__(self, h, w, new_h, new_w):
        self.h = h
        self.w = w
        self.new_h = new_h
        self.new_w = new_w

    def apply_image(self, image):
        import cv2

        return cv2.resize(image, (self.new_w, self.new_h))


class DefaultPredictor:
    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            image.to(self.cfg.MODEL.DEVICE)

            inputs = {"image": image, "height": height, "width": width}

            predictions = self.model([inputs])[0]
            return predictions


if __name__ == "__main__":
    # Load a standard Panoptic segmentation model from the model zoo
    cfg = get_cfg()
    cfg.merge_from_file(
        get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
    )
    cfg.MODEL.WEIGHTS = get_checkpoint_url(
        "COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"
    )
    cfg.MODEL.DEVICE = "cpu"  # Set the device to CPU

# Add confidence threshold to config
cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

# Set up metadata for COCO panoptic dataset
metadata = MetadataCatalog.get("coco_2017_val_panoptic")
metadata.thing_classes = [
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
metadata.stuff_classes = [
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
