# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from .utils import (
    MODELS,
    BaseModel,
    ConfigDict,
    List,
    Optional,
    DetDataSample,
    InstanceData,
    samplelist_boxtype2tensor,
    print_log,
    get_world_size,
    BaseModule,
    _BatchNorm,
    SyncBatchNorm,
    _get_norm,
    digit_version,
    BaseBoxes,
    Registry,
    get_dist_info,
    T,
    DeviceType,
    MaskType,
    PolygonMasks,
    BitmapMasks,
    _box_type_to_name,
    box_types,
    DATASETS,
    load,
    FileClient,
    get_file_backend,
    imread_backend,
    is_str,
    _scale_size,
    imrescale,
    imresize,
    get_box_type,
    impad,
    PixelData,
    nms,
    CLASS_TEXTS,
)
import pycocotools.mask as maskUtils
import gc
import io

try:
    from PIL import Image, ImageOps
except ImportError:
    Image = None
try:
    import tifffile
except ImportError:
    tifffile = None
from pathlib import Path
from contextlib import contextmanager
import pycocotools
from pycocotools.coco import COCO as _COCO
from io import StringIO
import os.path as osp
import logging
import json
import pickle
from torch.utils.data import Dataset
import functools
from torch.nn.modules.utils import _pair
import cv2
import numpy as np
import torch.nn.functional as F
from inspect import signature
import copy
import itertools
from transformers import AutoTokenizer, CLIPTextConfig
from transformers import CLIPTextModelWithProjection as CLIPTP
import math
from functools import partial
import warnings
import re
import inspect
from torch import Tensor, BoolTensor
from typing import (
    List,
    Tuple,
    Union,
    Optional,
    Dict,
    Sequence,
    Type,
    Callable,
    Any,
    Generator,
)
from abc import ABCMeta, abstractmethod
import torch
import torch.nn as nn
from cv2 import (
    IMREAD_COLOR,
    IMREAD_GRAYSCALE,
    IMREAD_IGNORE_ORIENTATION,
    IMREAD_UNCHANGED,
)

supported_backends = ["cv2", "turbojpeg", "pillow", "tifffile"]
imread_flags = {
    "color": IMREAD_COLOR,
    "grayscale": IMREAD_GRAYSCALE,
    "unchanged": IMREAD_UNCHANGED,
    "color_ignore_orientation": IMREAD_IGNORE_ORIENTATION | IMREAD_COLOR,
    "grayscale_ignore_orientation": IMREAD_IGNORE_ORIENTATION | IMREAD_GRAYSCALE,
}
ConfigType = Union[ConfigDict, dict]
OptConfigType = Optional[ConfigType]
MultiConfig = Union[ConfigType, List[ConfigType]]
OptMultiConfig = Optional[MultiConfig]
MultiConfig = Union[ConfigType, List[ConfigType]]
ForwardResults = Union[
    Dict[str, torch.Tensor], List[DetDataSample], Tuple[torch.Tensor], torch.Tensor
]
SampleList = List[DetDataSample]
OptSampleList = Optional[SampleList]
InstanceList = List[InstanceData]
OptInstanceList = Optional[InstanceList]


class BaseDetector(BaseModel, metaclass=ABCMeta):
    def __init__(
        self, data_preprocessor: OptConfigType = None, init_cfg: OptMultiConfig = None
    ):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

    @property
    def with_neck(self) -> bool:
        return hasattr(self, "neck") and self.neck is not None

    @property
    def with_shared_head(self) -> bool:
        return hasattr(self, "roi_head") and self.roi_head.with_shared_head

    @property
    def with_bbox(self) -> bool:
        return (hasattr(self, "roi_head") and self.roi_head.with_bbox) or (
            hasattr(self, "bbox_head") and self.bbox_head is not None
        )

    @property
    def with_mask(self) -> bool:
        return (hasattr(self, "roi_head") and self.roi_head.with_mask) or (
            hasattr(self, "mask_head") and self.mask_head is not None
        )

    def forward(
        self,
        inputs: torch.Tensor,
        data_samples: OptSampleList = None,
        mode: str = "tensor",
    ) -> ForwardResults:
        if mode == "loss":
            return self.loss(inputs, data_samples)
        elif mode == "predict":
            return self.predict(inputs, data_samples)
        elif mode == "tensor":
            return self._forward(inputs, data_samples)
        else:
            raise RuntimeError(
                f'Invalid mode "{mode}". ' "Only supports loss, predict and tensor mode"
            )

    @abstractmethod
    def loss(
        self, batch_inputs: Tensor, batch_data_samples: SampleList
    ) -> Union[dict, tuple]:
        pass

    @abstractmethod
    def predict(
        self, batch_inputs: Tensor, batch_data_samples: SampleList
    ) -> SampleList:
        pass

    @abstractmethod
    def _forward(self, batch_inputs: Tensor, batch_data_samples: OptSampleList = None):
        pass

    @abstractmethod
    def extract_feat(self, batch_inputs: Tensor):
        pass

    def add_pred_to_datasample(
        self, data_samples: SampleList, results_list: InstanceList
    ) -> SampleList:
        for data_sample, pred_instances in zip(data_samples, results_list):
            data_sample.pred_instances = pred_instances
        samplelist_boxtype2tensor(data_samples)
        return data_samples


@MODELS.register_module()
class SingleStageDetector(BaseDetector):
    def __init__(
        self,
        backbone: ConfigType,
        neck: OptConfigType = None,
        bbox_head: OptConfigType = None,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        data_preprocessor: OptConfigType = None,
        init_cfg: OptMultiConfig = None,
    ) -> None:
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = MODELS.build(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: Union[List[str], str],
        unexpected_keys: Union[List[str], str],
        error_msgs: Union[List[str], str],
    ) -> None:
        bbox_head_prefix = prefix + ".bbox_head" if prefix else "bbox_head"
        bbox_head_keys = [
            k for k in state_dict.keys() if k.startswith(bbox_head_prefix)
        ]
        rpn_head_prefix = prefix + ".rpn_head" if prefix else "rpn_head"
        rpn_head_keys = [k for k in state_dict.keys() if k.startswith(rpn_head_prefix)]
        if len(bbox_head_keys) == 0 and len(rpn_head_keys) != 0:
            for rpn_head_key in rpn_head_keys:
                bbox_head_key = bbox_head_prefix + rpn_head_key[len(rpn_head_prefix) :]
                state_dict[bbox_head_key] = state_dict.pop(rpn_head_key)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def loss(
        self, batch_inputs: Tensor, batch_data_samples: SampleList
    ) -> Union[dict, list]:
        x = self.extract_feat(batch_inputs)
        losses = self.bbox_head.loss(x, batch_data_samples)
        return losses

    def predict(
        self, batch_inputs: Tensor, batch_data_samples: SampleList, rescale: bool = True
    ) -> SampleList:
        x = self.extract_feat(batch_inputs)
        results_list = self.bbox_head.predict(x, batch_data_samples, rescale=rescale)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list
        )
        return batch_data_samples

    def _forward(
        self, batch_inputs: Tensor, batch_data_samples: OptSampleList = None
    ) -> Tuple[List[Tensor]]:
        x = self.extract_feat(batch_inputs)
        results = self.bbox_head.forward(x)
        return results

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        return x


@MODELS.register_module()
class YOLODetector(SingleStageDetector):
    def __init__(
        self,
        backbone: ConfigType,
        neck: ConfigType,
        bbox_head: ConfigType,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        data_preprocessor: OptConfigType = None,
        init_cfg: OptMultiConfig = None,
        use_syncbn: bool = True,
    ):
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
        )
        if use_syncbn and get_world_size() > 1:
            torch.nn.SyncBatchNorm.convert_sync_batchnorm(self)
            print_log("Using SyncBatchNorm()", "current")


@MODELS.register_module()
class YOLOWorldDetector(YOLODetector):
    def __init__(
        self,
        *args,
        mm_neck: bool = False,
        num_train_classes=80,
        num_test_classes=80,
        **kwargs,
    ) -> None:
        self.mm_neck = mm_neck
        self.num_train_classes = num_train_classes
        self.num_test_classes = num_test_classes
        super().__init__(*args, **kwargs)

    def loss(
        self, batch_inputs: Tensor, batch_data_samples: SampleList
    ) -> Union[dict, list]:
        self.bbox_head.num_classes = self.num_train_classes
        img_feats, txt_feats, txt_masks = self.extract_feat(
            batch_inputs, batch_data_samples
        )
        losses = self.bbox_head.loss(
            img_feats, txt_feats, txt_masks, batch_data_samples
        )
        return losses

    def predict(
        self, batch_inputs: Tensor, batch_data_samples: SampleList, rescale: bool = True
    ) -> SampleList:
        img_feats, txt_feats, txt_masks = self.extract_feat(
            batch_inputs, batch_data_samples
        )
        self.bbox_head.num_classes = txt_feats[0].shape[0]
        results_list = self.bbox_head.predict(
            img_feats, txt_feats, txt_masks, batch_data_samples, rescale=rescale
        )

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list
        )
        return batch_data_samples

    def reparameterize(self, texts: List[List[str]]) -> None:
        self.texts = texts
        self.text_feats, _ = self.backbone.forward_text(texts)

    def _forward(
        self, batch_inputs: Tensor, batch_data_samples: OptSampleList = None
    ) -> Tuple[List[Tensor]]:
        img_feats, txt_feats, txt_masks = self.extract_feat(
            batch_inputs, batch_data_samples
        )
        results = self.bbox_head.forward(img_feats, txt_feats, txt_masks)
        return results

    def extract_feat(
        self, batch_inputs: Tensor, batch_data_samples: SampleList
    ) -> Tuple[Tuple[Tensor], Tensor]:
        txt_feats = None
        if batch_data_samples is None:
            texts = self.texts
            txt_feats = self.text_feats
        elif isinstance(batch_data_samples, dict) and "texts" in batch_data_samples:
            texts = batch_data_samples["texts"]
        elif isinstance(batch_data_samples, list) and hasattr(
            batch_data_samples[0], "texts"
        ):
            texts = [data_sample.texts for data_sample in batch_data_samples]
        elif hasattr(self, "text_feats"):
            texts = self.texts
            txt_feats = self.text_feats
        else:
            raise TypeError("batch_data_samples should be dict or list.")
        if txt_feats is not None:
            img_feats = self.backbone.forward_image(batch_inputs)
        else:
            img_feats, (txt_feats, txt_masks) = self.backbone(batch_inputs, texts)
        if self.with_neck:
            if self.mm_neck:
                img_feats = self.neck(img_feats, txt_feats)
            else:
                img_feats = self.neck(img_feats)
        return img_feats, txt_feats, txt_masks


@MODELS.register_module()
class SimpleYOLOWorldDetector(YOLODetector):
    def __init__(
        self,
        *args,
        mm_neck: bool = False,
        num_train_classes=80,
        num_test_classes=80,
        prompt_dim=512,
        num_prompts=80,
        embedding_path="",
        reparameterized=False,
        freeze_prompt=False,
        use_mlp_adapter=False,
        **kwargs,
    ) -> None:
        self.mm_neck = mm_neck
        self.num_training_classes = num_train_classes
        self.num_test_classes = num_test_classes
        self.prompt_dim = prompt_dim
        self.num_prompts = num_prompts
        self.reparameterized = reparameterized
        self.freeze_prompt = freeze_prompt
        self.use_mlp_adapter = use_mlp_adapter
        super().__init__(*args, **kwargs)

        if not self.reparameterized:
            if len(embedding_path) > 0:
                import numpy as np

                self.embeddings = torch.nn.Parameter(
                    torch.from_numpy(np.load(embedding_path)).float()
                )
            else:
                embeddings = nn.functional.normalize(
                    torch.randn((num_prompts, prompt_dim)), dim=-1
                )
                self.embeddings = nn.Parameter(embeddings)

            if self.freeze_prompt:
                self.embeddings.requires_grad = False
            else:
                self.embeddings.requires_grad = True

            if use_mlp_adapter:
                self.adapter = nn.Sequential(
                    nn.Linear(prompt_dim, prompt_dim * 2),
                    nn.ReLU(True),
                    nn.Linear(prompt_dim * 2, prompt_dim),
                )
            else:
                self.adapter = None

    def loss(
        self, batch_inputs: Tensor, batch_data_samples: SampleList
    ) -> Union[dict, list]:
        self.bbox_head.num_classes = self.num_training_classes
        img_feats, txt_feats = self.extract_feat(batch_inputs, batch_data_samples)
        if self.reparameterized:
            losses = self.bbox_head.loss(img_feats, batch_data_samples)
        else:
            losses = self.bbox_head.loss(img_feats, txt_feats, batch_data_samples)
        return losses

    def predict(
        self, batch_inputs: Tensor, batch_data_samples: SampleList, rescale: bool = True
    ) -> SampleList:
        img_feats, txt_feats = self.extract_feat(batch_inputs, batch_data_samples)

        self.bbox_head.num_classes = self.num_test_classes
        if self.reparameterized:
            results_list = self.bbox_head.predict(
                img_feats, batch_data_samples, rescale=rescale
            )
        else:
            results_list = self.bbox_head.predict(
                img_feats, txt_feats, batch_data_samples, rescale=rescale
            )

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list
        )
        return batch_data_samples

    def _forward(
        self, batch_inputs: Tensor, batch_data_samples: OptSampleList = None
    ) -> Tuple[List[Tensor]]:
        img_feats, txt_feats = self.extract_feat(batch_inputs, batch_data_samples)
        if self.reparameterized:
            results = self.bbox_head.forward(img_feats)
        else:
            results = self.bbox_head.forward(img_feats, txt_feats)
        return results

    def extract_feat(
        self, batch_inputs: Tensor, batch_data_samples: SampleList
    ) -> Tuple[Tuple[Tensor], Tensor]:
        img_feats, _ = self.backbone(batch_inputs, None)

        if not self.reparameterized:
            txt_feats = self.embeddings[None]
            if self.adapter is not None:
                txt_feats = self.adapter(txt_feats) + txt_feats
                txt_feats = nn.functional.normalize(txt_feats, dim=-1, p=2)
            txt_feats = txt_feats.repeat(img_feats[0].shape[0], 1, 1)
        else:
            txt_feats = None
        if self.with_neck:
            if self.mm_neck:
                img_feats = self.neck(img_feats, txt_feats)
            else:
                img_feats = self.neck(img_feats)
        return img_feats, txt_feats


@MODELS.register_module()
class MultiModalYOLOBackbone(BaseModule):
    def __init__(
        self,
        image_model: ConfigType,
        text_model: ConfigType,
        frozen_stages: int = -1,
        with_text_model: bool = True,
        init_cfg: OptMultiConfig = None,
    ) -> None:
        super().__init__(init_cfg)
        self.with_text_model = with_text_model
        self.image_model = MODELS.build(image_model)
        if self.with_text_model:
            self.text_model = MODELS.build(text_model)
        else:
            self.text_model = None
        self.frozen_stages = frozen_stages
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages + 1):
                m = getattr(self.image_model, self.image_model.layers[i])
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_stages()

    def forward(
        self, image: Tensor, text: List[List[str]]
    ) -> Tuple[Tuple[Tensor], Tensor]:
        img_feats = self.image_model(image)
        if text is not None and self.with_text_model:
            txt_feats = self.text_model(text)
            return img_feats, txt_feats
        else:
            return img_feats, None

    def forward_text(self, text: List[List[str]]) -> Tensor:
        assert self.with_text_model, "forward_text() requires a text model"
        txt_feats = self.text_model(text)
        return txt_feats

    def forward_image(self, image: Tensor) -> Tuple[Tensor]:
        return self.image_model(image)


def infer_abbr(class_type: type) -> str:
    def camel2snack(word):
        word = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", word)
        word = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", word)
        word = word.replace("-", "_")
        return word.lower()

    if not inspect.isclass(class_type):
        raise TypeError(f"class_type must be a type, but got {type(class_type)}")
    if hasattr(class_type, "_abbr_"):
        return class_type._abbr_
    else:
        return camel2snack(class_type.__name__)


def build_plugin_layer(
    cfg: Dict, postfix: Union[int, str] = "", **kwargs
) -> Tuple[str, nn.Module]:
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be a dict")
    if "type" not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop("type")
    if inspect.isclass(layer_type):
        plugin_layer = layer_type
    else:
        with MODELS.switch_scope_and_registry(None) as registry:
            plugin_layer = registry.get(layer_type)
        if plugin_layer is None:
            raise KeyError(
                f"Cannot find {plugin_layer} in registry under scope "
                f"name {registry.scope}"
            )
    abbr = infer_abbr(plugin_layer)

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    layer = plugin_layer(**kwargs, **cfg_)

    return name, layer


@MODELS.register_module()
class BaseBackbone(BaseModule, metaclass=ABCMeta):
    def __init__(
        self,
        arch_setting: list,
        deepen_factor: float = 1.0,
        widen_factor: float = 1.0,
        input_channels: int = 3,
        out_indices: Sequence[int] = (2, 3, 4),
        frozen_stages: int = -1,
        plugins: Union[dict, List[dict]] = None,
        norm_cfg: ConfigType = None,
        act_cfg: ConfigType = None,
        norm_eval: bool = False,
        init_cfg: OptMultiConfig = None,
    ):
        super().__init__(init_cfg)
        self.num_stages = len(arch_setting)
        self.arch_setting = arch_setting

        assert set(out_indices).issubset(i for i in range(len(arch_setting) + 1))

        if frozen_stages not in range(-1, len(arch_setting) + 1):
            raise ValueError(
                '"frozen_stages" must be in range(-1, '
                "len(arch_setting) + 1). But received "
                f"{frozen_stages}"
            )

        self.input_channels = input_channels
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.widen_factor = widen_factor
        self.deepen_factor = deepen_factor
        self.norm_eval = norm_eval
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.plugins = plugins

        self.stem = self.build_stem_layer()
        self.layers = ["stem"]

        for idx, setting in enumerate(arch_setting):
            stage = []
            stage += self.build_stage_layer(idx, setting)
            if plugins is not None:
                stage += self.make_stage_plugins(plugins, idx, setting)
            self.add_module(f"stage{idx + 1}", nn.Sequential(*stage))
            self.layers.append(f"stage{idx + 1}")

    @abstractmethod
    def build_stem_layer(self):
        pass

    @abstractmethod
    def build_stage_layer(self, stage_idx: int, setting: list):
        pass

    def make_stage_plugins(self, plugins, stage_idx, setting):
        in_channels = int(setting[1] * self.widen_factor)
        plugin_layers = []
        for plugin in plugins:
            plugin = plugin.copy()
            stages = plugin.pop("stages", None)
            assert stages is None or len(stages) == self.num_stages
            if stages is None or stages[stage_idx]:
                name, layer = build_plugin_layer(plugin["cfg"], in_channels=in_channels)
                plugin_layers.append(layer)
        return plugin_layers

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages + 1):
                m = getattr(self, self.layers[i])
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def forward(self, x: torch.Tensor) -> tuple:
        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)


MODELS.register_module("Conv1d", module=nn.Conv1d)
MODELS.register_module("Conv2d", module=nn.Conv2d)
MODELS.register_module("Conv3d", module=nn.Conv3d)
MODELS.register_module("Conv", module=nn.Conv2d)
MODELS.register_module("zero", module=nn.ZeroPad2d)
MODELS.register_module("reflect", module=nn.ReflectionPad2d)
MODELS.register_module("replicate", module=nn.ReplicationPad2d)


def build_padding_layer(cfg: Dict, *args, **kwargs) -> nn.Module:
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be a dict")
    if "type" not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')

    cfg_ = cfg.copy()
    padding_type = cfg_.pop("type")
    if inspect.isclass(padding_type):
        return padding_type(*args, **kwargs, **cfg_)
    with MODELS.switch_scope_and_registry(None) as registry:
        padding_layer = registry.get(padding_type)
    if padding_layer is None:
        raise KeyError(
            f"Cannot find {padding_layer} in registry under scope "
            f"name {registry.scope}"
        )
    layer = padding_layer(*args, **kwargs, **cfg_)

    return layer


def build_conv_layer(cfg: Optional[Dict], *args, **kwargs) -> nn.Module:
    if cfg is None:
        cfg_ = dict(type="Conv2d")
    else:
        if not isinstance(cfg, dict):
            raise TypeError("cfg must be a dict")
        if "type" not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        cfg_ = cfg.copy()

    layer_type = cfg_.pop("type")
    if inspect.isclass(layer_type):
        return layer_type(*args, **kwargs, **cfg_)
    with MODELS.switch_scope_and_registry(None) as registry:
        conv_layer = registry.get(layer_type)
    if conv_layer is None:
        raise KeyError(
            f"Cannot find {conv_layer} in registry under scope "
            f"name {registry.scope}"
        )
    layer = conv_layer(*args, **kwargs, **cfg_)

    return layer


MODELS.register_module("BN", module=nn.BatchNorm2d)
MODELS.register_module("BN1d", module=nn.BatchNorm1d)
MODELS.register_module("BN2d", module=nn.BatchNorm2d)
MODELS.register_module("BN3d", module=nn.BatchNorm3d)
MODELS.register_module("SyncBN", module=SyncBatchNorm)
MODELS.register_module("GN", module=nn.GroupNorm)
MODELS.register_module("LN", module=nn.LayerNorm)
MODELS.register_module("IN", module=nn.InstanceNorm2d)
MODELS.register_module("IN1d", module=nn.InstanceNorm1d)
MODELS.register_module("IN2d", module=nn.InstanceNorm2d)
MODELS.register_module("IN3d", module=nn.InstanceNorm3d)


def build_norm_layer(
    cfg: Dict, num_features: int, postfix: Union[int, str] = ""
) -> Tuple[str, nn.Module]:
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be a dict")
    if "type" not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop("type")

    if inspect.isclass(layer_type):
        norm_layer = layer_type
    else:
        with MODELS.switch_scope_and_registry(None) as registry:
            norm_layer = registry.get(layer_type)
        if norm_layer is None:
            raise KeyError(
                f"Cannot find {norm_layer} in registry under "
                f"scope name {registry.scope}"
            )
    abbr = infer_abbr(norm_layer)

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    requires_grad = cfg_.pop("requires_grad", True)
    cfg_.setdefault("eps", 1e-5)
    if norm_layer is not nn.GroupNorm:
        layer = norm_layer(num_features, **cfg_)
        if layer_type == "SyncBN" and hasattr(layer, "_specify_ddp_gpu_num"):
            layer._specify_ddp_gpu_num(1)
    else:
        assert "num_groups" in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return name, layer


_BatchNorm, _InstanceNorm, SyncBatchNorm_ = _get_norm()


def build_activation_layer(cfg: Dict) -> nn.Module:
    return MODELS.build(cfg)


for module in [
    nn.ReLU,
    nn.LeakyReLU,
    nn.PReLU,
    nn.RReLU,
    nn.ReLU6,
    nn.ELU,
    nn.Sigmoid,
    nn.Tanh,
]:
    MODELS.register_module(module=module)

if digit_version(torch.__version__) >= digit_version("1.7.0"):
    MODELS.register_module(module=nn.SiLU, name="SiLU")
else:

    class SiLU(nn.Module):
        def __init__(self, inplace=False):
            super().__init__()
            self.inplace = inplace

        def forward(self, inputs) -> torch.Tensor:
            if self.inplace:
                return inputs.mul_(torch.sigmoid(inputs))
            else:
                return inputs * torch.sigmoid(inputs)

    MODELS.register_module(module=SiLU, name="SiLU")


def kaiming_init(
    module, a=0, mode="fan_out", nonlinearity="relu", bias=0, distribution="normal"
):
    assert distribution in ["uniform", "normal"]
    if hasattr(module, "weight") and module.weight is not None:
        if distribution == "uniform":
            nn.init.kaiming_uniform_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity
            )
        else:
            nn.init.kaiming_normal_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity
            )
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def fast_conv_bn_eval_forward(
    bn: _BatchNorm, conv: nn.modules.conv._ConvNd, x: torch.Tensor
):
    weight_on_the_fly = conv.weight
    if conv.bias is not None:
        bias_on_the_fly = conv.bias
    else:
        bias_on_the_fly = torch.zeros_like(bn.running_var)

    if bn.weight is not None:
        bn_weight = bn.weight
    else:
        bn_weight = torch.ones_like(bn.running_var)

    if bn.bias is not None:
        bn_bias = bn.bias
    else:
        bn_bias = torch.zeros_like(bn.running_var)

    weight_coeff = torch.rsqrt(bn.running_var + bn.eps).reshape(
        [-1] + [1] * (len(conv.weight.shape) - 1)
    )
    coefff_on_the_fly = bn_weight.view_as(weight_coeff) * weight_coeff
    weight_on_the_fly = weight_on_the_fly * coefff_on_the_fly
    bias_on_the_fly = bn_bias + coefff_on_the_fly.flatten() * (
        bias_on_the_fly - bn.running_mean
    )

    return conv._conv_forward(x, weight_on_the_fly, bias_on_the_fly)


@MODELS.register_module()
class ConvModule(nn.Module):
    _abbr_ = "conv_block"

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: Union[bool, str] = "auto",
        conv_cfg: Optional[Dict] = None,
        norm_cfg: Optional[Dict] = None,
        act_cfg: Optional[Dict] = dict(type="ReLU"),
        inplace: bool = True,
        with_spectral_norm: bool = False,
        padding_mode: str = "zeros",
        order: tuple = ("conv", "norm", "act"),
        fast_conv_bn_eval: bool = False,
    ):
        super().__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert act_cfg is None or isinstance(act_cfg, dict)
        official_padding_mode = ["zeros", "circular"]
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.inplace = inplace
        self.with_spectral_norm = with_spectral_norm
        self.with_explicit_padding = padding_mode not in official_padding_mode
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == {"conv", "norm", "act"}

        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        if bias == "auto":
            bias = not self.with_norm
        self.with_bias = bias

        if self.with_explicit_padding:
            pad_cfg = dict(type=padding_mode)
            self.padding_layer = build_padding_layer(pad_cfg, padding)
        conv_padding = 0 if self.with_explicit_padding else padding
        self.conv = build_conv_layer(
            conv_cfg,
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=conv_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups

        if self.with_spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)
        if self.with_norm:
            if order.index("norm") > order.index("conv"):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            self.norm_name, norm = build_norm_layer(norm_cfg, norm_channels)
            self.add_module(self.norm_name, norm)
            if self.with_bias:
                if isinstance(norm, (_BatchNorm, _InstanceNorm)):
                    warnings.warn("Unnecessary conv bias before batch/instance norm")
        else:
            self.norm_name = None

        self.turn_on_fast_conv_bn_eval(fast_conv_bn_eval)
        if self.with_activation:
            act_cfg_ = act_cfg.copy()
            if act_cfg_["type"] not in [
                "Tanh",
                "PReLU",
                "Sigmoid",
                "HSigmoid",
                "Swish",
                "GELU",
            ]:
                act_cfg_.setdefault("inplace", inplace)
            self.activate = build_activation_layer(act_cfg_)

        self.init_weights()

    @property
    def norm(self):
        if self.norm_name:
            return getattr(self, self.norm_name)
        else:
            return None

    def init_weights(self):
        if not hasattr(self.conv, "init_weights"):
            if self.with_activation and self.act_cfg["type"] == "LeakyReLU":
                nonlinearity = "leaky_relu"
                a = self.act_cfg.get("negative_slope", 0.01)
            else:
                nonlinearity = "relu"
                a = 0
            kaiming_init(self.conv, a=a, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.norm, 1, bias=0)

    def forward(
        self, x: torch.Tensor, activate: bool = True, norm: bool = True
    ) -> torch.Tensor:
        layer_index = 0
        while layer_index < len(self.order):
            layer = self.order[layer_index]
            if layer == "conv":
                if self.with_explicit_padding:
                    x = self.padding_layer(x)
                if (
                    layer_index + 1 < len(self.order)
                    and self.order[layer_index + 1] == "norm"
                    and norm
                    and self.with_norm
                    and not self.norm.training
                    and self.fast_conv_bn_eval_forward is not None
                ):
                    self.conv.forward = partial(
                        self.fast_conv_bn_eval_forward, self.norm, self.conv
                    )
                    layer_index += 1
                    x = self.conv(x)
                    del self.conv.forward
                else:
                    x = self.conv(x)
            elif layer == "norm" and norm and self.with_norm:
                x = self.norm(x)
            elif layer == "act" and activate and self.with_activation:
                x = self.activate(x)
            layer_index += 1
        return x

    def turn_on_fast_conv_bn_eval(self, fast_conv_bn_eval=True):
        if (
            fast_conv_bn_eval
            and self.norm
            and isinstance(self.norm, _BatchNorm)
            and self.norm.track_running_stats
        ):
            self.fast_conv_bn_eval_forward = fast_conv_bn_eval_forward
        else:
            self.fast_conv_bn_eval_forward = None

    @staticmethod
    def create_from_conv_bn(
        conv: torch.nn.modules.conv._ConvNd,
        bn: torch.nn.modules.batchnorm._BatchNorm,
        fast_conv_bn_eval=True,
    ) -> "ConvModule":
        self = ConvModule.__new__(ConvModule)
        super(ConvModule, self).__init__()

        self.conv_cfg = None
        self.norm_cfg = None
        self.act_cfg = None
        self.inplace = False
        self.with_spectral_norm = False
        self.with_explicit_padding = False
        self.order = ("conv", "norm", "act")

        self.with_norm = True
        self.with_activation = False
        self.with_bias = conv.bias is not None
        self.conv = conv
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = self.conv.padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups
        self.norm_name, norm = "bn", bn
        self.add_module(self.norm_name, norm)

        self.turn_on_fast_conv_bn_eval(fast_conv_bn_eval)

        return self


def make_divisible(x: float, widen_factor: float = 1.0, divisor: int = 8) -> int:
    return math.ceil(x * widen_factor / divisor) * divisor


def make_round(x: float, deepen_factor: float = 1.0) -> int:
    return max(round(x * deepen_factor), 1) if x > 1 else x


class DepthwiseSeparableConvModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        norm_cfg: Optional[Dict] = None,
        act_cfg: Dict = dict(type="ReLU"),
        dw_norm_cfg: Union[Dict, str] = "default",
        dw_act_cfg: Union[Dict, str] = "default",
        pw_norm_cfg: Union[Dict, str] = "default",
        pw_act_cfg: Union[Dict, str] = "default",
        **kwargs,
    ):
        super().__init__()
        assert "groups" not in kwargs, "groups should not be specified"

        dw_norm_cfg = dw_norm_cfg if dw_norm_cfg != "default" else norm_cfg
        dw_act_cfg = dw_act_cfg if dw_act_cfg != "default" else act_cfg
        pw_norm_cfg = pw_norm_cfg if pw_norm_cfg != "default" else norm_cfg
        pw_act_cfg = pw_act_cfg if pw_act_cfg != "default" else act_cfg

        self.depthwise_conv = ConvModule(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            norm_cfg=dw_norm_cfg,
            act_cfg=dw_act_cfg,
            **kwargs,
        )

        self.pointwise_conv = ConvModule(
            in_channels,
            out_channels,
            1,
            norm_cfg=pw_norm_cfg,
            act_cfg=pw_act_cfg,
            **kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


class MMDET_DarknetBottleneck(BaseModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: float = 0.5,
        add_identity: bool = True,
        use_depthwise: bool = False,
        conv_cfg: OptConfigType = None,
        norm_cfg: ConfigType = dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg: ConfigType = dict(type="Swish"),
        init_cfg: OptMultiConfig = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        hidden_channels = int(out_channels * expansion)
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        self.conv1 = ConvModule(
            in_channels,
            hidden_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.conv2 = conv(
            hidden_channels,
            out_channels,
            3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.add_identity = add_identity and in_channels == out_channels

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.add_identity:
            return out + identity
        else:
            return out


class DarknetBottleneck(MMDET_DarknetBottleneck):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: float = 0.5,
        kernel_size: Sequence[int] = (1, 3),
        padding: Sequence[int] = (0, 1),
        add_identity: bool = True,
        use_depthwise: bool = False,
        conv_cfg: OptConfigType = None,
        norm_cfg: ConfigType = dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg: ConfigType = dict(type="SiLU", inplace=True),
        init_cfg: OptMultiConfig = None,
    ) -> None:
        super().__init__(in_channels, out_channels, init_cfg=init_cfg)
        hidden_channels = int(out_channels * expansion)
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        assert isinstance(kernel_size, Sequence) and len(kernel_size) == 2

        self.conv1 = ConvModule(
            in_channels,
            hidden_channels,
            kernel_size[0],
            padding=padding[0],
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.conv2 = conv(
            hidden_channels,
            out_channels,
            kernel_size[1],
            stride=1,
            padding=padding[1],
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.add_identity = add_identity and in_channels == out_channels


class CSPLayerWithTwoConv(BaseModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: float = 0.5,
        num_blocks: int = 1,
        add_identity: bool = True,
        conv_cfg: OptConfigType = None,
        norm_cfg: ConfigType = dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg: ConfigType = dict(type="SiLU", inplace=True),
        init_cfg: OptMultiConfig = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.mid_channels = int(out_channels * expand_ratio)
        self.main_conv = ConvModule(
            in_channels,
            2 * self.mid_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.final_conv = ConvModule(
            (2 + num_blocks) * self.mid_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

        self.blocks = nn.ModuleList(
            DarknetBottleneck(
                self.mid_channels,
                self.mid_channels,
                expansion=1,
                kernel_size=(3, 3),
                padding=(1, 1),
                add_identity=add_identity,
                use_depthwise=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )
            for _ in range(num_blocks)
        )

    def forward(self, x: Tensor) -> Tensor:
        x_main = self.main_conv(x)
        x_main = list(x_main.split((self.mid_channels, self.mid_channels), 1))
        x_main.extend(blocks(x_main[-1]) for blocks in self.blocks)
        return self.final_conv(torch.cat(x_main, 1))


class SPPFBottleneck(BaseModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: Union[int, Sequence[int]] = 5,
        use_conv_first: bool = True,
        mid_channels_scale: float = 0.5,
        conv_cfg: ConfigType = None,
        norm_cfg: ConfigType = dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg: ConfigType = dict(type="SiLU", inplace=True),
        init_cfg: OptMultiConfig = None,
    ):
        super().__init__(init_cfg)

        if use_conv_first:
            mid_channels = int(in_channels * mid_channels_scale)
            self.conv1 = ConvModule(
                in_channels,
                mid_channels,
                1,
                stride=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )
        else:
            mid_channels = in_channels
            self.conv1 = None
        self.kernel_sizes = kernel_sizes
        if isinstance(kernel_sizes, int):
            self.poolings = nn.MaxPool2d(
                kernel_size=kernel_sizes, stride=1, padding=kernel_sizes // 2
            )
            conv2_in_channels = mid_channels * 4
        else:
            self.poolings = nn.ModuleList(
                [
                    nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                    for ks in kernel_sizes
                ]
            )
            conv2_in_channels = mid_channels * (len(kernel_sizes) + 1)

        self.conv2 = ConvModule(
            conv2_in_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.conv1:
            x = self.conv1(x)
        if isinstance(self.kernel_sizes, int):
            y1 = self.poolings(x)
            y2 = self.poolings(y1)
            x = torch.cat([x, y1, y2, self.poolings(y2)], dim=1)
        else:
            x = torch.cat([x] + [pooling(x) for pooling in self.poolings], dim=1)
        x = self.conv2(x)
        return x


@MODELS.register_module()
class YOLOv8CSPDarknet(BaseBackbone):
    arch_settings = {
        "P5": [
            [64, 128, 3, True, False],
            [128, 256, 6, True, False],
            [256, 512, 6, True, False],
            [512, None, 3, True, True],
        ],
    }

    def __init__(
        self,
        arch: str = "P5",
        last_stage_out_channels: int = 1024,
        plugins: Union[dict, List[dict]] = None,
        deepen_factor: float = 1.0,
        widen_factor: float = 1.0,
        input_channels: int = 3,
        out_indices: Tuple[int] = (2, 3, 4),
        frozen_stages: int = -1,
        norm_cfg: ConfigType = dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg: ConfigType = dict(type="SiLU", inplace=True),
        norm_eval: bool = False,
        init_cfg: OptMultiConfig = None,
    ):
        self.arch_settings[arch][-1][1] = last_stage_out_channels
        super().__init__(
            self.arch_settings[arch],
            deepen_factor,
            widen_factor,
            input_channels=input_channels,
            out_indices=out_indices,
            plugins=plugins,
            frozen_stages=frozen_stages,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            norm_eval=norm_eval,
            init_cfg=init_cfg,
        )

    def build_stem_layer(self) -> nn.Module:
        return ConvModule(
            self.input_channels,
            make_divisible(self.arch_setting[0][0], self.widen_factor),
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )

    def build_stage_layer(self, stage_idx: int, setting: list) -> list:
        in_channels, out_channels, num_blocks, add_identity, use_spp = setting

        in_channels = make_divisible(in_channels, self.widen_factor)
        out_channels = make_divisible(out_channels, self.widen_factor)
        num_blocks = make_round(num_blocks, self.deepen_factor)
        stage = []
        conv_layer = ConvModule(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )
        stage.append(conv_layer)
        csp_layer = CSPLayerWithTwoConv(
            out_channels,
            out_channels,
            num_blocks=num_blocks,
            add_identity=add_identity,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )
        stage.append(csp_layer)
        if use_spp:
            spp = SPPFBottleneck(
                out_channels,
                out_channels,
                kernel_sizes=5,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
            )
            stage.append(spp)
        return stage

    def init_weights(self):
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, torch.nn.Conv2d):
                    m.reset_parameters()
        else:
            super().init_weights()


@MODELS.register_module()
class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


@MODELS.register_module()
class HuggingCLIPLanguageBackbone(BaseModule):
    def __init__(
        self,
        model_name: str,
        frozen_modules: Sequence[str] = (),
        dropout: float = 0.0,
        add_mask: bool = False,
        training_use_cache: bool = False,
        init_cfg: OptMultiConfig = None,
    ) -> None:

        super().__init__(init_cfg=init_cfg)

        self.frozen_modules = frozen_modules
        self.training_use_cache = training_use_cache
        self.add_mask = add_mask
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        clip_config = CLIPTextConfig.from_pretrained(
            model_name, attention_dropout=dropout
        )
        self.model = CLIPTP.from_pretrained(model_name, config=clip_config)
        self._freeze_modules()

    def forward_tokenizer(self, texts):
        if not hasattr(self, "text"):
            text = list(itertools.chain(*texts))
            text = self.tokenizer(text=text, return_tensors="pt", padding=True)
            self.text = text
        return self.text

    def forward(self, text: List[List[str]]) -> Tensor:
        num_per_batch = [len(t) for t in text]
        assert max(num_per_batch) == min(
            num_per_batch
        ), "number of sequences not equal in batch"
        text = list(itertools.chain(*text))
        if self.add_mask:
            text_mask = torch.tensor(
                [x != self.pad_value for x in text], requires_grad=False
            )
        text = self.tokenizer(text=text, return_tensors=None, padding=True)
        input_ids = torch.tensor(text["input_ids"], dtype=torch.long).to(
            self.model.device
        )
        attention_mask = torch.tensor(text["attention_mask"], dtype=torch.long).to(
            self.model.device
        )

        text = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if len(self.frozen_modules) > 0:
            with torch.no_grad():
                txt_outputs = self.model(**text)
                txt_feats = txt_outputs.text_embeds
        else:
            txt_outputs = self.model(**text)
            txt_feats = txt_outputs.text_embeds

        txt_feats = txt_outputs.text_embeds
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
        txt_feats = txt_feats.reshape(-1, num_per_batch[0], txt_feats.shape[-1])
        if self.add_mask:
            text_mask = text_mask.reshape(-1, num_per_batch[0]).to(txt_feats)
        else:
            text_mask = None
        return txt_feats, text_mask

    def _freeze_modules(self):

        if len(self.frozen_modules) == 0:
            return
        if self.frozen_modules[0] == "all":
            self.model.eval()
            for _, module in self.model.named_modules():
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False
            return
        for name, module in self.model.named_modules():
            for frozen_name in self.frozen_modules:
                if name.startswith(frozen_name):
                    module.eval()
                    for param in module.parameters():
                        param.requires_grad = False
                    break

    def train(self, mode=True):
        super().train(mode)
        self._freeze_modules()


@MODELS.register_module()
class BaseYOLONeck(BaseModule, metaclass=ABCMeta):
    def __init__(
        self,
        in_channels: List[int],
        out_channels: Union[int, List[int]],
        deepen_factor: float = 1.0,
        widen_factor: float = 1.0,
        upsample_feats_cat_first: bool = True,
        freeze_all: bool = False,
        norm_cfg: ConfigType = None,
        act_cfg: ConfigType = None,
        init_cfg: OptMultiConfig = None,
        **kwargs,
    ):
        super().__init__(init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.deepen_factor = deepen_factor
        self.widen_factor = widen_factor
        self.upsample_feats_cat_first = upsample_feats_cat_first
        self.freeze_all = freeze_all
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.reduce_layers = nn.ModuleList()
        for idx in range(len(in_channels)):
            self.reduce_layers.append(self.build_reduce_layer(idx))

        self.upsample_layers = nn.ModuleList()
        self.top_down_layers = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.upsample_layers.append(self.build_upsample_layer(idx))
            self.top_down_layers.append(self.build_top_down_layer(idx))

        self.downsample_layers = nn.ModuleList()
        self.bottom_up_layers = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            self.downsample_layers.append(self.build_downsample_layer(idx))
            self.bottom_up_layers.append(self.build_bottom_up_layer(idx))

        self.out_layers = nn.ModuleList()
        for idx in range(len(in_channels)):
            self.out_layers.append(self.build_out_layer(idx))

    @abstractmethod
    def build_reduce_layer(self, idx: int):
        pass

    @abstractmethod
    def build_upsample_layer(self, idx: int):
        pass

    @abstractmethod
    def build_top_down_layer(self, idx: int):
        pass

    @abstractmethod
    def build_downsample_layer(self, idx: int):
        pass

    @abstractmethod
    def build_bottom_up_layer(self, idx: int):
        pass

    @abstractmethod
    def build_out_layer(self, idx: int):
        pass

    def _freeze_all(self):
        for m in self.modules():
            if isinstance(m, _BatchNorm):
                m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_all:
            self._freeze_all()

    def forward(self, inputs: List[torch.Tensor]) -> tuple:
        assert len(inputs) == len(self.in_channels)
        reduce_outs = []
        for idx in range(len(self.in_channels)):
            reduce_outs.append(self.reduce_layers[idx](inputs[idx]))

        inner_outs = [reduce_outs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = reduce_outs[idx - 1]
            upsample_feat = self.upsample_layers[len(self.in_channels) - 1 - idx](
                feat_high
            )
            if self.upsample_feats_cat_first:
                top_down_layer_inputs = torch.cat([upsample_feat, feat_low], 1)
            else:
                top_down_layer_inputs = torch.cat([feat_low, upsample_feat], 1)
            inner_out = self.top_down_layers[len(self.in_channels) - 1 - idx](
                top_down_layer_inputs
            )
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_layers[idx](feat_low)
            out = self.bottom_up_layers[idx](torch.cat([downsample_feat, feat_high], 1))
            outs.append(out)

        results = []
        for idx in range(len(self.in_channels)):
            results.append(self.out_layers[idx](outs[idx]))

        return tuple(results)


class CSPNeXtBlock(BaseModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: float = 0.5,
        add_identity: bool = True,
        use_depthwise: bool = False,
        kernel_size: int = 5,
        conv_cfg: OptConfigType = None,
        norm_cfg: ConfigType = dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg: ConfigType = dict(type="SiLU"),
        init_cfg: OptMultiConfig = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        hidden_channels = int(out_channels * expansion)
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        self.conv1 = conv(
            in_channels,
            hidden_channels,
            3,
            stride=1,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.conv2 = DepthwiseSeparableConvModule(
            hidden_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.add_identity = add_identity and in_channels == out_channels

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.add_identity:
            return out + identity
        else:
            return out


class ChannelAttention(BaseModule):
    def __init__(self, channels: int, init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        if digit_version(torch.__version__) < (1, 7, 0):
            self.act = nn.Hardsigmoid()
        else:
            self.act = nn.Hardsigmoid(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        with torch.cuda.amp.autocast(enabled=False):
            out = self.global_avgpool(x)
        out = self.fc(out)
        out = self.act(out)
        return x * out


class CSPLayer(BaseModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: float = 0.5,
        num_blocks: int = 1,
        add_identity: bool = True,
        use_depthwise: bool = False,
        use_cspnext_block: bool = False,
        channel_attention: bool = False,
        conv_cfg: OptConfigType = None,
        norm_cfg: ConfigType = dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg: ConfigType = dict(type="Swish"),
        init_cfg: OptMultiConfig = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        block = CSPNeXtBlock if use_cspnext_block else DarknetBottleneck
        mid_channels = int(out_channels * expand_ratio)
        self.channel_attention = channel_attention
        self.main_conv = ConvModule(
            in_channels,
            mid_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.short_conv = ConvModule(
            in_channels,
            mid_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.final_conv = ConvModule(
            2 * mid_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

        self.blocks = nn.Sequential(
            *[
                block(
                    mid_channels,
                    mid_channels,
                    1.0,
                    add_identity,
                    use_depthwise,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                )
                for _ in range(num_blocks)
            ]
        )
        if channel_attention:
            self.attention = ChannelAttention(2 * mid_channels)

    def forward(self, x: Tensor) -> Tensor:
        x_short = self.short_conv(x)

        x_main = self.main_conv(x)
        x_main = self.blocks(x_main)

        x_final = torch.cat((x_main, x_short), dim=1)

        if self.channel_attention:
            x_final = self.attention(x_final)
        return self.final_conv(x_final)


@MODELS.register_module()
class YOLOv5PAFPN(BaseYOLONeck):
    def __init__(
        self,
        in_channels: List[int],
        out_channels: Union[List[int], int],
        deepen_factor: float = 1.0,
        widen_factor: float = 1.0,
        num_csp_blocks: int = 1,
        freeze_all: bool = False,
        norm_cfg: ConfigType = dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg: ConfigType = dict(type="SiLU", inplace=True),
        init_cfg: OptMultiConfig = None,
    ):
        self.num_csp_blocks = num_csp_blocks
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            freeze_all=freeze_all,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg,
        )

    def init_weights(self):
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, torch.nn.Conv2d):
                    m.reset_parameters()
        else:
            super().init_weights()

    def build_reduce_layer(self, idx: int) -> nn.Module:
        if idx == len(self.in_channels) - 1:
            layer = ConvModule(
                make_divisible(self.in_channels[idx], self.widen_factor),
                make_divisible(self.in_channels[idx - 1], self.widen_factor),
                1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
            )
        else:
            layer = nn.Identity()

        return layer

    def build_upsample_layer(self, *args, **kwargs) -> nn.Module:
        return nn.Upsample(scale_factor=2, mode="nearest")

    def build_top_down_layer(self, idx: int):

        if idx == 1:
            return CSPLayer(
                make_divisible(self.in_channels[idx - 1] * 2, self.widen_factor),
                make_divisible(self.in_channels[idx - 1], self.widen_factor),
                num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
                add_identity=False,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
            )
        else:
            return nn.Sequential(
                CSPLayer(
                    make_divisible(self.in_channels[idx - 1] * 2, self.widen_factor),
                    make_divisible(self.in_channels[idx - 1], self.widen_factor),
                    num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
                    add_identity=False,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                ),
                ConvModule(
                    make_divisible(self.in_channels[idx - 1], self.widen_factor),
                    make_divisible(self.in_channels[idx - 2], self.widen_factor),
                    kernel_size=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                ),
            )

    def build_downsample_layer(self, idx: int) -> nn.Module:
        return ConvModule(
            make_divisible(self.in_channels[idx], self.widen_factor),
            make_divisible(self.in_channels[idx], self.widen_factor),
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )

    def build_bottom_up_layer(self, idx: int) -> nn.Module:
        return CSPLayer(
            make_divisible(self.in_channels[idx] * 2, self.widen_factor),
            make_divisible(self.in_channels[idx + 1], self.widen_factor),
            num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
            add_identity=False,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )

    def build_out_layer(self, *args, **kwargs) -> nn.Module:
        return nn.Identity()


@MODELS.register_module()
class YOLOv8PAFPN(YOLOv5PAFPN):
    def __init__(
        self,
        in_channels: List[int],
        out_channels: Union[List[int], int],
        deepen_factor: float = 1.0,
        widen_factor: float = 1.0,
        num_csp_blocks: int = 3,
        freeze_all: bool = False,
        norm_cfg: ConfigType = dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg: ConfigType = dict(type="SiLU", inplace=True),
        init_cfg: OptMultiConfig = None,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            num_csp_blocks=num_csp_blocks,
            freeze_all=freeze_all,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg,
        )

    def build_reduce_layer(self, idx: int) -> nn.Module:
        return nn.Identity()

    def build_top_down_layer(self, idx: int) -> nn.Module:
        return CSPLayerWithTwoConv(
            make_divisible(
                (self.in_channels[idx - 1] + self.in_channels[idx]), self.widen_factor
            ),
            make_divisible(self.out_channels[idx - 1], self.widen_factor),
            num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
            add_identity=False,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )

    def build_bottom_up_layer(self, idx: int) -> nn.Module:
        return CSPLayerWithTwoConv(
            make_divisible(
                (self.out_channels[idx] + self.out_channels[idx + 1]), self.widen_factor
            ),
            make_divisible(self.out_channels[idx + 1], self.widen_factor),
            num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
            add_identity=False,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )


@MODELS.register_module()
class YOLOWorldPAFPN(YOLOv8PAFPN):
    def __init__(
        self,
        in_channels: List[int],
        out_channels: Union[List[int], int],
        guide_channels: int,
        embed_channels: List[int],
        num_heads: List[int],
        deepen_factor: float = 1.0,
        widen_factor: float = 1.0,
        num_csp_blocks: int = 3,
        freeze_all: bool = False,
        block_cfg: ConfigType = dict(type="CSPLayerWithTwoConv"),
        norm_cfg: ConfigType = dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg: ConfigType = dict(type="SiLU", inplace=True),
        init_cfg: OptMultiConfig = None,
    ) -> None:
        self.guide_channels = guide_channels
        self.embed_channels = embed_channels
        self.num_heads = num_heads
        self.block_cfg = block_cfg
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            num_csp_blocks=num_csp_blocks,
            freeze_all=freeze_all,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg,
        )

    def build_top_down_layer(self, idx: int) -> nn.Module:
        block_cfg = copy.deepcopy(self.block_cfg)
        block_cfg.update(
            dict(
                in_channels=make_divisible(
                    (self.in_channels[idx - 1] + self.in_channels[idx]),
                    self.widen_factor,
                ),
                out_channels=make_divisible(
                    self.out_channels[idx - 1], self.widen_factor
                ),
                guide_channels=self.guide_channels,
                embed_channels=make_round(
                    self.embed_channels[idx - 1], self.widen_factor
                ),
                num_heads=make_round(self.num_heads[idx - 1], self.widen_factor),
                num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
                add_identity=False,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
            )
        )
        return MODELS.build(block_cfg)

    def build_bottom_up_layer(self, idx: int) -> nn.Module:
        block_cfg = copy.deepcopy(self.block_cfg)
        block_cfg.update(
            dict(
                in_channels=make_divisible(
                    (self.out_channels[idx] + self.out_channels[idx + 1]),
                    self.widen_factor,
                ),
                out_channels=make_divisible(
                    self.out_channels[idx + 1], self.widen_factor
                ),
                guide_channels=self.guide_channels,
                embed_channels=make_round(
                    self.embed_channels[idx + 1], self.widen_factor
                ),
                num_heads=make_round(self.num_heads[idx + 1], self.widen_factor),
                num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
                add_identity=False,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
            )
        )
        return MODELS.build(block_cfg)

    def forward(self, img_feats: List[Tensor], txt_feats: Tensor = None) -> tuple:
        assert len(img_feats) == len(self.in_channels)
        reduce_outs = []
        for idx in range(len(self.in_channels)):
            reduce_outs.append(self.reduce_layers[idx](img_feats[idx]))

        inner_outs = [reduce_outs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = reduce_outs[idx - 1]
            upsample_feat = self.upsample_layers[len(self.in_channels) - 1 - idx](
                feat_high
            )
            if self.upsample_feats_cat_first:
                top_down_layer_inputs = torch.cat([upsample_feat, feat_low], 1)
            else:
                top_down_layer_inputs = torch.cat([feat_low, upsample_feat], 1)
            inner_out = self.top_down_layers[len(self.in_channels) - 1 - idx](
                top_down_layer_inputs, txt_feats
            )
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_layers[idx](feat_low)
            out = self.bottom_up_layers[idx](
                torch.cat([downsample_feat, feat_high], 1), txt_feats
            )
            outs.append(out)

        results = []
        for idx in range(len(self.in_channels)):
            results.append(self.out_layers[idx](outs[idx]))

        return tuple(results)


class NewEmptyTensorOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, new_shape: tuple) -> torch.Tensor:
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> tuple:
        shape = ctx.shape
        return NewEmptyTensorOp.apply(grad, shape), None


if torch.__version__ == "parrots":
    TORCH_VERSION = torch.__version__
else:
    TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])


def obsolete_torch_version(torch_version, version_threshold) -> bool:
    return torch_version == "parrots" or torch_version <= version_threshold


class Linear(torch.nn.Linear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0 and obsolete_torch_version(TORCH_VERSION, (1, 5)):
            out_shape = [x.shape[0], self.out_features]
            empty = NewEmptyTensorOp.apply(x, out_shape)
            if self.training:
                dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
                return empty + dummy
            else:
                return empty

        return super().forward(x)


@MODELS.register_module()
class MaxSigmoidAttnBlock(BaseModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        guide_channels: int,
        embed_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        num_heads: int = 1,
        use_depthwise: bool = False,
        with_scale: bool = False,
        conv_cfg: OptConfigType = None,
        norm_cfg: ConfigType = dict(type="BN", momentum=0.03, eps=0.001),
        init_cfg: OptMultiConfig = None,
        use_einsum: bool = True,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule

        assert (
            out_channels % num_heads == 0 and embed_channels % num_heads == 0
        ), "out_channels and embed_channels should be divisible by num_heads."
        self.num_heads = num_heads
        self.head_channels = embed_channels // num_heads
        self.use_einsum = use_einsum

        self.embed_conv = (
            ConvModule(
                in_channels,
                embed_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None,
            )
            if embed_channels != in_channels
            else None
        )
        self.guide_fc = Linear(guide_channels, embed_channels)
        self.bias = nn.Parameter(torch.zeros(num_heads))
        if with_scale:
            self.scale = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        else:
            self.scale = 1.0

        self.project_conv = conv(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=padding,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None,
        )

    def forward(self, x: Tensor, guide: Tensor) -> Tensor:
        B, _, H, W = x.shape

        guide = self.guide_fc(guide)
        guide = guide.reshape(B, -1, self.num_heads, self.head_channels)
        embed = self.embed_conv(x) if self.embed_conv is not None else x
        embed = embed.reshape(B, self.num_heads, self.head_channels, H, W)

        if self.use_einsum:
            attn_weight = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        else:
            batch, m, channel, height, width = embed.shape
            _, n, _, _ = guide.shape
            embed = embed.permute(0, 1, 3, 4, 2)
            embed = embed.reshape(batch, m, -1, channel)
            guide = guide.permute(0, 2, 3, 1)
            attn_weight = torch.matmul(embed, guide)
            attn_weight = attn_weight.reshape(batch, m, height, width, n)

        attn_weight = attn_weight.max(dim=-1)[0]
        attn_weight = attn_weight / (self.head_channels**0.5)
        attn_weight = attn_weight + self.bias[None, :, None, None]
        attn_weight = attn_weight.sigmoid() * self.scale

        x = self.project_conv(x)
        x = x.reshape(B, self.num_heads, -1, H, W)
        x = x * attn_weight.unsqueeze(2)
        x = x.reshape(B, -1, H, W)
        return x


@MODELS.register_module()
class MaxSigmoidCSPLayerWithTwoConv(CSPLayerWithTwoConv):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        guide_channels: int,
        embed_channels: int,
        num_heads: int = 1,
        expand_ratio: float = 0.5,
        num_blocks: int = 1,
        with_scale: bool = False,
        add_identity: bool = True,
        conv_cfg: OptConfigType = None,
        norm_cfg: ConfigType = dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg: ConfigType = dict(type="SiLU", inplace=True),
        init_cfg: OptMultiConfig = None,
        use_einsum: bool = True,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            expand_ratio=expand_ratio,
            num_blocks=num_blocks,
            add_identity=add_identity,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg,
        )

        self.final_conv = ConvModule(
            (3 + num_blocks) * self.mid_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

        self.attn_block = MaxSigmoidAttnBlock(
            self.mid_channels,
            self.mid_channels,
            guide_channels=guide_channels,
            embed_channels=embed_channels,
            num_heads=num_heads,
            with_scale=with_scale,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            use_einsum=use_einsum,
        )

    def forward(self, x: Tensor, guide: Tensor) -> Tensor:
        x_main = self.main_conv(x)
        x_main = list(x_main.split((self.mid_channels, self.mid_channels), 1))
        x_main.extend(blocks(x_main[-1]) for blocks in self.blocks)
        x_main.append(self.attn_block(x_main[-1], guide))
        return self.final_conv(torch.cat(x_main, 1))


def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def unpack_gt_instances(batch_data_samples: SampleList) -> tuple:
    batch_gt_instances = []
    batch_gt_instances_ignore = []
    batch_img_metas = []
    for data_sample in batch_data_samples:
        batch_img_metas.append(data_sample.metainfo)
        batch_gt_instances.append(data_sample.gt_instances)
        if "ignored_instances" in data_sample:
            batch_gt_instances_ignore.append(data_sample.ignored_instances)
        else:
            batch_gt_instances_ignore.append(None)

    return batch_gt_instances, batch_gt_instances_ignore, batch_img_metas


def select_single_mlvl(mlvl_tensors, batch_id, detach=True):
    assert isinstance(mlvl_tensors, (list, tuple))
    num_levels = len(mlvl_tensors)

    if detach:
        mlvl_tensor_list = [
            mlvl_tensors[i][batch_id].detach() for i in range(num_levels)
        ]
    else:
        mlvl_tensor_list = [mlvl_tensors[i][batch_id] for i in range(num_levels)]
    return mlvl_tensor_list


def filter_scores_and_topk(scores, score_thr, topk, results=None):
    valid_mask = scores > score_thr
    scores = scores[valid_mask]
    valid_idxs = torch.nonzero(valid_mask)

    num_topk = min(topk, valid_idxs.size(0))
    scores, idxs = scores.sort(descending=True)
    scores = scores[:num_topk]
    topk_idxs = valid_idxs[idxs[:num_topk]]
    keep_idxs, labels = topk_idxs.unbind(dim=1)

    filtered_results = None
    if results is not None:
        if isinstance(results, dict):
            filtered_results = {k: v[keep_idxs] for k, v in results.items()}
        elif isinstance(results, list):
            filtered_results = [result[keep_idxs] for result in results]
        elif isinstance(results, torch.Tensor):
            filtered_results = results[keep_idxs]
        else:
            raise NotImplementedError(
                f"Only supports dict or list or Tensor, " f"but get {type(results)}."
            )
    return scores, labels, keep_idxs, filtered_results


def cat_boxes(
    data_list: List[Union[Tensor, BaseBoxes]], dim: int = 0
) -> Union[Tensor, BaseBoxes]:
    if data_list and isinstance(data_list[0], BaseBoxes):
        return data_list[0].cat(data_list, dim=dim)
    else:
        return torch.cat(data_list, dim=dim)


def scale_boxes(
    boxes: Union[Tensor, BaseBoxes], scale_factor: Tuple[float, float]
) -> Union[Tensor, BaseBoxes]:
    if isinstance(boxes, BaseBoxes):
        boxes.rescale_(scale_factor)
        return boxes
    else:
        repeat_num = int(boxes.size(-1) / 2)
        scale_factor = boxes.new_tensor(scale_factor).repeat((1, repeat_num))
        return boxes * scale_factor


def get_box_wh(boxes: Union[Tensor, BaseBoxes]) -> Tuple[Tensor, Tensor]:
    if isinstance(boxes, BaseBoxes):
        w = boxes.widths
        h = boxes.heights
    else:
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
    return w, h


def get_box_tensor(boxes: Union[Tensor, BaseBoxes]) -> Tensor:
    if isinstance(boxes, BaseBoxes):
        boxes = boxes.tensor
    return boxes


def batched_nms(
    boxes: Tensor,
    scores: Tensor,
    idxs: Tensor,
    nms_cfg: Optional[Dict],
    class_agnostic: bool = False,
) -> Tuple[Tensor, Tensor]:
    if nms_cfg is None:
        scores, inds = scores.sort(descending=True)
        boxes = boxes[inds]
        return torch.cat([boxes, scores[:, None]], -1), inds

    nms_cfg_ = nms_cfg.copy()
    class_agnostic = nms_cfg_.pop("class_agnostic", class_agnostic)
    if class_agnostic:
        boxes_for_nms = boxes
    else:
        if boxes.size(-1) == 5:
            max_coordinate = boxes[..., :2].max() + boxes[..., 2:4].max()
            offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
            boxes_ctr_for_nms = boxes[..., :2] + offsets[:, None]
            boxes_for_nms = torch.cat([boxes_ctr_for_nms, boxes[..., 2:5]], dim=-1)
        else:
            max_coordinate = boxes.max()
            offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
            boxes_for_nms = boxes + offsets[:, None]

    nms_op = nms_cfg_.pop("type", "nms")
    if isinstance(nms_op, str):
        nms_op = eval(nms_op)
    split_thr = nms_cfg_.pop("split_thr", 10000)
    if boxes_for_nms.shape[0] < split_thr:
        dets, keep = nms_op(boxes_for_nms, scores, **nms_cfg_)
        boxes = boxes[keep]
        scores = dets[:, -1]
    else:
        max_num = nms_cfg_.pop("max_num", -1)
        total_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
        scores_after_nms = scores.new_zeros(scores.size())
        for id in torch.unique(idxs):
            mask = (idxs == id).nonzero(as_tuple=False).view(-1)
            dets, keep = nms_op(boxes_for_nms[mask], scores[mask], **nms_cfg_)
            total_mask[mask[keep]] = True
            scores_after_nms[mask[keep]] = dets[:, -1]
        keep = total_mask.nonzero(as_tuple=False).view(-1)

        scores, inds = scores_after_nms[keep].sort(descending=True)
        keep = keep[inds]
        boxes = boxes[keep]

        if max_num > 0:
            keep = keep[:max_num]
            boxes = boxes[:max_num]
            scores = scores[:max_num]

    boxes = torch.cat([boxes, scores[:, None]], -1)
    return boxes, keep


def bbox_flip(
    bboxes: Tensor, img_shape: Tuple[int], direction: str = "horizontal"
) -> Tensor:

    assert bboxes.shape[-1] % 4 == 0
    assert direction in ["horizontal", "vertical", "diagonal"]
    flipped = bboxes.clone()
    if direction == "horizontal":
        flipped[..., 0::4] = img_shape[1] - bboxes[..., 2::4]
        flipped[..., 2::4] = img_shape[1] - bboxes[..., 0::4]
    elif direction == "vertical":
        flipped[..., 1::4] = img_shape[0] - bboxes[..., 3::4]
        flipped[..., 3::4] = img_shape[0] - bboxes[..., 1::4]
    else:
        flipped[..., 0::4] = img_shape[1] - bboxes[..., 2::4]
        flipped[..., 1::4] = img_shape[0] - bboxes[..., 3::4]
        flipped[..., 2::4] = img_shape[1] - bboxes[..., 0::4]
        flipped[..., 3::4] = img_shape[0] - bboxes[..., 1::4]
    return flipped


def bbox_mapping_back(
    bboxes: Tensor,
    img_shape: Tuple[int],
    scale_factor: Union[float, Tuple[float]],
    flip: bool,
    flip_direction: str = "horizontal",
) -> Tensor:

    new_bboxes = bbox_flip(bboxes, img_shape, flip_direction) if flip else bboxes
    new_bboxes = new_bboxes.view(-1, 4) / new_bboxes.new_tensor(scale_factor)
    return new_bboxes.view(bboxes.shape)


def merge_aug_results(aug_batch_results, aug_batch_img_metas):

    num_augs = len(aug_batch_results)
    num_imgs = len(aug_batch_results[0])

    batch_results = []
    aug_batch_results = copy.deepcopy(aug_batch_results)
    for img_id in range(num_imgs):
        aug_results = []
        for aug_id in range(num_augs):
            img_metas = aug_batch_img_metas[aug_id][img_id]
            results = aug_batch_results[aug_id][img_id]

            img_shape = img_metas["img_shape"]
            scale_factor = img_metas["scale_factor"]
            flip = img_metas["flip"]
            flip_direction = img_metas["flip_direction"]
            bboxes = bbox_mapping_back(
                results.bboxes, img_shape, scale_factor, flip, flip_direction
            )
            results.bboxes = bboxes
            aug_results.append(results)
        merged_aug_results = results.cat(aug_results)
        batch_results.append(merged_aug_results)

    return batch_results


class BaseDenseHead(BaseModule, metaclass=ABCMeta):
    def __init__(self, init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)

        self._raw_positive_infos = dict()

    def init_weights(self) -> None:
        super().init_weights()
        for m in self.modules():
            if hasattr(m, "conv_offset"):
                constant_init(m.conv_offset, 0)

    def get_positive_infos(self) -> InstanceList:
        if len(self._raw_positive_infos) == 0:
            return None

        sampling_results = self._raw_positive_infos.get("sampling_results", None)
        assert sampling_results is not None
        positive_infos = []
        for sampling_result in enumerate(sampling_results):
            pos_info = InstanceData()
            pos_info.bboxes = sampling_result.pos_gt_bboxes
            pos_info.labels = sampling_result.pos_gt_labels
            pos_info.priors = sampling_result.pos_priors
            pos_info.pos_assigned_gt_inds = sampling_result.pos_assigned_gt_inds
            pos_info.pos_inds = sampling_result.pos_inds
            positive_infos.append(pos_info)
        return positive_infos

    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList) -> dict:
        outs = self(x)

        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore, batch_img_metas) = outputs

        loss_inputs = outs + (
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore,
        )
        losses = self.loss_by_feat(*loss_inputs)
        return losses

    @abstractmethod
    def loss_by_feat(self, **kwargs) -> dict:
        pass

    def loss_and_predict(
        self,
        x: Tuple[Tensor],
        batch_data_samples: SampleList,
        proposal_cfg: Optional[ConfigDict] = None,
    ) -> Tuple[dict, InstanceList]:
        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore, batch_img_metas) = outputs

        outs = self(x)

        loss_inputs = outs + (
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore,
        )
        losses = self.loss_by_feat(*loss_inputs)

        predictions = self.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas, cfg=proposal_cfg
        )
        return losses, predictions

    def predict(
        self, x: Tuple[Tensor], batch_data_samples: SampleList, rescale: bool = False
    ) -> InstanceList:
        batch_img_metas = [data_samples.metainfo for data_samples in batch_data_samples]

        outs = self(x)

        predictions = self.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas, rescale=rescale
        )
        return predictions

    def predict_by_feat(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        score_factors: Optional[List[Tensor]] = None,
        batch_img_metas: Optional[List[dict]] = None,
        cfg: Optional[ConfigDict] = None,
        rescale: bool = False,
        with_nms: bool = True,
    ) -> InstanceList:
        assert len(cls_scores) == len(bbox_preds)

        if score_factors is None:
            with_score_factors = False
        else:
            with_score_factors = True
            assert len(cls_scores) == len(score_factors)

        num_levels = len(cls_scores)

        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes, dtype=cls_scores[0].dtype
        )

        result_list = []

        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            cls_score_list = select_single_mlvl(cls_scores, img_id, detach=True)
            bbox_pred_list = select_single_mlvl(bbox_preds, img_id, detach=True)
            if with_score_factors:
                score_factor_list = select_single_mlvl(
                    score_factors, img_id, detach=True
                )
            else:
                score_factor_list = [None for _ in range(num_levels)]

            results = self._predict_by_feat_single(
                cls_score_list=cls_score_list,
                bbox_pred_list=bbox_pred_list,
                score_factor_list=score_factor_list,
                mlvl_priors=mlvl_priors,
                img_meta=img_meta,
                cfg=cfg,
                rescale=rescale,
                with_nms=with_nms,
            )
            result_list.append(results)
        return result_list

    def _predict_by_feat_single(
        self,
        cls_score_list: List[Tensor],
        bbox_pred_list: List[Tensor],
        score_factor_list: List[Tensor],
        mlvl_priors: List[Tensor],
        img_meta: dict,
        cfg: ConfigDict,
        rescale: bool = False,
        with_nms: bool = True,
    ) -> InstanceData:
        if score_factor_list[0] is None:
            with_score_factors = False
        else:
            with_score_factors = True

        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        img_shape = img_meta["img_shape"]
        nms_pre = cfg.get("nms_pre", -1)

        mlvl_bbox_preds = []
        mlvl_valid_priors = []
        mlvl_scores = []
        mlvl_labels = []
        if with_score_factors:
            mlvl_score_factors = []
        else:
            mlvl_score_factors = None
        for level_idx, (cls_score, bbox_pred, score_factor, priors) in enumerate(
            zip(cls_score_list, bbox_pred_list, score_factor_list, mlvl_priors)
        ):

            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            dim = self.bbox_coder.encode_size
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, dim)
            if with_score_factors:
                score_factor = score_factor.permute(1, 2, 0).reshape(-1).sigmoid()
            cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)[:, :-1]
            score_thr = cfg.get("score_thr", 0)

            results = filter_scores_and_topk(
                scores, score_thr, nms_pre, dict(bbox_pred=bbox_pred, priors=priors)
            )
            scores, labels, keep_idxs, filtered_results = results

            bbox_pred = filtered_results["bbox_pred"]
            priors = filtered_results["priors"]

            if with_score_factors:
                score_factor = score_factor[keep_idxs]

            mlvl_bbox_preds.append(bbox_pred)
            mlvl_valid_priors.append(priors)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)

            if with_score_factors:
                mlvl_score_factors.append(score_factor)

        bbox_pred = torch.cat(mlvl_bbox_preds)
        priors = cat_boxes(mlvl_valid_priors)
        bboxes = self.bbox_coder.decode(priors, bbox_pred, max_shape=img_shape)

        results = InstanceData()
        results.bboxes = bboxes
        results.scores = torch.cat(mlvl_scores)
        results.labels = torch.cat(mlvl_labels)
        if with_score_factors:
            results.score_factors = torch.cat(mlvl_score_factors)

        return self._bbox_post_process(
            results=results,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms,
            img_meta=img_meta,
        )

    def _bbox_post_process(
        self,
        results: InstanceData,
        cfg: ConfigDict,
        rescale: bool = False,
        with_nms: bool = True,
        img_meta: Optional[dict] = None,
    ) -> InstanceData:
        if rescale:
            assert img_meta.get("scale_factor") is not None
            scale_factor = [1 / s for s in img_meta["scale_factor"]]
            results.bboxes = scale_boxes(results.bboxes, scale_factor)

        if hasattr(results, "score_factors"):
            score_factors = results.pop("score_factors")
            results.scores = results.scores * score_factors
        if cfg.get("min_bbox_size", -1) >= 0:
            w, h = get_box_wh(results.bboxes)
            valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
            if not valid_mask.all():
                results = results[valid_mask]
        if with_nms and results.bboxes.numel() > 0:
            bboxes = get_box_tensor(results.bboxes)
            det_bboxes, keep_idxs = batched_nms(
                bboxes, results.scores, results.labels, cfg.nms
            )
            results = results[keep_idxs]
            results.scores = det_bboxes[:, -1]
            results = results[: cfg.max_per_img]

        return results

    def aug_test(
        self,
        aug_batch_feats,
        aug_batch_img_metas,
        rescale=False,
        with_ori_nms=False,
        **kwargs,
    ):
        sig_of_get_results = signature(self.get_results)
        get_results_args = [p.name for p in sig_of_get_results.parameters.values()]
        get_results_single_sig = signature(self._get_results_single)
        get_results_single_sig_args = [
            p.name for p in get_results_single_sig.parameters.values()
        ]
        assert ("with_nms" in get_results_args) and (
            "with_nms" in get_results_single_sig_args
        ), (f"{self.__class__.__name__}" "does not support test-time augmentation ")

        num_imgs = len(aug_batch_img_metas[0])
        aug_batch_results = []
        for x, img_metas in zip(aug_batch_feats, aug_batch_img_metas):
            outs = self.forward(x)
            batch_instance_results = self.get_results(
                *outs,
                img_metas=img_metas,
                cfg=self.test_cfg,
                rescale=False,
                with_nms=with_ori_nms,
                **kwargs,
            )
            aug_batch_results.append(batch_instance_results)
        batch_results = merge_aug_results(aug_batch_results, aug_batch_img_metas)

        final_results = []
        for img_id in range(num_imgs):
            results = batch_results[img_id]
            det_bboxes, keep_idxs = batched_nms(
                results.bboxes, results.scores, results.labels, self.test_cfg.nms
            )
            results = results[keep_idxs]
            results.scores = det_bboxes[:, -1]
            results = results[: self.test_cfg.max_per_img]
            if rescale:
                pass
            else:
                scale_factor = results.bboxes.new_tensor(
                    aug_batch_img_metas[0][img_id]["scale_factor"]
                )
                results.bboxes = results.bboxes * scale_factor

            final_results.append(results)

        return final_results


TASK_UTILS = Registry("task util")


@MODELS.register_module()
class YOLOv5Head(BaseDenseHead):
    def __init__(
        self,
        head_module: ConfigType,
        prior_generator: ConfigType = dict(
            type="YOLOAnchorGenerator",
            base_sizes=[
                [(10, 13), (16, 30), (33, 23)],
                [(30, 61), (62, 45), (59, 119)],
                [(116, 90), (156, 198), (373, 326)],
            ],
            strides=[8, 16, 32],
        ),
        bbox_coder: ConfigType = dict(type="YOLOv5BBoxCoder"),
        loss_cls: ConfigType = dict(
            type="CrossEntropyLoss",
            use_sigmoid=True,
            reduction="mean",
            loss_weight=0.5,
        ),
        loss_bbox: ConfigType = dict(
            type="IoULoss",
            iou_mode="ciou",
            bbox_format="xywh",
            eps=1e-7,
            reduction="mean",
            loss_weight=0.05,
            return_iou=True,
        ),
        loss_obj: ConfigType = dict(
            type="CrossEntropyLoss",
            use_sigmoid=True,
            reduction="mean",
            loss_weight=1.0,
        ),
        prior_match_thr: float = 4.0,
        near_neighbor_thr: float = 0.5,
        ignore_iof_thr: float = -1.0,
        obj_level_weights: List[float] = [4.0, 1.0, 0.4],
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        init_cfg: OptMultiConfig = None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.head_module = MODELS.build(head_module)
        self.num_classes = self.head_module.num_classes
        self.featmap_strides = self.head_module.featmap_strides
        self.num_levels = len(self.featmap_strides)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.loss_cls: nn.Module = MODELS.build(loss_cls)
        self.loss_bbox: nn.Module = MODELS.build(loss_bbox)
        self.loss_obj: nn.Module = MODELS.build(loss_obj)

        self.prior_generator = TASK_UTILS.build(prior_generator)
        self.bbox_coder = TASK_UTILS.build(bbox_coder)
        self.num_base_priors = self.prior_generator.num_base_priors[0]

        self.featmap_sizes = [torch.empty(1)] * self.num_levels

        self.prior_match_thr = prior_match_thr
        self.near_neighbor_thr = near_neighbor_thr
        self.obj_level_weights = obj_level_weights
        self.ignore_iof_thr = ignore_iof_thr

        self.special_init()

    def special_init(self):
        assert (
            len(self.obj_level_weights) == len(self.featmap_strides) == self.num_levels
        )
        if self.prior_match_thr != 4.0:
            print_log(
                "!!!Now, you've changed the prior_match_thr "
                "parameter to something other than 4.0. Please make sure "
                "that you have modified both the regression formula in "
                "bbox_coder and before loss_box computation, "
                "otherwise the accuracy may be degraded!!!"
            )

        if self.num_classes == 1:
            print_log(
                "!!!You are using `YOLOv5Head` with num_classes == 1."
                " The loss_cls will be 0. This is a normal phenomenon."
            )

        priors_base_sizes = torch.tensor(
            self.prior_generator.base_sizes, dtype=torch.float
        )
        featmap_strides = torch.tensor(self.featmap_strides, dtype=torch.float)[
            :, None, None
        ]
        self.register_buffer(
            "priors_base_sizes", priors_base_sizes / featmap_strides, persistent=False
        )

        grid_offset = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],
            ]
        ).float()
        self.register_buffer("grid_offset", grid_offset[:, None], persistent=False)

        prior_inds = (
            torch.arange(self.num_base_priors).float().view(self.num_base_priors, 1)
        )
        self.register_buffer("prior_inds", prior_inds, persistent=False)

    def forward(self, x: Tuple[Tensor]) -> Tuple[List]:
        return self.head_module(x)

    def predict_by_feat(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        objectnesses: Optional[List[Tensor]] = None,
        batch_img_metas: Optional[List[dict]] = None,
        cfg: Optional[ConfigDict] = None,
        rescale: bool = True,
        with_nms: bool = True,
    ) -> List[InstanceData]:
        assert len(cls_scores) == len(bbox_preds)
        if objectnesses is None:
            with_objectnesses = False
        else:
            with_objectnesses = True
            assert len(cls_scores) == len(objectnesses)

        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)

        multi_label = cfg.multi_label
        multi_label &= self.num_classes > 1
        cfg.multi_label = multi_label

        num_imgs = len(batch_img_metas)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        if featmap_sizes != self.featmap_sizes:
            self.mlvl_priors = self.prior_generator.grid_priors(
                featmap_sizes, dtype=cls_scores[0].dtype
            )
            self.featmap_sizes = featmap_sizes
        flatten_priors = torch.cat(self.mlvl_priors)

        mlvl_strides = [
            flatten_priors.new_full(
                (featmap_size.numel() * self.num_base_priors,), stride
            )
            for featmap_size, stride in zip(featmap_sizes, self.featmap_strides)
        ]
        flatten_stride = torch.cat(mlvl_strides)
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.num_classes)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]

        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_decoded_bboxes = self.bbox_coder.decode(
            flatten_priors[None], flatten_bbox_preds, flatten_stride
        )

        if with_objectnesses:
            flatten_objectness = [
                objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
                for objectness in objectnesses
            ]
            flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()
        else:
            flatten_objectness = [None for _ in range(num_imgs)]

        results_list = []
        for bboxes, scores, objectness, img_meta in zip(
            flatten_decoded_bboxes,
            flatten_cls_scores,
            flatten_objectness,
            batch_img_metas,
        ):
            ori_shape = img_meta["ori_shape"]
            scale_factor = img_meta["scale_factor"]
            if "pad_param" in img_meta:
                pad_param = img_meta["pad_param"]
            else:
                pad_param = None

            score_thr = cfg.get("score_thr", -1)
            if (
                objectness is not None
                and score_thr > 0
                and not cfg.get("yolox_style", False)
            ):
                conf_inds = objectness > score_thr
                bboxes = bboxes[conf_inds, :]
                scores = scores[conf_inds, :]
                objectness = objectness[conf_inds]

            if objectness is not None:
                scores *= objectness[:, None]

            if scores.shape[0] == 0:
                empty_results = InstanceData()
                empty_results.bboxes = bboxes
                empty_results.scores = scores[:, 0]
                empty_results.labels = scores[:, 0].int()
                results_list.append(empty_results)
                continue

            nms_pre = cfg.get("nms_pre", 100000)
            if cfg.multi_label is False:
                scores, labels = scores.max(1, keepdim=True)
                scores, _, keep_idxs, results = filter_scores_and_topk(
                    scores, score_thr, nms_pre, results=dict(labels=labels[:, 0])
                )
                labels = results["labels"]
            else:
                scores, labels, keep_idxs, _ = filter_scores_and_topk(
                    scores, score_thr, nms_pre
                )

            results = InstanceData(
                scores=scores, labels=labels, bboxes=bboxes[keep_idxs]
            )

            if rescale:
                if pad_param is not None:
                    results.bboxes -= results.bboxes.new_tensor(
                        [pad_param[2], pad_param[0], pad_param[2], pad_param[0]]
                    )
                results.bboxes /= results.bboxes.new_tensor(scale_factor).repeat((1, 2))

            if cfg.get("yolox_style", False):
                cfg.max_per_img = len(results)

            results = self._bbox_post_process(
                results=results,
                cfg=cfg,
                rescale=False,
                with_nms=with_nms,
                img_meta=img_meta,
            )
            results.bboxes[:, 0::2].clamp_(0, ori_shape[1])
            results.bboxes[:, 1::2].clamp_(0, ori_shape[0])

            results_list.append(results)
        return results_list

    def loss(self, x: Tuple[Tensor], batch_data_samples: Union[list, dict]) -> dict:
        if isinstance(batch_data_samples, list):
            losses = super().loss(x, batch_data_samples)
        else:
            outs = self(x)
            loss_inputs = outs + (
                batch_data_samples["bboxes_labels"],
                batch_data_samples["img_metas"],
            )
            losses = self.loss_by_feat(*loss_inputs)

        return losses

    def loss_by_feat(
        self,
        cls_scores: Sequence[Tensor],
        bbox_preds: Sequence[Tensor],
        objectnesses: Sequence[Tensor],
        batch_gt_instances: Sequence[InstanceData],
        batch_img_metas: Sequence[dict],
        batch_gt_instances_ignore: OptInstanceList = None,
    ) -> dict:
        if self.ignore_iof_thr != -1:
            batch_target_ignore_list = []
            for i, gt_instances_ignore in enumerate(batch_gt_instances_ignore):
                bboxes = gt_instances_ignore.bboxes
                labels = gt_instances_ignore.labels
                index = bboxes.new_full((len(bboxes), 1), i)
                target = torch.cat((index, labels[:, None].float(), bboxes), dim=1)
                batch_target_ignore_list.append(target)

            batch_gt_targets_ignore = torch.cat(batch_target_ignore_list, dim=0)
            if batch_gt_targets_ignore.shape[0] != 0:
                return self._loss_by_feat_with_ignore(
                    cls_scores,
                    bbox_preds,
                    objectnesses,
                    batch_gt_instances=batch_gt_instances,
                    batch_img_metas=batch_img_metas,
                    batch_gt_instances_ignore=batch_gt_targets_ignore,
                )
        batch_targets_normed = self._convert_gt_to_norm_format(
            batch_gt_instances, batch_img_metas
        )

        device = cls_scores[0].device
        loss_cls = torch.zeros(1)
        loss_box = torch.zeros(1)
        loss_obj = torch.zeros(1)
        scaled_factor = torch.ones(7)

        for i in range(self.num_levels):
            batch_size, _, h, w = bbox_preds[i].shape
            target_obj = torch.zeros_like(objectnesses[i])
            if batch_targets_normed.shape[1] == 0:
                loss_box += bbox_preds[i].sum() * 0
                loss_cls += cls_scores[i].sum() * 0
                loss_obj += (
                    self.loss_obj(objectnesses[i], target_obj)
                    * self.obj_level_weights[i]
                )
                continue

            priors_base_sizes_i = self.priors_base_sizes[i]
            scaled_factor[2:6] = torch.tensor(bbox_preds[i].shape)[[3, 2, 3, 2]]
            batch_targets_scaled = batch_targets_normed * scaled_factor

            wh_ratio = batch_targets_scaled[..., 4:6] / priors_base_sizes_i[:, None]
            match_inds = (
                torch.max(wh_ratio, 1 / wh_ratio).max(2)[0] < self.prior_match_thr
            )
            batch_targets_scaled = batch_targets_scaled[match_inds]
            if batch_targets_scaled.shape[0] == 0:
                loss_box += bbox_preds[i].sum() * 0
                loss_cls += cls_scores[i].sum() * 0
                loss_obj += (
                    self.loss_obj(objectnesses[i], target_obj)
                    * self.obj_level_weights[i]
                )
                continue

            batch_targets_cxcy = batch_targets_scaled[:, 2:4]
            grid_xy = scaled_factor[[2, 3]] - batch_targets_cxcy
            left, up = (
                (batch_targets_cxcy % 1 < self.near_neighbor_thr)
                & (batch_targets_cxcy > 1)
            ).T
            right, bottom = ((grid_xy % 1 < self.near_neighbor_thr) & (grid_xy > 1)).T
            offset_inds = torch.stack((torch.ones_like(left), left, up, right, bottom))

            batch_targets_scaled = batch_targets_scaled.repeat((5, 1, 1))[offset_inds]
            retained_offsets = self.grid_offset.repeat(1, offset_inds.shape[1], 1)[
                offset_inds
            ]

            _chunk_targets = batch_targets_scaled.chunk(4, 1)
            img_class_inds, grid_xy, grid_wh, priors_inds = _chunk_targets
            priors_inds, (img_inds, class_inds) = (
                priors_inds.long().view(-1),
                img_class_inds.long().T,
            )

            grid_xy_long = (grid_xy - retained_offsets * self.near_neighbor_thr).long()
            grid_x_inds, grid_y_inds = grid_xy_long.T
            bboxes_targets = torch.cat((grid_xy - grid_xy_long, grid_wh), 1)

            retained_bbox_pred = bbox_preds[i].reshape(
                batch_size, self.num_base_priors, -1, h, w
            )[img_inds, priors_inds, :, grid_y_inds, grid_x_inds]
            priors_base_sizes_i = priors_base_sizes_i[priors_inds]
            decoded_bbox_pred = self._decode_bbox_to_xywh(
                retained_bbox_pred, priors_base_sizes_i
            )
            loss_box_i, iou = self.loss_bbox(decoded_bbox_pred, bboxes_targets)
            loss_box += loss_box_i

            iou = iou.detach().clamp(0)
            target_obj[img_inds, priors_inds, grid_y_inds, grid_x_inds] = iou.type(
                target_obj.dtype
            )
            loss_obj += (
                self.loss_obj(objectnesses[i], target_obj) * self.obj_level_weights[i]
            )

            if self.num_classes > 1:
                pred_cls_scores = cls_scores[i].reshape(
                    batch_size, self.num_base_priors, -1, h, w
                )[img_inds, priors_inds, :, grid_y_inds, grid_x_inds]

                target_class = torch.full_like(pred_cls_scores, 0.0)
                target_class[range(batch_targets_scaled.shape[0]), class_inds] = 1.0
                loss_cls += self.loss_cls(pred_cls_scores, target_class)
            else:
                loss_cls += cls_scores[i].sum() * 0

        _, world_size = get_dist_info()
        return dict(
            loss_cls=loss_cls * batch_size * world_size,
            loss_obj=loss_obj * batch_size * world_size,
            loss_bbox=loss_box * batch_size * world_size,
        )

    def _convert_gt_to_norm_format(
        self,
        batch_gt_instances: Sequence[InstanceData],
        batch_img_metas: Sequence[dict],
    ) -> Tensor:
        if isinstance(batch_gt_instances, torch.Tensor):

            img_shape = batch_img_metas[0]["batch_input_shape"]
            gt_bboxes_xyxy = batch_gt_instances[:, 2:]
            xy1, xy2 = gt_bboxes_xyxy.split((2, 2), dim=-1)
            gt_bboxes_xywh = torch.cat([(xy2 + xy1) / 2, (xy2 - xy1)], dim=-1)
            gt_bboxes_xywh[:, 1::2] /= img_shape[0]
            gt_bboxes_xywh[:, 0::2] /= img_shape[1]
            batch_gt_instances[:, 2:] = gt_bboxes_xywh
            batch_targets_normed = batch_gt_instances.repeat(self.num_base_priors, 1, 1)
        else:
            batch_target_list = []
            for i, gt_instances in enumerate(batch_gt_instances):
                img_shape = batch_img_metas[i]["batch_input_shape"]
                bboxes = gt_instances.bboxes
                labels = gt_instances.labels

                xy1, xy2 = bboxes.split((2, 2), dim=-1)
                bboxes = torch.cat([(xy2 + xy1) / 2, (xy2 - xy1)], dim=-1)
                bboxes[:, 1::2] /= img_shape[0]
                bboxes[:, 0::2] /= img_shape[1]

                index = bboxes.new_full((len(bboxes), 1), i)
                target = torch.cat((index, labels[:, None].float(), bboxes), dim=1)
                batch_target_list.append(target)

            batch_targets_normed = torch.cat(batch_target_list, dim=0).repeat(
                self.num_base_priors, 1, 1
            )
        batch_targets_prior_inds = self.prior_inds.repeat(
            1, batch_targets_normed.shape[1]
        )[..., None]
        batch_targets_normed = torch.cat(
            (batch_targets_normed, batch_targets_prior_inds), 2
        )
        return batch_targets_normed

    def _decode_bbox_to_xywh(self, bbox_pred, priors_base_sizes) -> Tensor:
        bbox_pred = bbox_pred.sigmoid()
        pred_xy = bbox_pred[:, :2] * 2 - 0.5
        pred_wh = (bbox_pred[:, 2:] * 2) ** 2 * priors_base_sizes
        decoded_bbox_pred = torch.cat((pred_xy, pred_wh), dim=-1)
        return decoded_bbox_pred

    def _loss_by_feat_with_ignore(
        self,
        cls_scores: Sequence[Tensor],
        bbox_preds: Sequence[Tensor],
        objectnesses: Sequence[Tensor],
        batch_gt_instances: Sequence[InstanceData],
        batch_img_metas: Sequence[dict],
        batch_gt_instances_ignore: Sequence[Tensor],
    ) -> dict:
        batch_targets_normed = self._convert_gt_to_norm_format(
            batch_gt_instances, batch_img_metas
        )

        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        if featmap_sizes != self.featmap_sizes:
            self.mlvl_priors = self.prior_generator.grid_priors(
                featmap_sizes, dtype=cls_scores[0].dtype
            )
            self.featmap_sizes = featmap_sizes

        device = cls_scores[0].device
        loss_cls = torch.zeros(1)
        loss_box = torch.zeros(1)
        loss_obj = torch.zeros(1)
        scaled_factor = torch.ones(7)

        for i in range(self.num_levels):
            batch_size, _, h, w = bbox_preds[i].shape
            target_obj = torch.zeros_like(objectnesses[i])

            not_ignore_flags = bbox_preds[i].new_ones(
                batch_size, self.num_base_priors, h, w
            )

            ignore_overlaps = bbox_overlaps(
                self.mlvl_priors[i], batch_gt_instances_ignore[..., 2:], "iof"
            )
            ignore_max_overlaps, ignore_max_ignore_index = ignore_overlaps.max(dim=1)

            batch_inds = batch_gt_instances_ignore[:, 0][ignore_max_ignore_index]
            ignore_inds = (ignore_max_overlaps > self.ignore_iof_thr).nonzero(
                as_tuple=True
            )[0]
            batch_inds = batch_inds[ignore_inds].long()
            ignore_priors, ignore_grid_xs, ignore_grid_ys = get_prior_xy_info(
                ignore_inds, self.num_base_priors, self.featmap_sizes[i]
            )
            not_ignore_flags[
                batch_inds, ignore_priors, ignore_grid_ys, ignore_grid_xs
            ] = 0
            if batch_targets_normed.shape[1] == 0:
                loss_box += bbox_preds[i].sum() * 0
                loss_cls += cls_scores[i].sum() * 0
                loss_obj += (
                    self.loss_obj(
                        objectnesses[i],
                        target_obj,
                        weight=not_ignore_flags,
                        avg_factor=max(not_ignore_flags.sum(), 1),
                    )
                    * self.obj_level_weights[i]
                )
                continue

            priors_base_sizes_i = self.priors_base_sizes[i]
            scaled_factor[2:6] = torch.tensor(bbox_preds[i].shape)[[3, 2, 3, 2]]

            batch_targets_scaled = batch_targets_normed * scaled_factor
            wh_ratio = batch_targets_scaled[..., 4:6] / priors_base_sizes_i[:, None]
            match_inds = (
                torch.max(wh_ratio, 1 / wh_ratio).max(2)[0] < self.prior_match_thr
            )
            batch_targets_scaled = batch_targets_scaled[match_inds]
            if batch_targets_scaled.shape[0] == 0:
                loss_box += bbox_preds[i].sum() * 0
                loss_cls += cls_scores[i].sum() * 0
                loss_obj += (
                    self.loss_obj(
                        objectnesses[i],
                        target_obj,
                        weight=not_ignore_flags,
                        avg_factor=max(not_ignore_flags.sum(), 1),
                    )
                    * self.obj_level_weights[i]
                )
                continue

            batch_targets_cxcy = batch_targets_scaled[:, 2:4]
            grid_xy = scaled_factor[[2, 3]] - batch_targets_cxcy
            left, up = (
                (batch_targets_cxcy % 1 < self.near_neighbor_thr)
                & (batch_targets_cxcy > 1)
            ).T
            right, bottom = ((grid_xy % 1 < self.near_neighbor_thr) & (grid_xy > 1)).T
            offset_inds = torch.stack((torch.ones_like(left), left, up, right, bottom))

            batch_targets_scaled = batch_targets_scaled.repeat((5, 1, 1))[offset_inds]
            retained_offsets = self.grid_offset.repeat(1, offset_inds.shape[1], 1)[
                offset_inds
            ]
            _chunk_targets = batch_targets_scaled.chunk(4, 1)
            img_class_inds, grid_xy, grid_wh, priors_inds = _chunk_targets
            priors_inds, (img_inds, class_inds) = (
                priors_inds.long().view(-1),
                img_class_inds.long().T,
            )

            grid_xy_long = (grid_xy - retained_offsets * self.near_neighbor_thr).long()
            grid_x_inds, grid_y_inds = grid_xy_long.T
            bboxes_targets = torch.cat((grid_xy - grid_xy_long, grid_wh), 1)
            retained_bbox_pred = bbox_preds[i].reshape(
                batch_size, self.num_base_priors, -1, h, w
            )[img_inds, priors_inds, :, grid_y_inds, grid_x_inds]
            priors_base_sizes_i = priors_base_sizes_i[priors_inds]
            decoded_bbox_pred = self._decode_bbox_to_xywh(
                retained_bbox_pred, priors_base_sizes_i
            )

            not_ignore_weights = not_ignore_flags[
                img_inds, priors_inds, grid_y_inds, grid_x_inds
            ]
            loss_box_i, iou = self.loss_bbox(
                decoded_bbox_pred,
                bboxes_targets,
                weight=not_ignore_weights,
                avg_factor=max(not_ignore_weights.sum(), 1),
            )
            loss_box += loss_box_i
            iou = iou.detach().clamp(0)
            target_obj[img_inds, priors_inds, grid_y_inds, grid_x_inds] = iou.type(
                target_obj.dtype
            )
            loss_obj += (
                self.loss_obj(
                    objectnesses[i],
                    target_obj,
                    weight=not_ignore_flags,
                    avg_factor=max(not_ignore_flags.sum(), 1),
                )
                * self.obj_level_weights[i]
            )

            if self.num_classes > 1:
                pred_cls_scores = cls_scores[i].reshape(
                    batch_size, self.num_base_priors, -1, h, w
                )[img_inds, priors_inds, :, grid_y_inds, grid_x_inds]

                target_class = torch.full_like(pred_cls_scores, 0.0)
                target_class[range(batch_targets_scaled.shape[0]), class_inds] = 1.0
                loss_cls += self.loss_cls(
                    pred_cls_scores,
                    target_class,
                    weight=not_ignore_weights[:, None].repeat(1, self.num_classes),
                    avg_factor=max(not_ignore_weights.sum(), 1),
                )
            else:
                loss_cls += cls_scores[i].sum() * 0

        _, world_size = get_dist_info()
        return dict(
            loss_cls=loss_cls * batch_size * world_size,
            loss_obj=loss_obj * batch_size * world_size,
            loss_bbox=loss_box * batch_size * world_size,
        )


def gt_instances_preprocess(
    batch_gt_instances: Union[Tensor, Sequence], batch_size: int
) -> Tensor:
    if isinstance(batch_gt_instances, Sequence):
        max_gt_bbox_len = max(
            [len(gt_instances) for gt_instances in batch_gt_instances]
        )
        batch_instance_list = []
        for index, gt_instance in enumerate(batch_gt_instances):
            bboxes = gt_instance.bboxes
            labels = gt_instance.labels
            box_dim = get_box_tensor(bboxes).size(-1)
            batch_instance_list.append(torch.cat((labels[:, None], bboxes), dim=-1))

            if bboxes.shape[0] >= max_gt_bbox_len:
                continue

            fill_tensor = bboxes.new_full(
                [max_gt_bbox_len - bboxes.shape[0], box_dim + 1], 0
            )
            batch_instance_list[index] = torch.cat(
                (batch_instance_list[index], fill_tensor), dim=0
            )

        return torch.stack(batch_instance_list)
    else:
        assert isinstance(batch_gt_instances, Tensor)
        box_dim = batch_gt_instances.size(-1) - 2
        if len(batch_gt_instances) > 0:
            gt_images_indexes = batch_gt_instances[:, 0]
            max_gt_bbox_len = gt_images_indexes.unique(return_counts=True)[1].max()
            batch_instance = torch.zeros(
                (batch_size, max_gt_bbox_len, box_dim + 1),
                dtype=batch_gt_instances.dtype,
            )

            for i in range(batch_size):
                match_indexes = gt_images_indexes == i
                gt_num = match_indexes.sum()
                if gt_num:
                    batch_instance[i, :gt_num] = batch_gt_instances[match_indexes, 1:]
        else:
            batch_instance = torch.zeros(
                (batch_size, 0, box_dim + 1),
                dtype=batch_gt_instances.dtype,
            )

        return batch_instance


@MODELS.register_module()
class YOLOv8Head(YOLOv5Head):
    def __init__(
        self,
        head_module: ConfigType,
        prior_generator: ConfigType = dict(
            type="MlvlPointGenerator", offset=0.5, strides=[8, 16, 32]
        ),
        bbox_coder: ConfigType = dict(type="DistancePointBBoxCoder"),
        loss_cls: ConfigType = dict(
            type="CrossEntropyLoss",
            use_sigmoid=True,
            reduction="none",
            loss_weight=0.5,
        ),
        loss_bbox: ConfigType = dict(
            type="IoULoss",
            iou_mode="ciou",
            bbox_format="xyxy",
            reduction="sum",
            loss_weight=7.5,
            return_iou=False,
        ),
        loss_dfl=dict(
            type="DistributionFocalLoss", reduction="mean", loss_weight=1.5 / 4
        ),
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        init_cfg: OptMultiConfig = None,
    ):
        super().__init__(
            head_module=head_module,
            prior_generator=prior_generator,
            bbox_coder=bbox_coder,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
        )
        self.loss_dfl = MODELS.build(loss_dfl)
        self.loss_obj = None

    def special_init(self):
        if self.train_cfg:
            self.assigner = TASK_UTILS.build(self.train_cfg.assigner)

            self.featmap_sizes_train = None
            self.num_level_priors = None
            self.flatten_priors_train = None
            self.stride_tensor = None

    def loss_by_feat(
        self,
        cls_scores: Sequence[Tensor],
        bbox_preds: Sequence[Tensor],
        bbox_dist_preds: Sequence[Tensor],
        batch_gt_instances: Sequence[InstanceData],
        batch_img_metas: Sequence[dict],
        batch_gt_instances_ignore: OptInstanceList = None,
    ) -> dict:
        num_imgs = len(batch_img_metas)

        current_featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        if current_featmap_sizes != self.featmap_sizes_train:
            self.featmap_sizes_train = current_featmap_sizes

            mlvl_priors_with_stride = self.prior_generator.grid_priors(
                self.featmap_sizes_train,
                dtype=cls_scores[0].dtype,
                with_stride=True,
            )

            self.num_level_priors = [len(n) for n in mlvl_priors_with_stride]
            self.flatten_priors_train = torch.cat(mlvl_priors_with_stride, dim=0)
            self.stride_tensor = self.flatten_priors_train[..., [2]]
        gt_info = gt_instances_preprocess(batch_gt_instances, num_imgs)
        gt_labels = gt_info[:, :, :1]
        gt_bboxes = gt_info[:, :, 1:]
        pad_bbox_flag = (gt_bboxes.sum(-1, keepdim=True) > 0).float()

        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.num_classes)
            for cls_pred in cls_scores
        ]
        flatten_pred_bboxes = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_pred_dists = [
            bbox_pred_org.reshape(num_imgs, -1, self.head_module.reg_max * 4)
            for bbox_pred_org in bbox_dist_preds
        ]

        flatten_dist_preds = torch.cat(flatten_pred_dists, dim=1)
        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_pred_bboxes = torch.cat(flatten_pred_bboxes, dim=1)
        flatten_pred_bboxes = self.bbox_coder.decode(
            self.flatten_priors_train[..., :2],
            flatten_pred_bboxes,
            self.stride_tensor[..., 0],
        )

        assigned_result = self.assigner(
            (flatten_pred_bboxes.detach()).type(gt_bboxes.dtype),
            flatten_cls_preds.detach().sigmoid(),
            self.flatten_priors_train,
            gt_labels,
            gt_bboxes,
            pad_bbox_flag,
        )

        assigned_bboxes = assigned_result["assigned_bboxes"]
        assigned_scores = assigned_result["assigned_scores"]
        fg_mask_pre_prior = assigned_result["fg_mask_pre_prior"]

        assigned_scores_sum = assigned_scores.sum().clamp(min=1)

        loss_cls = self.loss_cls(flatten_cls_preds, assigned_scores).sum()
        loss_cls /= assigned_scores_sum
        assigned_bboxes /= self.stride_tensor
        flatten_pred_bboxes /= self.stride_tensor
        num_pos = fg_mask_pre_prior.sum()
        if num_pos > 0:
            prior_bbox_mask = fg_mask_pre_prior.unsqueeze(-1).repeat([1, 1, 4])
            pred_bboxes_pos = torch.masked_select(
                flatten_pred_bboxes, prior_bbox_mask
            ).reshape([-1, 4])
            assigned_bboxes_pos = torch.masked_select(
                assigned_bboxes, prior_bbox_mask
            ).reshape([-1, 4])
            bbox_weight = torch.masked_select(
                assigned_scores.sum(-1), fg_mask_pre_prior
            ).unsqueeze(-1)
            loss_bbox = (
                self.loss_bbox(pred_bboxes_pos, assigned_bboxes_pos, weight=bbox_weight)
                / assigned_scores_sum
            )
            pred_dist_pos = flatten_dist_preds[fg_mask_pre_prior]
            assigned_ltrb = self.bbox_coder.encode(
                self.flatten_priors_train[..., :2] / self.stride_tensor,
                assigned_bboxes,
                max_dis=self.head_module.reg_max - 1,
                eps=0.01,
            )
            assigned_ltrb_pos = torch.masked_select(
                assigned_ltrb, prior_bbox_mask
            ).reshape([-1, 4])
            loss_dfl = self.loss_dfl(
                pred_dist_pos.reshape(-1, self.head_module.reg_max),
                assigned_ltrb_pos.reshape(-1),
                weight=bbox_weight.expand(-1, 4).reshape(-1),
                avg_factor=assigned_scores_sum,
            )
        else:
            loss_bbox = flatten_pred_bboxes.sum() * 0
            loss_dfl = flatten_pred_bboxes.sum() * 0
        _, world_size = get_dist_info()
        return dict(
            loss_cls=loss_cls * num_imgs * world_size,
            loss_bbox=loss_bbox * num_imgs * world_size,
            loss_dfl=loss_dfl * num_imgs * world_size,
        )


@MODELS.register_module()
class YOLOWorldHead(YOLOv8Head):
    def __init__(self, world_size=-1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.world_size = world_size

    def loss(
        self,
        img_feats: Tuple[Tensor],
        txt_feats: Tensor,
        txt_masks: Tensor,
        batch_data_samples: Union[list, dict],
    ) -> dict:

        outs = self(img_feats, txt_feats, txt_masks)
        loss_inputs = outs + (
            batch_data_samples["bboxes_labels"],
            batch_data_samples["img_metas"],
        )
        losses = self.loss_by_feat(*loss_inputs)

        return losses

    def loss_and_predict(
        self,
        img_feats: Tuple[Tensor],
        txt_feats: Tensor,
        txt_masks: Tensor,
        batch_data_samples: SampleList,
        proposal_cfg: Optional[ConfigDict] = None,
    ) -> Tuple[dict, InstanceList]:
        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore, batch_img_metas) = outputs

        outs = self(img_feats, txt_feats, txt_masks)

        loss_inputs = outs + (
            txt_masks,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore,
        )
        losses = self.loss_by_feat(*loss_inputs)

        predictions = self.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas, cfg=proposal_cfg
        )
        return losses, predictions

    def forward(
        self, img_feats: Tuple[Tensor], txt_feats: Tensor, txt_masks: Tensor
    ) -> Tuple[List]:
        return self.head_module(img_feats, txt_feats, txt_masks)

    def predict(
        self,
        img_feats: Tuple[Tensor],
        txt_feats: Tensor,
        txt_masks: Tensor,
        batch_data_samples: SampleList,
        rescale: bool = False,
    ) -> InstanceList:
        batch_img_metas = [data_samples.metainfo for data_samples in batch_data_samples]
        outs = self(img_feats, txt_feats, txt_masks)
        predictions = self.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas, rescale=rescale
        )
        return predictions

    def aug_test(
        self,
        aug_batch_feats,
        aug_batch_img_metas,
        rescale=False,
        with_ori_nms=False,
        **kwargs,
    ):
        raise NotImplementedError("aug_test is not implemented yet.")

    def loss_by_feat(
        self,
        cls_scores: Sequence[Tensor],
        bbox_preds: Sequence[Tensor],
        bbox_dist_preds: Sequence[Tensor],
        batch_text_masks: Tensor,
        batch_gt_instances: Sequence[InstanceData],
        batch_img_metas: Sequence[dict],
        batch_gt_instances_ignore: OptInstanceList = None,
    ) -> dict:
        num_imgs = len(batch_img_metas)

        current_featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        if current_featmap_sizes != self.featmap_sizes_train:
            self.featmap_sizes_train = current_featmap_sizes

            mlvl_priors_with_stride = self.prior_generator.grid_priors(
                self.featmap_sizes_train,
                dtype=cls_scores[0].dtype,
                with_stride=True,
            )

            self.num_level_priors = [len(n) for n in mlvl_priors_with_stride]
            self.flatten_priors_train = torch.cat(mlvl_priors_with_stride, dim=0)
            self.stride_tensor = self.flatten_priors_train[..., [2]]
        gt_info = gt_instances_preprocess(batch_gt_instances, num_imgs)
        gt_labels = gt_info[:, :, :1]
        gt_bboxes = gt_info[:, :, 1:]
        pad_bbox_flag = (gt_bboxes.sum(-1, keepdim=True) > 0).float()
        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.num_classes)
            for cls_pred in cls_scores
        ]
        flatten_pred_bboxes = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_pred_dists = [
            bbox_pred_org.reshape(num_imgs, -1, self.head_module.reg_max * 4)
            for bbox_pred_org in bbox_dist_preds
        ]

        flatten_dist_preds = torch.cat(flatten_pred_dists, dim=1)
        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_pred_bboxes = torch.cat(flatten_pred_bboxes, dim=1)
        flatten_pred_bboxes = self.bbox_coder.decode(
            self.flatten_priors_train[..., :2],
            flatten_pred_bboxes,
            self.stride_tensor[..., 0],
        )

        assigned_result = self.assigner(
            (flatten_pred_bboxes.detach()).type(gt_bboxes.dtype),
            flatten_cls_preds.detach().sigmoid(),
            self.flatten_priors_train,
            gt_labels,
            gt_bboxes,
            pad_bbox_flag,
        )

        assigned_bboxes = assigned_result["assigned_bboxes"]
        assigned_scores = assigned_result["assigned_scores"]
        fg_mask_pre_prior = assigned_result["fg_mask_pre_prior"]

        assigned_scores_sum = assigned_scores.sum().clamp(min=1)

        if batch_text_masks is not None:
            cls_weight = (
                batch_text_masks.view(num_imgs, 1, -1)
                .expand(-1, flatten_cls_preds.shape[1], -1)
                .to(flatten_cls_preds)
            )

            loss_cls = self.loss_cls(flatten_cls_preds, assigned_scores)
            _loss_cls = (loss_cls * cls_weight).sum(dim=-1)
            loss_cls = _loss_cls.sum()
        else:
            loss_cls = self.loss_cls(flatten_cls_preds, assigned_scores).sum()
        loss_cls /= assigned_scores_sum
        assigned_bboxes /= self.stride_tensor
        flatten_pred_bboxes /= self.stride_tensor
        num_pos = fg_mask_pre_prior.sum()
        if num_pos > 0:
            prior_bbox_mask = fg_mask_pre_prior.unsqueeze(-1).repeat([1, 1, 4])
            pred_bboxes_pos = torch.masked_select(
                flatten_pred_bboxes, prior_bbox_mask
            ).reshape([-1, 4])
            assigned_bboxes_pos = torch.masked_select(
                assigned_bboxes, prior_bbox_mask
            ).reshape([-1, 4])
            bbox_weight = torch.masked_select(
                assigned_scores.sum(-1), fg_mask_pre_prior
            ).unsqueeze(-1)
            loss_bbox = (
                self.loss_bbox(pred_bboxes_pos, assigned_bboxes_pos, weight=bbox_weight)
                / assigned_scores_sum
            )

            pred_dist_pos = flatten_dist_preds[fg_mask_pre_prior]
            assigned_ltrb = self.bbox_coder.encode(
                self.flatten_priors_train[..., :2] / self.stride_tensor,
                assigned_bboxes,
                max_dis=self.head_module.reg_max - 1,
                eps=0.01,
            )
            assigned_ltrb_pos = torch.masked_select(
                assigned_ltrb, prior_bbox_mask
            ).reshape([-1, 4])
            loss_dfl = self.loss_dfl(
                pred_dist_pos.reshape(-1, self.head_module.reg_max),
                assigned_ltrb_pos.reshape(-1),
                weight=bbox_weight.expand(-1, 4).reshape(-1),
                avg_factor=assigned_scores_sum,
            )
        else:
            loss_bbox = flatten_pred_bboxes.sum() * 0
            loss_dfl = flatten_pred_bboxes.sum() * 0
        if self.world_size == -1:
            _, world_size = get_dist_info()
        else:
            world_size = self.world_size
        return dict(
            loss_cls=loss_cls * num_imgs * world_size,
            loss_bbox=loss_bbox * num_imgs * world_size,
            loss_dfl=loss_dfl * num_imgs * world_size,
        )

    def predict_by_feat(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        objectnesses: Optional[List[Tensor]] = None,
        batch_img_metas: Optional[List[dict]] = None,
        cfg: Optional[ConfigDict] = None,
        rescale: bool = True,
        with_nms: bool = True,
    ) -> List[InstanceData]:
        assert len(cls_scores) == len(bbox_preds)
        if objectnesses is None:
            with_objectnesses = False
        else:
            with_objectnesses = True
            assert len(cls_scores) == len(objectnesses)

        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)

        multi_label = cfg.multi_label
        multi_label &= self.num_classes > 1
        cfg.multi_label = multi_label

        num_imgs = len(batch_img_metas)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        if featmap_sizes != self.featmap_sizes:
            self.mlvl_priors = self.prior_generator.grid_priors(
                featmap_sizes, dtype=cls_scores[0].dtype
            )
            self.featmap_sizes = featmap_sizes
        flatten_priors = torch.cat(self.mlvl_priors)
        mlvl_strides = [
            flatten_priors.new_full(
                (featmap_size.numel() * self.num_base_priors,), stride
            )
            for featmap_size, stride in zip(featmap_sizes, self.featmap_strides)
        ]
        flatten_stride = torch.cat(mlvl_strides)
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.num_classes)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_decoded_bboxes = self.bbox_coder.decode(
            flatten_priors[None], flatten_bbox_preds, flatten_stride
        )
        if with_objectnesses:
            flatten_objectness = [
                objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
                for objectness in objectnesses
            ]
            flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()

        else:
            flatten_objectness = [None for _ in range(num_imgs)]

        results_list = []
        for img_i, (bboxes, scores, objectness, img_meta) in enumerate(
            zip(
                flatten_decoded_bboxes,
                flatten_cls_scores,
                flatten_objectness,
                batch_img_metas,
            )
        ):

            ori_shape = img_meta["ori_shape"]
            scale_factor = img_meta["scale_factor"]
            if "pad_param" in img_meta:
                pad_param = img_meta["pad_param"]
            else:
                pad_param = None

            score_thr = cfg.get("score_thr", -1)
            if (
                objectness is not None
                and score_thr > 0
                and not cfg.get("yolox_style", False)
            ):
                conf_inds = objectness > score_thr
                bboxes = bboxes[conf_inds, :]
                scores = scores[conf_inds, :]
                objectness = objectness[conf_inds]

            if objectness is not None:
                scores *= objectness[:, None]

            if scores.shape[0] == 0:
                empty_results = InstanceData()
                empty_results.bboxes = bboxes
                empty_results.scores = scores[:, 0]
                empty_results.labels = scores[:, 0].int()
                results_list.append(empty_results)
                continue

            nms_pre = cfg.get("nms_pre", 100000)
            if cfg.multi_label is False:
                scores, labels = scores.max(1, keepdim=True)
                scores, _, keep_idxs, results = filter_scores_and_topk(
                    scores, score_thr, nms_pre, results=dict(labels=labels[:, 0])
                )
                labels = results["labels"]
            else:
                scores, labels, keep_idxs, _ = filter_scores_and_topk(
                    scores, score_thr, nms_pre
                )

            results = InstanceData(
                scores=scores, labels=labels, bboxes=bboxes[keep_idxs]
            )

            if rescale:
                if pad_param is not None:
                    results.bboxes -= results.bboxes.new_tensor(
                        [pad_param[2], pad_param[0], pad_param[2], pad_param[0]]
                    )
                results.bboxes /= results.bboxes.new_tensor(scale_factor).repeat((1, 2))

            if cfg.get("yolox_style", False):
                cfg.max_per_img = len(results)

            results = self._bbox_post_process(
                results=results,
                cfg=cfg,
                rescale=False,
                with_nms=with_nms,
                img_meta=img_meta,
            )

            results.bboxes[:, 0::2].clamp_(0, ori_shape[1])
            results.bboxes[:, 1::2].clamp_(0, ori_shape[0])

            results_list.append(results)
        return results_list


@MODELS.register_module()
class YOLOv8HeadModule(BaseModule):
    def __init__(
        self,
        num_classes: int,
        in_channels: Union[int, Sequence],
        widen_factor: float = 1.0,
        num_base_priors: int = 1,
        featmap_strides: Sequence[int] = (8, 16, 32),
        reg_max: int = 16,
        norm_cfg: ConfigType = dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg: ConfigType = dict(type="SiLU", inplace=True),
        init_cfg: OptMultiConfig = None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.featmap_strides = featmap_strides
        self.num_levels = len(self.featmap_strides)
        self.num_base_priors = num_base_priors
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_channels = in_channels
        self.reg_max = reg_max

        in_channels = []
        for channel in self.in_channels:
            channel = make_divisible(channel, widen_factor)
            in_channels.append(channel)
        self.in_channels = in_channels

        self._init_layers()

    def init_weights(self, prior_prob=0.01):
        super().init_weights()
        for reg_pred, cls_pred, stride in zip(
            self.reg_preds, self.cls_preds, self.featmap_strides
        ):
            reg_pred[-1].bias.data[:] = 1.0
            cls_pred[-1].bias.data[: self.num_classes] = math.log(
                5 / self.num_classes / (640 / stride) ** 2
            )

    def _init_layers(self):
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()

        reg_out_channels = max((16, self.in_channels[0] // 4, self.reg_max * 4))
        cls_out_channels = max(self.in_channels[0], self.num_classes)

        for i in range(self.num_levels):
            self.reg_preds.append(
                nn.Sequential(
                    ConvModule(
                        in_channels=self.in_channels[i],
                        out_channels=reg_out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                    ),
                    ConvModule(
                        in_channels=reg_out_channels,
                        out_channels=reg_out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                    ),
                    nn.Conv2d(
                        in_channels=reg_out_channels,
                        out_channels=4 * self.reg_max,
                        kernel_size=1,
                    ),
                )
            )
            self.cls_preds.append(
                nn.Sequential(
                    ConvModule(
                        in_channels=self.in_channels[i],
                        out_channels=cls_out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                    ),
                    ConvModule(
                        in_channels=cls_out_channels,
                        out_channels=cls_out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                    ),
                    nn.Conv2d(
                        in_channels=cls_out_channels,
                        out_channels=self.num_classes,
                        kernel_size=1,
                    ),
                )
            )

        proj = torch.arange(self.reg_max, dtype=torch.float)
        self.register_buffer("proj", proj, persistent=False)

    def forward(self, x: Tuple[Tensor]) -> Tuple[List]:
        assert len(x) == self.num_levels
        return multi_apply(self.forward_single, x, self.cls_preds, self.reg_preds)

    def forward_single(
        self, x: torch.Tensor, cls_pred: nn.ModuleList, reg_pred: nn.ModuleList
    ) -> Tuple:
        b, _, h, w = x.shape
        cls_logit = cls_pred(x)
        bbox_dist_preds = reg_pred(x)
        if self.reg_max > 1:
            bbox_dist_preds = bbox_dist_preds.reshape(
                [-1, 4, self.reg_max, h * w]
            ).permute(0, 3, 1, 2)
            bbox_preds = (
                bbox_dist_preds.softmax(3).matmul(self.proj.view([-1, 1])).squeeze(-1)
            )
            bbox_preds = bbox_preds.transpose(1, 2).reshape(b, -1, h, w)
        else:
            bbox_preds = bbox_dist_preds
        if self.training:
            return cls_logit, bbox_preds, bbox_dist_preds
        else:
            return cls_logit, bbox_preds


@MODELS.register_module()
class ContrastiveHead(BaseModule):
    def __init__(
        self, embed_dims: int, init_cfg: OptConfigType = None, use_einsum: bool = True
    ) -> None:

        super().__init__(init_cfg=init_cfg)

        self.bias = nn.Parameter(torch.zeros([]))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.use_einsum = use_einsum

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)

        if self.use_einsum:
            x = torch.einsum("bchw,bkc->bkhw", x, w)
        else:
            batch, channel, height, width = x.shape
            _, k, _ = w.shape
            x = x.permute(0, 2, 3, 1)
            x = x.reshape(batch, -1, channel)
            w = w.permute(0, 2, 1)
            x = torch.matmul(x, w)
            x = x.reshape(batch, height, width, k)
            x = x.permute(0, 3, 1, 2)

        x = x * self.logit_scale.exp() + self.bias
        return x


@MODELS.register_module()
class BNContrastiveHead(BaseModule):
    def __init__(
        self,
        embed_dims: int,
        norm_cfg: ConfigDict,
        init_cfg: OptConfigType = None,
        use_einsum: bool = True,
    ) -> None:

        super().__init__(init_cfg=init_cfg)
        self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        self.bias = nn.Parameter(torch.zeros([]))
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))
        self.use_einsum = use_einsum

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)

        if self.use_einsum:
            x = torch.einsum("bchw,bkc->bkhw", x, w)
        else:
            batch, channel, height, width = x.shape
            _, k, _ = w.shape
            x = x.permute(0, 2, 3, 1)
            x = x.reshape(batch, -1, channel)
            w = w.permute(0, 2, 1)
            x = torch.matmul(x, w)
            x = x.reshape(batch, height, width, k)
            x = x.permute(0, 3, 1, 2)

        x = x * self.logit_scale.exp() + self.bias
        return x


@MODELS.register_module()
class YOLOWorldHeadModule(YOLOv8HeadModule):
    def __init__(
        self,
        *args,
        embed_dims: int,
        use_bn_head: bool = False,
        use_einsum: bool = True,
        freeze_all: bool = False,
        **kwargs,
    ) -> None:
        self.embed_dims = embed_dims
        self.use_bn_head = use_bn_head
        self.use_einsum = use_einsum
        self.freeze_all = freeze_all
        super().__init__(*args, **kwargs)

    def init_weights(self, prior_prob=0.01):
        super().init_weights()
        for cls_pred, cls_contrast, stride in zip(
            self.cls_preds, self.cls_contrasts, self.featmap_strides
        ):
            cls_pred[-1].bias.data[:] = 0.0
            if hasattr(cls_contrast, "bias"):
                nn.init.constant_(
                    cls_contrast.bias.data,
                    math.log(5 / self.num_classes / (640 / stride) ** 2),
                )

    def _init_layers(self) -> None:
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.cls_contrasts = nn.ModuleList()

        reg_out_channels = max((16, self.in_channels[0] // 4, self.reg_max * 4))
        cls_out_channels = max(self.in_channels[0], self.num_classes)

        for i in range(self.num_levels):
            self.reg_preds.append(
                nn.Sequential(
                    ConvModule(
                        in_channels=self.in_channels[i],
                        out_channels=reg_out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                    ),
                    ConvModule(
                        in_channels=reg_out_channels,
                        out_channels=reg_out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                    ),
                    nn.Conv2d(
                        in_channels=reg_out_channels,
                        out_channels=4 * self.reg_max,
                        kernel_size=1,
                    ),
                )
            )
            self.cls_preds.append(
                nn.Sequential(
                    ConvModule(
                        in_channels=self.in_channels[i],
                        out_channels=cls_out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                    ),
                    ConvModule(
                        in_channels=cls_out_channels,
                        out_channels=cls_out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                    ),
                    nn.Conv2d(
                        in_channels=cls_out_channels,
                        out_channels=self.embed_dims,
                        kernel_size=1,
                    ),
                )
            )
            if self.use_bn_head:
                self.cls_contrasts.append(
                    BNContrastiveHead(
                        self.embed_dims, self.norm_cfg, use_einsum=self.use_einsum
                    )
                )
            else:
                self.cls_contrasts.append(
                    ContrastiveHead(self.embed_dims, use_einsum=self.use_einsum)
                )

        proj = torch.arange(self.reg_max, dtype=torch.float)
        self.register_buffer("proj", proj, persistent=False)

        if self.freeze_all:
            self._freeze_all()

    def _freeze_all(self):
        for m in self.modules():
            if isinstance(m, _BatchNorm):
                m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_all:
            self._freeze_all()

    def forward(
        self, img_feats: Tuple[Tensor], txt_feats: Tensor, txt_masks: Tensor
    ) -> Tuple[List]:
        assert len(img_feats) == self.num_levels
        txt_feats = [txt_feats for _ in range(self.num_levels)]
        txt_masks = [txt_masks for _ in range(self.num_levels)]
        return multi_apply(
            self.forward_single,
            img_feats,
            txt_feats,
            txt_masks,
            self.cls_preds,
            self.reg_preds,
            self.cls_contrasts,
        )

    def forward_single(
        self,
        img_feat: Tensor,
        txt_feat: Tensor,
        txt_masks: Tensor,
        cls_pred: nn.ModuleList,
        reg_pred: nn.ModuleList,
        cls_contrast: nn.ModuleList,
    ) -> Tuple:
        b, _, h, w = img_feat.shape
        cls_embed = cls_pred(img_feat)
        cls_logit = cls_contrast(cls_embed, txt_feat)

        if txt_masks is not None:
            txt_masks = txt_masks.view(b, -1, 1, 1).expand(-1, -1, h, w)
            if self.training:
                cls_logit = cls_logit * txt_masks
                cls_logit[txt_masks == 0] = -10e6
            else:
                cls_logit[txt_masks == 0] = -10e6

        bbox_dist_preds = reg_pred(img_feat)
        if self.reg_max > 1:
            bbox_dist_preds = bbox_dist_preds.reshape(
                [-1, 4, self.reg_max, h * w]
            ).permute(0, 3, 1, 2)
            bbox_preds = (
                bbox_dist_preds.softmax(3).matmul(self.proj.view([-1, 1])).squeeze(-1)
            )
            bbox_preds = bbox_preds.transpose(1, 2).reshape(b, -1, h, w)
        else:
            bbox_preds = bbox_dist_preds
        if self.training:
            return cls_logit, bbox_preds, bbox_dist_preds
        else:
            return cls_logit, bbox_preds


def _expand_onehot_labels(labels, label_weights, label_channels, ignore_index):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    valid_mask = (labels >= 0) & (labels != ignore_index)
    inds = torch.nonzero(valid_mask & (labels < label_channels), as_tuple=False)

    if inds.numel() > 0:
        bin_labels[inds, labels[inds]] = 1

    valid_mask = valid_mask.view(-1, 1).expand(labels.size(0), label_channels).float()
    if label_weights is None:
        bin_label_weights = valid_mask
    else:
        bin_label_weights = label_weights.view(-1, 1).repeat(1, label_channels)
        bin_label_weights *= valid_mask

    return bin_labels, bin_label_weights, valid_mask


def reduce_loss(loss: Tensor, reduction: str) -> Tensor:
    reduction_enum = F._Reduction.get_enum(reduction)
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(
    loss: Tensor,
    weight: Optional[Tensor] = None,
    reduction: str = "mean",
    avg_factor: Optional[float] = None,
) -> Tensor:
    if weight is not None:
        loss = loss * weight
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        if reduction == "mean":
            eps = torch.finfo(torch.float32).eps
            loss = loss.sum() / (avg_factor + eps)

        elif reduction != "none":
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def binary_cross_entropy(
    pred,
    label,
    weight=None,
    reduction="mean",
    avg_factor=None,
    class_weight=None,
    ignore_index=-100,
    avg_non_ignore=False,
):
    ignore_index = -100 if ignore_index is None else ignore_index

    if pred.dim() != label.dim():
        label, weight, valid_mask = _expand_onehot_labels(
            label, weight, pred.size(-1), ignore_index
        )
    else:
        valid_mask = ((label >= 0) & (label != ignore_index)).float()
        if weight is not None:
            weight = weight * valid_mask
        else:
            weight = valid_mask
    if (avg_factor is None) and avg_non_ignore and reduction == "mean":
        avg_factor = valid_mask.sum().item()
    weight = weight.float()
    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), pos_weight=class_weight, reduction="none"
    )
    loss = weight_reduce_loss(loss, weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def mask_cross_entropy(
    pred,
    target,
    label,
    reduction="mean",
    avg_factor=None,
    class_weight=None,
    ignore_index=None,
    **kwargs,
):
    assert ignore_index is None, "BCE loss does not support ignore_index"
    assert reduction == "mean" and avg_factor is None
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long)
    pred_slice = pred[inds, label].squeeze(1)
    return F.binary_cross_entropy_with_logits(
        pred_slice, target, weight=class_weight, reduction="mean"
    )[None]


def cross_entropy(
    pred,
    label,
    weight=None,
    reduction="mean",
    avg_factor=None,
    class_weight=None,
    ignore_index=-100,
    avg_non_ignore=False,
):
    ignore_index = -100 if ignore_index is None else ignore_index
    loss = F.cross_entropy(
        pred, label, weight=class_weight, reduction="none", ignore_index=ignore_index
    )
    if (avg_factor is None) and avg_non_ignore and reduction == "mean":
        avg_factor = label.numel() - (label == ignore_index).sum().item()
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor
    )

    return loss


@MODELS.register_module()
class CrossEntropyLoss(nn.Module):
    def __init__(
        self,
        use_sigmoid=False,
        use_mask=False,
        reduction="mean",
        class_weight=None,
        ignore_index=None,
        loss_weight=1.0,
        avg_non_ignore=False,
    ):
        super(CrossEntropyLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.ignore_index = ignore_index
        self.avg_non_ignore = avg_non_ignore
        if (
            (ignore_index is not None)
            and not self.avg_non_ignore
            and self.reduction == "mean"
        ):
            warnings.warn(
                "Default ``avg_non_ignore`` is False, if you would like to "
                "ignore the certain label and average loss over non-ignore "
                "labels, which is the same with PyTorch official "
                "cross_entropy, set ``avg_non_ignore=True``."
            )

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = cross_entropy

    def extra_repr(self):
        s = f"avg_non_ignore={self.avg_non_ignore}"
        return s

    def forward(
        self,
        cls_score,
        label,
        weight=None,
        avg_factor=None,
        reduction_override=None,
        ignore_index=None,
        **kwargs,
    ):
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        if ignore_index is None:
            ignore_index = self.ignore_index

        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            ignore_index=ignore_index,
            avg_non_ignore=self.avg_non_ignore,
            **kwargs,
        )
        return loss_cls


def _register_box(name: str, box_type: Type, force: bool = False) -> None:
    assert issubclass(box_type, BaseBoxes)
    name = name.lower()

    if not force and (name in box_types or box_type in _box_type_to_name):
        raise KeyError(f"box type {name} has been registered")
    elif name in box_types:
        _box_type = box_types.pop(name)
        _box_type_to_name.pop(_box_type)
    elif box_type in _box_type_to_name:
        _name = _box_type_to_name.pop(box_type)
        box_types.pop(_name)

    box_types[name] = box_type
    _box_type_to_name[box_type] = name


def register_box(
    name: str, box_type: Type = None, force: bool = False
) -> Union[Type, Callable]:
    if not isinstance(force, bool):
        raise TypeError(f"force must be a boolean, but got {type(force)}")
    if box_type is not None:
        _register_box(name=name, box_type=box_type, force=force)
        return box_type

    def _register(cls):
        _register_box(name=name, box_type=cls, force=force)
        return cls

    return _register


@register_box(name="hbox")
class HorizontalBoxes(BaseBoxes):
    box_dim: int = 4

    def __init__(
        self,
        data: Union[Tensor, np.ndarray],
        dtype: torch.dtype = None,
        device: DeviceType = None,
        clone: bool = True,
        in_mode: Optional[str] = None,
    ) -> None:
        super().__init__(data=data, dtype=dtype, clone=clone)
        if isinstance(in_mode, str):
            if in_mode not in ("xyxy", "cxcywh"):
                raise ValueError(f"Get invalid mode {in_mode}.")
            if in_mode == "cxcywh":
                self.tensor = self.cxcywh_to_xyxy(self.tensor)

    @staticmethod
    def cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
        ctr, wh = boxes.split((2, 2), dim=-1)
        return torch.cat([(ctr - wh / 2), (ctr + wh / 2)], dim=-1)

    @staticmethod
    def xyxy_to_cxcywh(boxes: Tensor) -> Tensor:
        xy1, xy2 = boxes.split((2, 2), dim=-1)
        return torch.cat([(xy2 + xy1) / 2, (xy2 - xy1)], dim=-1)

    @property
    def cxcywh(self) -> Tensor:
        return self.xyxy_to_cxcywh(self.tensor)

    @property
    def centers(self) -> Tensor:
        boxes = self.tensor
        return (boxes[..., :2] + boxes[..., 2:]) / 2

    @property
    def areas(self) -> Tensor:
        boxes = self.tensor
        return (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])

    @property
    def widths(self) -> Tensor:
        boxes = self.tensor
        return boxes[..., 2] - boxes[..., 0]

    @property
    def heights(self) -> Tensor:
        boxes = self.tensor
        return boxes[..., 3] - boxes[..., 1]

    def flip_(self, img_shape: Tuple[int, int], direction: str = "horizontal") -> None:
        assert direction in ["horizontal", "vertical", "diagonal"]
        flipped = self.tensor
        boxes = flipped.clone()
        if direction == "horizontal":
            flipped[..., 0] = img_shape[1] - boxes[..., 2]
            flipped[..., 2] = img_shape[1] - boxes[..., 0]
        elif direction == "vertical":
            flipped[..., 1] = img_shape[0] - boxes[..., 3]
            flipped[..., 3] = img_shape[0] - boxes[..., 1]
        else:
            flipped[..., 0] = img_shape[1] - boxes[..., 2]
            flipped[..., 1] = img_shape[0] - boxes[..., 3]
            flipped[..., 2] = img_shape[1] - boxes[..., 0]
            flipped[..., 3] = img_shape[0] - boxes[..., 1]

    def translate_(self, distances: Tuple[float, float]) -> None:
        boxes = self.tensor
        assert len(distances) == 2
        self.tensor = boxes + boxes.new_tensor(distances).repeat(2)

    def clip_(self, img_shape: Tuple[int, int]) -> None:
        boxes = self.tensor
        boxes[..., 0::2] = boxes[..., 0::2].clamp(0, img_shape[1])
        boxes[..., 1::2] = boxes[..., 1::2].clamp(0, img_shape[0])

    def rotate_(self, center: Tuple[float, float], angle: float) -> None:
        boxes = self.tensor
        rotation_matrix = boxes.new_tensor(cv2.getRotationMatrix2D(center, -angle, 1))

        corners = self.hbox2corner(boxes)
        corners = torch.cat([corners, corners.new_ones(*corners.shape[:-1], 1)], dim=-1)
        corners_T = torch.transpose(corners, -1, -2)
        corners_T = torch.matmul(rotation_matrix, corners_T)
        corners = torch.transpose(corners_T, -1, -2)
        self.tensor = self.corner2hbox(corners)

    def project_(self, homography_matrix: Union[Tensor, np.ndarray]) -> None:
        boxes = self.tensor
        if isinstance(homography_matrix, np.ndarray):
            homography_matrix = boxes.new_tensor(homography_matrix)
        corners = self.hbox2corner(boxes)
        corners = torch.cat([corners, corners.new_ones(*corners.shape[:-1], 1)], dim=-1)
        corners_T = torch.transpose(corners, -1, -2)
        corners_T = torch.matmul(homography_matrix, corners_T)
        corners = torch.transpose(corners_T, -1, -2)
        corners = corners[..., :2] / corners[..., 2:3]
        self.tensor = self.corner2hbox(corners)

    @staticmethod
    def hbox2corner(boxes: Tensor) -> Tensor:
        x1, y1, x2, y2 = torch.split(boxes, 1, dim=-1)
        corners = torch.cat([x1, y1, x2, y1, x1, y2, x2, y2], dim=-1)
        return corners.reshape(*corners.shape[:-1], 4, 2)

    @staticmethod
    def corner2hbox(corners: Tensor) -> Tensor:
        if corners.numel() == 0:
            return corners.new_zeros((0, 4))
        min_xy = corners.min(dim=-2)[0]
        max_xy = corners.max(dim=-2)[0]
        return torch.cat([min_xy, max_xy], dim=-1)

    def rescale_(self, scale_factor: Tuple[float, float]) -> None:
        boxes = self.tensor
        assert len(scale_factor) == 2
        scale_factor = boxes.new_tensor(scale_factor).repeat(2)
        self.tensor = boxes * scale_factor

    def resize_(self, scale_factor: Tuple[float, float]) -> None:
        boxes = self.tensor
        assert len(scale_factor) == 2
        ctrs = (boxes[..., 2:] + boxes[..., :2]) / 2
        wh = boxes[..., 2:] - boxes[..., :2]
        scale_factor = boxes.new_tensor(scale_factor)
        wh = wh * scale_factor
        xy1 = ctrs - 0.5 * wh
        xy2 = ctrs + 0.5 * wh
        self.tensor = torch.cat([xy1, xy2], dim=-1)

    def is_inside(
        self,
        img_shape: Tuple[int, int],
        all_inside: bool = False,
        allowed_border: int = 0,
    ) -> BoolTensor:
        img_h, img_w = img_shape
        boxes = self.tensor
        if all_inside:
            return (
                (boxes[:, 0] >= -allowed_border)
                & (boxes[:, 1] >= -allowed_border)
                & (boxes[:, 2] < img_w + allowed_border)
                & (boxes[:, 3] < img_h + allowed_border)
            )
        else:
            return (
                (boxes[..., 0] < img_w + allowed_border)
                & (boxes[..., 1] < img_h + allowed_border)
                & (boxes[..., 2] > -allowed_border)
                & (boxes[..., 3] > -allowed_border)
            )

    def find_inside_points(
        self, points: Tensor, is_aligned: bool = False
    ) -> BoolTensor:
        boxes = self.tensor
        assert boxes.dim() == 2, "boxes dimension must be 2."

        if not is_aligned:
            boxes = boxes[None, :, :]
            points = points[:, None, :]
        else:
            assert boxes.size(0) == points.size(0)

        x_min, y_min, x_max, y_max = boxes.unbind(dim=-1)
        return (
            (points[..., 0] >= x_min)
            & (points[..., 0] <= x_max)
            & (points[..., 1] >= y_min)
            & (points[..., 1] <= y_max)
        )

    @staticmethod
    def overlaps(
        boxes1: BaseBoxes,
        boxes2: BaseBoxes,
        mode: str = "iou",
        is_aligned: bool = False,
        eps: float = 1e-6,
    ) -> Tensor:
        boxes1 = boxes1.convert_to("hbox")
        boxes2 = boxes2.convert_to("hbox")
        return bbox_overlaps(
            boxes1.tensor, boxes2.tensor, mode=mode, is_aligned=is_aligned, eps=eps
        )

    @staticmethod
    def from_instance_masks(masks: MaskType) -> "HorizontalBoxes":
        num_masks = len(masks)
        boxes = np.zeros((num_masks, 4), dtype=np.float32)
        if isinstance(masks, BitmapMasks):
            x_any = masks.masks.any(axis=1)
            y_any = masks.masks.any(axis=2)
            for idx in range(num_masks):
                x = np.where(x_any[idx, :])[0]
                y = np.where(y_any[idx, :])[0]
                if len(x) > 0 and len(y) > 0:
                    boxes[idx, :] = np.array(
                        [x[0], y[0], x[-1] + 1, y[-1] + 1], dtype=np.float32
                    )
        elif isinstance(masks, PolygonMasks):
            for idx, poly_per_obj in enumerate(masks.masks):
                xy_min = np.array([masks.width * 2, masks.height * 2], dtype=np.float32)
                xy_max = np.zeros(2, dtype=np.float32)
                for p in poly_per_obj:
                    xy = np.array(p).reshape(-1, 2).astype(np.float32)
                    xy_min = np.minimum(xy_min, np.min(xy, axis=0))
                    xy_max = np.maximum(xy_max, np.max(xy, axis=0))
                boxes[idx, :2] = xy_min
                boxes[idx, 2:] = xy_max
        else:
            raise TypeError(
                "`masks` must be `BitmapMasks`  or `PolygonMasks`, "
                f"but got {type(masks)}."
            )
        return HorizontalBoxes(boxes)


def bbox_overlaps(
    pred: torch.Tensor,
    target: torch.Tensor,
    iou_mode: str = "ciou",
    bbox_format: str = "xywh",
    siou_theta: float = 4.0,
    eps: float = 1e-7,
) -> torch.Tensor:
    assert iou_mode in ("iou", "ciou", "giou", "siou")
    assert bbox_format in ("xyxy", "xywh")
    if bbox_format == "xywh":
        pred = HorizontalBoxes.cxcywh_to_xyxy(pred)
        target = HorizontalBoxes.cxcywh_to_xyxy(target)

    bbox1_x1, bbox1_y1 = pred[..., 0], pred[..., 1]
    bbox1_x2, bbox1_y2 = pred[..., 2], pred[..., 3]
    bbox2_x1, bbox2_y1 = target[..., 0], target[..., 1]
    bbox2_x2, bbox2_y2 = target[..., 2], target[..., 3]
    overlap = (torch.min(bbox1_x2, bbox2_x2) - torch.max(bbox1_x1, bbox2_x1)).clamp(
        0
    ) * (torch.min(bbox1_y2, bbox2_y2) - torch.max(bbox1_y1, bbox2_y1)).clamp(0)
    w1, h1 = bbox1_x2 - bbox1_x1, bbox1_y2 - bbox1_y1
    w2, h2 = bbox2_x2 - bbox2_x1, bbox2_y2 - bbox2_y1
    union = (w1 * h1) + (w2 * h2) - overlap + eps

    h1 = bbox1_y2 - bbox1_y1 + eps
    h2 = bbox2_y2 - bbox2_y1 + eps
    ious = overlap / union
    enclose_x1y1 = torch.min(pred[..., :2], target[..., :2])
    enclose_x2y2 = torch.max(pred[..., 2:], target[..., 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)

    enclose_w = enclose_wh[..., 0]
    enclose_h = enclose_wh[..., 1]

    if iou_mode == "ciou":
        enclose_area = enclose_w**2 + enclose_h**2 + eps
        rho2_left_item = ((bbox2_x1 + bbox2_x2) - (bbox1_x1 + bbox1_x2)) ** 2 / 4
        rho2_right_item = ((bbox2_y1 + bbox2_y2) - (bbox1_y1 + bbox1_y2)) ** 2 / 4
        rho2 = rho2_left_item + rho2_right_item
        wh_ratio = (4 / (math.pi**2)) * torch.pow(
            torch.atan(w2 / h2) - torch.atan(w1 / h1), 2
        )

        with torch.no_grad():
            alpha = wh_ratio / (wh_ratio - ious + (1 + eps))
        ious = ious - ((rho2 / enclose_area) + (alpha * wh_ratio))

    elif iou_mode == "giou":
        convex_area = enclose_w * enclose_h + eps
        ious = ious - (convex_area - union) / convex_area

    elif iou_mode == "siou":
        sigma_cw = (bbox2_x1 + bbox2_x2) / 2 - (bbox1_x1 + bbox1_x2) / 2 + eps
        sigma_ch = (bbox2_y1 + bbox2_y2) / 2 - (bbox1_y1 + bbox1_y2) / 2 + eps
        sigma = torch.pow(sigma_cw**2 + sigma_ch**2, 0.5)
        sin_alpha = torch.abs(sigma_ch) / sigma
        sin_beta = torch.abs(sigma_cw) / sigma
        sin_alpha = torch.where(sin_alpha <= math.sin(math.pi / 4), sin_alpha, sin_beta)
        angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)

        rho_x = (sigma_cw / enclose_w) ** 2
        rho_y = (sigma_ch / enclose_h) ** 2
        gamma = 2 - angle_cost
        distance_cost = (1 - torch.exp(-1 * gamma * rho_x)) + (
            1 - torch.exp(-1 * gamma * rho_y)
        )
        omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
        omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
        shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), siou_theta) + torch.pow(
            1 - torch.exp(-1 * omiga_h), siou_theta
        )

        ious = ious - ((distance_cost + shape_cost) * 0.5)

    return ious.clamp(min=-1.0, max=1.0)


@MODELS.register_module()
class IoULoss(nn.Module):
    def __init__(
        self,
        iou_mode: str = "ciou",
        bbox_format: str = "xywh",
        eps: float = 1e-7,
        reduction: str = "mean",
        loss_weight: float = 1.0,
        return_iou: bool = True,
    ):
        super().__init__()
        assert bbox_format in ("xywh", "xyxy")
        assert iou_mode in ("ciou", "siou", "giou")
        self.iou_mode = iou_mode
        self.bbox_format = bbox_format
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.return_iou = return_iou

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        avg_factor: Optional[float] = None,
        reduction_override: Optional[Union[str, bool]] = None,
    ) -> Tuple[Union[torch.Tensor, torch.Tensor], torch.Tensor]:
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction

        if weight is not None and weight.dim() > 1:
            weight = weight.mean(-1)

        iou = bbox_overlaps(
            pred,
            target,
            iou_mode=self.iou_mode,
            bbox_format=self.bbox_format,
            eps=self.eps,
        )
        loss = self.loss_weight * weight_reduce_loss(
            1.0 - iou, weight, reduction, avg_factor
        )

        if self.return_iou:
            return loss, iou
        else:
            return loss


@TASK_UTILS.register_module()
class MlvlPointGenerator:
    def __init__(
        self, strides: Union[List[int], List[Tuple[int, int]]], offset: float = 0.5
    ) -> None:
        self.strides = [_pair(stride) for stride in strides]
        self.offset = offset

    @property
    def num_levels(self) -> int:
        return len(self.strides)

    @property
    def num_base_priors(self) -> List[int]:
        return [1 for _ in range(len(self.strides))]

    def _meshgrid(
        self, x: Tensor, y: Tensor, row_major: bool = True
    ) -> Tuple[Tensor, Tensor]:
        yy, xx = torch.meshgrid(y, x)
        if row_major:
            return xx.reshape(-1), yy.reshape(-1)

        else:
            return yy.reshape(-1), xx.reshape(-1)

    def grid_priors(
        self,
        featmap_sizes: List[Tuple],
        dtype: torch.dtype = torch.float32,
        device: DeviceType = "cuda",
        with_stride: bool = False,
    ) -> List[Tensor]:
        assert self.num_levels == len(featmap_sizes)
        multi_level_priors = []
        for i in range(self.num_levels):
            priors = self.single_level_grid_priors(
                featmap_sizes[i], level_idx=i, dtype=dtype, with_stride=with_stride
            )
            multi_level_priors.append(priors)
        return multi_level_priors

    def single_level_grid_priors(
        self,
        featmap_size: Tuple[int],
        level_idx: int,
        dtype: torch.dtype = torch.float32,
        device: DeviceType = "cuda",
        with_stride: bool = False,
    ) -> Tensor:
        feat_h, feat_w = featmap_size
        stride_w, stride_h = self.strides[level_idx]
        shift_x = (torch.arange(0, feat_w) + self.offset) * stride_w
        shift_x = shift_x.to(dtype)

        shift_y = (torch.arange(0, feat_h) + self.offset) * stride_h
        shift_y = shift_y.to(dtype)
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        if not with_stride:
            shifts = torch.stack([shift_xx, shift_yy], dim=-1)
        else:
            stride_w = shift_xx.new_full((shift_xx.shape[0],), stride_w).to(dtype)
            stride_h = shift_xx.new_full((shift_yy.shape[0],), stride_h).to(dtype)
            shifts = torch.stack([shift_xx, shift_yy, stride_w, stride_h], dim=-1)
        all_points = shifts
        return all_points

    def valid_flags(
        self,
        featmap_sizes: List[Tuple[int, int]],
        pad_shape: Tuple[int],
        device: DeviceType = "cuda",
    ) -> List[Tensor]:
        assert self.num_levels == len(featmap_sizes)
        multi_level_flags = []
        for i in range(self.num_levels):
            point_stride = self.strides[i]
            feat_h, feat_w = featmap_sizes[i]
            h, w = pad_shape[:2]
            valid_feat_h = min(int(np.ceil(h / point_stride[1])), feat_h)
            valid_feat_w = min(int(np.ceil(w / point_stride[0])), feat_w)
            flags = self.single_level_valid_flags(
                (feat_h, feat_w), (valid_feat_h, valid_feat_w), device=device
            )
            multi_level_flags.append(flags)
        return multi_level_flags

    def single_level_valid_flags(
        self,
        featmap_size: Tuple[int, int],
        valid_size: Tuple[int, int],
        device: DeviceType = "cuda",
    ) -> Tensor:
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = torch.zeros(feat_w, dtype=torch.bool)
        valid_y = torch.zeros(feat_h, dtype=torch.bool)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        return valid

    def sparse_priors(
        self,
        prior_idxs: Tensor,
        featmap_size: Tuple[int],
        level_idx: int,
        dtype: torch.dtype = torch.float32,
        device: DeviceType = "cuda",
    ) -> Tensor:
        height, width = featmap_size
        x = (prior_idxs % width + self.offset) * self.strides[level_idx][0]
        y = ((prior_idxs // width) % height + self.offset) * self.strides[level_idx][1]
        prioris = torch.stack([x, y], 1).to(dtype)
        return prioris


class BaseBBoxCoder(metaclass=ABCMeta):
    encode_size = 4

    def __init__(self, use_box_type: bool = False, **kwargs):
        self.use_box_type = use_box_type

    @abstractmethod
    def encode(self, bboxes, gt_bboxes):
        pass

    @abstractmethod
    def decode(self, bboxes, bboxes_pred):
        pass


def bbox2distance(
    points: Tensor, bbox: Tensor, max_dis: Optional[float] = None, eps: float = 0.1
) -> Tensor:
    left = points[..., 0] - bbox[..., 0]
    top = points[..., 1] - bbox[..., 1]
    right = bbox[..., 2] - points[..., 0]
    bottom = bbox[..., 3] - points[..., 1]
    if max_dis is not None:
        left = left.clamp(min=0, max=max_dis - eps)
        top = top.clamp(min=0, max=max_dis - eps)
        right = right.clamp(min=0, max=max_dis - eps)
        bottom = bottom.clamp(min=0, max=max_dis - eps)
    return torch.stack([left, top, right, bottom], -1)


def distance2bbox(
    points: Tensor,
    distance: Tensor,
    max_shape: Optional[Union[Sequence[int], Tensor, Sequence[Sequence[int]]]] = None,
) -> Tensor:
    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]

    bboxes = torch.stack([x1, y1, x2, y2], -1)

    if max_shape is not None:
        if bboxes.dim() == 2 and not torch.onnx.is_in_onnx_export():
            bboxes[:, 0::2].clamp_(min=0, max=max_shape[1])
            bboxes[:, 1::2].clamp_(min=0, max=max_shape[0])
            return bboxes
        if not isinstance(max_shape, torch.Tensor):
            max_shape = x1.new_tensor(max_shape)
        max_shape = max_shape[..., :2].type_as(x1)
        if max_shape.ndim == 2:
            assert bboxes.ndim == 3
            assert max_shape.size(0) == bboxes.size(0)

        min_xy = x1.new_tensor(0)
        max_xy = torch.cat([max_shape, max_shape], dim=-1).flip(-1).unsqueeze(-2)
        bboxes = torch.where(bboxes < min_xy, min_xy, bboxes)
        bboxes = torch.where(bboxes > max_xy, max_xy, bboxes)

    return bboxes


@TASK_UTILS.register_module()
class MMDET_DistancePointBBoxCoder(BaseBBoxCoder):
    def __init__(self, clip_border: Optional[bool] = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.clip_border = clip_border

    def encode(
        self,
        points: Tensor,
        gt_bboxes: Union[Tensor, BaseBoxes],
        max_dis: Optional[float] = None,
        eps: float = 0.1,
    ) -> Tensor:
        gt_bboxes = get_box_tensor(gt_bboxes)
        assert points.size(0) == gt_bboxes.size(0)
        assert points.size(-1) == 2
        assert gt_bboxes.size(-1) == 4
        return bbox2distance(points, gt_bboxes, max_dis, eps)

    def decode(
        self,
        points: Tensor,
        pred_bboxes: Tensor,
        max_shape: Optional[
            Union[Sequence[int], Tensor, Sequence[Sequence[int]]]
        ] = None,
    ) -> Union[Tensor, BaseBoxes]:
        assert points.size(0) == pred_bboxes.size(0)
        assert points.size(-1) == 2
        assert pred_bboxes.size(-1) == 4
        if self.clip_border is False:
            max_shape = None
        bboxes = distance2bbox(points, pred_bboxes, max_shape)

        if self.use_box_type:
            bboxes = HorizontalBoxes(bboxes)
        return bboxes


@TASK_UTILS.register_module()
class DistancePointBBoxCoder(MMDET_DistancePointBBoxCoder):
    def decode(
        self,
        points: torch.Tensor,
        pred_bboxes: torch.Tensor,
        stride: torch.Tensor,
        max_shape: Optional[
            Union[Sequence[int], torch.Tensor, Sequence[Sequence[int]]]
        ] = None,
    ) -> torch.Tensor:
        assert points.size(-2) == pred_bboxes.size(-2)
        assert points.size(-1) == 2
        assert pred_bboxes.size(-1) == 4
        if self.clip_border is False:
            max_shape = None

        pred_bboxes = pred_bboxes * stride[None, :, None]

        return distance2bbox(points, pred_bboxes, max_shape)

    def encode(
        self,
        points: torch.Tensor,
        gt_bboxes: torch.Tensor,
        max_dis: float = 16.0,
        eps: float = 0.01,
    ) -> torch.Tensor:
        assert points.size(-2) == gt_bboxes.size(-2)
        assert points.size(-1) == 2
        assert gt_bboxes.size(-1) == 4
        return bbox2distance(points, gt_bboxes, max_dis, eps)


def yolov6_iou_calculator(bbox1: Tensor, bbox2: Tensor, eps: float = 1e-9) -> Tensor:
    bbox1 = bbox1.unsqueeze(2)
    bbox2 = bbox2.unsqueeze(1)
    bbox1_x1y1, bbox1_x2y2 = bbox1[:, :, :, 0:2], bbox1[:, :, :, 2:4]
    bbox2_x1y1, bbox2_x2y2 = bbox2[:, :, :, 0:2], bbox2[:, :, :, 2:4]
    overlap = (
        (torch.minimum(bbox1_x2y2, bbox2_x2y2) - torch.maximum(bbox1_x1y1, bbox2_x1y1))
        .clip(0)
        .prod(-1)
    )
    bbox1_area = (bbox1_x2y2 - bbox1_x1y1).clip(0).prod(-1)
    bbox2_area = (bbox2_x2y2 - bbox2_x1y1).clip(0).prod(-1)

    union = bbox1_area + bbox2_area - overlap + eps

    return overlap / union


def select_highest_overlaps(
    pos_mask: Tensor, overlaps: Tensor, num_gt: int
) -> Tuple[Tensor, Tensor, Tensor]:
    fg_mask_pre_prior = pos_mask.sum(axis=-2)
    if fg_mask_pre_prior.max() > 1:
        mask_multi_gts = (fg_mask_pre_prior.unsqueeze(1) > 1).repeat([1, num_gt, 1])
        index = overlaps.argmax(axis=1)
        is_max_overlaps = F.one_hot(index, num_gt)
        is_max_overlaps = is_max_overlaps.permute(0, 2, 1).to(overlaps.dtype)

        pos_mask = torch.where(mask_multi_gts, is_max_overlaps, pos_mask)
        fg_mask_pre_prior = pos_mask.sum(axis=-2)

    gt_idx_pre_prior = pos_mask.argmax(axis=-2)
    return gt_idx_pre_prior, fg_mask_pre_prior, pos_mask


def select_candidates_in_gts(
    priors_points: Tensor, gt_bboxes: Tensor, eps: float = 1e-9
) -> Tensor:
    batch_size, num_gt, _ = gt_bboxes.size()
    gt_bboxes = gt_bboxes.reshape([-1, 4])

    priors_number = priors_points.size(0)
    priors_points = priors_points.unsqueeze(0).repeat(batch_size * num_gt, 1, 1)
    gt_bboxes_lt = gt_bboxes[:, 0:2].unsqueeze(1).repeat(1, priors_number, 1)
    gt_bboxes_rb = gt_bboxes[:, 2:4].unsqueeze(1).repeat(1, priors_number, 1)
    bbox_deltas = torch.cat(
        [priors_points - gt_bboxes_lt, gt_bboxes_rb - priors_points], dim=-1
    )
    bbox_deltas = bbox_deltas.reshape([batch_size, num_gt, priors_number, -1])

    return (bbox_deltas.min(axis=-1)[0] > eps).to(gt_bboxes.dtype)


@TASK_UTILS.register_module()
class BatchTaskAlignedAssigner(nn.Module):
    def __init__(
        self,
        num_classes: int,
        topk: int = 13,
        alpha: float = 1.0,
        beta: float = 6.0,
        eps: float = 1e-7,
        use_ciou: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.use_ciou = use_ciou

    @torch.no_grad()
    def forward(
        self,
        pred_bboxes: Tensor,
        pred_scores: Tensor,
        priors: Tensor,
        gt_labels: Tensor,
        gt_bboxes: Tensor,
        pad_bbox_flag: Tensor,
    ) -> dict:
        priors = priors[:, :2]

        batch_size = pred_scores.size(0)
        num_gt = gt_bboxes.size(1)

        assigned_result = {
            "assigned_labels": gt_bboxes.new_full(
                pred_scores[..., 0].shape, self.num_classes
            ),
            "assigned_bboxes": gt_bboxes.new_full(pred_bboxes.shape, 0),
            "assigned_scores": gt_bboxes.new_full(pred_scores.shape, 0),
            "fg_mask_pre_prior": gt_bboxes.new_full(pred_scores[..., 0].shape, 0),
        }

        if num_gt == 0:
            return assigned_result

        pos_mask, alignment_metrics, overlaps = self.get_pos_mask(
            pred_bboxes,
            pred_scores,
            priors,
            gt_labels,
            gt_bboxes,
            pad_bbox_flag,
            batch_size,
            num_gt,
        )

        (assigned_gt_idxs, fg_mask_pre_prior, pos_mask) = select_highest_overlaps(
            pos_mask, overlaps, num_gt
        )
        assigned_labels, assigned_bboxes, assigned_scores = self.get_targets(
            gt_labels,
            gt_bboxes,
            assigned_gt_idxs,
            fg_mask_pre_prior,
            batch_size,
            num_gt,
        )
        alignment_metrics *= pos_mask
        pos_align_metrics = alignment_metrics.max(axis=-1, keepdim=True)[0]
        pos_overlaps = (overlaps * pos_mask).max(axis=-1, keepdim=True)[0]
        norm_align_metric = (
            (alignment_metrics * pos_overlaps / (pos_align_metrics + self.eps))
            .max(-2)[0]
            .unsqueeze(-1)
        )
        assigned_scores = assigned_scores * norm_align_metric

        assigned_result["assigned_labels"] = assigned_labels
        assigned_result["assigned_bboxes"] = assigned_bboxes
        assigned_result["assigned_scores"] = assigned_scores
        assigned_result["fg_mask_pre_prior"] = fg_mask_pre_prior.bool()
        return assigned_result

    def get_pos_mask(
        self,
        pred_bboxes: Tensor,
        pred_scores: Tensor,
        priors: Tensor,
        gt_labels: Tensor,
        gt_bboxes: Tensor,
        pad_bbox_flag: Tensor,
        batch_size: int,
        num_gt: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        alignment_metrics, overlaps = self.get_box_metrics(
            pred_bboxes, pred_scores, gt_labels, gt_bboxes, batch_size, num_gt
        )
        is_in_gts = select_candidates_in_gts(priors, gt_bboxes)
        topk_metric = self.select_topk_candidates(
            alignment_metrics * is_in_gts,
            topk_mask=pad_bbox_flag.repeat([1, 1, self.topk]).bool(),
        )
        pos_mask = topk_metric * is_in_gts * pad_bbox_flag

        return pos_mask, alignment_metrics, overlaps

    def get_box_metrics(
        self,
        pred_bboxes: Tensor,
        pred_scores: Tensor,
        gt_labels: Tensor,
        gt_bboxes: Tensor,
        batch_size: int,
        num_gt: int,
    ) -> Tuple[Tensor, Tensor]:
        pred_scores = pred_scores.permute(0, 2, 1)
        gt_labels = gt_labels.to(torch.long)
        idx = torch.zeros([2, batch_size, num_gt], dtype=torch.long)
        idx[0] = torch.arange(end=batch_size).view(-1, 1).repeat(1, num_gt)
        idx[1] = gt_labels.squeeze(-1)
        bbox_scores = pred_scores[idx[0], idx[1]]
        if self.use_ciou:
            overlaps = bbox_overlaps(
                pred_bboxes.unsqueeze(1),
                gt_bboxes.unsqueeze(2),
                iou_mode="ciou",
                bbox_format="xyxy",
            ).clamp(0)
        else:
            overlaps = yolov6_iou_calculator(gt_bboxes, pred_bboxes)

        alignment_metrics = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)

        return alignment_metrics, overlaps

    def select_topk_candidates(
        self,
        alignment_gt_metrics: Tensor,
        using_largest_topk: bool = True,
        topk_mask: Optional[Tensor] = None,
    ) -> Tensor:
        num_priors = alignment_gt_metrics.shape[-1]
        topk_metrics, topk_idxs = torch.topk(
            alignment_gt_metrics, self.topk, axis=-1, largest=using_largest_topk
        )
        if topk_mask is None:
            topk_mask = (topk_metrics.max(axis=-1, keepdim=True) > self.eps).tile(
                [1, 1, self.topk]
            )
        topk_idxs = torch.where(topk_mask, topk_idxs, torch.zeros_like(topk_idxs))
        is_in_topk = F.one_hot(topk_idxs, num_priors).sum(axis=-2)
        is_in_topk = torch.where(
            is_in_topk > 1, torch.zeros_like(is_in_topk), is_in_topk
        )
        return is_in_topk.to(alignment_gt_metrics.dtype)

    def get_targets(
        self,
        gt_labels: Tensor,
        gt_bboxes: Tensor,
        assigned_gt_idxs: Tensor,
        fg_mask_pre_prior: Tensor,
        batch_size: int,
        num_gt: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        batch_ind = torch.arange(end=batch_size, dtype=torch.int64)[..., None]
        assigned_gt_idxs = assigned_gt_idxs + batch_ind * num_gt
        assigned_labels = gt_labels.long().flatten()[assigned_gt_idxs]
        assigned_bboxes = gt_bboxes.reshape([-1, 4])[assigned_gt_idxs]
        assigned_labels[assigned_labels < 0] = 0
        assigned_scores = F.one_hot(assigned_labels, self.num_classes)
        force_gt_scores_mask = fg_mask_pre_prior[:, :, None].repeat(
            1, 1, self.num_classes
        )
        assigned_scores = torch.where(
            force_gt_scores_mask > 0,
            assigned_scores,
            torch.full_like(assigned_scores, 0),
        )

        return assigned_labels, assigned_bboxes, assigned_scores


def weighted_loss(loss_func: Callable) -> Callable:
    @functools.wraps(loss_func)
    def wrapper(
        pred: Tensor,
        target: Tensor,
        weight: Optional[Tensor] = None,
        reduction: str = "mean",
        avg_factor: Optional[int] = None,
        **kwargs,
    ) -> Tensor:
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    return wrapper


@weighted_loss
def distribution_focal_loss(pred, label):
    dis_left = label.long()
    dis_right = dis_left + 1
    weight_left = dis_right.float() - label
    weight_right = label - dis_left.float()
    loss = (
        F.cross_entropy(pred, dis_left, reduction="none") * weight_left
        + F.cross_entropy(pred, dis_right, reduction="none") * weight_right
    )
    return loss


@MODELS.register_module()
class DistributionFocalLoss(nn.Module):
    def __init__(self, reduction="mean", loss_weight=1.0):
        super(DistributionFocalLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self, pred, target, weight=None, avg_factor=None, reduction_override=None
    ):
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        loss_cls = self.loss_weight * distribution_focal_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor
        )
        return loss_cls


TRANSFORMS = Registry("transform")


class Compose:
    def __init__(self, transforms: Optional[Sequence[Union[dict, Callable]]]):
        self.transforms: List[Callable] = []

        if transforms is None:
            transforms = []

        for transform in transforms:
            if isinstance(transform, dict):
                transform = TRANSFORMS.build(transform)
                if not callable(transform):
                    raise TypeError(
                        f"transform should be a callable object, "
                        f"but got {type(transform)}"
                    )
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError(
                    f"transform must be a callable object or dict, "
                    f"but got {type(transform)}"
                )

    def __call__(self, data: dict) -> Optional[dict]:
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


def force_full_init(old_func: Callable) -> Any:
    @functools.wraps(old_func)
    def wrapper(obj: object, *args, **kwargs):
        if not hasattr(obj, "full_init"):
            raise AttributeError(f"{type(obj)} does not have full_init " "method.")
        if not getattr(obj, "_fully_initialized", False):
            print_log(
                f"Attribute `_fully_initialized` is not defined in "
                f"{type(obj)} or `type(obj)._fully_initialized is "
                "False, `full_init` will be called and "
                f"{type(obj)}._fully_initialized will be set to True",
                logger="current",
                level=logging.WARNING,
            )
            obj.full_init()
            obj._fully_initialized = True

        return old_func(obj, *args, **kwargs)

    return wrapper


def list_from_file(
    filename,
    prefix="",
    offset=0,
    max_num=0,
    encoding="utf-8",
    file_client_args=None,
    backend_args=None,
):
    if file_client_args is not None:
        warnings.warn(
            '"file_client_args" will be deprecated in future. '
            'Please use "backend_args" instead',
            DeprecationWarning,
        )
        if backend_args is not None:
            raise ValueError(
                '"file_client_args" and "backend_args" cannot be set at the '
                "same time."
            )
    cnt = 0
    item_list = []

    if file_client_args is not None:
        file_client = FileClient.infer_client(file_client_args, filename)
        text = file_client.get_text(filename, encoding)
    else:
        text = get_text(filename, encoding, backend_args=backend_args)

    with StringIO(text) as f:
        for _ in range(offset):
            f.readline()
        for line in f:
            if 0 < max_num <= cnt:
                break
            item_list.append(prefix + line.rstrip("\n\r"))
            cnt += 1
    return item_list


def is_abs(path: str) -> bool:
    if osp.isabs(path) or path.startswith(("http://", "https://", "s3://")):
        return True
    else:
        return False


class BaseDataset(Dataset):
    METAINFO: dict = dict()
    _fully_initialized: bool = False

    def __init__(
        self,
        ann_file: str = "",
        metainfo: Optional[dict] = None,
        data_root: str = "",
        data_prefix: dict = dict(img_path=""),
        filter_cfg: Optional[dict] = None,
        indices: Optional[Union[int, Sequence[int]]] = None,
        serialize_data: bool = True,
        pipeline: List[Union[dict, Callable]] = [],
        test_mode: bool = False,
        lazy_init: bool = False,
        max_refetch: int = 1000,
    ):

        self.data_root = data_root
        self.data_prefix = copy.copy(data_prefix)
        self.ann_file = ann_file
        self.filter_cfg = copy.deepcopy(filter_cfg)
        self._indices = indices
        self.serialize_data = serialize_data
        self.test_mode = test_mode
        self.max_refetch = max_refetch
        self.data_list: List[dict] = []
        self.data_bytes: np.ndarray
        self._metainfo = self._load_metainfo(copy.deepcopy(metainfo))
        self._join_prefix()
        self.pipeline = Compose(pipeline)
        if not lazy_init:
            self.full_init()

    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        if self.serialize_data:
            start_addr = 0 if idx == 0 else self.data_address[idx - 1].item()
            end_addr = self.data_address[idx].item()
            bytes = memoryview(self.data_bytes[start_addr:end_addr])
            data_info = pickle.loads(bytes)
        else:
            data_info = copy.deepcopy(self.data_list[idx])
        if idx >= 0:
            data_info["sample_idx"] = idx
        else:
            data_info["sample_idx"] = len(self) + idx

        return data_info

    def full_init(self):
        if self._fully_initialized:
            return
        self.data_list = self.load_data_list()
        self.data_list = self.filter_data()
        if self._indices is not None:
            self.data_list = self._get_unserialized_subset(self._indices)
        if self.serialize_data:
            self.data_bytes, self.data_address = self._serialize_data()

        self._fully_initialized = True

    @property
    def metainfo(self) -> dict:
        return copy.deepcopy(self._metainfo)

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        for prefix_key, prefix in self.data_prefix.items():
            assert prefix_key in raw_data_info, (
                f"raw_data_info: {raw_data_info} dose not contain prefix key"
                f"{prefix_key}, please check your data_prefix."
            )
            raw_data_info[prefix_key] = osp.join(prefix, raw_data_info[prefix_key])
        return raw_data_info

    def filter_data(self) -> List[dict]:
        return self.data_list

    def get_cat_ids(self, idx: int) -> List[int]:
        raise NotImplementedError(
            f"{type(self)} must implement `get_cat_ids` " "method"
        )

    def __getitem__(self, idx: int) -> dict:
        if not self._fully_initialized:
            print_log(
                "Please call `full_init()` method manually to accelerate " "the speed.",
                logger="current",
                level=logging.WARNING,
            )
            self.full_init()

        if self.test_mode:
            data = self.prepare_data(idx)
            if data is None:
                raise Exception(
                    "Test time pipline should not get `None` " "data_sample"
                )
            return data

        for _ in range(self.max_refetch + 1):
            data = self.prepare_data(idx)
            if data is None:
                idx = self._rand_another()
                continue
            return data

        raise Exception(
            f"Cannot find valid image after {self.max_refetch}! "
            "Please check your image path and pipeline"
        )

    def load_data_list(self) -> List[dict]:
        annotations = load(self.ann_file)
        if not isinstance(annotations, dict):
            raise TypeError(
                f"The annotations loaded from annotation file "
                f"should be a dict, but got {type(annotations)}!"
            )
        if "data_list" not in annotations or "metainfo" not in annotations:
            raise ValueError("Annotation must have data_list and metainfo " "keys")
        metainfo = annotations["metainfo"]
        raw_data_list = annotations["data_list"]
        for k, v in metainfo.items():
            self._metainfo.setdefault(k, v)
        data_list = []
        for raw_data_info in raw_data_list:
            data_info = self.parse_data_info(raw_data_info)
            if isinstance(data_info, dict):
                data_list.append(data_info)
            elif isinstance(data_info, list):
                for item in data_info:
                    if not isinstance(item, dict):
                        raise TypeError(
                            "data_info must be list of dict, but " f"got {type(item)}"
                        )
                data_list.extend(data_info)
            else:
                raise TypeError(
                    "data_info should be a dict or list of dict, "
                    f"but got {type(data_info)}"
                )

        return data_list

    @classmethod
    def _load_metainfo(cls, metainfo: dict = None) -> dict:
        cls_metainfo = copy.deepcopy(cls.METAINFO)
        if metainfo is None:
            return cls_metainfo
        if not isinstance(metainfo, dict):
            raise TypeError(f"metainfo should be a dict, but got {type(metainfo)}")

        for k, v in metainfo.items():
            if isinstance(v, str):
                try:
                    cls_metainfo[k] = list_from_file(v)
                except (TypeError, FileNotFoundError):
                    print_log(
                        f"{v} is not a meta file, simply parsed as meta " "information",
                        logger="current",
                        level=logging.WARNING,
                    )
                    cls_metainfo[k] = v
            else:
                cls_metainfo[k] = v
        return cls_metainfo

    def _join_prefix(self):
        if not is_abs(self.ann_file) and self.ann_file:
            self.ann_file = osp.join(self.data_root, self.ann_file)
        for data_key, prefix in self.data_prefix.items():
            if isinstance(prefix, str):
                if not is_abs(prefix):
                    self.data_prefix[data_key] = osp.join(self.data_root, prefix)
                else:
                    self.data_prefix[data_key] = prefix
            else:
                raise TypeError("prefix should be a string, but got " f"{type(prefix)}")

    @force_full_init
    def get_subset_(self, indices: Union[Sequence[int], int]) -> None:
        if self.serialize_data:
            self.data_bytes, self.data_address = self._get_serialized_subset(indices)
        else:
            self.data_list = self._get_unserialized_subset(indices)

    @force_full_init
    def get_subset(self, indices: Union[Sequence[int], int]) -> "BaseDataset":
        sub_dataset = self._copy_without_annotation()
        if self.serialize_data:
            data_bytes, data_address = self._get_serialized_subset(indices)
            sub_dataset.data_bytes = data_bytes.copy()
            sub_dataset.data_address = data_address.copy()
        else:
            data_list = self._get_unserialized_subset(indices)
            sub_dataset.data_list = copy.deepcopy(data_list)
        return sub_dataset

    def _get_serialized_subset(
        self, indices: Union[Sequence[int], int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        sub_data_bytes: Union[List, np.ndarray]
        sub_data_address: Union[List, np.ndarray]
        if isinstance(indices, int):
            if indices >= 0:
                assert indices < len(
                    self.data_address
                ), f"{indices} is out of dataset length({len(self)}"
                end_addr = self.data_address[indices - 1].item() if indices > 0 else 0
                sub_data_bytes = self.data_bytes[:end_addr]
                sub_data_address = self.data_address[:indices]
            else:
                assert -indices <= len(
                    self.data_address
                ), f"{indices} is out of dataset length({len(self)}"
                ignored_bytes_size = self.data_address[indices - 1]
                start_addr = self.data_address[indices - 1].item()
                sub_data_bytes = self.data_bytes[start_addr:]
                sub_data_address = self.data_address[indices:]
                sub_data_address = sub_data_address - ignored_bytes_size
        elif isinstance(indices, Sequence):
            sub_data_bytes = []
            sub_data_address = []
            for idx in indices:
                assert len(self) > idx >= -len(self)
                start_addr = 0 if idx == 0 else self.data_address[idx - 1].item()
                end_addr = self.data_address[idx].item()
                sub_data_bytes.append(self.data_bytes[start_addr:end_addr])
                sub_data_address.append(end_addr - start_addr)
            if sub_data_bytes:
                sub_data_bytes = np.concatenate(sub_data_bytes)
                sub_data_address = np.cumsum(sub_data_address)
            else:
                sub_data_bytes = np.array([])
                sub_data_address = np.array([])
        else:
            raise TypeError(
                "indices should be a int or sequence of int, "
                f"but got {type(indices)}"
            )
        return sub_data_bytes, sub_data_address

    def _get_unserialized_subset(self, indices: Union[Sequence[int], int]) -> list:
        if isinstance(indices, int):
            if indices >= 0:
                sub_data_list = self.data_list[:indices]
            else:
                sub_data_list = self.data_list[indices:]
        elif isinstance(indices, Sequence):
            sub_data_list = []
            for idx in indices:
                sub_data_list.append(self.data_list[idx])
        else:
            raise TypeError(
                "indices should be a int or sequence of int, "
                f"but got {type(indices)}"
            )
        return sub_data_list

    def _serialize_data(self) -> Tuple[np.ndarray, np.ndarray]:
        def _serialize(data):
            buffer = pickle.dumps(data, protocol=4)
            return np.frombuffer(buffer, dtype=np.uint8)

        data_list = [_serialize(x) for x in self.data_list]
        address_list = np.asarray([len(x) for x in data_list], dtype=np.int64)
        data_address: np.ndarray = np.cumsum(address_list)
        data_bytes = np.concatenate(data_list)
        self.data_list.clear()
        gc.collect()
        return data_bytes, data_address

    def _rand_another(self) -> int:
        return np.random.randint(0, len(self))

    def prepare_data(self, idx) -> Any:
        data_info = self.get_data_info(idx)
        return self.pipeline(data_info)

    @force_full_init
    def __len__(self) -> int:
        if self.serialize_data:
            return len(self.data_address)
        else:
            return len(self.data_list)

    def _copy_without_annotation(self, memo=dict()) -> "BaseDataset":
        cls = self.__class__
        other = cls.__new__(cls)
        memo[id(self)] = other

        for key, value in self.__dict__.items():
            if key in ["data_list", "data_address", "data_bytes"]:
                continue
            super(BaseDataset, other).__setattr__(key, copy.deepcopy(value, memo))

        return other


@DATASETS.register_module()
class MultiModalDataset:
    def __init__(
        self,
        dataset: Union[BaseDataset, dict],
        class_text_path: str = None,
        test_mode: bool = True,
        pipeline: List[Union[dict, Callable]] = [],
        lazy_init: bool = False,
    ) -> None:
        self.dataset: BaseDataset
        if isinstance(dataset, dict):
            self.dataset = DATASETS.build(dataset)
        elif isinstance(dataset, BaseDataset):
            self.dataset = dataset
        else:
            raise TypeError(
                "dataset must be a dict or a BaseDataset, " f"but got {dataset}"
            )

        if class_text_path is not None:
            self.class_texts = CLASS_TEXTS
        else:
            self.class_texts = None

        self.test_mode = test_mode
        self._metainfo = self.dataset.metainfo
        self.pipeline = Compose(pipeline)

        self._fully_initialized = False
        if not lazy_init:
            self.full_init()

    @property
    def metainfo(self) -> dict:
        return copy.deepcopy(self._metainfo)

    def full_init(self) -> None:
        if self._fully_initialized:
            return

        self.dataset.full_init()
        self._ori_len = len(self.dataset)
        self._fully_initialized = True

    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        data_info = self.dataset.get_data_info(idx)
        if self.class_texts is not None:
            data_info.update({"texts": self.class_texts})
        return data_info

    def __getitem__(self, idx):
        if not self._fully_initialized:
            print_log(
                "Please call `full_init` method manually to " "accelerate the speed.",
                logger="current",
                level=logging.WARNING,
            )
            self.full_init()

        data_info = self.get_data_info(idx)

        if hasattr(self.dataset, "test_mode") and not self.dataset.test_mode:
            data_info["dataset"] = self
        elif not self.test_mode:
            data_info["dataset"] = self
        return self.pipeline(data_info)

    @force_full_init
    def __len__(self) -> int:
        return self._ori_len


@DATASETS.register_module()
class BaseDetDataset(BaseDataset):
    def __init__(
        self,
        *args,
        seg_map_suffix: str = ".png",
        proposal_file: Optional[str] = None,
        file_client_args: dict = None,
        backend_args: dict = None,
        **kwargs,
    ) -> None:
        self.seg_map_suffix = seg_map_suffix
        self.proposal_file = proposal_file
        self.backend_args = backend_args
        if file_client_args is not None:
            raise RuntimeError(
                "The `file_client_args` is deprecated, "
                "please use `backend_args` instead, please refer to"
                "https://github.com/open-mmlab/mmdetection/blob/main/configs/_base_/datasets/coco_detection.py"
            )
        super().__init__(*args, **kwargs)

    def full_init(self) -> None:
        if self._fully_initialized:
            return
        self.data_list = self.load_data_list()
        if self.proposal_file is not None:
            self.load_proposals()
        self.data_list = self.filter_data()
        if self._indices is not None:
            self.data_list = self._get_unserialized_subset(self._indices)
        if self.serialize_data:
            self.data_bytes, self.data_address = self._serialize_data()

        self._fully_initialized = True

    def load_proposals(self) -> None:
        if not is_abs(self.proposal_file):
            self.proposal_file = osp.join(self.data_root, self.proposal_file)
        proposals_list = load(self.proposal_file, backend_args=self.backend_args)
        assert len(self.data_list) == len(proposals_list)
        for data_info in self.data_list:
            img_path = data_info["img_path"]
            file_name = osp.join(
                osp.split(osp.split(img_path)[0])[-1], osp.split(img_path)[-1]
            )
            proposals = proposals_list[file_name]
            data_info["proposals"] = proposals

    def get_cat_ids(self, idx: int) -> List[int]:
        instances = self.get_data_info(idx)["instances"]
        return [instance["bbox_label"] for instance in instances]


class COCO(_COCO):
    def __init__(self, annotation_file=None):
        if getattr(pycocotools, "__version__", "0") >= "12.0.2":
            warnings.warn(
                'mmpycocotools is deprecated. Please install official pycocotools by "pip install pycocotools"',
                UserWarning,
            )
        super().__init__(annotation_file=annotation_file)
        self.img_ann_map = self.imgToAnns
        self.cat_img_map = self.catToImgs

    def get_ann_ids(self, img_ids=[], cat_ids=[], area_rng=[], iscrowd=None):
        return self.getAnnIds(img_ids, cat_ids, area_rng, iscrowd)

    def get_cat_ids(self, cat_names=[], sup_names=[], cat_ids=[]):
        return self.getCatIds(cat_names, sup_names, cat_ids)

    def get_img_ids(self, img_ids=[], cat_ids=[]):
        return self.getImgIds(img_ids, cat_ids)

    def load_anns(self, ids):
        return self.loadAnns(ids)

    def load_cats(self, ids):
        return self.loadCats(ids)

    def load_imgs(self, ids):
        return self.loadImgs(ids)


@contextmanager
def get_local_path(
    filepath: Union[str, Path],
    backend_args: Optional[dict] = None,
) -> Generator[Union[str, Path], None, None]:
    backend = get_file_backend(
        filepath, backend_args=backend_args, enable_singleton=True
    )
    with backend.get_local_path(str(filepath)) as local_path:
        yield local_path


@contextmanager
def get_local_path(
    filepath: Union[str, Path],
    backend_args: Optional[dict] = None,
) -> Generator[Union[str, Path], None, None]:
    backend = get_file_backend(
        filepath, backend_args=backend_args, enable_singleton=True
    )
    with backend.get_local_path(str(filepath)) as local_path:
        yield local_path


@DATASETS.register_module()
class CocoDataset(BaseDetDataset):

    METAINFO = {
        "classes": (
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
        ),
        "palette": [
            (220, 20, 60),
            (119, 11, 32),
            (0, 0, 142),
            (0, 0, 230),
            (106, 0, 228),
            (0, 60, 100),
            (0, 80, 100),
            (0, 0, 70),
            (0, 0, 192),
            (250, 170, 30),
            (100, 170, 30),
            (220, 220, 0),
            (175, 116, 175),
            (250, 0, 30),
            (165, 42, 42),
            (255, 77, 255),
            (0, 226, 252),
            (182, 182, 255),
            (0, 82, 0),
            (120, 166, 157),
            (110, 76, 0),
            (174, 57, 255),
            (199, 100, 0),
            (72, 0, 118),
            (255, 179, 240),
            (0, 125, 92),
            (209, 0, 151),
            (188, 208, 182),
            (0, 220, 176),
            (255, 99, 164),
            (92, 0, 73),
            (133, 129, 255),
            (78, 180, 255),
            (0, 228, 0),
            (174, 255, 243),
            (45, 89, 255),
            (134, 134, 103),
            (145, 148, 174),
            (255, 208, 186),
            (197, 226, 255),
            (171, 134, 1),
            (109, 63, 54),
            (207, 138, 255),
            (151, 0, 95),
            (9, 80, 61),
            (84, 105, 51),
            (74, 65, 105),
            (166, 196, 102),
            (208, 195, 210),
            (255, 109, 65),
            (0, 143, 149),
            (179, 0, 194),
            (209, 99, 106),
            (5, 121, 0),
            (227, 255, 205),
            (147, 186, 208),
            (153, 69, 1),
            (3, 95, 161),
            (163, 255, 0),
            (119, 0, 170),
            (0, 182, 199),
            (0, 165, 120),
            (183, 130, 88),
            (95, 32, 0),
            (130, 114, 135),
            (110, 129, 133),
            (166, 74, 118),
            (219, 142, 185),
            (79, 210, 114),
            (178, 90, 62),
            (65, 70, 15),
            (127, 167, 115),
            (59, 105, 106),
            (142, 108, 45),
            (196, 172, 0),
            (95, 54, 80),
            (128, 76, 255),
            (201, 57, 1),
            (246, 0, 122),
            (191, 162, 208),
        ],
    }
    COCOAPI = COCO
    ANN_ID_UNIQUE = True

    def load_data_list(self) -> List[dict]:
        with get_local_path(
            self.ann_file, backend_args=self.backend_args
        ) as local_path:
            self.coco = self.COCOAPI(local_path)
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.metainfo["classes"])
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)

        img_ids = self.coco.get_img_ids()
        data_list = []
        total_ann_ids = []
        for img_id in img_ids:
            raw_img_info = self.coco.load_imgs([img_id])[0]
            raw_img_info["img_id"] = img_id

            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            parsed_data_info = self.parse_data_info(
                {"raw_ann_info": raw_ann_info, "raw_img_info": raw_img_info}
            )
            data_list.append(parsed_data_info)
        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"

        del self.coco

        return data_list

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        img_info = raw_data_info["raw_img_info"]
        ann_info = raw_data_info["raw_ann_info"]

        data_info = {}
        img_path = osp.join(self.data_prefix["img"], img_info["file_name"])
        if self.data_prefix.get("seg", None):
            seg_map_path = osp.join(
                self.data_prefix["seg"],
                img_info["file_name"].rsplit(".", 1)[0] + self.seg_map_suffix,
            )
        else:
            seg_map_path = None
        data_info["img_path"] = img_path
        data_info["img_id"] = img_info["img_id"]
        data_info["seg_map_path"] = seg_map_path
        data_info["height"] = img_info["height"]
        data_info["width"] = img_info["width"]

        instances = []
        for i, ann in enumerate(ann_info):
            instance = {}

            if ann.get("ignore", False):
                continue
            x1, y1, w, h = ann["bbox"]
            inter_w = max(0, min(x1 + w, img_info["width"]) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info["height"]) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann["area"] <= 0 or w < 1 or h < 1:
                continue
            if ann["category_id"] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get("iscrowd", False):
                instance["ignore_flag"] = 1
            else:
                instance["ignore_flag"] = 0
            instance["bbox"] = bbox
            instance["bbox_label"] = self.cat2label[ann["category_id"]]

            if ann.get("segmentation", None):
                instance["mask"] = ann["segmentation"]

            instances.append(instance)
        data_info["instances"] = instances
        return data_info

    def filter_data(self) -> List[dict]:
        if self.test_mode:
            return self.data_list

        if self.filter_cfg is None:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get("filter_empty_gt", False)
        min_size = self.filter_cfg.get("min_size", 0)
        ids_with_ann = set(data_info["img_id"] for data_info in self.data_list)
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.cat_img_map[class_id])
        ids_in_cat &= ids_with_ann

        valid_data_infos = []
        for i, data_info in enumerate(self.data_list):
            img_id = data_info["img_id"]
            width = data_info["width"]
            height = data_info["height"]
            if filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(width, height) >= min_size:
                valid_data_infos.append(data_info)

        return valid_data_infos


@DATASETS.register_module()
class LVISV05Dataset(CocoDataset):
    METAINFO = {
        "classes": (
            "acorn",
            "aerosol_can",
            "air_conditioner",
            "airplane",
            "alarm_clock",
            "alcohol",
            "alligator",
            "almond",
            "ambulance",
            "amplifier",
            "anklet",
            "antenna",
            "apple",
            "apple_juice",
            "applesauce",
            "apricot",
            "apron",
            "aquarium",
            "armband",
            "armchair",
            "armoire",
            "armor",
            "artichoke",
            "trash_can",
            "ashtray",
            "asparagus",
            "atomizer",
            "avocado",
            "award",
            "awning",
            "ax",
            "baby_buggy",
            "basketball_backboard",
            "backpack",
            "handbag",
            "suitcase",
            "bagel",
            "bagpipe",
            "baguet",
            "bait",
            "ball",
            "ballet_skirt",
            "balloon",
            "bamboo",
            "banana",
            "Band_Aid",
            "bandage",
            "bandanna",
            "banjo",
            "banner",
            "barbell",
            "barge",
            "barrel",
            "barrette",
            "barrow",
            "baseball_base",
            "baseball",
            "baseball_bat",
            "baseball_cap",
            "baseball_glove",
            "basket",
            "basketball_hoop",
            "basketball",
            "bass_horn",
            "bat_(animal)",
            "bath_mat",
            "bath_towel",
            "bathrobe",
            "bathtub",
            "batter_(food)",
            "battery",
            "beachball",
            "bead",
            "beaker",
            "bean_curd",
            "beanbag",
            "beanie",
            "bear",
            "bed",
            "bedspread",
            "cow",
            "beef_(food)",
            "beeper",
            "beer_bottle",
            "beer_can",
            "beetle",
            "bell",
            "bell_pepper",
            "belt",
            "belt_buckle",
            "bench",
            "beret",
            "bib",
            "Bible",
            "bicycle",
            "visor",
            "binder",
            "binoculars",
            "bird",
            "birdfeeder",
            "birdbath",
            "birdcage",
            "birdhouse",
            "birthday_cake",
            "birthday_card",
            "biscuit_(bread)",
            "pirate_flag",
            "black_sheep",
            "blackboard",
            "blanket",
            "blazer",
            "blender",
            "blimp",
            "blinker",
            "blueberry",
            "boar",
            "gameboard",
            "boat",
            "bobbin",
            "bobby_pin",
            "boiled_egg",
            "bolo_tie",
            "deadbolt",
            "bolt",
            "bonnet",
            "book",
            "book_bag",
            "bookcase",
            "booklet",
            "bookmark",
            "boom_microphone",
            "boot",
            "bottle",
            "bottle_opener",
            "bouquet",
            "bow_(weapon)",
            "bow_(decorative_ribbons)",
            "bow-tie",
            "bowl",
            "pipe_bowl",
            "bowler_hat",
            "bowling_ball",
            "bowling_pin",
            "boxing_glove",
            "suspenders",
            "bracelet",
            "brass_plaque",
            "brassiere",
            "bread-bin",
            "breechcloth",
            "bridal_gown",
            "briefcase",
            "bristle_brush",
            "broccoli",
            "broach",
            "broom",
            "brownie",
            "brussels_sprouts",
            "bubble_gum",
            "bucket",
            "horse_buggy",
            "bull",
            "bulldog",
            "bulldozer",
            "bullet_train",
            "bulletin_board",
            "bulletproof_vest",
            "bullhorn",
            "corned_beef",
            "bun",
            "bunk_bed",
            "buoy",
            "burrito",
            "bus_(vehicle)",
            "business_card",
            "butcher_knife",
            "butter",
            "butterfly",
            "button",
            "cab_(taxi)",
            "cabana",
            "cabin_car",
            "cabinet",
            "locker",
            "cake",
            "calculator",
            "calendar",
            "calf",
            "camcorder",
            "camel",
            "camera",
            "camera_lens",
            "camper_(vehicle)",
            "can",
            "can_opener",
            "candelabrum",
            "candle",
            "candle_holder",
            "candy_bar",
            "candy_cane",
            "walking_cane",
            "canister",
            "cannon",
            "canoe",
            "cantaloup",
            "canteen",
            "cap_(headwear)",
            "bottle_cap",
            "cape",
            "cappuccino",
            "car_(automobile)",
            "railcar_(part_of_a_train)",
            "elevator_car",
            "car_battery",
            "identity_card",
            "card",
            "cardigan",
            "cargo_ship",
            "carnation",
            "horse_carriage",
            "carrot",
            "tote_bag",
            "cart",
            "carton",
            "cash_register",
            "casserole",
            "cassette",
            "cast",
            "cat",
            "cauliflower",
            "caviar",
            "cayenne_(spice)",
            "CD_player",
            "celery",
            "cellular_telephone",
            "chain_mail",
            "chair",
            "chaise_longue",
            "champagne",
            "chandelier",
            "chap",
            "checkbook",
            "checkerboard",
            "cherry",
            "chessboard",
            "chest_of_drawers_(furniture)",
            "chicken_(animal)",
            "chicken_wire",
            "chickpea",
            "Chihuahua",
            "chili_(vegetable)",
            "chime",
            "chinaware",
            "crisp_(potato_chip)",
            "poker_chip",
            "chocolate_bar",
            "chocolate_cake",
            "chocolate_milk",
            "chocolate_mousse",
            "choker",
            "chopping_board",
            "chopstick",
            "Christmas_tree",
            "slide",
            "cider",
            "cigar_box",
            "cigarette",
            "cigarette_case",
            "cistern",
            "clarinet",
            "clasp",
            "cleansing_agent",
            "clementine",
            "clip",
            "clipboard",
            "clock",
            "clock_tower",
            "clothes_hamper",
            "clothespin",
            "clutch_bag",
            "coaster",
            "coat",
            "coat_hanger",
            "coatrack",
            "cock",
            "coconut",
            "coffee_filter",
            "coffee_maker",
            "coffee_table",
            "coffeepot",
            "coil",
            "coin",
            "colander",
            "coleslaw",
            "coloring_material",
            "combination_lock",
            "pacifier",
            "comic_book",
            "computer_keyboard",
            "concrete_mixer",
            "cone",
            "control",
            "convertible_(automobile)",
            "sofa_bed",
            "cookie",
            "cookie_jar",
            "cooking_utensil",
            "cooler_(for_food)",
            "cork_(bottle_plug)",
            "corkboard",
            "corkscrew",
            "edible_corn",
            "cornbread",
            "cornet",
            "cornice",
            "cornmeal",
            "corset",
            "romaine_lettuce",
            "costume",
            "cougar",
            "coverall",
            "cowbell",
            "cowboy_hat",
            "crab_(animal)",
            "cracker",
            "crape",
            "crate",
            "crayon",
            "cream_pitcher",
            "credit_card",
            "crescent_roll",
            "crib",
            "crock_pot",
            "crossbar",
            "crouton",
            "crow",
            "crown",
            "crucifix",
            "cruise_ship",
            "police_cruiser",
            "crumb",
            "crutch",
            "cub_(animal)",
            "cube",
            "cucumber",
            "cufflink",
            "cup",
            "trophy_cup",
            "cupcake",
            "hair_curler",
            "curling_iron",
            "curtain",
            "cushion",
            "custard",
            "cutting_tool",
            "cylinder",
            "cymbal",
            "dachshund",
            "dagger",
            "dartboard",
            "date_(fruit)",
            "deck_chair",
            "deer",
            "dental_floss",
            "desk",
            "detergent",
            "diaper",
            "diary",
            "die",
            "dinghy",
            "dining_table",
            "tux",
            "dish",
            "dish_antenna",
            "dishrag",
            "dishtowel",
            "dishwasher",
            "dishwasher_detergent",
            "diskette",
            "dispenser",
            "Dixie_cup",
            "dog",
            "dog_collar",
            "doll",
            "dollar",
            "dolphin",
            "domestic_ass",
            "eye_mask",
            "doorbell",
            "doorknob",
            "doormat",
            "doughnut",
            "dove",
            "dragonfly",
            "drawer",
            "underdrawers",
            "dress",
            "dress_hat",
            "dress_suit",
            "dresser",
            "drill",
            "drinking_fountain",
            "drone",
            "dropper",
            "drum_(musical_instrument)",
            "drumstick",
            "duck",
            "duckling",
            "duct_tape",
            "duffel_bag",
            "dumbbell",
            "dumpster",
            "dustpan",
            "Dutch_oven",
            "eagle",
            "earphone",
            "earplug",
            "earring",
            "easel",
            "eclair",
            "eel",
            "egg",
            "egg_roll",
            "egg_yolk",
            "eggbeater",
            "eggplant",
            "electric_chair",
            "refrigerator",
            "elephant",
            "elk",
            "envelope",
            "eraser",
            "escargot",
            "eyepatch",
            "falcon",
            "fan",
            "faucet",
            "fedora",
            "ferret",
            "Ferris_wheel",
            "ferry",
            "fig_(fruit)",
            "fighter_jet",
            "figurine",
            "file_cabinet",
            "file_(tool)",
            "fire_alarm",
            "fire_engine",
            "fire_extinguisher",
            "fire_hose",
            "fireplace",
            "fireplug",
            "fish",
            "fish_(food)",
            "fishbowl",
            "fishing_boat",
            "fishing_rod",
            "flag",
            "flagpole",
            "flamingo",
            "flannel",
            "flash",
            "flashlight",
            "fleece",
            "flip-flop_(sandal)",
            "flipper_(footwear)",
            "flower_arrangement",
            "flute_glass",
            "foal",
            "folding_chair",
            "food_processor",
            "football_(American)",
            "football_helmet",
            "footstool",
            "fork",
            "forklift",
            "freight_car",
            "French_toast",
            "freshener",
            "frisbee",
            "frog",
            "fruit_juice",
            "fruit_salad",
            "frying_pan",
            "fudge",
            "funnel",
            "futon",
            "gag",
            "garbage",
            "garbage_truck",
            "garden_hose",
            "gargle",
            "gargoyle",
            "garlic",
            "gasmask",
            "gazelle",
            "gelatin",
            "gemstone",
            "giant_panda",
            "gift_wrap",
            "ginger",
            "giraffe",
            "cincture",
            "glass_(drink_container)",
            "globe",
            "glove",
            "goat",
            "goggles",
            "goldfish",
            "golf_club",
            "golfcart",
            "gondola_(boat)",
            "goose",
            "gorilla",
            "gourd",
            "surgical_gown",
            "grape",
            "grasshopper",
            "grater",
            "gravestone",
            "gravy_boat",
            "green_bean",
            "green_onion",
            "griddle",
            "grillroom",
            "grinder_(tool)",
            "grits",
            "grizzly",
            "grocery_bag",
            "guacamole",
            "guitar",
            "gull",
            "gun",
            "hair_spray",
            "hairbrush",
            "hairnet",
            "hairpin",
            "ham",
            "hamburger",
            "hammer",
            "hammock",
            "hamper",
            "hamster",
            "hair_dryer",
            "hand_glass",
            "hand_towel",
            "handcart",
            "handcuff",
            "handkerchief",
            "handle",
            "handsaw",
            "hardback_book",
            "harmonium",
            "hat",
            "hatbox",
            "hatch",
            "veil",
            "headband",
            "headboard",
            "headlight",
            "headscarf",
            "headset",
            "headstall_(for_horses)",
            "hearing_aid",
            "heart",
            "heater",
            "helicopter",
            "helmet",
            "heron",
            "highchair",
            "hinge",
            "hippopotamus",
            "hockey_stick",
            "hog",
            "home_plate_(baseball)",
            "honey",
            "fume_hood",
            "hook",
            "horse",
            "hose",
            "hot-air_balloon",
            "hotplate",
            "hot_sauce",
            "hourglass",
            "houseboat",
            "hummingbird",
            "hummus",
            "polar_bear",
            "icecream",
            "popsicle",
            "ice_maker",
            "ice_pack",
            "ice_skate",
            "ice_tea",
            "igniter",
            "incense",
            "inhaler",
            "iPod",
            "iron_(for_clothing)",
            "ironing_board",
            "jacket",
            "jam",
            "jean",
            "jeep",
            "jelly_bean",
            "jersey",
            "jet_plane",
            "jewelry",
            "joystick",
            "jumpsuit",
            "kayak",
            "keg",
            "kennel",
            "kettle",
            "key",
            "keycard",
            "kilt",
            "kimono",
            "kitchen_sink",
            "kitchen_table",
            "kite",
            "kitten",
            "kiwi_fruit",
            "knee_pad",
            "knife",
            "knight_(chess_piece)",
            "knitting_needle",
            "knob",
            "knocker_(on_a_door)",
            "koala",
            "lab_coat",
            "ladder",
            "ladle",
            "ladybug",
            "lamb_(animal)",
            "lamb-chop",
            "lamp",
            "lamppost",
            "lampshade",
            "lantern",
            "lanyard",
            "laptop_computer",
            "lasagna",
            "latch",
            "lawn_mower",
            "leather",
            "legging_(clothing)",
            "Lego",
            "lemon",
            "lemonade",
            "lettuce",
            "license_plate",
            "life_buoy",
            "life_jacket",
            "lightbulb",
            "lightning_rod",
            "lime",
            "limousine",
            "linen_paper",
            "lion",
            "lip_balm",
            "lipstick",
            "liquor",
            "lizard",
            "Loafer_(type_of_shoe)",
            "log",
            "lollipop",
            "lotion",
            "speaker_(stereo_equipment)",
            "loveseat",
            "machine_gun",
            "magazine",
            "magnet",
            "mail_slot",
            "mailbox_(at_home)",
            "mallet",
            "mammoth",
            "mandarin_orange",
            "manger",
            "manhole",
            "map",
            "marker",
            "martini",
            "mascot",
            "mashed_potato",
            "masher",
            "mask",
            "mast",
            "mat_(gym_equipment)",
            "matchbox",
            "mattress",
            "measuring_cup",
            "measuring_stick",
            "meatball",
            "medicine",
            "melon",
            "microphone",
            "microscope",
            "microwave_oven",
            "milestone",
            "milk",
            "minivan",
            "mint_candy",
            "mirror",
            "mitten",
            "mixer_(kitchen_tool)",
            "money",
            "monitor_(computer_equipment) computer_monitor",
            "monkey",
            "motor",
            "motor_scooter",
            "motor_vehicle",
            "motorboat",
            "motorcycle",
            "mound_(baseball)",
            "mouse_(animal_rodent)",
            "mouse_(computer_equipment)",
            "mousepad",
            "muffin",
            "mug",
            "mushroom",
            "music_stool",
            "musical_instrument",
            "nailfile",
            "nameplate",
            "napkin",
            "neckerchief",
            "necklace",
            "necktie",
            "needle",
            "nest",
            "newsstand",
            "nightshirt",
            "nosebag_(for_animals)",
            "noseband_(for_animals)",
            "notebook",
            "notepad",
            "nut",
            "nutcracker",
            "oar",
            "octopus_(food)",
            "octopus_(animal)",
            "oil_lamp",
            "olive_oil",
            "omelet",
            "onion",
            "orange_(fruit)",
            "orange_juice",
            "oregano",
            "ostrich",
            "ottoman",
            "overalls_(clothing)",
            "owl",
            "packet",
            "inkpad",
            "pad",
            "paddle",
            "padlock",
            "paintbox",
            "paintbrush",
            "painting",
            "pajamas",
            "palette",
            "pan_(for_cooking)",
            "pan_(metal_container)",
            "pancake",
            "pantyhose",
            "papaya",
            "paperclip",
            "paper_plate",
            "paper_towel",
            "paperback_book",
            "paperweight",
            "parachute",
            "parakeet",
            "parasail_(sports)",
            "parchment",
            "parka",
            "parking_meter",
            "parrot",
            "passenger_car_(part_of_a_train)",
            "passenger_ship",
            "passport",
            "pastry",
            "patty_(food)",
            "pea_(food)",
            "peach",
            "peanut_butter",
            "pear",
            "peeler_(tool_for_fruit_and_vegetables)",
            "pegboard",
            "pelican",
            "pen",
            "pencil",
            "pencil_box",
            "pencil_sharpener",
            "pendulum",
            "penguin",
            "pennant",
            "penny_(coin)",
            "pepper",
            "pepper_mill",
            "perfume",
            "persimmon",
            "baby",
            "pet",
            "petfood",
            "pew_(church_bench)",
            "phonebook",
            "phonograph_record",
            "piano",
            "pickle",
            "pickup_truck",
            "pie",
            "pigeon",
            "piggy_bank",
            "pillow",
            "pin_(non_jewelry)",
            "pineapple",
            "pinecone",
            "ping-pong_ball",
            "pinwheel",
            "tobacco_pipe",
            "pipe",
            "pistol",
            "pita_(bread)",
            "pitcher_(vessel_for_liquid)",
            "pitchfork",
            "pizza",
            "place_mat",
            "plate",
            "platter",
            "playing_card",
            "playpen",
            "pliers",
            "plow_(farm_equipment)",
            "pocket_watch",
            "pocketknife",
            "poker_(fire_stirring_tool)",
            "pole",
            "police_van",
            "polo_shirt",
            "poncho",
            "pony",
            "pool_table",
            "pop_(soda)",
            "portrait",
            "postbox_(public)",
            "postcard",
            "poster",
            "pot",
            "flowerpot",
            "potato",
            "potholder",
            "pottery",
            "pouch",
            "power_shovel",
            "prawn",
            "printer",
            "projectile_(weapon)",
            "projector",
            "propeller",
            "prune",
            "pudding",
            "puffer_(fish)",
            "puffin",
            "pug-dog",
            "pumpkin",
            "puncher",
            "puppet",
            "puppy",
            "quesadilla",
            "quiche",
            "quilt",
            "rabbit",
            "race_car",
            "racket",
            "radar",
            "radiator",
            "radio_receiver",
            "radish",
            "raft",
            "rag_doll",
            "raincoat",
            "ram_(animal)",
            "raspberry",
            "rat",
            "razorblade",
            "reamer_(juicer)",
            "rearview_mirror",
            "receipt",
            "recliner",
            "record_player",
            "red_cabbage",
            "reflector",
            "remote_control",
            "rhinoceros",
            "rib_(food)",
            "rifle",
            "ring",
            "river_boat",
            "road_map",
            "robe",
            "rocking_chair",
            "roller_skate",
            "Rollerblade",
            "rolling_pin",
            "root_beer",
            "router_(computer_equipment)",
            "rubber_band",
            "runner_(carpet)",
            "plastic_bag",
            "saddle_(on_an_animal)",
            "saddle_blanket",
            "saddlebag",
            "safety_pin",
            "sail",
            "salad",
            "salad_plate",
            "salami",
            "salmon_(fish)",
            "salmon_(food)",
            "salsa",
            "saltshaker",
            "sandal_(type_of_shoe)",
            "sandwich",
            "satchel",
            "saucepan",
            "saucer",
            "sausage",
            "sawhorse",
            "saxophone",
            "scale_(measuring_instrument)",
            "scarecrow",
            "scarf",
            "school_bus",
            "scissors",
            "scoreboard",
            "scrambled_eggs",
            "scraper",
            "scratcher",
            "screwdriver",
            "scrubbing_brush",
            "sculpture",
            "seabird",
            "seahorse",
            "seaplane",
            "seashell",
            "seedling",
            "serving_dish",
            "sewing_machine",
            "shaker",
            "shampoo",
            "shark",
            "sharpener",
            "Sharpie",
            "shaver_(electric)",
            "shaving_cream",
            "shawl",
            "shears",
            "sheep",
            "shepherd_dog",
            "sherbert",
            "shield",
            "shirt",
            "shoe",
            "shopping_bag",
            "shopping_cart",
            "short_pants",
            "shot_glass",
            "shoulder_bag",
            "shovel",
            "shower_head",
            "shower_curtain",
            "shredder_(for_paper)",
            "sieve",
            "signboard",
            "silo",
            "sink",
            "skateboard",
            "skewer",
            "ski",
            "ski_boot",
            "ski_parka",
            "ski_pole",
            "skirt",
            "sled",
            "sleeping_bag",
            "sling_(bandage)",
            "slipper_(footwear)",
            "smoothie",
            "snake",
            "snowboard",
            "snowman",
            "snowmobile",
            "soap",
            "soccer_ball",
            "sock",
            "soda_fountain",
            "carbonated_water",
            "sofa",
            "softball",
            "solar_array",
            "sombrero",
            "soup",
            "soup_bowl",
            "soupspoon",
            "sour_cream",
            "soya_milk",
            "space_shuttle",
            "sparkler_(fireworks)",
            "spatula",
            "spear",
            "spectacles",
            "spice_rack",
            "spider",
            "sponge",
            "spoon",
            "sportswear",
            "spotlight",
            "squirrel",
            "stapler_(stapling_machine)",
            "starfish",
            "statue_(sculpture)",
            "steak_(food)",
            "steak_knife",
            "steamer_(kitchen_appliance)",
            "steering_wheel",
            "stencil",
            "stepladder",
            "step_stool",
            "stereo_(sound_system)",
            "stew",
            "stirrer",
            "stirrup",
            "stockings_(leg_wear)",
            "stool",
            "stop_sign",
            "brake_light",
            "stove",
            "strainer",
            "strap",
            "straw_(for_drinking)",
            "strawberry",
            "street_sign",
            "streetlight",
            "string_cheese",
            "stylus",
            "subwoofer",
            "sugar_bowl",
            "sugarcane_(plant)",
            "suit_(clothing)",
            "sunflower",
            "sunglasses",
            "sunhat",
            "sunscreen",
            "surfboard",
            "sushi",
            "mop",
            "sweat_pants",
            "sweatband",
            "sweater",
            "sweatshirt",
            "sweet_potato",
            "swimsuit",
            "sword",
            "syringe",
            "Tabasco_sauce",
            "table-tennis_table",
            "table",
            "table_lamp",
            "tablecloth",
            "tachometer",
            "taco",
            "tag",
            "taillight",
            "tambourine",
            "army_tank",
            "tank_(storage_vessel)",
            "tank_top_(clothing)",
            "tape_(sticky_cloth_or_paper)",
            "tape_measure",
            "tapestry",
            "tarp",
            "tartan",
            "tassel",
            "tea_bag",
            "teacup",
            "teakettle",
            "teapot",
            "teddy_bear",
            "telephone",
            "telephone_booth",
            "telephone_pole",
            "telephoto_lens",
            "television_camera",
            "television_set",
            "tennis_ball",
            "tennis_racket",
            "tequila",
            "thermometer",
            "thermos_bottle",
            "thermostat",
            "thimble",
            "thread",
            "thumbtack",
            "tiara",
            "tiger",
            "tights_(clothing)",
            "timer",
            "tinfoil",
            "tinsel",
            "tissue_paper",
            "toast_(food)",
            "toaster",
            "toaster_oven",
            "toilet",
            "toilet_tissue",
            "tomato",
            "tongs",
            "toolbox",
            "toothbrush",
            "toothpaste",
            "toothpick",
            "cover",
            "tortilla",
            "tow_truck",
            "towel",
            "towel_rack",
            "toy",
            "tractor_(farm_equipment)",
            "traffic_light",
            "dirt_bike",
            "trailer_truck",
            "train_(railroad_vehicle)",
            "trampoline",
            "tray",
            "tree_house",
            "trench_coat",
            "triangle_(musical_instrument)",
            "tricycle",
            "tripod",
            "trousers",
            "truck",
            "truffle_(chocolate)",
            "trunk",
            "vat",
            "turban",
            "turkey_(bird)",
            "turkey_(food)",
            "turnip",
            "turtle",
            "turtleneck_(clothing)",
            "typewriter",
            "umbrella",
            "underwear",
            "unicycle",
            "urinal",
            "urn",
            "vacuum_cleaner",
            "valve",
            "vase",
            "vending_machine",
            "vent",
            "videotape",
            "vinegar",
            "violin",
            "vodka",
            "volleyball",
            "vulture",
            "waffle",
            "waffle_iron",
            "wagon",
            "wagon_wheel",
            "walking_stick",
            "wall_clock",
            "wall_socket",
            "wallet",
            "walrus",
            "wardrobe",
            "wasabi",
            "automatic_washer",
            "watch",
            "water_bottle",
            "water_cooler",
            "water_faucet",
            "water_filter",
            "water_heater",
            "water_jug",
            "water_gun",
            "water_scooter",
            "water_ski",
            "water_tower",
            "watering_can",
            "watermelon",
            "weathervane",
            "webcam",
            "wedding_cake",
            "wedding_ring",
            "wet_suit",
            "wheel",
            "wheelchair",
            "whipped_cream",
            "whiskey",
            "whistle",
            "wick",
            "wig",
            "wind_chime",
            "windmill",
            "window_box_(for_plants)",
            "windshield_wiper",
            "windsock",
            "wine_bottle",
            "wine_bucket",
            "wineglass",
            "wing_chair",
            "blinder_(for_horses)",
            "wok",
            "wolf",
            "wooden_spoon",
            "wreath",
            "wrench",
            "wristband",
            "wristlet",
            "yacht",
            "yak",
            "yogurt",
            "yoke_(animal_equipment)",
            "zebra",
            "zucchini",
        ),
        "palette": None,
    }

    def load_data_list(self) -> List[dict]:
        try:
            import lvis

            if getattr(lvis, "__version__", "0") >= "10.5.3":
                warnings.warn(
                    'mmlvis is deprecated, please install official lvis-api by "pip install git+https://github.com/lvis-dataset/lvis-api.git"',
                    UserWarning,
                )
            from lvis import LVIS
        except ImportError:
            raise ImportError(
                'Package lvis is not installed. Please run "pip install git+https://github.com/lvis-dataset/lvis-api.git".'
            )
        with get_local_path(
            self.ann_file, backend_args=self.backend_args
        ) as local_path:
            self.lvis = LVIS(local_path)
        self.cat_ids = self.lvis.get_cat_ids()
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(self.lvis.cat_img_map)

        img_ids = self.lvis.get_img_ids()
        data_list = []
        total_ann_ids = []
        for img_id in img_ids:
            raw_img_info = self.lvis.load_imgs([img_id])[0]
            raw_img_info["img_id"] = img_id
            if raw_img_info["file_name"].startswith("COCO"):
                raw_img_info["file_name"] = raw_img_info["file_name"][-16:]
            ann_ids = self.lvis.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.lvis.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            parsed_data_info = self.parse_data_info(
                {"raw_ann_info": raw_ann_info, "raw_img_info": raw_img_info}
            )
            data_list.append(parsed_data_info)
        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"

        del self.lvis

        return data_list


LVISDataset = LVISV05Dataset
DATASETS.register_module(name="LVISDataset", module=LVISDataset)


@DATASETS.register_module()
class LVISV1Dataset(LVISDataset):
    METAINFO = {
        "classes": (
            "aerosol_can",
            "air_conditioner",
            "airplane",
            "alarm_clock",
            "alcohol",
            "alligator",
            "almond",
            "ambulance",
            "amplifier",
            "anklet",
            "antenna",
            "apple",
            "applesauce",
            "apricot",
            "apron",
            "aquarium",
            "arctic_(type_of_shoe)",
            "armband",
            "armchair",
            "armoire",
            "armor",
            "artichoke",
            "trash_can",
            "ashtray",
            "asparagus",
            "atomizer",
            "avocado",
            "award",
            "awning",
            "ax",
            "baboon",
            "baby_buggy",
            "basketball_backboard",
            "backpack",
            "handbag",
            "suitcase",
            "bagel",
            "bagpipe",
            "baguet",
            "bait",
            "ball",
            "ballet_skirt",
            "balloon",
            "bamboo",
            "banana",
            "Band_Aid",
            "bandage",
            "bandanna",
            "banjo",
            "banner",
            "barbell",
            "barge",
            "barrel",
            "barrette",
            "barrow",
            "baseball_base",
            "baseball",
            "baseball_bat",
            "baseball_cap",
            "baseball_glove",
            "basket",
            "basketball",
            "bass_horn",
            "bat_(animal)",
            "bath_mat",
            "bath_towel",
            "bathrobe",
            "bathtub",
            "batter_(food)",
            "battery",
            "beachball",
            "bead",
            "bean_curd",
            "beanbag",
            "beanie",
            "bear",
            "bed",
            "bedpan",
            "bedspread",
            "cow",
            "beef_(food)",
            "beeper",
            "beer_bottle",
            "beer_can",
            "beetle",
            "bell",
            "bell_pepper",
            "belt",
            "belt_buckle",
            "bench",
            "beret",
            "bib",
            "Bible",
            "bicycle",
            "visor",
            "billboard",
            "binder",
            "binoculars",
            "bird",
            "birdfeeder",
            "birdbath",
            "birdcage",
            "birdhouse",
            "birthday_cake",
            "birthday_card",
            "pirate_flag",
            "black_sheep",
            "blackberry",
            "blackboard",
            "blanket",
            "blazer",
            "blender",
            "blimp",
            "blinker",
            "blouse",
            "blueberry",
            "gameboard",
            "boat",
            "bob",
            "bobbin",
            "bobby_pin",
            "boiled_egg",
            "bolo_tie",
            "deadbolt",
            "bolt",
            "bonnet",
            "book",
            "bookcase",
            "booklet",
            "bookmark",
            "boom_microphone",
            "boot",
            "bottle",
            "bottle_opener",
            "bouquet",
            "bow_(weapon)",
            "bow_(decorative_ribbons)",
            "bow-tie",
            "bowl",
            "pipe_bowl",
            "bowler_hat",
            "bowling_ball",
            "box",
            "boxing_glove",
            "suspenders",
            "bracelet",
            "brass_plaque",
            "brassiere",
            "bread-bin",
            "bread",
            "breechcloth",
            "bridal_gown",
            "briefcase",
            "broccoli",
            "broach",
            "broom",
            "brownie",
            "brussels_sprouts",
            "bubble_gum",
            "bucket",
            "horse_buggy",
            "bull",
            "bulldog",
            "bulldozer",
            "bullet_train",
            "bulletin_board",
            "bulletproof_vest",
            "bullhorn",
            "bun",
            "bunk_bed",
            "buoy",
            "burrito",
            "bus_(vehicle)",
            "business_card",
            "butter",
            "butterfly",
            "button",
            "cab_(taxi)",
            "cabana",
            "cabin_car",
            "cabinet",
            "locker",
            "cake",
            "calculator",
            "calendar",
            "calf",
            "camcorder",
            "camel",
            "camera",
            "camera_lens",
            "camper_(vehicle)",
            "can",
            "can_opener",
            "candle",
            "candle_holder",
            "candy_bar",
            "candy_cane",
            "walking_cane",
            "canister",
            "canoe",
            "cantaloup",
            "canteen",
            "cap_(headwear)",
            "bottle_cap",
            "cape",
            "cappuccino",
            "car_(automobile)",
            "railcar_(part_of_a_train)",
            "elevator_car",
            "car_battery",
            "identity_card",
            "card",
            "cardigan",
            "cargo_ship",
            "carnation",
            "horse_carriage",
            "carrot",
            "tote_bag",
            "cart",
            "carton",
            "cash_register",
            "casserole",
            "cassette",
            "cast",
            "cat",
            "cauliflower",
            "cayenne_(spice)",
            "CD_player",
            "celery",
            "cellular_telephone",
            "chain_mail",
            "chair",
            "chaise_longue",
            "chalice",
            "chandelier",
            "chap",
            "checkbook",
            "checkerboard",
            "cherry",
            "chessboard",
            "chicken_(animal)",
            "chickpea",
            "chili_(vegetable)",
            "chime",
            "chinaware",
            "crisp_(potato_chip)",
            "poker_chip",
            "chocolate_bar",
            "chocolate_cake",
            "chocolate_milk",
            "chocolate_mousse",
            "choker",
            "chopping_board",
            "chopstick",
            "Christmas_tree",
            "slide",
            "cider",
            "cigar_box",
            "cigarette",
            "cigarette_case",
            "cistern",
            "clarinet",
            "clasp",
            "cleansing_agent",
            "cleat_(for_securing_rope)",
            "clementine",
            "clip",
            "clipboard",
            "clippers_(for_plants)",
            "cloak",
            "clock",
            "clock_tower",
            "clothes_hamper",
            "clothespin",
            "clutch_bag",
            "coaster",
            "coat",
            "coat_hanger",
            "coatrack",
            "cock",
            "cockroach",
            "cocoa_(beverage)",
            "coconut",
            "coffee_maker",
            "coffee_table",
            "coffeepot",
            "coil",
            "coin",
            "colander",
            "coleslaw",
            "coloring_material",
            "combination_lock",
            "pacifier",
            "comic_book",
            "compass",
            "computer_keyboard",
            "condiment",
            "cone",
            "control",
            "convertible_(automobile)",
            "sofa_bed",
            "cooker",
            "cookie",
            "cooking_utensil",
            "cooler_(for_food)",
            "cork_(bottle_plug)",
            "corkboard",
            "corkscrew",
            "edible_corn",
            "cornbread",
            "cornet",
            "cornice",
            "cornmeal",
            "corset",
            "costume",
            "cougar",
            "coverall",
            "cowbell",
            "cowboy_hat",
            "crab_(animal)",
            "crabmeat",
            "cracker",
            "crape",
            "crate",
            "crayon",
            "cream_pitcher",
            "crescent_roll",
            "crib",
            "crock_pot",
            "crossbar",
            "crouton",
            "crow",
            "crowbar",
            "crown",
            "crucifix",
            "cruise_ship",
            "police_cruiser",
            "crumb",
            "crutch",
            "cub_(animal)",
            "cube",
            "cucumber",
            "cufflink",
            "cup",
            "trophy_cup",
            "cupboard",
            "cupcake",
            "hair_curler",
            "curling_iron",
            "curtain",
            "cushion",
            "cylinder",
            "cymbal",
            "dagger",
            "dalmatian",
            "dartboard",
            "date_(fruit)",
            "deck_chair",
            "deer",
            "dental_floss",
            "desk",
            "detergent",
            "diaper",
            "diary",
            "die",
            "dinghy",
            "dining_table",
            "tux",
            "dish",
            "dish_antenna",
            "dishrag",
            "dishtowel",
            "dishwasher",
            "dishwasher_detergent",
            "dispenser",
            "diving_board",
            "Dixie_cup",
            "dog",
            "dog_collar",
            "doll",
            "dollar",
            "dollhouse",
            "dolphin",
            "domestic_ass",
            "doorknob",
            "doormat",
            "doughnut",
            "dove",
            "dragonfly",
            "drawer",
            "underdrawers",
            "dress",
            "dress_hat",
            "dress_suit",
            "dresser",
            "drill",
            "drone",
            "dropper",
            "drum_(musical_instrument)",
            "drumstick",
            "duck",
            "duckling",
            "duct_tape",
            "duffel_bag",
            "dumbbell",
            "dumpster",
            "dustpan",
            "eagle",
            "earphone",
            "earplug",
            "earring",
            "easel",
            "eclair",
            "eel",
            "egg",
            "egg_roll",
            "egg_yolk",
            "eggbeater",
            "eggplant",
            "electric_chair",
            "refrigerator",
            "elephant",
            "elk",
            "envelope",
            "eraser",
            "escargot",
            "eyepatch",
            "falcon",
            "fan",
            "faucet",
            "fedora",
            "ferret",
            "Ferris_wheel",
            "ferry",
            "fig_(fruit)",
            "fighter_jet",
            "figurine",
            "file_cabinet",
            "file_(tool)",
            "fire_alarm",
            "fire_engine",
            "fire_extinguisher",
            "fire_hose",
            "fireplace",
            "fireplug",
            "first-aid_kit",
            "fish",
            "fish_(food)",
            "fishbowl",
            "fishing_rod",
            "flag",
            "flagpole",
            "flamingo",
            "flannel",
            "flap",
            "flash",
            "flashlight",
            "fleece",
            "flip-flop_(sandal)",
            "flipper_(footwear)",
            "flower_arrangement",
            "flute_glass",
            "foal",
            "folding_chair",
            "food_processor",
            "football_(American)",
            "football_helmet",
            "footstool",
            "fork",
            "forklift",
            "freight_car",
            "French_toast",
            "freshener",
            "frisbee",
            "frog",
            "fruit_juice",
            "frying_pan",
            "fudge",
            "funnel",
            "futon",
            "gag",
            "garbage",
            "garbage_truck",
            "garden_hose",
            "gargle",
            "gargoyle",
            "garlic",
            "gasmask",
            "gazelle",
            "gelatin",
            "gemstone",
            "generator",
            "giant_panda",
            "gift_wrap",
            "ginger",
            "giraffe",
            "cincture",
            "glass_(drink_container)",
            "globe",
            "glove",
            "goat",
            "goggles",
            "goldfish",
            "golf_club",
            "golfcart",
            "gondola_(boat)",
            "goose",
            "gorilla",
            "gourd",
            "grape",
            "grater",
            "gravestone",
            "gravy_boat",
            "green_bean",
            "green_onion",
            "griddle",
            "grill",
            "grits",
            "grizzly",
            "grocery_bag",
            "guitar",
            "gull",
            "gun",
            "hairbrush",
            "hairnet",
            "hairpin",
            "halter_top",
            "ham",
            "hamburger",
            "hammer",
            "hammock",
            "hamper",
            "hamster",
            "hair_dryer",
            "hand_glass",
            "hand_towel",
            "handcart",
            "handcuff",
            "handkerchief",
            "handle",
            "handsaw",
            "hardback_book",
            "harmonium",
            "hat",
            "hatbox",
            "veil",
            "headband",
            "headboard",
            "headlight",
            "headscarf",
            "headset",
            "headstall_(for_horses)",
            "heart",
            "heater",
            "helicopter",
            "helmet",
            "heron",
            "highchair",
            "hinge",
            "hippopotamus",
            "hockey_stick",
            "hog",
            "home_plate_(baseball)",
            "honey",
            "fume_hood",
            "hook",
            "hookah",
            "hornet",
            "horse",
            "hose",
            "hot-air_balloon",
            "hotplate",
            "hot_sauce",
            "hourglass",
            "houseboat",
            "hummingbird",
            "hummus",
            "polar_bear",
            "icecream",
            "popsicle",
            "ice_maker",
            "ice_pack",
            "ice_skate",
            "igniter",
            "inhaler",
            "iPod",
            "iron_(for_clothing)",
            "ironing_board",
            "jacket",
            "jam",
            "jar",
            "jean",
            "jeep",
            "jelly_bean",
            "jersey",
            "jet_plane",
            "jewel",
            "jewelry",
            "joystick",
            "jumpsuit",
            "kayak",
            "keg",
            "kennel",
            "kettle",
            "key",
            "keycard",
            "kilt",
            "kimono",
            "kitchen_sink",
            "kitchen_table",
            "kite",
            "kitten",
            "kiwi_fruit",
            "knee_pad",
            "knife",
            "knitting_needle",
            "knob",
            "knocker_(on_a_door)",
            "koala",
            "lab_coat",
            "ladder",
            "ladle",
            "ladybug",
            "lamb_(animal)",
            "lamb-chop",
            "lamp",
            "lamppost",
            "lampshade",
            "lantern",
            "lanyard",
            "laptop_computer",
            "lasagna",
            "latch",
            "lawn_mower",
            "leather",
            "legging_(clothing)",
            "Lego",
            "legume",
            "lemon",
            "lemonade",
            "lettuce",
            "license_plate",
            "life_buoy",
            "life_jacket",
            "lightbulb",
            "lightning_rod",
            "lime",
            "limousine",
            "lion",
            "lip_balm",
            "liquor",
            "lizard",
            "log",
            "lollipop",
            "speaker_(stereo_equipment)",
            "loveseat",
            "machine_gun",
            "magazine",
            "magnet",
            "mail_slot",
            "mailbox_(at_home)",
            "mallard",
            "mallet",
            "mammoth",
            "manatee",
            "mandarin_orange",
            "manger",
            "manhole",
            "map",
            "marker",
            "martini",
            "mascot",
            "mashed_potato",
            "masher",
            "mask",
            "mast",
            "mat_(gym_equipment)",
            "matchbox",
            "mattress",
            "measuring_cup",
            "measuring_stick",
            "meatball",
            "medicine",
            "melon",
            "microphone",
            "microscope",
            "microwave_oven",
            "milestone",
            "milk",
            "milk_can",
            "milkshake",
            "minivan",
            "mint_candy",
            "mirror",
            "mitten",
            "mixer_(kitchen_tool)",
            "money",
            "monitor_(computer_equipment) computer_monitor",
            "monkey",
            "motor",
            "motor_scooter",
            "motor_vehicle",
            "motorcycle",
            "mound_(baseball)",
            "mouse_(computer_equipment)",
            "mousepad",
            "muffin",
            "mug",
            "mushroom",
            "music_stool",
            "musical_instrument",
            "nailfile",
            "napkin",
            "neckerchief",
            "necklace",
            "necktie",
            "needle",
            "nest",
            "newspaper",
            "newsstand",
            "nightshirt",
            "nosebag_(for_animals)",
            "noseband_(for_animals)",
            "notebook",
            "notepad",
            "nut",
            "nutcracker",
            "oar",
            "octopus_(food)",
            "octopus_(animal)",
            "oil_lamp",
            "olive_oil",
            "omelet",
            "onion",
            "orange_(fruit)",
            "orange_juice",
            "ostrich",
            "ottoman",
            "oven",
            "overalls_(clothing)",
            "owl",
            "packet",
            "inkpad",
            "pad",
            "paddle",
            "padlock",
            "paintbrush",
            "painting",
            "pajamas",
            "palette",
            "pan_(for_cooking)",
            "pan_(metal_container)",
            "pancake",
            "pantyhose",
            "papaya",
            "paper_plate",
            "paper_towel",
            "paperback_book",
            "paperweight",
            "parachute",
            "parakeet",
            "parasail_(sports)",
            "parasol",
            "parchment",
            "parka",
            "parking_meter",
            "parrot",
            "passenger_car_(part_of_a_train)",
            "passenger_ship",
            "passport",
            "pastry",
            "patty_(food)",
            "pea_(food)",
            "peach",
            "peanut_butter",
            "pear",
            "peeler_(tool_for_fruit_and_vegetables)",
            "wooden_leg",
            "pegboard",
            "pelican",
            "pen",
            "pencil",
            "pencil_box",
            "pencil_sharpener",
            "pendulum",
            "penguin",
            "pennant",
            "penny_(coin)",
            "pepper",
            "pepper_mill",
            "perfume",
            "persimmon",
            "person",
            "pet",
            "pew_(church_bench)",
            "phonebook",
            "phonograph_record",
            "piano",
            "pickle",
            "pickup_truck",
            "pie",
            "pigeon",
            "piggy_bank",
            "pillow",
            "pin_(non_jewelry)",
            "pineapple",
            "pinecone",
            "ping-pong_ball",
            "pinwheel",
            "tobacco_pipe",
            "pipe",
            "pistol",
            "pita_(bread)",
            "pitcher_(vessel_for_liquid)",
            "pitchfork",
            "pizza",
            "place_mat",
            "plate",
            "platter",
            "playpen",
            "pliers",
            "plow_(farm_equipment)",
            "plume",
            "pocket_watch",
            "pocketknife",
            "poker_(fire_stirring_tool)",
            "pole",
            "polo_shirt",
            "poncho",
            "pony",
            "pool_table",
            "pop_(soda)",
            "postbox_(public)",
            "postcard",
            "poster",
            "pot",
            "flowerpot",
            "potato",
            "potholder",
            "pottery",
            "pouch",
            "power_shovel",
            "prawn",
            "pretzel",
            "printer",
            "projectile_(weapon)",
            "projector",
            "propeller",
            "prune",
            "pudding",
            "puffer_(fish)",
            "puffin",
            "pug-dog",
            "pumpkin",
            "puncher",
            "puppet",
            "puppy",
            "quesadilla",
            "quiche",
            "quilt",
            "rabbit",
            "race_car",
            "racket",
            "radar",
            "radiator",
            "radio_receiver",
            "radish",
            "raft",
            "rag_doll",
            "raincoat",
            "ram_(animal)",
            "raspberry",
            "rat",
            "razorblade",
            "reamer_(juicer)",
            "rearview_mirror",
            "receipt",
            "recliner",
            "record_player",
            "reflector",
            "remote_control",
            "rhinoceros",
            "rib_(food)",
            "rifle",
            "ring",
            "river_boat",
            "road_map",
            "robe",
            "rocking_chair",
            "rodent",
            "roller_skate",
            "Rollerblade",
            "rolling_pin",
            "root_beer",
            "router_(computer_equipment)",
            "rubber_band",
            "runner_(carpet)",
            "plastic_bag",
            "saddle_(on_an_animal)",
            "saddle_blanket",
            "saddlebag",
            "safety_pin",
            "sail",
            "salad",
            "salad_plate",
            "salami",
            "salmon_(fish)",
            "salmon_(food)",
            "salsa",
            "saltshaker",
            "sandal_(type_of_shoe)",
            "sandwich",
            "satchel",
            "saucepan",
            "saucer",
            "sausage",
            "sawhorse",
            "saxophone",
            "scale_(measuring_instrument)",
            "scarecrow",
            "scarf",
            "school_bus",
            "scissors",
            "scoreboard",
            "scraper",
            "screwdriver",
            "scrubbing_brush",
            "sculpture",
            "seabird",
            "seahorse",
            "seaplane",
            "seashell",
            "sewing_machine",
            "shaker",
            "shampoo",
            "shark",
            "sharpener",
            "Sharpie",
            "shaver_(electric)",
            "shaving_cream",
            "shawl",
            "shears",
            "sheep",
            "shepherd_dog",
            "sherbert",
            "shield",
            "shirt",
            "shoe",
            "shopping_bag",
            "shopping_cart",
            "short_pants",
            "shot_glass",
            "shoulder_bag",
            "shovel",
            "shower_head",
            "shower_cap",
            "shower_curtain",
            "shredder_(for_paper)",
            "signboard",
            "silo",
            "sink",
            "skateboard",
            "skewer",
            "ski",
            "ski_boot",
            "ski_parka",
            "ski_pole",
            "skirt",
            "skullcap",
            "sled",
            "sleeping_bag",
            "sling_(bandage)",
            "slipper_(footwear)",
            "smoothie",
            "snake",
            "snowboard",
            "snowman",
            "snowmobile",
            "soap",
            "soccer_ball",
            "sock",
            "sofa",
            "softball",
            "solar_array",
            "sombrero",
            "soup",
            "soup_bowl",
            "soupspoon",
            "sour_cream",
            "soya_milk",
            "space_shuttle",
            "sparkler_(fireworks)",
            "spatula",
            "spear",
            "spectacles",
            "spice_rack",
            "spider",
            "crawfish",
            "sponge",
            "spoon",
            "sportswear",
            "spotlight",
            "squid_(food)",
            "squirrel",
            "stagecoach",
            "stapler_(stapling_machine)",
            "starfish",
            "statue_(sculpture)",
            "steak_(food)",
            "steak_knife",
            "steering_wheel",
            "stepladder",
            "step_stool",
            "stereo_(sound_system)",
            "stew",
            "stirrer",
            "stirrup",
            "stool",
            "stop_sign",
            "brake_light",
            "stove",
            "strainer",
            "strap",
            "straw_(for_drinking)",
            "strawberry",
            "street_sign",
            "streetlight",
            "string_cheese",
            "stylus",
            "subwoofer",
            "sugar_bowl",
            "sugarcane_(plant)",
            "suit_(clothing)",
            "sunflower",
            "sunglasses",
            "sunhat",
            "surfboard",
            "sushi",
            "mop",
            "sweat_pants",
            "sweatband",
            "sweater",
            "sweatshirt",
            "sweet_potato",
            "swimsuit",
            "sword",
            "syringe",
            "Tabasco_sauce",
            "table-tennis_table",
            "table",
            "table_lamp",
            "tablecloth",
            "tachometer",
            "taco",
            "tag",
            "taillight",
            "tambourine",
            "army_tank",
            "tank_(storage_vessel)",
            "tank_top_(clothing)",
            "tape_(sticky_cloth_or_paper)",
            "tape_measure",
            "tapestry",
            "tarp",
            "tartan",
            "tassel",
            "tea_bag",
            "teacup",
            "teakettle",
            "teapot",
            "teddy_bear",
            "telephone",
            "telephone_booth",
            "telephone_pole",
            "telephoto_lens",
            "television_camera",
            "television_set",
            "tennis_ball",
            "tennis_racket",
            "tequila",
            "thermometer",
            "thermos_bottle",
            "thermostat",
            "thimble",
            "thread",
            "thumbtack",
            "tiara",
            "tiger",
            "tights_(clothing)",
            "timer",
            "tinfoil",
            "tinsel",
            "tissue_paper",
            "toast_(food)",
            "toaster",
            "toaster_oven",
            "toilet",
            "toilet_tissue",
            "tomato",
            "tongs",
            "toolbox",
            "toothbrush",
            "toothpaste",
            "toothpick",
            "cover",
            "tortilla",
            "tow_truck",
            "towel",
            "towel_rack",
            "toy",
            "tractor_(farm_equipment)",
            "traffic_light",
            "dirt_bike",
            "trailer_truck",
            "train_(railroad_vehicle)",
            "trampoline",
            "tray",
            "trench_coat",
            "triangle_(musical_instrument)",
            "tricycle",
            "tripod",
            "trousers",
            "truck",
            "truffle_(chocolate)",
            "trunk",
            "vat",
            "turban",
            "turkey_(food)",
            "turnip",
            "turtle",
            "turtleneck_(clothing)",
            "typewriter",
            "umbrella",
            "underwear",
            "unicycle",
            "urinal",
            "urn",
            "vacuum_cleaner",
            "vase",
            "vending_machine",
            "vent",
            "vest",
            "videotape",
            "vinegar",
            "violin",
            "vodka",
            "volleyball",
            "vulture",
            "waffle",
            "waffle_iron",
            "wagon",
            "wagon_wheel",
            "walking_stick",
            "wall_clock",
            "wall_socket",
            "wallet",
            "walrus",
            "wardrobe",
            "washbasin",
            "automatic_washer",
            "watch",
            "water_bottle",
            "water_cooler",
            "water_faucet",
            "water_heater",
            "water_jug",
            "water_gun",
            "water_scooter",
            "water_ski",
            "water_tower",
            "watering_can",
            "watermelon",
            "weathervane",
            "webcam",
            "wedding_cake",
            "wedding_ring",
            "wet_suit",
            "wheel",
            "wheelchair",
            "whipped_cream",
            "whistle",
            "wig",
            "wind_chime",
            "windmill",
            "window_box_(for_plants)",
            "windshield_wiper",
            "windsock",
            "wine_bottle",
            "wine_bucket",
            "wineglass",
            "blinder_(for_horses)",
            "wok",
            "wolf",
            "wooden_spoon",
            "wreath",
            "wrench",
            "wristband",
            "wristlet",
            "yacht",
            "yogurt",
            "yoke_(animal_equipment)",
            "zebra",
            "zucchini",
        ),
        "palette": None,
    }

    def load_data_list(self) -> List[dict]:
        try:
            import lvis

            if getattr(lvis, "__version__", "0") >= "10.5.3":
                warnings.warn(
                    'mmlvis is deprecated, please install official lvis-api by "pip install git+https://github.com/lvis-dataset/lvis-api.git"',
                    UserWarning,
                )
            from lvis import LVIS
        except ImportError:
            raise ImportError(
                'Package lvis is not installed. Please run "pip install git+https://github.com/lvis-dataset/lvis-api.git".'
            )
        with get_local_path(
            self.ann_file, backend_args=self.backend_args
        ) as local_path:
            self.lvis = LVIS(local_path)
        self.cat_ids = self.lvis.get_cat_ids()
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(self.lvis.cat_img_map)

        img_ids = self.lvis.get_img_ids()
        data_list = []
        total_ann_ids = []
        for img_id in img_ids:
            raw_img_info = self.lvis.load_imgs([img_id])[0]
            raw_img_info["img_id"] = img_id
            raw_img_info["file_name"] = raw_img_info["coco_url"].replace(
                "http://images.cocodataset.org/", ""
            )
            ann_ids = self.lvis.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.lvis.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)
            parsed_data_info = self.parse_data_info(
                {"raw_ann_info": raw_ann_info, "raw_img_info": raw_img_info}
            )
            data_list.append(parsed_data_info)
        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"

        del self.lvis

        return data_list


class BatchShapePolicyDataset(BaseDetDataset):
    def __init__(self, *args, batch_shapes_cfg: Optional[dict] = None, **kwargs):
        self.batch_shapes_cfg = batch_shapes_cfg
        super().__init__(*args, **kwargs)

    def full_init(self):
        if self._fully_initialized:
            return
        self.data_list = self.load_data_list()
        if self.batch_shapes_cfg:
            batch_shapes_policy = TASK_UTILS.build(self.batch_shapes_cfg)
            self.data_list = batch_shapes_policy(self.data_list)
            del batch_shapes_policy
        self.data_list = self.filter_data()
        if self._indices is not None:
            self.data_list = self._get_unserialized_subset(self._indices)

        if self.serialize_data:
            self.data_bytes, self.data_address = self._serialize_data()

        self._fully_initialized = True

    def prepare_data(self, idx: int) -> Any:
        if self.test_mode is False:
            data_info = self.get_data_info(idx)
            data_info["dataset"] = self
            return self.pipeline(data_info)
        else:
            return super().prepare_data(idx)


@DATASETS.register_module()
class YOLOv5LVISV1Dataset(BatchShapePolicyDataset, LVISV1Dataset):
    pass


class BaseTransform(metaclass=ABCMeta):
    def __call__(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:

        return self.transform(results)

    @abstractmethod
    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        pass


def get(
    filepath: Union[str, Path],
    backend_args: Optional[dict] = None,
) -> bytes:
    backend = get_file_backend(
        filepath, backend_args=backend_args, enable_singleton=True
    )
    return backend.get(filepath)


def _pillow2array(img, flag: str = "color", channel_order: str = "bgr") -> np.ndarray:
    channel_order = channel_order.lower()
    if channel_order not in ["rgb", "bgr"]:
        raise ValueError('channel order must be either "rgb" or "bgr"')

    if flag == "unchanged":
        array = np.array(img)
        if array.ndim >= 3 and array.shape[2] >= 3:
            array[:, :, :3] = array[:, :, (2, 1, 0)]
    else:
        if flag in ["color", "grayscale"]:
            img = ImageOps.exif_transpose(img)
        if img.mode != "RGB":
            if img.mode != "LA":
                img = img.convert("RGB")
            else:
                img_rgba = img.convert("RGBA")
                img = Image.new("RGB", img_rgba.size, (124, 117, 104))
                img.paste(img_rgba, mask=img_rgba.split()[3])
        if flag in ["color", "color_ignore_orientation"]:
            array = np.array(img)
            if channel_order != "rgb":
                array = array[:, :, ::-1]
        elif flag in ["grayscale", "grayscale_ignore_orientation"]:
            img = img.convert("L")
            array = np.array(img)
        else:
            raise ValueError(
                'flag must be "color", "grayscale", "unchanged", '
                f'"color_ignore_orientation" or "grayscale_ignore_orientation"'
                f" but got {flag}"
            )
    return array


def imfrombytes(
    content: bytes,
    flag: str = "color",
    channel_order: str = "bgr",
    backend: Optional[str] = None,
) -> np.ndarray:
    if backend is None:
        backend = imread_backend
    if backend not in supported_backends:
        raise ValueError(
            f"backend: {backend} is not supported. Supported "
            "backends are 'cv2', 'turbojpeg', 'pillow', 'tifffile'"
        )
    if backend == "turbojpeg":
        img = jpeg.decode(content, _jpegflag(flag, channel_order))
        if img.shape[-1] == 1:
            img = img[:, :, 0]
        return img
    elif backend == "pillow":
        with io.BytesIO(content) as buff:
            img = Image.open(buff)
            img = _pillow2array(img, flag, channel_order)
        return img
    elif backend == "tifffile":
        with io.BytesIO(content) as buff:
            img = tifffile.imread(buff)
        return img
    else:
        img_np = np.frombuffer(content, np.uint8)
        flag = imread_flags[flag] if is_str(flag) else flag
        img = cv2.imdecode(img_np, flag)
        if flag == IMREAD_COLOR and channel_order == "rgb":
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        return img


@TRANSFORMS.register_module()
class LoadImageFromFile(BaseTransform):
    def __init__(
        self,
        to_float32: bool = False,
        color_type: str = "color",
        imdecode_backend: str = "cv2",
        file_client_args: Optional[dict] = None,
        ignore_empty: bool = False,
        *,
        backend_args: Optional[dict] = None,
    ) -> None:
        self.ignore_empty = ignore_empty
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend

        self.file_client_args: Optional[dict] = None
        self.backend_args: Optional[dict] = None
        if file_client_args is not None:
            warnings.warn(
                '"file_client_args" will be deprecated in future. '
                'Please use "backend_args" instead',
                DeprecationWarning,
            )
            if backend_args is not None:
                raise ValueError(
                    '"file_client_args" and "backend_args" cannot be set '
                    "at the same time."
                )

            self.file_client_args = file_client_args.copy()
        if backend_args is not None:
            self.backend_args = backend_args.copy()

    def transform(self, results: dict) -> Optional[dict]:
        filename = results["img_path"]
        try:
            if self.file_client_args is not None:
                file_client = FileClient.infer_client(self.file_client_args, filename)
                img_bytes = file_client.get(filename)
            else:
                img_bytes = get(filename, backend_args=self.backend_args)
            img = imfrombytes(
                img_bytes, flag=self.color_type, backend=self.imdecode_backend
            )
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        assert img is not None, f"failed to load image: {filename}"
        if self.to_float32:
            img = img.astype(np.float32)

        results["img"] = img
        results["img_shape"] = img.shape[:2]
        results["ori_shape"] = img.shape[:2]
        return results

    def __repr__(self):
        repr_str = (
            f"{self.__class__.__name__}("
            f"ignore_empty={self.ignore_empty}, "
            f"to_float32={self.to_float32}, "
            f"color_type='{self.color_type}', "
            f"imdecode_backend='{self.imdecode_backend}', "
        )

        if self.file_client_args is not None:
            repr_str += f"file_client_args={self.file_client_args})"
        else:
            repr_str += f"backend_args={self.backend_args})"

        return repr_str


@TRANSFORMS.register_module()
class MMCV_Resize(BaseTransform):
    def __init__(
        self,
        scale: Optional[Union[int, Tuple[int, int]]] = None,
        scale_factor: Optional[Union[float, Tuple[float, float]]] = None,
        keep_ratio: bool = False,
        clip_object_border: bool = True,
        backend: str = "cv2",
        interpolation="bilinear",
    ) -> None:
        assert scale is not None or scale_factor is not None, (
            "`scale` and" "`scale_factor` can not both be `None`"
        )
        if scale is None:
            self.scale = None
        else:
            if isinstance(scale, int):
                self.scale = (scale, scale)
            else:
                self.scale = scale

        self.backend = backend
        self.interpolation = interpolation
        self.keep_ratio = keep_ratio
        self.clip_object_border = clip_object_border
        if scale_factor is None:
            self.scale_factor = None
        elif isinstance(scale_factor, float):
            self.scale_factor = (scale_factor, scale_factor)
        elif isinstance(scale_factor, tuple):
            assert (len(scale_factor)) == 2
            self.scale_factor = scale_factor
        else:
            raise TypeError(
                f"expect scale_factor is float or Tuple(float), but"
                f"get {type(scale_factor)}"
            )

    def _resize_img(self, results: dict) -> None:
        if results.get("img", None) is not None:
            if self.keep_ratio:
                img, scale_factor = imrescale(
                    results["img"],
                    results["scale"],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend,
                )
                new_h, new_w = img.shape[:2]
                h, w = results["img"].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                img, w_scale, h_scale = imresize(
                    results["img"],
                    results["scale"],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend,
                )
            results["img"] = img
            results["img_shape"] = img.shape[:2]
            results["scale_factor"] = (w_scale, h_scale)
            results["keep_ratio"] = self.keep_ratio

    def _resize_bboxes(self, results: dict) -> None:
        if results.get("gt_bboxes", None) is not None:
            bboxes = results["gt_bboxes"] * np.tile(
                np.array(results["scale_factor"]), 2
            )
            if self.clip_object_border:
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, results["img_shape"][1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, results["img_shape"][0])
            results["gt_bboxes"] = bboxes

    def _resize_seg(self, results: dict) -> None:
        if results.get("gt_seg_map", None) is not None:
            if self.keep_ratio:
                gt_seg = imrescale(
                    results["gt_seg_map"],
                    results["scale"],
                    interpolation="nearest",
                    backend=self.backend,
                )
            else:
                gt_seg = imresize(
                    results["gt_seg_map"],
                    results["scale"],
                    interpolation="nearest",
                    backend=self.backend,
                )
            results["gt_seg_map"] = gt_seg

    def _resize_keypoints(self, results: dict) -> None:
        if results.get("gt_keypoints", None) is not None:
            keypoints = results["gt_keypoints"]

            keypoints[:, :, :2] = keypoints[:, :, :2] * np.array(
                results["scale_factor"]
            )
            if self.clip_object_border:
                keypoints[:, :, 0] = np.clip(
                    keypoints[:, :, 0], 0, results["img_shape"][1]
                )
                keypoints[:, :, 1] = np.clip(
                    keypoints[:, :, 1], 0, results["img_shape"][0]
                )
            results["gt_keypoints"] = keypoints

    def transform(self, results: dict) -> dict:

        if self.scale:
            results["scale"] = self.scale
        else:
            img_shape = results["img"].shape[:2]
            results["scale"] = _scale_size(img_shape[::-1], self.scale_factor)
        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_seg(results)
        self._resize_keypoints(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(scale={self.scale}, "
        repr_str += f"scale_factor={self.scale_factor}, "
        repr_str += f"keep_ratio={self.keep_ratio}, "
        repr_str += f"clip_object_border={self.clip_object_border}), "
        repr_str += f"backend={self.backend}), "
        repr_str += f"interpolation={self.interpolation})"
        return repr_str


def autocast_box_type(dst_box_type="hbox") -> Callable:
    _, box_type_cls = get_box_type(dst_box_type)

    def decorator(func: Callable) -> Callable:
        def wrapper(self, results: dict, *args, **kwargs) -> dict:
            if "gt_bboxes" not in results or isinstance(
                results["gt_bboxes"], BaseBoxes
            ):
                return func(self, results)
            elif isinstance(results["gt_bboxes"], np.ndarray):
                results["gt_bboxes"] = box_type_cls(results["gt_bboxes"], clone=False)
                if "mix_results" in results:
                    for res in results["mix_results"]:
                        if isinstance(res["gt_bboxes"], np.ndarray):
                            res["gt_bboxes"] = box_type_cls(
                                res["gt_bboxes"], clone=False
                            )

                _results = func(self, results, *args, **kwargs)
                if isinstance(_results, dict) and "gt_bboxes" in _results:
                    if isinstance(_results["gt_bboxes"], BaseBoxes):
                        _results["gt_bboxes"] = _results["gt_bboxes"].numpy()
                if isinstance(results["gt_bboxes"], BaseBoxes):
                    results["gt_bboxes"] = results["gt_bboxes"].numpy()
                return _results
            else:
                raise TypeError(
                    "auto_box_type requires results['gt_bboxes'] to "
                    "be BaseBoxes or np.ndarray, but got "
                    f"{type(results['gt_bboxes'])}"
                )

        return wrapper

    return decorator


@TRANSFORMS.register_module()
class MMDET_Resize(MMCV_Resize):
    def _resize_masks(self, results: dict) -> None:
        if results.get("gt_masks", None) is not None:
            if self.keep_ratio:
                results["gt_masks"] = results["gt_masks"].rescale(results["scale"])
            else:
                results["gt_masks"] = results["gt_masks"].resize(results["img_shape"])

    def _resize_bboxes(self, results: dict) -> None:
        if results.get("gt_bboxes", None) is not None:
            results["gt_bboxes"].rescale_(results["scale_factor"])
            if self.clip_object_border:
                results["gt_bboxes"].clip_(results["img_shape"])

    def _resize_seg(self, results: dict) -> None:
        if results.get("gt_seg_map", None) is not None:
            if self.keep_ratio:
                gt_seg = imrescale(
                    results["gt_seg_map"],
                    results["scale"],
                    interpolation="nearest",
                    backend=self.backend,
                )
            else:
                gt_seg = imresize(
                    results["gt_seg_map"],
                    results["scale"],
                    interpolation="nearest",
                    backend=self.backend,
                )
            results["gt_seg_map"] = gt_seg

    def _record_homography_matrix(self, results: dict) -> None:
        w_scale, h_scale = results["scale_factor"]
        homography_matrix = np.array(
            [[w_scale, 0, 0], [0, h_scale, 0], [0, 0, 1]], dtype=np.float32
        )
        if results.get("homography_matrix", None) is None:
            results["homography_matrix"] = homography_matrix
        else:
            results["homography_matrix"] = (
                homography_matrix @ results["homography_matrix"]
            )

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        if self.scale:
            results["scale"] = self.scale
        else:
            img_shape = results["img"].shape[:2]
            results["scale"] = _scale_size(img_shape[::-1], self.scale_factor)
        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_masks(results)
        self._resize_seg(results)
        self._record_homography_matrix(results)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(scale={self.scale}, "
        repr_str += f"scale_factor={self.scale_factor}, "
        repr_str += f"keep_ratio={self.keep_ratio}, "
        repr_str += f"clip_object_border={self.clip_object_border}), "
        repr_str += f"backend={self.backend}), "
        repr_str += f"interpolation={self.interpolation})"
        return repr_str


@TRANSFORMS.register_module()
class YOLOv5KeepRatioResize(MMDET_Resize):
    def __init__(
        self, scale: Union[int, Tuple[int, int]], keep_ratio: bool = True, **kwargs
    ):
        assert keep_ratio is True
        super().__init__(scale=scale, keep_ratio=True, **kwargs)

    @staticmethod
    def _get_rescale_ratio(
        old_size: Tuple[int, int], scale: Union[float, Tuple[int]]
    ) -> float:
        w, h = old_size
        if isinstance(scale, (float, int)):
            if scale <= 0:
                raise ValueError(f"Invalid scale {scale}, must be positive.")
            scale_factor = scale
        elif isinstance(scale, tuple):
            max_long_edge = max(scale)
            max_short_edge = min(scale)
            scale_factor = min(max_long_edge / max(h, w), max_short_edge / min(h, w))
        else:
            raise TypeError(
                "Scale must be a number or tuple of int, " f"but got {type(scale)}"
            )

        return scale_factor

    def _resize_img(self, results: dict):
        assert self.keep_ratio is True

        if results.get("img", None) is not None:
            image = results["img"]
            original_h, original_w = image.shape[:2]
            ratio = self._get_rescale_ratio((original_h, original_w), self.scale)

            if ratio != 1:
                image = imresize(
                    img=image,
                    size=(int(original_w * ratio), int(original_h * ratio)),
                    interpolation="area" if ratio < 1 else "bilinear",
                    backend=self.backend,
                )

            resized_h, resized_w = image.shape[:2]
            scale_ratio_h = resized_h / original_h
            scale_ratio_w = resized_w / original_w
            scale_factor = (scale_ratio_w, scale_ratio_h)

            results["img"] = image
            results["img_shape"] = image.shape[:2]
            results["scale_factor"] = scale_factor


@TRANSFORMS.register_module()
class LetterResize(MMDET_Resize):
    def __init__(
        self,
        scale: Union[int, Tuple[int, int]],
        pad_val: dict = dict(img=0, mask=0, seg=255),
        use_mini_pad: bool = False,
        stretch_only: bool = False,
        allow_scale_up: bool = True,
        half_pad_param: bool = False,
        **kwargs,
    ):
        super().__init__(scale=scale, keep_ratio=True, **kwargs)

        self.pad_val = pad_val
        if isinstance(pad_val, (int, float)):
            pad_val = dict(img=pad_val, seg=255)
        assert isinstance(
            pad_val, dict
        ), f"pad_val must be dict, but got {type(pad_val)}"

        self.use_mini_pad = use_mini_pad
        self.stretch_only = stretch_only
        self.allow_scale_up = allow_scale_up
        self.half_pad_param = half_pad_param

    def _resize_img(self, results: dict):
        image = results.get("img", None)
        if image is None:
            return
        if "batch_shape" in results:
            scale = tuple(results["batch_shape"])
        else:
            scale = self.scale[::-1]

        image_shape = image.shape[:2]
        ratio = min(scale[0] / image_shape[0], scale[1] / image_shape[1])
        if not self.allow_scale_up:
            ratio = min(ratio, 1.0)

        ratio = [ratio, ratio]
        no_pad_shape = (
            int(round(image_shape[0] * ratio[0])),
            int(round(image_shape[1] * ratio[1])),
        )
        padding_h, padding_w = [scale[0] - no_pad_shape[0], scale[1] - no_pad_shape[1]]
        if self.use_mini_pad:
            padding_w, padding_h = np.mod(padding_w, 32), np.mod(padding_h, 32)

        elif self.stretch_only:
            padding_h, padding_w = 0.0, 0.0
            no_pad_shape = (scale[0], scale[1])
            ratio = [
                scale[0] / image_shape[0],
                scale[1] / image_shape[1],
            ]

        if image_shape != no_pad_shape:
            image = imresize(
                image,
                (no_pad_shape[1], no_pad_shape[0]),
                interpolation=self.interpolation,
                backend=self.backend,
            )

        scale_factor = (
            no_pad_shape[1] / image_shape[1],
            no_pad_shape[0] / image_shape[0],
        )

        if "scale_factor" in results:
            results["scale_factor_origin"] = results["scale_factor"]
        results["scale_factor"] = scale_factor
        top_padding, left_padding = int(round(padding_h // 2 - 0.1)), int(
            round(padding_w // 2 - 0.1)
        )
        bottom_padding = padding_h - top_padding
        right_padding = padding_w - left_padding

        padding_list = [top_padding, bottom_padding, left_padding, right_padding]
        if (
            top_padding != 0
            or bottom_padding != 0
            or left_padding != 0
            or right_padding != 0
        ):

            pad_val = self.pad_val.get("img", 0)
            if isinstance(pad_val, int) and image.ndim == 3:
                pad_val = tuple(pad_val for _ in range(image.shape[2]))

            image = impad(
                img=image,
                padding=(
                    padding_list[2],
                    padding_list[0],
                    padding_list[3],
                    padding_list[1],
                ),
                pad_val=pad_val,
                padding_mode="constant",
            )

        results["img"] = image
        results["img_shape"] = image.shape
        if "pad_param" in results:
            results["pad_param_origin"] = results["pad_param"] * np.repeat(ratio, 2)

        if self.half_pad_param:
            results["pad_param"] = np.array(
                [padding_h / 2, padding_h / 2, padding_w / 2, padding_w / 2],
                dtype=np.float32,
            )
        else:
            results["pad_param"] = np.array(padding_list, dtype=np.float32)

    def _resize_masks(self, results: dict):
        if results.get("gt_masks", None) is None:
            return

        gt_masks = results["gt_masks"]
        assert isinstance(
            gt_masks, PolygonMasks
        ), f"Only supports PolygonMasks, but got {type(gt_masks)}"
        gt_mask_h = results["gt_masks"].height * results["scale_factor"][1]
        gt_mask_w = results["gt_masks"].width * results["scale_factor"][0]
        gt_masks = results["gt_masks"].resize(
            (int(round(gt_mask_h)), int(round(gt_mask_w)))
        )

        top_padding, _, left_padding, _ = results["pad_param"]
        if int(left_padding) != 0:
            gt_masks = gt_masks.translate(
                out_shape=results["img_shape"][:2],
                offset=int(left_padding),
                direction="horizontal",
            )
        if int(top_padding) != 0:
            gt_masks = gt_masks.translate(
                out_shape=results["img_shape"][:2],
                offset=int(top_padding),
                direction="vertical",
            )
        results["gt_masks"] = gt_masks

    def _resize_bboxes(self, results: dict):
        if results.get("gt_bboxes", None) is None:
            return
        results["gt_bboxes"].rescale_(results["scale_factor"])

        if len(results["pad_param"]) != 4:
            return
        results["gt_bboxes"].translate_(
            (results["pad_param"][2], results["pad_param"][0])
        )

        if self.clip_object_border:
            results["gt_bboxes"].clip_(results["img_shape"])

    def transform(self, results: dict) -> dict:
        results = super().transform(results)
        if "scale_factor_origin" in results:
            scale_factor_origin = results.pop("scale_factor_origin")
            results["scale_factor"] = (
                results["scale_factor"][0] * scale_factor_origin[0],
                results["scale_factor"][1] * scale_factor_origin[1],
            )
        if "pad_param_origin" in results:
            pad_param_origin = results.pop("pad_param_origin")
            results["pad_param"] += pad_param_origin
        return results


@TRANSFORMS.register_module()
class MMCV_LoadAnnotations(BaseTransform):
    def __init__(
        self,
        with_bbox: bool = True,
        with_label: bool = True,
        with_seg: bool = False,
        with_keypoints: bool = False,
        imdecode_backend: str = "cv2",
        file_client_args: Optional[dict] = None,
        *,
        backend_args: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_seg = with_seg
        self.with_keypoints = with_keypoints
        self.imdecode_backend = imdecode_backend

        self.file_client_args: Optional[dict] = None
        self.backend_args: Optional[dict] = None
        if file_client_args is not None:
            warnings.warn(
                '"file_client_args" will be deprecated in future. '
                'Please use "backend_args" instead',
                DeprecationWarning,
            )
            if backend_args is not None:
                raise ValueError(
                    '"file_client_args" and "backend_args" cannot be set '
                    "at the same time."
                )

            self.file_client_args = file_client_args.copy()
        if backend_args is not None:
            self.backend_args = backend_args.copy()

    def _load_bboxes(self, results: dict) -> None:
        gt_bboxes = []
        for instance in results["instances"]:
            gt_bboxes.append(instance["bbox"])
        results["gt_bboxes"] = np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4)

    def _load_labels(self, results: dict) -> None:
        gt_bboxes_labels = []
        for instance in results["instances"]:
            gt_bboxes_labels.append(instance["bbox_label"])
        results["gt_bboxes_labels"] = np.array(gt_bboxes_labels, dtype=np.int64)

    def _load_seg_map(self, results: dict) -> None:
        if self.file_client_args is not None:
            file_client = FileClient.infer_client(
                self.file_client_args, results["seg_map_path"]
            )
            img_bytes = file_client.get(results["seg_map_path"])
        else:
            img_bytes = get(results["seg_map_path"], backend_args=self.backend_args)

        results["gt_seg_map"] = imfrombytes(
            img_bytes, flag="unchanged", backend=self.imdecode_backend
        ).squeeze()

    def _load_kps(self, results: dict) -> None:
        gt_keypoints = []
        for instance in results["instances"]:
            gt_keypoints.append(instance["keypoints"])
        results["gt_keypoints"] = np.array(gt_keypoints, np.float32).reshape(
            (len(gt_keypoints), -1, 3)
        )

    def transform(self, results: dict) -> dict:

        if self.with_bbox:
            self._load_bboxes(results)
        if self.with_label:
            self._load_labels(results)
        if self.with_seg:
            self._load_seg_map(results)
        if self.with_keypoints:
            self._load_kps(results)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(with_bbox={self.with_bbox}, "
        repr_str += f"with_label={self.with_label}, "
        repr_str += f"with_seg={self.with_seg}, "
        repr_str += f"with_keypoints={self.with_keypoints}, "
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "

        if self.file_client_args is not None:
            repr_str += f"file_client_args={self.file_client_args})"
        else:
            repr_str += f"backend_args={self.backend_args})"

        return repr_str


@TRANSFORMS.register_module()
class LoadAnnotations(MMCV_LoadAnnotations):
    def __init__(
        self,
        with_mask: bool = False,
        poly2mask: bool = True,
        box_type: str = "hbox",
        **kwargs,
    ) -> None:
        super(LoadAnnotations, self).__init__(**kwargs)
        self.with_mask = with_mask
        self.poly2mask = poly2mask
        self.box_type = box_type

    def _load_bboxes(self, results: dict) -> None:
        gt_bboxes = []
        gt_ignore_flags = []
        for instance in results.get("instances", []):
            gt_bboxes.append(instance["bbox"])
            gt_ignore_flags.append(instance["ignore_flag"])
        if self.box_type is None:
            results["gt_bboxes"] = np.array(gt_bboxes, dtype=np.float32).reshape(
                (-1, 4)
            )
        else:
            _, box_type_cls = get_box_type(self.box_type)
            results["gt_bboxes"] = box_type_cls(gt_bboxes, dtype=torch.float32)
        results["gt_ignore_flags"] = np.array(gt_ignore_flags, dtype=bool)

    def _load_labels(self, results: dict) -> None:
        gt_bboxes_labels = []
        for instance in results.get("instances", []):
            gt_bboxes_labels.append(instance["bbox_label"])
        results["gt_bboxes_labels"] = np.array(gt_bboxes_labels, dtype=np.int64)

    def _poly2mask(
        self, mask_ann: Union[list, dict], img_h: int, img_w: int
    ) -> np.ndarray:

        if isinstance(mask_ann, list):
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann["counts"], list):
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

    def _process_masks(self, results: dict) -> list:
        gt_masks = []
        gt_ignore_flags = []
        for instance in results.get("instances", []):
            gt_mask = instance["mask"]

            if isinstance(gt_mask, list):
                gt_mask = [
                    np.array(polygon)
                    for polygon in gt_mask
                    if len(polygon) % 2 == 0 and len(polygon) >= 6
                ]
                if len(gt_mask) == 0:

                    instance["ignore_flag"] = 1
                    gt_mask = [np.zeros(6)]
            elif not self.poly2mask:
                instance["ignore_flag"] = 1
                gt_mask = [np.zeros(6)]
            elif isinstance(gt_mask, dict) and not (
                gt_mask.get("counts") is not None
                and gt_mask.get("size") is not None
                and isinstance(gt_mask["counts"], (list, str))
            ):
                instance["ignore_flag"] = 1
                gt_mask = [np.zeros(6)]
            gt_masks.append(gt_mask)
            gt_ignore_flags.append(instance["ignore_flag"])
        results["gt_ignore_flags"] = np.array(gt_ignore_flags, dtype=bool)
        return gt_masks

    def _load_masks(self, results: dict) -> None:
        h, w = results["ori_shape"]
        gt_masks = self._process_masks(results)
        if self.poly2mask:
            gt_masks = BitmapMasks(
                [self._poly2mask(mask, h, w) for mask in gt_masks], h, w
            )
        else:
            gt_masks = PolygonMasks([mask for mask in gt_masks], h, w)
        results["gt_masks"] = gt_masks

    def transform(self, results: dict) -> dict:
        if self.with_bbox:
            self._load_bboxes(results)
        if self.with_label:
            self._load_labels(results)
        if self.with_mask:
            self._load_masks(results)
        if self.with_seg:
            self._load_seg_map(results)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(with_bbox={self.with_bbox}, "
        repr_str += f"with_label={self.with_label}, "
        repr_str += f"with_mask={self.with_mask}, "
        repr_str += f"with_seg={self.with_seg}, "
        repr_str += f"poly2mask={self.poly2mask}, "
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f"backend_args={self.backend_args})"
        return repr_str


@TRANSFORMS.register_module()
class LoadText:
    def __init__(
        self,
        text_path: str = None,
        prompt_format: str = "{}",
        multi_prompt_flag: str = "/",
    ) -> None:
        self.prompt_format = prompt_format
        self.multi_prompt_flag = multi_prompt_flag
        if text_path is not None:
            with open(text_path, "r") as f:
                self.class_texts = json.load(f)

    def __call__(self, results: dict) -> dict:
        assert "texts" in results or hasattr(
            self, "class_texts"
        ), "No texts found in results."
        class_texts = results.get("texts", getattr(self, "class_texts", None))

        texts = []
        for idx, cls_caps in enumerate(class_texts):
            assert len(cls_caps) > 0
            sel_cls_cap = cls_caps[0]
            sel_cls_cap = self.prompt_format.format(sel_cls_cap)
            texts.append(sel_cls_cap)

        results["texts"] = texts

        return results


def to_tensor(
    data: Union[torch.Tensor, np.ndarray, Sequence, int, float]
) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f"type {type(data)} cannot be converted to tensor.")


@TRANSFORMS.register_module()
class PackDetInputs(BaseTransform):
    mapping_table = {
        "gt_bboxes": "bboxes",
        "gt_bboxes_labels": "labels",
        "gt_masks": "masks",
    }

    def __init__(
        self,
        meta_keys=(
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "scale_factor",
            "flip",
            "flip_direction",
        ),
    ):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        packed_results = dict()
        if "img" in results:
            img = results["img"]
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            if not img.flags.c_contiguous:
                img = np.ascontiguousarray(img.transpose(2, 0, 1))
                img = to_tensor(img)
            else:
                img = to_tensor(img).permute(2, 0, 1).contiguous()

            packed_results["inputs"] = img

        if "gt_ignore_flags" in results:
            valid_idx = np.where(results["gt_ignore_flags"] == 0)[0]
            ignore_idx = np.where(results["gt_ignore_flags"] == 1)[0]

        data_sample = DetDataSample()
        instance_data = InstanceData()
        ignore_instance_data = InstanceData()

        for key in self.mapping_table.keys():
            if key not in results:
                continue
            if key == "gt_masks" or isinstance(results[key], BaseBoxes):
                if "gt_ignore_flags" in results:
                    instance_data[self.mapping_table[key]] = results[key][valid_idx]
                    ignore_instance_data[self.mapping_table[key]] = results[key][
                        ignore_idx
                    ]
                else:
                    instance_data[self.mapping_table[key]] = results[key]
            else:
                if "gt_ignore_flags" in results:
                    instance_data[self.mapping_table[key]] = to_tensor(
                        results[key][valid_idx]
                    )
                    ignore_instance_data[self.mapping_table[key]] = to_tensor(
                        results[key][ignore_idx]
                    )
                else:
                    instance_data[self.mapping_table[key]] = to_tensor(results[key])
        data_sample.gt_instances = instance_data
        data_sample.ignored_instances = ignore_instance_data

        if "proposals" in results:
            proposals = InstanceData(
                bboxes=to_tensor(results["proposals"]),
                scores=to_tensor(results["proposals_scores"]),
            )
            data_sample.proposals = proposals

        if "gt_seg_map" in results:
            gt_sem_seg_data = dict(
                sem_seg=to_tensor(results["gt_seg_map"][None, ...].copy())
            )
            data_sample.gt_sem_seg = PixelData(**gt_sem_seg_data)

        img_meta = {}
        for key in self.meta_keys:
            assert key in results, (
                f"`{key}` is not found in `results`, "
                f"the valid keys are {list(results)}."
            )
            img_meta[key] = results[key]

        data_sample.set_metainfo(img_meta)
        packed_results["data_samples"] = data_sample

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(meta_keys={self.meta_keys})"
        return repr_str
