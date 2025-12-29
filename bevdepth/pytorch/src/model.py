# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BEVDepth model implementation

Apdapted from: https://github.com/Megvii-BaseDetection/BEVDepth

MIT License

Copyright (c) 2022 Megvii-BaseDetection

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software,and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIESOF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERSBE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""
import copy
import numpy as np
from abc import ABCMeta, abstractmethod
from functools import partial
from typing import Dict, Optional, Tuple, Union
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function
from torch.nn.modules.utils import _pair
import torchvision

# Base configuration parameters
H = 900
W = 1600
final_dim = (256, 704)
img_conf = dict(
    img_mean=[123.675, 116.28, 103.53], img_std=[58.395, 57.12, 57.375], to_rgb=True
)

ida_aug_conf = {
    "resize_lim": (0.386, 0.55),
    "final_dim": final_dim,
    "rot_lim": (-5.4, 5.4),
    "H": H,
    "W": W,
    "rand_flip": True,
    "bot_pct_lim": (0.0, 0.0),
    "cams": [
        "CAM_FRONT_LEFT",
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT",
        "CAM_BACK",
        "CAM_BACK_RIGHT",
    ],
    "Ncams": 6,
}

bda_aug_conf = {
    "rot_lim": (-22.5, 22.5),
    "scale_lim": (0.95, 1.05),
    "flip_dx_ratio": 0.5,
    "flip_dy_ratio": 0.5,
}

bev_backbone = dict(
    type="ResNet",
    in_channels=80,
    depth=18,
    num_stages=3,
    strides=(1, 2, 2),
    dilations=(1, 1, 1),
    out_indices=[0, 1, 2],
    norm_eval=False,
    base_channels=160,
)

bev_neck = dict(
    type="SECONDFPN",
    in_channels=[80, 160, 320, 640],
    upsample_strides=[1, 2, 4, 8],
    out_channels=[64, 64, 64, 64],
)

CLASSES = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

TASKS = [
    dict(num_class=1, class_names=["car"]),
    dict(num_class=2, class_names=["truck", "construction_vehicle"]),
    dict(num_class=2, class_names=["bus", "trailer"]),
    dict(num_class=1, class_names=["barrier"]),
    dict(num_class=2, class_names=["motorcycle", "bicycle"]),
    dict(num_class=2, class_names=["pedestrian", "traffic_cone"]),
]

common_heads = dict(reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2))

bbox_coder = dict(
    type="CenterPointBBoxCoder",
    post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
    max_num=500,
    score_threshold=0.1,
    out_size_factor=4,
    voxel_size=[0.2, 0.2, 8],
    pc_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
    code_size=9,
)

train_cfg = dict(
    point_cloud_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
    grid_size=[512, 512, 1],
    voxel_size=[0.2, 0.2, 8],
    out_size_factor=4,
    dense_reg=1,
    gaussian_overlap=0.1,
    max_objs=500,
    min_radius=2,
    code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5],
)

test_cfg = dict(
    post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
    max_per_img=500,
    max_pool_nms=False,
    min_radius=[4, 12, 10, 1, 0.85, 0.175],
    score_threshold=0.1,
    out_size_factor=4,
    voxel_size=[0.2, 0.2, 8],
    nms_type="circle",
    pre_max_size=1000,
    post_max_size=83,
    nms_thr=0.2,
)

# Base backbone configuration
backbone_conf = {
    "x_bound": [-51.2, 51.2, 0.8],
    "y_bound": [-51.2, 51.2, 0.8],
    "z_bound": [-5, 3, 8],
    "d_bound": [2.0, 58.0, 0.5],
    "final_dim": final_dim,
    "output_channels": 80,
    "downsample_factor": 16,
    "img_backbone_conf": {
        "type": "ResNet",
        "depth": 50,
        "frozen_stages": 0,
        "out_indices": [0, 1, 2, 3],
        "norm_eval": False,
        "init_cfg": {"type": "Pretrained", "checkpoint": "torchvision://resnet50"},
    },
    "img_neck_conf": {
        "type": "SECONDFPN",
        "in_channels": [256, 512, 1024, 2048],
        "upsample_strides": [0.25, 0.5, 1, 2],
        "out_channels": [128, 128, 128, 128],
    },
    "depth_net_conf": {"in_channels": 512, "mid_channels": 512},
}

# # Base head configuration
head_conf = {
    "bev_backbone_conf": bev_backbone,
    "bev_neck_conf": bev_neck,
    "tasks": TASKS,
    "common_heads": common_heads,
    "bbox_coder": bbox_coder,
    "train_cfg": train_cfg,
    "test_cfg": test_cfg,
    "in_channels": 256,  # Equal to bev_neck output_channels.
    "loss_cls": {"type": "GaussianFocalLoss", "reduction": "mean"},
    "loss_bbox": {"type": "L1Loss", "reduction": "mean", "loss_weight": 0.25},
    "gaussian_overlap": 0.1,
    "min_radius": 2,
}


def get_bevdepth_config(variant: str = "bev_depth_lss_r50_256x704_128x128_24e_2key"):
    """
    Get configuration for different BEVDepth variants.

    Args:
        variant (str): One of the supported BEVDepth variants:
            - "bev_depth_lss_r50_256x704_128x128_24e_2key" (base)
            - "bev_depth_lss_r50_256x704_128x128_24e_2key_ema"
            - "bev_depth_lss_r50_256x704_128x128_20e_cbgs_2key_da"
            - "bev_depth_lss_r50_256x704_128x128_20e_cbgs_2key_da_ema"

    Returns:
        dict: Configuration with backbone_conf, head_conf, and other parameters
    """
    # Deep copy base configurations to avoid mutation
    config = {
        "backbone_conf": copy.deepcopy(backbone_conf),
        "head_conf": copy.deepcopy(head_conf),
        "ida_aug_conf": copy.deepcopy(ida_aug_conf),
        "bda_aug_conf": copy.deepcopy(bda_aug_conf),
        "img_conf": copy.deepcopy(img_conf),
        "classes": CLASSES,
        "use_ema": False,
        "use_da": False,
        "use_cbgs": False,
        "basic_lr_per_img": 2e-4 / 64,
        "weight_decay": 1e-7,
        "epochs": 24,
        "lr_schedule_milestones": [19, 23],
        "key_idxes": [-1],  # Default for 2key variants
    }

    # Apply variant-specific overrides
    if variant == "bev_depth_lss_r50_256x704_128x128_24e_2key":
        # Base configuration - no changes needed
        pass

    elif variant == "bev_depth_lss_r50_256x704_128x128_24e_2key_ema":
        # EMA variant
        config["use_ema"] = True

    elif variant == "bev_depth_lss_r50_256x704_128x128_20e_cbgs_2key_da":
        # DA + CBGS variant
        config["use_da"] = True
        config["use_cbgs"] = True
        config["basic_lr_per_img"] = (
            2e-4 / 64
        )  # Match upstream depth DA (no LR change from base)
        config["weight_decay"] = 1e-7  # Different weight decay
        config["epochs"] = 20
        config["lr_schedule_milestones"] = [16, 19]
        config["backbone_conf"]["use_da"] = True

    elif variant == "bev_depth_lss_r50_256x704_128x128_20e_cbgs_2key_da_ema":
        # DA + CBGS + EMA variant
        config["use_ema"] = True
        config["use_da"] = True
        config["use_cbgs"] = True
        config["basic_lr_per_img"] = 2e-4 / 32  # Different learning rate
        config["weight_decay"] = 1e-7  # Different weight decay
        config["epochs"] = 20
        config["lr_schedule_milestones"] = [16, 19]
        config["backbone_conf"]["use_da"] = True

    else:
        raise ValueError(
            f"Unsupported variant: {variant}. Supported variants are: "
            f"bev_depth_lss_r50_256x704_128x128_24e_2key, "
            f"bev_depth_lss_r50_256x704_128x128_24e_2key_ema, "
            f"bev_depth_lss_r50_256x704_128x128_20e_cbgs_2key_da, "
            f"bev_depth_lss_r50_256x704_128x128_20e_cbgs_2key_da_ema"
        )

    # Apply common 2key modifications to head configuration
    num_key_frames = len(config["key_idxes"]) + 1  # +1 for current frame
    config["head_conf"]["bev_backbone_conf"]["in_channels"] = 80 * num_key_frames
    config["head_conf"]["bev_neck_conf"]["in_channels"] = [
        80 * num_key_frames,
        160,
        320,
        640,
    ]
    config["head_conf"]["train_cfg"]["code_weights"] = [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]

    return config


def load_checkpoint(model: torch.nn.Module, ckpt_path: str):
    if not ckpt_path:
        return
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model.module."):
            new_key = k[len("model.module.") :]
        elif k.startswith("model."):
            new_key = k[len("model.") :]
        else:
            new_key = k
        new_state_dict[new_key] = v
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)


def build_norm_hardcoded(cfg, num_features, postfix=None):
    cfg = cfg.copy()
    layer_type = cfg.pop("type")
    requires_grad = cfg.pop("requires_grad", None)
    if layer_type in ("BN", "BN2d"):
        module = nn.BatchNorm2d(num_features, **cfg)
        abbr = "bn"
    if requires_grad is not None:
        for p in module.parameters():
            p.requires_grad = requires_grad
    if postfix is None:
        postfix = ""
    name = f"{abbr}{postfix}"
    return name, module


class BaseModule(nn.Module, metaclass=ABCMeta):
    def __init__(self, init_cfg=None):
        super(BaseModule, self).__init__()
        self._is_init = False

        self.init_cfg = copy.deepcopy(init_cfg)

    @property
    def is_init(self):
        return self._is_init

    def init_weights(self):
        is_top_level_module = False
        if not hasattr(self, "_params_init_info"):
            self._params_init_info = defaultdict(dict)
            is_top_level_module = True
            for name, param in self.named_parameters():
                self._params_init_info[param]["init_info"] = (
                    f"The value is the same before and "
                    f"after calling `init_weights` "
                    f"of {self.__class__.__name__} "
                )
                self._params_init_info[param]["tmp_mean_value"] = param.data.mean()
            for sub_module in self.modules():
                sub_module._params_init_info = self._params_init_info
        if not self._is_init:
            if self.init_cfg:
                if isinstance(self.init_cfg, dict):
                    if self.init_cfg["type"] == "Pretrained":
                        return

            for m in self.children():
                if hasattr(m, "init_weights"):
                    m.init_weights()

            self._is_init = True

        if is_top_level_module:
            for sub_module in self.modules():
                del sub_module._params_init_info


class Sequential(BaseModule, nn.Sequential):
    def __init__(self, *args, init_cfg=None):
        BaseModule.__init__(self, init_cfg)
        nn.Sequential.__init__(self, *args)


class ResLayer(Sequential):
    def __init__(
        self,
        block,
        inplanes,
        planes,
        num_blocks,
        stride=1,
        avg_down=False,
        conv_cfg=None,
        norm_cfg=dict(type="BN"),
        downsample_first=True,
        **kwargs,
    ):
        self.block = block

        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = []
            conv_stride = stride
            downsample.extend(
                [
                    nn.Conv2d(
                        inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=conv_stride,
                        bias=False,
                    ),
                    build_norm_hardcoded(norm_cfg, planes * block.expansion)[1],
                ]
            )
            downsample = nn.Sequential(*downsample)

        layers = []
        if downsample_first:
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    downsample=downsample,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs,
                )
            )
            inplanes = planes * block.expansion
            for _ in range(1, num_blocks):
                layers.append(
                    block(
                        inplanes=inplanes,
                        planes=planes,
                        stride=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        **kwargs,
                    )
                )

        super(ResLayer, self).__init__(*layers)


class BasicBlock(BaseModule):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        dilation=1,
        downsample=None,
        style="pytorch",
        with_cp=False,
        conv_cfg=None,
        norm_cfg=dict(type="BN"),
        dcn=None,
        plugins=None,
        init_cfg=None,
    ):
        super(BasicBlock, self).__init__(init_cfg)
        self.norm1_name, norm1 = build_norm_hardcoded(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_hardcoded(norm_cfg, planes, postfix=2)
        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.add_module(self.norm1_name, norm1)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):
        def _inner_forward(x):
            identity = x
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
            return out

        out = _inner_forward(x)
        out = self.relu(out)

        return out


class Bottleneck(BaseModule):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        dilation=1,
        downsample=None,
        style="pytorch",
        with_cp=False,
        conv_cfg=None,
        norm_cfg=dict(type="BN"),
        dcn=None,
        plugins=None,
        init_cfg=None,
    ):
        super(Bottleneck, self).__init__(init_cfg)
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.dcn = dcn
        self.with_dcn = dcn is not None
        self.plugins = plugins
        self.with_plugins = plugins is not None

        if self.style == "pytorch":
            self.conv1_stride = 1
            self.conv2_stride = stride

        self.norm1_name, norm1 = build_norm_hardcoded(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_hardcoded(norm_cfg, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_hardcoded(
            norm_cfg, planes * self.expansion, postfix=3
        )

        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=1, stride=self.conv1_stride, bias=False
        )
        self.add_module(self.norm1_name, norm1)
        fallback_on_stride = False
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = nn.Conv2d(
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False,
            )

        self.add_module(self.norm2_name, norm2)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        return getattr(self, self.norm3_name)

    def forward(self, x):
        def _inner_forward(x):
            identity = x
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)
            out = self.conv3(out)
            out = self.norm3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
            return out

        out = _inner_forward(x)

        out = self.relu(out)

        return out


class ResNet(BaseModule):
    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3)),
    }

    def __init__(
        self,
        depth,
        in_channels=3,
        stem_channels=None,
        base_channels=64,
        num_stages=4,
        strides=(1, 2, 2, 2),
        dilations=(1, 1, 1, 1),
        out_indices=(0, 1, 2, 3),
        style="pytorch",
        deep_stem=False,
        avg_down=False,
        frozen_stages=-1,
        conv_cfg=None,
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=True,
        dcn=None,
        stage_with_dcn=(False, False, False, False),
        plugins=None,
        with_cp=False,
        zero_init_residual=True,
        pretrained=None,
        init_cfg=None,
    ):
        super(ResNet, self).__init__(init_cfg)
        self.zero_init_residual = zero_init_residual
        block_init_cfg = None
        if pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type="Kaiming", layer="Conv2d"),
                    dict(type="Constant", val=1, layer=["_BatchNorm", "GroupNorm"]),
                ]
                block = self.arch_settings[depth][0]
                if self.zero_init_residual:
                    if block is BasicBlock:
                        block_init_cfg = dict(
                            type="Constant", val=0, override=dict(name="norm2")
                        )

        self.depth = depth
        if stem_channels is None:
            stem_channels = base_channels
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        self.strides = strides
        self.dilations = dilations
        self.out_indices = out_indices
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        self.plugins = plugins
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = stem_channels

        self._make_stem_layer(in_channels, stem_channels)

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            stage_plugins = None
            planes = base_channels * 2**i
            res_layer = self.make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn,
                plugins=stage_plugins,
                init_cfg=block_init_cfg,
            )
            self.inplanes = planes * self.block.expansion
            layer_name = f"layer{i + 1}"
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

        self.feat_dim = (
            self.block.expansion * base_channels * 2 ** (len(self.stage_blocks) - 1)
        )

    def make_res_layer(self, **kwargs):
        return ResLayer(**kwargs)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, stem_channels):
        self.conv1 = nn.Conv2d(
            in_channels,
            stem_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.norm1_name, norm1 = build_norm_hardcoded(
            self.norm_cfg, stem_channels, postfix=1
        )
        self.add_module(self.norm1_name, norm1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.norm1.eval()
            for m in [self.conv1, self.norm1]:
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f"layer{i}")
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)


class ASPP(nn.Module):
    def __init__(self, inplanes, mid_channels=256, BatchNorm=nn.BatchNorm2d):
        super(ASPP, self).__init__()

        dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(
            inplanes,
            mid_channels,
            1,
            padding=0,
            dilation=dilations[0],
            BatchNorm=BatchNorm,
        )
        self.aspp2 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[1],
            dilation=dilations[1],
            BatchNorm=BatchNorm,
        )
        self.aspp3 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[2],
            dilation=dilations[2],
            BatchNorm=BatchNorm,
        )
        self.aspp4 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[3],
            dilation=dilations[3],
            BatchNorm=BatchNorm,
        )

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            BatchNorm(mid_channels),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(int(mid_channels * 5), mid_channels, 1, bias=False)
        self.bn1 = BatchNorm(mid_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode="bilinear", align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.ReLU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class DeformConv2dPack(nn.Module):
    """A CPU/GPU compatible version of MMCV's DeformConv2dPack using torchvision.ops.deform_conv2d.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple): Stride of the convolution.
        padding (int or tuple): Zero-padding added to both sides of the input.
        dilation (int or tuple): Spacing between kernel elements.
        groups (int): Number of blocked connections.
        deform_groups (int): Number of deformable group partitions.
        bias (bool): If True, adds a learnable bias to the output.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        deform_groups=1,
        bias=False,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deform_groups = deform_groups

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, *self.kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None

        self.conv_offset = nn.Conv2d(
            in_channels,
            2 * deform_groups * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True,
        )

        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=1)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        nn.init.constant_(self.conv_offset.weight, 0)
        nn.init.constant_(self.conv_offset.bias, 0)

    def forward(self, x):
        offset = self.conv_offset(x)

        return torchvision.ops.deform_conv2d(
            input=x,
            offset=offset,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )


class DepthNet(nn.Module):
    def __init__(self, in_channels, mid_channels, context_channels, depth_channels):
        super(DepthNet, self).__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.context_conv = nn.Conv2d(
            mid_channels, context_channels, kernel_size=1, stride=1, padding=0
        )
        self.bn = nn.BatchNorm1d(27)
        self.depth_mlp = Mlp(27, mid_channels, mid_channels)
        self.depth_se = SELayer(mid_channels)  # NOTE: add camera-aware
        self.context_mlp = Mlp(27, mid_channels, mid_channels)
        self.context_se = SELayer(mid_channels)  # NOTE: add camera-aware
        self.depth_conv = nn.Sequential(
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            ASPP(mid_channels, mid_channels),
            DeformConv2dPack(
                mid_channels,
                mid_channels,
                kernel_size=3,
                padding=1,
                groups=4,
                deform_groups=1,
                bias=False,
            ),
            nn.Conv2d(mid_channels, depth_channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x, mats_dict):
        intrins = mats_dict["intrin_mats"][:, 0:1, ..., :3, :3]
        batch_size = intrins.shape[0]
        num_cams = intrins.shape[2]
        ida = mats_dict["ida_mats"][:, 0:1, ...]
        sensor2ego = mats_dict["sensor2ego_mats"][:, 0:1, ..., :3, :]
        bda = (
            mats_dict["bda_mat"]
            .view(batch_size, 1, 1, 4, 4)
            .repeat(1, 1, num_cams, 1, 1)
        )
        mlp_input = torch.cat(
            [
                torch.stack(
                    [
                        intrins[:, 0:1, ..., 0, 0],
                        intrins[:, 0:1, ..., 1, 1],
                        intrins[:, 0:1, ..., 0, 2],
                        intrins[:, 0:1, ..., 1, 2],
                        ida[:, 0:1, ..., 0, 0],
                        ida[:, 0:1, ..., 0, 1],
                        ida[:, 0:1, ..., 0, 3],
                        ida[:, 0:1, ..., 1, 0],
                        ida[:, 0:1, ..., 1, 1],
                        ida[:, 0:1, ..., 1, 3],
                        bda[:, 0:1, ..., 0, 0],
                        bda[:, 0:1, ..., 0, 1],
                        bda[:, 0:1, ..., 1, 0],
                        bda[:, 0:1, ..., 1, 1],
                        bda[:, 0:1, ..., 2, 2],
                    ],
                    dim=-1,
                ),
                sensor2ego.view(batch_size, 1, num_cams, -1),
            ],
            -1,
        )
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
        x = self.reduce_conv(x)
        context_se = self.context_mlp(mlp_input)[..., None, None]
        context = self.context_se(x, context_se)
        context = self.context_conv(context)
        depth_se = self.depth_mlp(mlp_input)[..., None, None]
        depth = self.depth_se(x, depth_se)
        depth = self.depth_conv(depth)
        return torch.cat([depth, context], dim=1)


class DepthAggregation(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(DepthAggregation, self).__init__()

        self.reduce_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
        )

    def forward(self, x):
        x = self.reduce_conv(x)
        x = self.conv(x) + x
        x = self.out_conv(x)
        return x


def obsolete_torch_version(torch_version, version_threshold) -> bool:
    return torch_version == "parrots" or torch_version <= version_threshold


def build_upsample_layer(cfg, *args, **kwargs) -> nn.Module:
    if cfg is None:
        cfg_ = dict(type="deconv", bias=False)
    else:
        cfg_ = cfg.copy()

    layer_type = cfg_.pop("type")

    if layer_type == "deconv":
        in_channels = kwargs.pop("in_channels")
        out_channels = kwargs.pop("out_channels")
        kernel_size = kwargs.pop("kernel_size")
        stride = kwargs.pop("stride")
        bias = cfg_.pop("bias", False)
        return nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=bias
        )
    elif layer_type in ("nearest", "bilinear"):
        return nn.Upsample(mode=layer_type, **cfg_)


def build_conv_layer(cfg, *args, **kwargs) -> nn.Module:
    if cfg is None:
        cfg_ = dict(type="Conv2d", bias=False)
    else:
        cfg_ = cfg.copy()

    layer_type = cfg_.pop("type")

    if layer_type == "Conv2d":
        in_channels = kwargs.pop("in_channels")
        out_channels = kwargs.pop("out_channels")
        kernel_size = kwargs.pop("kernel_size")
        stride = kwargs.pop("stride", 1)
        bias = cfg_.pop("bias", False)
        padding = cfg_.pop("padding", 0)
        dilation = cfg_.pop("dilation", 1)
        groups = cfg_.pop("groups", 1)
        padding_mode = cfg_.pop("padding_mode", "zeros")
        return nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )


class NewEmptyTensorOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, new_shape: tuple) -> torch.Tensor:
        ctx.shape = x.shape
        return x.new_empty(new_shape)


class ConvTranspose2d(nn.ConvTranspose2d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            out_shape = [x.shape[0], self.out_channels]
            for i, k, p, s, d, op in zip(
                x.shape[-2:],
                self.kernel_size,
                self.padding,
                self.stride,
                self.dilation,
                self.output_padding,
            ):
                out_shape.append((i - 1) * s - 2 * p + (d * (k - 1) + 1) + op)
            empty = NewEmptyTensorOp.apply(x, out_shape)
            return empty

        return super().forward(x)


class SECONDFPN(BaseModule):
    def __init__(
        self,
        in_channels=[128, 128, 256],
        out_channels=[256, 256, 256],
        upsample_strides=[1, 2, 4],
        norm_cfg=dict(type="BN", eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type="deconv", bias=False),
        conv_cfg=dict(type="Conv2d", bias=False),
        use_conv_for_no_stride=False,
        init_cfg=None,
    ):
        super(SECONDFPN, self).__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels

        deblocks = []
        for i, out_channel in enumerate(out_channels):
            stride = upsample_strides[i]
            if stride > 1 or (stride == 1 and not use_conv_for_no_stride):
                upsample_layer = build_upsample_layer(
                    upsample_cfg,
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=upsample_strides[i],
                    stride=upsample_strides[i],
                )
            else:
                stride = np.round(1 / stride).astype(np.int64)
                upsample_layer = build_conv_layer(
                    conv_cfg,
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=stride,
                    stride=stride,
                )

            name, norm = build_norm_hardcoded(norm_cfg, out_channel)
            deblock = nn.Sequential(upsample_layer, norm, nn.ReLU(inplace=True))
            deblocks.append(deblock)
        self.deblocks = nn.ModuleList(deblocks)

        if init_cfg is None:
            self.init_cfg = [
                dict(type="Kaiming", layer="ConvTranspose2d"),
                dict(type="Constant", layer="NaiveSyncBatchNorm2d", val=1.0),
            ]

    def forward(self, x):
        ups = [deblock(x[i]) for i, deblock in enumerate(self.deblocks)]

        if len(ups) > 1:
            out = torch.cat(ups, dim=1)
        else:
            out = ups[0]
        return [out]


class VoxelPoolingTrain(Function):
    @staticmethod
    def forward(
        ctx,
        geom_xyz: torch.Tensor,
        input_features: torch.Tensor,
        voxel_num: torch.Tensor,
    ) -> torch.Tensor:
        ctx.mark_non_differentiable(geom_xyz)

        # Save original shape before reshape for gradient computation
        original_input_shape = input_features.shape

        geom_xyz = geom_xyz.reshape(geom_xyz.shape[0], -1, geom_xyz.shape[-1])
        input_features = input_features.reshape(
            (geom_xyz.shape[0], -1, input_features.shape[-1])
        )
        ctx.original_input_shape = original_input_shape

        batch_size = input_features.shape[0]
        num_points = input_features.shape[1]
        num_channels = input_features.shape[2]

        voxel_num_x = int(voxel_num[0].detach().cpu().item())
        voxel_num_y = int(voxel_num[1].detach().cpu().item())
        voxel_num_z = int(voxel_num[2].detach().cpu().item())

        # Output BEV grid (B, Vy, Vx, C)
        output_features = input_features.new_zeros(
            batch_size, voxel_num_y, voxel_num_x, num_channels
        )

        # Save the position in BEV feature map for each input point. Init to -1 (long dtype for indexing)
        pos_memo = torch.full(
            (batch_size, num_points, 3), -1, dtype=torch.long, device=geom_xyz.device
        )

        # Flatten (B, N, ...) to vectorized form
        geom_flat = geom_xyz.view(-1, 3)
        feats_flat = input_features.view(-1, num_channels)
        total = geom_flat.shape[0]

        # Compute batch indices for each flattened point
        b_idx = (
            torch.arange(batch_size, device=geom_xyz.device)
            .unsqueeze(1)
            .expand(batch_size, num_points)
            .reshape(-1)
        )

        sx_all = geom_flat[:, 0]
        sy_all = geom_flat[:, 1]
        sz_all = geom_flat[:, 2]

        # Valid mask within voxel bounds
        valid = (
            (sx_all >= 0)
            & (sx_all < voxel_num_x)
            & (sy_all >= 0)
            & (sy_all < voxel_num_y)
            & (sz_all >= 0)
            & (sz_all < voxel_num_z)
        )

        if valid.any():
            sx_v = sx_all[valid].to(torch.long)
            sy_v = sy_all[valid].to(torch.long)
            b_v = b_idx[valid].to(torch.long)
            feats_v = feats_flat[valid]

            # Accumulate into output grid: flatten BEV to [B*Vy*Vx, C]
            out_flat = output_features.view(
                batch_size * voxel_num_y * voxel_num_x, num_channels
            )
            lin_idx = b_v * (voxel_num_y * voxel_num_x) + sy_v * voxel_num_x + sx_v
            out_flat.index_add_(0, lin_idx, feats_v)

            # Populate pos_memo for backward (store [b, y, x])
            pos_view = pos_memo.view(-1, 3)
            pos_view[valid, 0] = b_v
            pos_view[valid, 1] = sy_v
            pos_view[valid, 2] = sx_v

        # Save zero-initialized grad_input_features and pos_memo for backward
        grad_input_features = torch.zeros_like(input_features)
        ctx.save_for_backward(grad_input_features, pos_memo)
        return output_features.permute(0, 3, 1, 2)

    @staticmethod
    def backward(ctx, grad_output_features):
        (grad_input_features, pos_memo) = ctx.saved_tensors
        kept = (pos_memo != -1)[..., 0]
        # Use the original input shape saved in forward, not the reshaped tensor's shape
        original_input_shape = ctx.original_input_shape
        grad_input_features = grad_input_features.reshape(
            grad_input_features.shape[0], -1, grad_input_features.shape[-1]
        )
        grad_input_features = grad_input_features.clone()
        grad_input_features[kept] = grad_output_features[
            pos_memo[kept][..., 0].long(),
            :,
            pos_memo[kept][..., 1].long(),
            pos_memo[kept][..., 2].long(),
        ]
        grad_input_features = grad_input_features.reshape(original_input_shape)
        return None, grad_input_features, None


voxel_pooling_train = VoxelPoolingTrain.apply


class VoxelPoolingInference(Function):
    @staticmethod
    def forward(
        ctx,
        geom_xyz: torch.Tensor,
        depth_features: torch.Tensor,
        context_features: torch.Tensor,
        voxel_num: torch.Tensor,
    ) -> torch.Tensor:
        ctx.mark_non_differentiable(geom_xyz)
        batch_size = geom_xyz.shape[0]
        num_cams = geom_xyz.shape[1]
        num_depth = geom_xyz.shape[2]
        num_height = geom_xyz.shape[3]
        num_width = geom_xyz.shape[4]
        num_channels = context_features.shape[1]

        voxel_num_x = int(voxel_num[0].detach().cpu().item())
        voxel_num_y = int(voxel_num[1].detach().cpu().item())
        voxel_num_z = int(voxel_num[2].detach().cpu().item())
        output_features = context_features.new_zeros(
            (batch_size, voxel_num_y, voxel_num_x, num_channels)
        )

        B = batch_size
        Cams = num_cams
        D = num_depth
        H = num_height
        W = num_width
        Vy = voxel_num_y
        Vx = voxel_num_x
        Vz = voxel_num_z

        total_samples = B * Cams * D * H * W
        geom_flat = geom_xyz.view(total_samples, 3)
        sx_all = geom_flat[:, 0]
        sy_all = geom_flat[:, 1]
        sz_all = geom_flat[:, 2]

        valid = (
            (sx_all >= 0)
            & (sx_all < Vx)
            & (sy_all >= 0)
            & (sy_all < Vy)
            & (sz_all >= 0)
            & (sz_all < Vz)
        )
        if valid.any():
            sx_v = sx_all[valid].to(torch.long)
            sy_v = sy_all[valid].to(torch.long)
            # derive indices (b, cam, d, h, w) from flat indices
            idx = torch.arange(total_samples, device=geom_xyz.device)[valid]
            # compute divisions on CPU tensors
            denom_cdhw = Cams * D * H * W
            denom_dhw = D * H * W
            denom_hw = H * W
            b_idx = torch.div(idx, denom_cdhw, rounding_mode="floor")
            cam_idx = torch.div(idx, denom_dhw, rounding_mode="floor") % Cams
            d_idx = torch.div(idx, denom_hw, rounding_mode="floor") % D
            h_idx = torch.div(idx, W, rounding_mode="floor") % H
            w_idx = idx % W
            bc_idx = b_idx * Cams + cam_idx

            # Gather depth [N]
            depth_vals = depth_features[bc_idx, d_idx, h_idx, w_idx]
            nonzero = depth_vals != 0
            if nonzero.any():
                # Filter by non-zero depth
                depth_vals = depth_vals[nonzero]
                bc_nz = bc_idx[nonzero]
                b_nz = b_idx[nonzero]
                sx_nz = sx_v[nonzero]
                sy_nz = sy_v[nonzero]
                h_nz = h_idx[nonzero]
                w_nz = w_idx[nonzero]
                # Gather context vectors [N, C]
                ctx_vecs = context_features[bc_nz, :, h_nz, w_nz]
                contrib = ctx_vecs * depth_vals.unsqueeze(1)
                # Flatten output over (B, Vy, Vx) -> [B*Vy*Vx, C]
                out_flat = output_features.view(B * Vy * Vx, num_channels)
                out_lin_idx = b_nz * (Vy * Vx) + sy_nz * Vx + sx_nz
                out_flat.index_add_(0, out_lin_idx, contrib)

        return output_features.permute(0, 3, 1, 2)


voxel_pooling_inference = VoxelPoolingInference.apply


class BaseLSSFPN(nn.Module):
    def __init__(
        self,
        x_bound,
        y_bound,
        z_bound,
        d_bound,
        final_dim,
        downsample_factor,
        output_channels,
        img_backbone_conf,
        img_neck_conf,
        depth_net_conf,
        use_da=False,
    ):
        super(BaseLSSFPN, self).__init__()
        self.downsample_factor = downsample_factor
        self.d_bound = d_bound
        self.final_dim = final_dim
        self.output_channels = output_channels

        self.register_buffer(
            "voxel_size", torch.Tensor([row[2] for row in [x_bound, y_bound, z_bound]])
        )
        self.register_buffer(
            "voxel_coord",
            torch.Tensor(
                [row[0] + row[2] / 2.0 for row in [x_bound, y_bound, z_bound]]
            ),
        )
        self.register_buffer(
            "voxel_num",
            torch.LongTensor(
                [(row[1] - row[0]) / row[2] for row in [x_bound, y_bound, z_bound]]
            ),
        )
        self.register_buffer("frustum", self.create_frustum())
        self.depth_channels, _, _, _ = self.frustum.shape
        self.use_da = use_da
        if self.use_da:
            self.depth_aggregation_net = self._configure_depth_aggregation_net()

        if isinstance(img_backbone_conf, dict):
            cfg = img_backbone_conf.copy()
            cfg.pop("type", None)
            self.img_backbone = ResNet(**cfg)
        if isinstance(img_neck_conf, dict):
            cfg = img_neck_conf.copy()
            cfg.pop("type", None)
            self.img_neck = SECONDFPN(**cfg)
        self.depth_net = self._configure_depth_net(depth_net_conf)

        self.img_neck.init_weights()
        self.img_backbone.init_weights()

    def _configure_depth_aggregation_net(self):
        return DepthAggregation(
            self.output_channels, self.output_channels, self.output_channels
        )

    def _forward_voxel_net(self, img_feat_with_depth):
        if self.use_da:
            # BEVConv2D [n, c, d, h, w] -> [n, h, c, w, d]
            img_feat_with_depth = img_feat_with_depth.permute(
                0, 3, 1, 4, 2
            ).contiguous()  # [n, c, d, h, w] -> [n, h, c, w, d]
            n, h, c, w, d = img_feat_with_depth.shape
            img_feat_with_depth = img_feat_with_depth.view(-1, c, w, d)
            img_feat_with_depth = (
                self.depth_aggregation_net(img_feat_with_depth)
                .view(n, h, c, w, d)
                .permute(0, 2, 4, 1, 3)
                .contiguous()
            )
        return img_feat_with_depth

    def _configure_depth_net(self, depth_net_conf):
        return DepthNet(
            depth_net_conf["in_channels"],
            depth_net_conf["mid_channels"],
            self.output_channels,
            self.depth_channels,
        )

    def create_frustum(self):
        ogfH, ogfW = self.final_dim
        fH, fW = ogfH // self.downsample_factor, ogfW // self.downsample_factor
        d_coords = (
            torch.arange(*self.d_bound, dtype=torch.float)
            .view(-1, 1, 1)
            .expand(-1, fH, fW)
        )
        D, _, _ = d_coords.shape
        x_coords = (
            torch.linspace(0, ogfW - 1, fW, dtype=torch.float)
            .view(1, 1, fW)
            .expand(D, fH, fW)
        )
        y_coords = (
            torch.linspace(0, ogfH - 1, fH, dtype=torch.float)
            .view(1, fH, 1)
            .expand(D, fH, fW)
        )
        paddings = torch.ones_like(d_coords)

        frustum = torch.stack((x_coords, y_coords, d_coords, paddings), -1)
        return frustum

    def get_geometry(self, sensor2ego_mat, intrin_mat, ida_mat, bda_mat):
        batch_size, num_cams, _, _ = sensor2ego_mat.shape
        points = self.frustum
        ida_mat = ida_mat.view(batch_size, num_cams, 1, 1, 1, 4, 4)
        points = ida_mat.inverse().matmul(points.unsqueeze(-1))
        points = torch.cat(
            (
                points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                points[:, :, :, :, :, 2:],
            ),
            5,
        )

        combine = sensor2ego_mat.matmul(torch.inverse(intrin_mat))
        points = combine.view(batch_size, num_cams, 1, 1, 1, 4, 4).matmul(points)
        if bda_mat is not None:
            bda_mat = (
                bda_mat.unsqueeze(1)
                .repeat(1, num_cams, 1, 1)
                .view(batch_size, num_cams, 1, 1, 1, 4, 4)
            )
            points = (bda_mat @ points).squeeze(-1)
        else:
            points = points.squeeze(-1)
        return points[..., :3]

    def get_cam_feats(self, imgs):
        batch_size, num_sweeps, num_cams, num_channels, imH, imW = imgs.shape

        imgs = imgs.flatten().view(
            batch_size * num_sweeps * num_cams, num_channels, imH, imW
        )
        img_feats = self.img_neck(self.img_backbone(imgs))[0]
        img_feats = img_feats.reshape(
            batch_size,
            num_sweeps,
            num_cams,
            img_feats.shape[1],
            img_feats.shape[2],
            img_feats.shape[3],
        )
        return img_feats

    def _forward_depth_net(self, feat, mats_dict):
        return self.depth_net(feat, mats_dict)

    def _forward_single_sweep(
        self, sweep_index, sweep_imgs, mats_dict, is_return_depth=False
    ):
        (
            batch_size,
            num_sweeps,
            num_cams,
            num_channels,
            img_height,
            img_width,
        ) = sweep_imgs.shape
        img_feats = self.get_cam_feats(sweep_imgs)
        source_features = img_feats[:, 0, ...]
        depth_feature = self._forward_depth_net(
            source_features.reshape(
                batch_size * num_cams,
                source_features.shape[2],
                source_features.shape[3],
                source_features.shape[4],
            ),
            mats_dict,
        )
        depth = depth_feature[:, : self.depth_channels].softmax(
            dim=1, dtype=depth_feature.dtype
        )
        geom_xyz = self.get_geometry(
            mats_dict["sensor2ego_mats"][:, sweep_index, ...],
            mats_dict["intrin_mats"][:, sweep_index, ...],
            mats_dict["ida_mats"][:, sweep_index, ...],
            mats_dict.get("bda_mat", None),
        )
        geom_xyz = (
            (geom_xyz - (self.voxel_coord - self.voxel_size / 2.0)) / self.voxel_size
        ).int()
        if self.training or self.use_da:
            img_feat_with_depth = depth.unsqueeze(1) * depth_feature[
                :, self.depth_channels : (self.depth_channels + self.output_channels)
            ].unsqueeze(2)

            img_feat_with_depth = self._forward_voxel_net(img_feat_with_depth)

            img_feat_with_depth = img_feat_with_depth.reshape(
                batch_size,
                num_cams,
                img_feat_with_depth.shape[1],
                img_feat_with_depth.shape[2],
                img_feat_with_depth.shape[3],
                img_feat_with_depth.shape[4],
            )

            img_feat_with_depth = img_feat_with_depth.permute(0, 1, 3, 4, 5, 2)

            feature_map = voxel_pooling_train(
                geom_xyz, img_feat_with_depth.contiguous(), self.voxel_num
            )
        else:
            feature_map = voxel_pooling_inference(
                geom_xyz,
                depth,
                depth_feature[
                    :,
                    self.depth_channels : (self.depth_channels + self.output_channels),
                ].contiguous(),
                self.voxel_num,
            )
        return feature_map.contiguous()

    def forward(self, sweep_imgs, mats_dict, timestamps=None, is_return_depth=False):
        (
            batch_size,
            num_sweeps,
            num_cams,
            num_channels,
            img_height,
            img_width,
        ) = sweep_imgs.shape

        key_frame_res = self._forward_single_sweep(
            0, sweep_imgs[:, 0:1, ...], mats_dict, is_return_depth=is_return_depth
        )
        if num_sweeps == 1:
            return key_frame_res

        key_frame_feature = key_frame_res[0] if is_return_depth else key_frame_res

        ret_feature_list = [key_frame_feature]
        for sweep_index in range(1, num_sweeps):
            with torch.no_grad():
                feature_map = self._forward_single_sweep(
                    sweep_index,
                    sweep_imgs[:, sweep_index : sweep_index + 1, ...],
                    mats_dict,
                    is_return_depth=False,
                )
                ret_feature_list.append(feature_map)

        return torch.cat(ret_feature_list, 1)


bev_backbone_conf = dict(
    type="ResNet",
    in_channels=80,
    depth=18,
    num_stages=3,
    strides=(1, 2, 2),
    dilations=(1, 1, 1),
    out_indices=[0, 1, 2],
    norm_eval=False,
    base_channels=160,
)

bev_neck_conf = dict(
    type="SECONDFPN",
    in_channels=[160, 320, 640],
    upsample_strides=[2, 4, 8],
    out_channels=[64, 64, 128],
)


def constant_init(module: nn.Module, val: float, bias: float = 0) -> None:
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(
    module, a=0, mode="fan_out", nonlinearity="relu", bias=0, distribution="normal"
):
    if hasattr(module, "weight") and module.weight is not None:
        if distribution == "uniform":
            nn.init.kaiming_uniform_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity
            )
        else:
            nn.init.kaiming_normal_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity
            )


class SeparateHead(BaseModule):
    def __init__(
        self,
        in_channels,
        heads,
        head_conv=64,
        final_kernel=1,
        init_bias=-2.19,
        conv_cfg=dict(type="Conv2d"),
        norm_cfg=dict(type="BN2d"),
        bias="auto",
        init_cfg=None,
        **kwargs,
    ):
        super(SeparateHead, self).__init__(init_cfg=init_cfg)
        self.heads = heads
        self.init_bias = init_bias
        for head in self.heads:
            classes, num_conv = self.heads[head]

            conv_layers = []
            c_in = in_channels
            for i in range(num_conv - 1):
                conv_layers.append(
                    ConvModule(
                        c_in,
                        head_conv,
                        kernel_size=final_kernel,
                        stride=1,
                        padding=final_kernel // 2,
                        bias=bias,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                    )
                )
                c_in = head_conv

            conv_layers.append(
                nn.Conv2d(
                    head_conv,
                    classes,
                    kernel_size=final_kernel,
                    stride=1,
                    padding=final_kernel // 2,
                    bias=True,
                )
            )
            conv_layers = nn.Sequential(*conv_layers)

            self.__setattr__(head, conv_layers)

            if init_cfg is None:
                self.init_cfg = dict(type="Kaiming", layer="Conv2d")

    def forward(self, x):
        ret_dict = dict()
        for head in self.heads:
            ret_dict[head] = self.__getattr__(head)(x)

        return ret_dict


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
    ):
        super().__init__()
        official_padding_mode = ["zeros", "circular"]
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.inplace = inplace
        self.with_spectral_norm = with_spectral_norm
        self.with_explicit_padding = padding_mode not in official_padding_mode
        self.order = order

        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        if bias == "auto":
            bias = not self.with_norm
        self.with_bias = bias

        conv_padding = 0 if self.with_explicit_padding else padding
        self.conv = nn.Conv2d(
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
            self.norm_name, norm = build_norm_hardcoded(
                norm_cfg, norm_channels
            )  # type: ignore
            self.add_module(self.norm_name, norm)

        if self.with_activation:
            act_cfg_ = act_cfg.copy()  # type: ignore
            if act_cfg_["type"] not in [
                "Tanh",
                "PReLU",
                "Sigmoid",
                "HSigmoid",
                "Swish",
                "GELU",
            ]:
                act_cfg_.setdefault("inplace", inplace)
            ACTIVATION_MAP = {
                "ReLU": nn.ReLU,
                "LeakyReLU": nn.LeakyReLU,
                "PReLU": nn.PReLU,
                "RReLU": nn.RReLU,
                "ReLU6": nn.ReLU6,
                "ELU": nn.ELU,
                "Sigmoid": nn.Sigmoid,
                "Tanh": nn.Tanh,
            }
            cfg = act_cfg.copy()
            act_type = cfg.pop("type")
            self.activate = ACTIVATION_MAP[act_type](**cfg)

        self.init_weights()

    @property
    def norm(self):
        if self.norm_name:
            return getattr(self, self.norm_name)
        else:
            return None

    def init_weights(self):
        if not hasattr(self.conv, "init_weights"):
            nonlinearity = "relu"
            a = 0
            kaiming_init(self.conv, a=a, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.norm, 1, bias=0)

    def forward(
        self, x: torch.Tensor, activate: bool = True, norm: bool = True
    ) -> torch.Tensor:
        for layer in self.order:
            if layer == "conv":
                x = self.conv(x)
            elif layer == "norm" and norm and self.with_norm:
                x = self.norm(x)
            elif layer == "act" and activate and self.with_activation:
                x = self.activate(x)
        return x


def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


class BaseBBoxCoder(metaclass=ABCMeta):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def encode(self, bboxes, gt_bboxes):
        """Encode deltas between bboxes and ground truth boxes."""

    @abstractmethod
    def decode(self, bboxes, bboxes_pred):
        """Decode the predicted bboxes according to prediction and base
        boxes."""


class CenterPointBBoxCoder(BaseBBoxCoder):
    def __init__(
        self,
        pc_range,
        out_size_factor,
        voxel_size,
        post_center_range=None,
        max_num=100,
        score_threshold=None,
        code_size=9,
    ):

        self.pc_range = pc_range
        self.out_size_factor = out_size_factor
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.code_size = code_size

    def _gather_feat(self, feats, inds, feat_masks=None):
        dim = feats.size(2)
        inds = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), dim)
        feats = feats.gather(1, inds)
        if feat_masks is not None:
            feat_masks = feat_masks.unsqueeze(2).expand_as(feats)
            feats = feats[feat_masks]
            feats = feats.view(-1, dim)
        return feats

    def _topk(self, scores, K=80):
        batch, cat, height, width = scores.size()

        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

        topk_inds = topk_inds % (height * width)
        topk_ys = (
            (topk_inds.float() / torch.tensor(width, dtype=torch.float)).int().float()
        )
        topk_xs = (topk_inds % width).int().float()

        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
        topk_clses = (topk_ind / torch.tensor(K, dtype=torch.float)).int()
        topk_inds = self._gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(
            batch, K
        )
        topk_ys = self._gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
        topk_xs = self._gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

    def _transpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def encode(self):
        pass

    def decode(self, heat, rot_sine, rot_cosine, hei, dim, vel, reg=None, task_id=-1):
        batch, cat, _, _ = heat.size()

        scores, inds, clses, ys, xs = self._topk(heat, K=self.max_num)

        if reg is not None:
            reg = self._transpose_and_gather_feat(reg, inds)
            reg = reg.view(batch, self.max_num, 2)
            xs = xs.view(batch, self.max_num, 1) + reg[:, :, 0:1]
            ys = ys.view(batch, self.max_num, 1) + reg[:, :, 1:2]
        else:
            xs = xs.view(batch, self.max_num, 1) + 0.5
            ys = ys.view(batch, self.max_num, 1) + 0.5

        rot_sine = self._transpose_and_gather_feat(rot_sine, inds)
        rot_sine = rot_sine.view(batch, self.max_num, 1)

        rot_cosine = self._transpose_and_gather_feat(rot_cosine, inds)
        rot_cosine = rot_cosine.view(batch, self.max_num, 1)
        rot = torch.atan2(rot_sine, rot_cosine)

        hei = self._transpose_and_gather_feat(hei, inds)
        hei = hei.view(batch, self.max_num, 1)

        dim = self._transpose_and_gather_feat(dim, inds)
        dim = dim.view(batch, self.max_num, 3)

        clses = clses.view(batch, self.max_num).float()
        scores = scores.view(batch, self.max_num)

        xs = (
            xs.view(batch, self.max_num, 1) * self.out_size_factor * self.voxel_size[0]
            + self.pc_range[0]
        )
        ys = (
            ys.view(batch, self.max_num, 1) * self.out_size_factor * self.voxel_size[1]
            + self.pc_range[1]
        )

        if vel is None:  # KITTI FORMAT
            final_box_preds = torch.cat([xs, ys, hei, dim, rot], dim=2)
        else:  # exist velocity, nuscene format
            vel = self._transpose_and_gather_feat(vel, inds)
            vel = vel.view(batch, self.max_num, 2)
            final_box_preds = torch.cat([xs, ys, hei, dim, rot, vel], dim=2)

        final_scores = scores
        final_preds = clses

        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold

        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(
                self.post_center_range, device=heat.device
            )
            mask = (final_box_preds[..., :3] >= self.post_center_range[:3]).all(2)
            mask &= (final_box_preds[..., :3] <= self.post_center_range[3:]).all(2)

            predictions_dicts = []
            for i in range(batch):
                cmask = mask[i, :]
                if self.score_threshold:
                    cmask &= thresh_mask[i]

                boxes3d = final_box_preds[i, cmask]
                scores = final_scores[i, cmask]
                labels = final_preds[i, cmask]
                predictions_dict = {
                    "bboxes": boxes3d,
                    "scores": scores,
                    "labels": labels,
                }

                predictions_dicts.append(predictions_dict)

        return predictions_dicts


class CenterHead(BaseModule):
    def __init__(
        self,
        in_channels=[128],
        tasks=None,
        train_cfg=None,
        test_cfg=None,
        bbox_coder=None,
        common_heads=dict(),
        loss_cls=dict(type="GaussianFocalLoss", reduction="mean"),
        loss_bbox=dict(type="L1Loss", reduction="none", loss_weight=0.25),
        separate_head=dict(type="SeparateHead", init_bias=-2.19, final_kernel=3),
        share_conv_channel=64,
        num_heatmap_convs=2,
        conv_cfg=dict(type="Conv2d"),
        norm_cfg=dict(type="BN2d"),
        bias="auto",
        norm_bbox=True,
        init_cfg=None,
    ):
        super(CenterHead, self).__init__(init_cfg=init_cfg)
        num_classes = [len(t["class_names"]) for t in tasks]
        self.class_names = [t["class_names"] for t in tasks]
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.norm_bbox = norm_bbox
        if isinstance(bbox_coder, dict):
            cfg = bbox_coder.copy()
            cfg.pop("type", None)
            self.bbox_coder = CenterPointBBoxCoder(**cfg)
        self.num_anchor_per_locs = [n for n in num_classes]

        self.shared_conv = ConvModule(
            in_channels,
            share_conv_channel,
            kernel_size=3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            bias=bias,
        )

        self.task_heads = nn.ModuleList()

        for num_cls in num_classes:
            heads = copy.deepcopy(common_heads)
            heads.update(dict(heatmap=(num_cls, num_heatmap_convs)))
            separate_head.update(
                in_channels=share_conv_channel, heads=heads, num_cls=num_cls
            )
            sh_cfg = copy.deepcopy(separate_head)
            sh_cfg.pop("type", None)
            self.task_heads.append(SeparateHead(**sh_cfg))

        self.with_velocity = "vel" in common_heads.keys()

    def forward_single(self, x):
        ret_dicts = []
        x = self.shared_conv(x)
        for task in self.task_heads:
            ret_dicts.append(task(x))

        return ret_dicts

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)


class BEVDepthHead(CenterHead):
    def __init__(
        self,
        in_channels=256,
        tasks=None,
        bbox_coder=None,
        common_heads=dict(),
        loss_cls=dict(type="GaussianFocalLoss", reduction="mean"),
        loss_bbox=dict(type="L1Loss", reduction="mean", loss_weight=0.25),
        gaussian_overlap=0.1,
        min_radius=2,
        train_cfg=None,
        test_cfg=None,
        bev_backbone_conf=bev_backbone_conf,
        bev_neck_conf=bev_neck_conf,
        separate_head=dict(type="SeparateHead", init_bias=-2.19, final_kernel=3),
    ):
        super(BEVDepthHead, self).__init__(
            in_channels=in_channels,
            tasks=tasks,
            bbox_coder=bbox_coder,
            common_heads=common_heads,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            separate_head=separate_head,
        )
        if isinstance(bev_backbone_conf, dict):
            cfg = bev_backbone_conf.copy()
            cfg.pop("type", None)
            self.trunk = ResNet(**cfg)
        self.trunk.init_weights()
        if isinstance(bev_neck_conf, dict):
            cfg = bev_neck_conf.copy()
            cfg.pop("type", None)
            self.neck = SECONDFPN(**cfg)
        self.neck.init_weights()
        del self.trunk.maxpool
        self.gaussian_overlap = gaussian_overlap
        self.min_radius = min_radius
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def forward(self, x):
        x = x.float()
        trunk_outs = [x]
        if self.trunk.deep_stem:
            x = self.trunk.stem(x)
        else:
            x = self.trunk.conv1(x)
            x = self.trunk.norm1(x)
            x = self.trunk.relu(x)
        for i, layer_name in enumerate(self.trunk.res_layers):
            res_layer = getattr(self.trunk, layer_name)
            x = res_layer(x)
            if i in self.trunk.out_indices:
                trunk_outs.append(x)
        fpn_output = self.neck(trunk_outs)
        ret_values = super().forward(fpn_output)
        return ret_values


class BaseBEVDepth(nn.Module):
    def __init__(self, backbone_conf, head_conf, is_train_depth=False):
        super(BaseBEVDepth, self).__init__()
        self.backbone = BaseLSSFPN(**backbone_conf)
        self.head = BEVDepthHead(**head_conf)

    def forward(
        self,
        x,
        mats_dict,
        timestamps=None,
    ):
        x = self.backbone(x, mats_dict, timestamps)
        preds = self.head(x)
        return preds


def create_bevdepth_model(
    variant: str = "bev_depth_lss_r50_256x704_128x128_24e_2key",
    is_train_depth: bool = False,
):
    """
    Create a BEVDepth model with variant-specific configuration.

    Args:
        variant (str): One of the supported BEVDepth variants:
            - "bev_depth_lss_r50_256x704_128x128_24e_2key" (base)
            - "bev_depth_lss_r50_256x704_128x128_24e_2key_ema"
            - "bev_depth_lss_r50_256x704_128x128_20e_cbgs_2key_da"
            - "bev_depth_lss_r50_256x704_128x128_20e_cbgs_2key_da_ema"
        is_train_depth (bool): Whether to enable depth training mode

    Returns:
        BaseBEVDepth: Configured model instance
        dict: Complete configuration used for the model
    """
    config = get_bevdepth_config(variant)
    model = BaseBEVDepth(
        backbone_conf=config["backbone_conf"],
        head_conf=config["head_conf"],
        is_train_depth=is_train_depth,
    )
    return model
