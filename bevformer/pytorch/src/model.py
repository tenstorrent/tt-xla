# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BEVFormer model implementation

Apdapted from: https://github.com/fundamentalvision/BEVFormer.git

Apache-2.0 License: https://github.com/fundamentalvision/BEVFormer/blob/master/LICENSE
"""
import copy
import math
import re
from typing import Tuple, Union
from abc import ABCMeta, abstractmethod

import numpy as np
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn.modules.utils import _pair, _single
from torchvision.transforms.functional import rotate
from addict import Dict
from collections import OrderedDict
import torch.utils.checkpoint as cp
from PIL import Image
from detectron2.layers import Conv2d, get_norm, ShapeSpec
from packaging.version import parse

# ============================================================================
# MODEL CONFIG
# ============================================================================


def get_bevformer_v2_model(variant_str):

    base_config = {
        "point_cloud_range": [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        "class_names": [
            "barrier",
            "bicycle",
            "bus",
            "car",
            "construction_vehicle",
            "motorcycle",
            "pedestrian",
            "traffic_cone",
            "trailer",
            "truck",
        ],
        "bev_h_": 200,
        "bev_w_": 200,
        "frames": (0,),
        "voxel_size": [102.4 / 200, 102.4 / 200, 8],
        "img_norm_cfg": dict(
            mean=[103.53, 116.28, 123.675], std=[1, 1, 1], to_rgb=False
        ),
        "_dim_": 256,
        "_pos_dim_": 128,
        "_ffn_dim_": 512,
        "_num_levels_": 4,
        "_num_mono_levels_": 5,
        "ida_aug_conf": {
            "reisze": [
                640,
            ],
            "crop": (0, 260, 1600, 900),
            "H": 900,
            "W": 1600,
            "rand_flip": False,
        },
        "ida_aug_conf_eval": {
            "reisze": [
                640,
            ],
            "crop": (0, 260, 1600, 900),
            "H": 900,
            "W": 1600,
            "rand_flip": False,
        },
    }

    if variant_str == "bevformerv2-r50-t1-base":
        frames = base_config["frames"]
        group_detr = None
        pts_bbox_head_type = "BEVFormerHead"
        self_attn_type = "MultiheadAttention"

    elif variant_str == "bevformerv2-r50-t1":
        base_config["frames"] = (0,)
        base_config["group_detr"] = 11
        base_config["ida_aug_conf"] = {
            "reisze": [512, 544, 576, 608, 640, 672, 704, 736, 768],
            "crop": (0, 260, 1600, 900),
            "H": 900,
            "W": 1600,
            "rand_flip": True,
        }
        frames = base_config["frames"]
        group_detr = base_config["group_detr"]
        pts_bbox_head_type = "BEVFormerHead_GroupDETR"
        self_attn_type = "GroupMultiheadAttention"

    elif variant_str == "bevformerv2-r50-t2":
        base_config["frames"] = (
            -1,
            0,
        )
        base_config["group_detr"] = 11
        base_config["ida_aug_conf"] = {
            "reisze": [512, 544, 576, 608, 640, 672, 704, 736, 768],
            "crop": (0, 260, 1600, 900),
            "H": 900,
            "W": 1600,
            "rand_flip": True,
        }
        frames = base_config["frames"]
        group_detr = base_config["group_detr"]
        pts_bbox_head_type = "BEVFormerHead_GroupDETR"
        self_attn_type = "GroupMultiheadAttention"

    elif variant_str == "bevformerv2-r50-t8":
        base_config["frames"] = (-7, -6, -5, -4, -3, -2, -1, 0)
        base_config["group_detr"] = 11
        base_config["ida_aug_conf"] = {
            "reisze": [512, 544, 576, 608, 640, 672, 704, 736, 768],
            "crop": (0, 260, 1600, 900),
            "H": 900,
            "W": 1600,
            "rand_flip": True,
        }
        frames = base_config["frames"]
        group_detr = base_config["group_detr"]
        pts_bbox_head_type = "BEVFormerHead_GroupDETR"
        self_attn_type = "GroupMultiheadAttention"

    img_backbone = {
        "type": "ResNet",
        "depth": 50,
        "num_stages": 4,
        "out_indices": (1, 2, 3),
        "frozen_stages": -1,
        "norm_cfg": {"type": "SyncBN"},
        "norm_eval": False,
        "style": "caffe",
    }

    img_neck = {
        "type": "FPN",
        "in_channels": [512, 1024, 2048],
        "out_channels": base_config["_dim_"],
        "start_level": 0,
        "add_extra_convs": "on_output",
        "num_outs": base_config["_num_mono_levels_"],
        "relu_before_extra_convs": True,
    }

    if pts_bbox_head_type == "BEVFormerHead":
        pts_bbox_head = _get_base_pts_bbox_head(base_config)
    else:
        pts_bbox_head = _get_group_detr_pts_bbox_head(
            base_config, group_detr, self_attn_type
        )

    fcos3d_bbox_head = _get_fcos3d_bbox_head(base_config)
    return img_backbone, pts_bbox_head, img_neck, fcos3d_bbox_head, frames


def _get_base_pts_bbox_head(config):
    """Get base BEVFormerHead configuration"""
    return {
        "type": "BEVFormerHead",
        "bev_h": config["bev_h_"],
        "bev_w": config["bev_w_"],
        "num_query": 900,
        "num_classes": 10,
        "in_channels": config["_dim_"],
        "sync_cls_avg_factor": True,
        "with_box_refine": True,
        "as_two_stage": False,
        "transformer": {
            "type": "PerceptionTransformerV2",
            "embed_dims": config["_dim_"],
            "frames": config["frames"],
            "encoder": {
                "type": "BEVFormerEncoder",
                "num_layers": 6,
                "pc_range": config["point_cloud_range"],
                "num_points_in_pillar": 4,
                "return_intermediate": False,
                "transformerlayers": {
                    "type": "BEVFormerLayer",
                    "attn_cfgs": [
                        {
                            "type": "TemporalSelfAttention",
                            "embed_dims": config["_dim_"],
                            "num_levels": 1,
                        },
                        {
                            "type": "SpatialCrossAttention",
                            "pc_range": config["point_cloud_range"],
                            "deformable_attention": {
                                "type": "MSDeformableAttention3D",
                                "embed_dims": config["_dim_"],
                                "num_points": 8,
                                "num_levels": 4,
                            },
                            "embed_dims": config["_dim_"],
                        },
                    ],
                    "feedforward_channels": config["_ffn_dim_"],
                    "ffn_dropout": 0.1,
                    "operation_order": (
                        "self_attn",
                        "norm",
                        "cross_attn",
                        "norm",
                        "ffn",
                        "norm",
                    ),
                },
            },
            "decoder": {
                "type": "DetectionTransformerDecoder",
                "num_layers": 6,
                "return_intermediate": True,
                "transformerlayers": {
                    "type": "DetrTransformerDecoderLayer",
                    "attn_cfgs": [
                        {
                            "type": "MultiheadAttention",
                            "embed_dims": config["_dim_"],
                            "num_heads": 8,
                            "dropout": 0.1,
                        },
                        {
                            "type": "CustomMSDeformableAttention",
                            "embed_dims": config["_dim_"],
                            "num_levels": 1,
                        },
                    ],
                    "feedforward_channels": config["_ffn_dim_"],
                    "ffn_dropout": 0.1,
                    "operation_order": (
                        "self_attn",
                        "norm",
                        "cross_attn",
                        "norm",
                        "ffn",
                        "norm",
                    ),
                },
            },
        },
        "bbox_coder": {
            "type": "NMSFreeCoder",
            "post_center_range": [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            "pc_range": config["point_cloud_range"],
            "max_num": 300,
            "voxel_size": config["voxel_size"],
            "num_classes": 10,
        },
        "positional_encoding": {
            "type": "LearnedPositionalEncoding",
            "num_feats": config["_pos_dim_"],
            "row_num_embed": config["bev_h_"],
            "col_num_embed": config["bev_w_"],
        },
        "loss_cls": {
            "type": "FocalLoss",
            "use_sigmoid": True,
            "gamma": 2.0,
            "alpha": 0.25,
            "loss_weight": 2.0,
        },
        "loss_bbox": {"type": "SmoothL1Loss", "loss_weight": 0.75, "beta": 1.0},
        "loss_iou": {"type": "GIoULoss", "loss_weight": 0.0},
    }


def _get_group_detr_pts_bbox_head(config, group_detr, self_attn_type):
    """Get GroupDETR variant of BEVFormerHead"""
    pts_bbox_head = _get_base_pts_bbox_head(config)
    pts_bbox_head["type"] = "BEVFormerHead_GroupDETR"
    pts_bbox_head["group_detr"] = group_detr

    pts_bbox_head["transformer"]["decoder"]["transformerlayers"]["attn_cfgs"][0] = {
        "type": self_attn_type,
        "group": group_detr,
        "embed_dims": config["_dim_"],
        "num_heads": 8,
        "dropout": 0.1,
    }

    if len(config["frames"]) > 2:
        pts_bbox_head["transformer"]["inter_channels"] = config["_dim_"] * 2

    return pts_bbox_head


def _get_fcos3d_bbox_head(config):
    """Get FCOS3D bbox head configuration"""
    return {
        "type": "NuscenesDD3D",
        "num_classes": 10,
        "in_channels": config["_dim_"],
        "strides": [8, 16, 32, 64, 128],
        "box3d_on": True,
        "feature_locations_offset": "none",
        "fcos2d_cfg": {
            "num_cls_convs": 4,
            "num_box_convs": 4,
            "norm": "SyncBN",
            "use_deformable": False,
            "use_scale": True,
            "box2d_scale_init_factor": 1.0,
        },
        "fcos2d_loss_cfg": {
            "focal_loss_alpha": 0.25,
            "focal_loss_gamma": 2.0,
            "loc_loss_type": "giou",
        },
        "fcos3d_cfg": {
            "num_convs": 4,
            "norm": "SyncBN",
            "use_scale": True,
            "depth_scale_init_factor": 0.3,
            "proj_ctr_scale_init_factor": 1.0,
            "use_per_level_predictors": False,
            "class_agnostic": False,
            "use_deformable": False,
            "mean_depth_per_level": [44.921, 20.252, 11.712, 7.166, 8.548],
            "std_depth_per_level": [24.331, 9.833, 6.223, 4.611, 8.275],
        },
        "fcos3d_loss_cfg": {
            "min_depth": 0.1,
            "max_depth": 80.0,
            "box3d_loss_weight": 2.0,
            "conf3d_loss_weight": 1.0,
            "conf_3d_temperature": 1.0,
            "smooth_l1_loss_beta": 0.05,
            "max_loss_per_group": 20,
            "predict_allocentric_rot": True,
            "scale_depth_by_focal_lengths": True,
            "scale_depth_by_focal_lengths_factor": 500.0,
            "class_agnostic": False,
            "predict_distance": False,
            "canon_box_sizes": [
                [2.3524184, 0.5062202, 1.0413622],
                [0.61416006, 1.7016163, 1.3054738],
                [2.9139307, 10.725025, 3.2832346],
                [1.9751819, 4.641267, 1.74352],
                [2.772134, 6.565072, 3.2474296],
                [0.7800532, 2.138673, 1.4437162],
                [0.6667362, 0.7181772, 1.7616143],
                [0.40246472, 0.4027083, 1.0084083],
                [3.0059454, 12.8197, 4.1213827],
                [2.4986045, 6.9310856, 2.8382742],
            ],
        },
        "target_assign_cfg": {
            "center_sample": True,
            "pos_radius": 1.5,
            "sizes_of_interest": [
                (-1, 64),
                (64, 128),
                (128, 256),
                (256, 512),
                (512, 1e8),
            ],
        },
        "nusc_loss_weight": {"attr_loss_weight": 0.2, "speed_loss_weight": 0.2},
    }


class BEVFormerBaseConfig:
    """Base configuration for BEVFormer model components"""

    point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    voxel_size = [0.2, 0.2, 8]

    class_names = [
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

    _dim_ = 256
    _pos_dim_ = _dim_ // 2
    _ffn_dim_ = _dim_ * 2
    _num_levels_ = 4
    bev_h_ = 200
    bev_w_ = 200
    queue_length = 4

    img_norm_cfg = dict(
        mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False
    )

    @classmethod
    def get_img_backbone(cls, variant="base"):
        """Get image backbone configuration"""
        if variant == "BEVFormer-tiny":
            return dict(
                type="ResNet",
                depth=50,
                num_stages=4,
                out_indices=(3,),
                frozen_stages=1,
                norm_cfg=dict(type="BN", requires_grad=False),
                norm_eval=True,
                style="pytorch",
            )
        elif variant == "BEVFormer-small":
            return dict(
                type="ResNet",
                depth=101,
                num_stages=4,
                out_indices=(3,),
                frozen_stages=1,
                norm_cfg=dict(type="BN2d", requires_grad=False),
                norm_eval=True,
                style="caffe",
                with_cp=True,
                dcn=dict(type="DCNv2", deform_groups=1, fallback_on_stride=False),
                stage_with_dcn=(False, False, True, True),
            )
        else:  # base
            return dict(
                type="ResNet",
                depth=101,
                num_stages=4,
                out_indices=(1, 2, 3),
                frozen_stages=1,
                norm_cfg=dict(type="BN2d", requires_grad=False),
                norm_eval=True,
                style="caffe",
                dcn=dict(type="DCNv2", deform_groups=1, fallback_on_stride=False),
                stage_with_dcn=(False, False, True, True),
            )

    @classmethod
    def get_img_neck(cls, variant="base"):
        """Get image neck configuration"""
        if variant == "BEVFormer-tiny":
            return dict(
                type="FPN",
                in_channels=[2048],
                out_channels=cls._dim_,
                start_level=0,
                add_extra_convs="on_output",
                num_outs=1,
                relu_before_extra_convs=True,
            )
        elif variant == "BEVFormer-small":
            return dict(
                type="FPN",
                in_channels=[2048],
                out_channels=cls._dim_,
                start_level=0,
                add_extra_convs="on_output",
                num_outs=1,
                relu_before_extra_convs=True,
            )
        else:  # base
            return dict(
                type="FPN",
                in_channels=[512, 1024, 2048],
                out_channels=cls._dim_,
                start_level=0,
                add_extra_convs="on_output",
                num_outs=4,
                relu_before_extra_convs=True,
            )

    @classmethod
    def get_pts_bbox_head(cls, variant="base"):
        params = cls._get_variant_params(variant)

        bbox_head = dict(
            type="BEVFormerHead",
            bev_h=params["bev_h"],
            bev_w=params["bev_w"],
            num_query=900,
            num_classes=10,
            in_channels=cls._dim_,
            sync_cls_avg_factor=True,
            with_box_refine=True,
            as_two_stage=False,
            transformer=dict(
                type="PerceptionTransformer",
                rotate_prev_bev=True,
                use_shift=True,
                use_can_bus=True,
                embed_dims=cls._dim_,
                encoder=dict(
                    type="BEVFormerEncoder",
                    num_layers=params["encoder_layers"],
                    pc_range=cls.point_cloud_range,
                    num_points_in_pillar=4,
                    return_intermediate=False,
                    transformerlayers=dict(
                        type="BEVFormerLayer",
                        attn_cfgs=[
                            dict(
                                type="TemporalSelfAttention",
                                embed_dims=cls._dim_,
                                num_levels=1,
                            ),
                            dict(
                                type="SpatialCrossAttention",
                                pc_range=cls.point_cloud_range,
                                deformable_attention=dict(
                                    type="MSDeformableAttention3D",
                                    embed_dims=cls._dim_,
                                    num_points=8,
                                    num_levels=params["num_levels"],
                                ),
                                embed_dims=cls._dim_,
                            ),
                        ],
                        feedforward_channels=cls._ffn_dim_,
                        ffn_dropout=0.1,
                        operation_order=(
                            "self_attn",
                            "norm",
                            "cross_attn",
                            "norm",
                            "ffn",
                            "norm",
                        ),
                    ),
                ),
                decoder=dict(
                    type="DetectionTransformerDecoder",
                    num_layers=6,
                    return_intermediate=True,
                    transformerlayers=dict(
                        type="DetrTransformerDecoderLayer",
                        attn_cfgs=[
                            dict(
                                type="MultiheadAttention",
                                embed_dims=cls._dim_,
                                num_heads=8,
                                dropout=0.1,
                            ),
                            dict(
                                type="CustomMSDeformableAttention",
                                embed_dims=cls._dim_,
                                num_levels=1,
                            ),
                        ],
                        feedforward_channels=cls._ffn_dim_,
                        ffn_dropout=0.1,
                        operation_order=(
                            "self_attn",
                            "norm",
                            "cross_attn",
                            "norm",
                            "ffn",
                            "norm",
                        ),
                    ),
                ),
            ),
            bbox_coder=dict(
                type="NMSFreeCoder",
                post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                pc_range=cls.point_cloud_range,
                max_num=300,
                voxel_size=cls.voxel_size,
                num_classes=10,
            ),
            positional_encoding=dict(
                type="LearnedPositionalEncoding",
                num_feats=cls._pos_dim_,
                row_num_embed=params["bev_h"],
                col_num_embed=params["bev_w"],
            ),
            loss_cls=dict(
                type="FocalLoss",
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=2.0,
            ),
            loss_bbox=dict(type="L1Loss", loss_weight=0.25),
            loss_iou=dict(type="GIoULoss", loss_weight=0.0),
        )

        return bbox_head

    @classmethod
    def _get_variant_params(cls, variant):
        if variant == "BEVFormer-tiny":
            return {
                "bev_h": 50,
                "bev_w": 50,
                "encoder_layers": 3,
                "num_levels": 1,
                "queue_length": 3,
            }
        elif variant == "BEVFormer-small":
            return {
                "bev_h": 150,
                "bev_w": 150,
                "encoder_layers": 3,
                "num_levels": 1,
                "queue_length": 3,
            }
        else:  # base
            return {
                "bev_h": 200,
                "bev_w": 200,
                "encoder_layers": 6,
                "num_levels": 4,
                "queue_length": 4,
            }

    @classmethod
    def get_img_norm_cfg(cls, variant="base"):
        """Get image normalization configuration"""
        if variant == "BEVFormer-tiny":
            return dict(
                mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
            )
        else:  # BEVFormer-small and BEVFormer-base
            return cls.img_norm_cfg


def get_bevformer_model(variant="BEVFormer-tiny"):

    img_backbone = BEVFormerBaseConfig.get_img_backbone(variant)
    img_neck = BEVFormerBaseConfig.get_img_neck(variant)
    pts_bbox_head = BEVFormerBaseConfig.get_pts_bbox_head(variant)

    return img_backbone, pts_bbox_head, img_neck


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


def load_state_dict(module, state_dict, strict=False, logger=None):
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            True,
            all_missing_keys,
            unexpected_keys,
            err_msg,
        )
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    load(module)
    load = None  # break load->load reference cycle


class CheckpointLoader:
    _schemes = {}

    @classmethod
    def _register_scheme(cls, prefixes, loader, force=False):
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        for prefix in prefixes:
            if (prefix not in cls._schemes) or force:
                cls._schemes[prefix] = loader
        cls._schemes = OrderedDict(
            sorted(cls._schemes.items(), key=lambda t: t[0], reverse=True)
        )

    @classmethod
    def register_scheme(cls, prefixes, loader=None, force=False):

        if loader is not None:
            cls._register_scheme(prefixes, loader, force=force)
            return

        def _register(loader_cls):
            cls._register_scheme(prefixes, loader_cls, force=force)
            return loader_cls

        return _register

    @classmethod
    def _get_checkpoint_loader(cls, path):

        for p in cls._schemes:
            if path.startswith(p):
                return cls._schemes[p]

    @classmethod
    def load_checkpoint(cls, filename, map_location=None, logger=None):

        checkpoint_loader = cls._get_checkpoint_loader(filename)
        class_name = checkpoint_loader.__name__
        print(f"load checkpoint from {class_name[10:]} path: {filename}")
        return checkpoint_loader(filename, map_location)


@CheckpointLoader.register_scheme(prefixes="")
def load_from_local(filename, map_location):
    checkpoint = torch.load(filename, map_location=map_location)
    return checkpoint


def _load_checkpoint(filename, map_location=None, logger=None):
    return CheckpointLoader.load_checkpoint(filename, map_location, logger)


def load_checkpoint_bev(
    model,
    filename,
    map_location=None,
    strict=False,
    logger=None,
    revise_keys=[(r"^module\.", "")],
):
    checkpoint = _load_checkpoint(filename, map_location, logger)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    metadata = getattr(state_dict, "_metadata", OrderedDict())
    for p, r in revise_keys:
        state_dict = OrderedDict({re.sub(p, r, k): v for k, v in state_dict.items()})
    state_dict._metadata = metadata

    load_state_dict(model, state_dict, strict, logger)
    return checkpoint


class BaseModule(nn.Module, metaclass=ABCMeta):
    def __init__(self, init_cfg=None):
        super(BaseModule, self).__init__()
        self._is_init = False
        self.init_cfg = copy.deepcopy(init_cfg)


def bbox3d2result(bboxes, scores, labels, attrs=None):
    result_dict = dict(
        boxes_3d=bboxes.to("cpu"), scores_3d=scores.cpu(), labels_3d=labels.cpu()
    )

    if attrs is not None:
        result_dict["attrs_3d"] = attrs.cpu()

    return result_dict


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class Sequential(BaseModule, nn.Sequential):
    def __init__(self, *args, init_cfg=None):
        BaseModule.__init__(self, init_cfg)
        nn.Sequential.__init__(self, *args)


class ModuleList(BaseModule, nn.ModuleList):
    def __init__(self, modules=None, init_cfg=None):
        BaseModule.__init__(self, init_cfg)
        nn.ModuleList.__init__(self, modules)


def build_conv_layer(cfg, *args, **kwargs):
    if cfg is None:
        cfg_ = dict(type="Conv2d")
    else:
        if not isinstance(cfg, dict):
            raise TypeError("cfg must be a dict")
        if "type" not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        cfg_ = cfg.copy()

    layer_type = cfg_.pop("type")

    if layer_type in ("Conv2d", "Conv"):
        layer_cls = nn.Conv2d
    elif layer_type == "Conv1d":
        layer_cls = nn.Conv1d
    elif layer_type == "Conv3d":
        layer_cls = nn.Conv3d
    elif layer_type in ("DCNv2", "DCN"):
        layer_cls = ModulatedDeformConv2dPackCPU

    return layer_cls(*args, **kwargs, **cfg_)


class BaseDetector(BaseModule, metaclass=ABCMeta):
    def __init__(self, init_cfg=None):
        super(BaseDetector, self).__init__(init_cfg)

    def forward(self, img, img_metas, return_loss=True, **kwargs):
        return self.forward_test(img, img_metas, **kwargs)


class Base3DDetector(BaseDetector):
    def forward(self, return_loss=True, **kwargs):
        return self.forward_test(**kwargs)


# ============================================================================
# DETECTORS
# ============================================================================


class MVXTwoStageDetector(Base3DDetector):
    def __init__(
        self,
        pts_voxel_layer=None,
        pts_voxel_encoder=None,
        pts_middle_encoder=None,
        pts_fusion_layer=None,
        img_backbone=None,
        pts_backbone=None,
        img_neck=None,
        pts_neck=None,
        pts_bbox_head=None,
        img_roi_head=None,
        img_rpn_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        init_cfg=None,
    ):
        super(MVXTwoStageDetector, self).__init__(init_cfg=init_cfg)

        if pts_bbox_head:
            pts_train_cfg = train_cfg.pts if train_cfg else None
            pts_bbox_head.update(train_cfg=pts_train_cfg)
            pts_test_cfg = test_cfg.pts if test_cfg else None
            pts_bbox_head.update(test_cfg=pts_test_cfg)
            trans_type = pts_bbox_head.get("type", None)
            if trans_type == "BEVFormerHead_GroupDETR":
                cfg = pts_bbox_head.copy()
                self.pts_bbox_head = BEVFormerHead_GroupDETR(**cfg)
            else:
                cfg = pts_bbox_head.copy()
                self.pts_bbox_head = BEVFormerHead(**cfg)
        if img_backbone:
            if isinstance(img_backbone, dict):
                cfg = img_backbone.copy()
                cfg.pop("type", None)
                self.img_backbone = ResNet(**cfg)
        if img_neck is not None:
            if isinstance(img_neck, dict):
                cfg = img_neck.copy()
                cfg.pop("type", None)
                self.img_neck = FPN(**cfg)
        self.test_cfg = test_cfg

    @property
    def with_img_neck(self):
        return hasattr(self, "img_neck") and self.img_neck is not None


class GridMask(nn.Module):
    def __init__(
        self, use_h, use_w, rotate=1, offset=False, ratio=0.5, mode=0, prob=1.0
    ):
        super(GridMask, self).__init__()
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob
        self.fp16_enable = False

    def forward(self, x):
        if np.random.rand() > self.prob or not self.training:
            return x
        n, c, h, w = x.size()
        x = x.view(-1, h, w)
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(2, h)
        self.l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.float32)

        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[
            (hh - h) // 2 : (hh - h) // 2 + h, (ww - w) // 2 : (ww - w) // 2 + w
        ]

        mask = torch.from_numpy(mask).to(x.dtype).cuda()
        if self.mode == 1:
            mask = 1 - mask
        mask = mask.expand_as(x)
        x = x * mask.cpu()

        return x.view(n, c, h, w)


class BaseInstance3DBoxes(object):
    def __init__(self, tensor, box_dim=7, with_yaw=True, origin=(0.5, 0.5, 0)):
        if isinstance(tensor, torch.Tensor):
            device = tensor.device
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            tensor = tensor.reshape((0, box_dim)).to(dtype=torch.float32, device=device)
        self.box_dim = box_dim
        self.with_yaw = with_yaw
        self.tensor = tensor.clone()

        if origin != (0.5, 0.5, 0):
            dst = self.tensor.new_tensor((0.5, 0.5, 0))
            src = self.tensor.new_tensor(origin)
            self.tensor[:, :3] += self.tensor[:, 3:6] * (dst - src)

    def __repr__(self):
        return self.__class__.__name__ + "(\n    " + str(self.tensor) + ")"

    def to(self, device):
        original_type = type(self)
        return original_type(
            self.tensor.to(device), box_dim=self.box_dim, with_yaw=self.with_yaw
        )


class LiDARInstance3DBoxes(BaseInstance3DBoxes):
    @property
    def bev(self):
        return self.tensor[:, [0, 1, 3, 4, 6]]


class BEVFormer(MVXTwoStageDetector):
    def __init__(
        self,
        use_grid_mask=False,
        pts_voxel_layer=None,
        pts_voxel_encoder=None,
        pts_middle_encoder=None,
        pts_fusion_layer=None,
        img_backbone=None,
        pts_backbone=None,
        img_neck=None,
        pts_neck=None,
        pts_bbox_head=None,
        img_roi_head=None,
        img_rpn_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        video_test_mode=False,
    ):

        super(BEVFormer, self).__init__(
            pts_voxel_layer,
            pts_voxel_encoder,
            pts_middle_encoder,
            pts_fusion_layer,
            img_backbone,
            pts_backbone,
            img_neck,
            pts_neck,
            pts_bbox_head,
            img_roi_head,
            img_rpn_head,
            train_cfg,
            test_cfg,
            pretrained,
        )
        self.use_grid_mask = use_grid_mask
        self.video_test_mode = video_test_mode
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
        )
        self.use_grid_mask = use_grid_mask
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            "prev_bev": None,
            "scene_token": None,
            "prev_pos": 0,
            "prev_angle": 0,
        }

    def extract_img_feat(self, img, img_metas, len_queue=None):
        B = img.size(0)
        if img is not None:
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(
                    img_feat.view(int(B / len_queue), len_queue, int(BN / B), C, H, W)
                )
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    def extract_feat(self, img, img_metas=None, len_queue=None):
        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        return img_feats

    def forward(self, return_loss=False, **kwargs):
        return self.forward_test(**kwargs)

    def forward_test(self, img_metas, img=None, **kwargs):

        img = [img] if img is None else img
        if img_metas[0][0]["scene_token"] != self.prev_frame_info["scene_token"]:
            self.prev_frame_info["prev_bev"] = None
        self.prev_frame_info["scene_token"] = img_metas[0][0]["scene_token"]

        if not self.video_test_mode:
            self.prev_frame_info["prev_bev"] = None

        tmp_pos = copy.deepcopy(img_metas[0][0]["can_bus"][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]["can_bus"][-1])
        if self.prev_frame_info["prev_bev"] is not None:
            img_metas[0][0]["can_bus"][:3] -= self.prev_frame_info["prev_pos"]
            img_metas[0][0]["can_bus"][-1] -= self.prev_frame_info["prev_angle"]
        else:
            img_metas[0][0]["can_bus"][-1] = 0
            img_metas[0][0]["can_bus"][:3] = 0

        new_prev_bev, bbox_results = self.simple_test(
            img_metas[0], img[0], prev_bev=self.prev_frame_info["prev_bev"], **kwargs
        )
        self.prev_frame_info["prev_pos"] = tmp_pos
        self.prev_frame_info["prev_angle"] = tmp_angle
        self.prev_frame_info["prev_bev"] = new_prev_bev
        return bbox_results

    def simple_test_pts(self, x, img_metas, prev_bev=None, rescale=False):
        outs = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev)
        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return outs["bev_embed"], bbox_results

    def simple_test(self, img_metas, img=None, prev_bev=None, rescale=False):
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, bbox_pts = self.simple_test_pts(
            img_feats, img_metas, prev_bev, rescale=rescale
        )
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict["pts_bbox"] = pts_bbox
        return new_prev_bev, bbox_list


class Linear(torch.nn.Linear):
    def forward(self, x):
        return super().forward(x)


class BEVFormerV2(MVXTwoStageDetector):
    def __init__(
        self,
        use_grid_mask=False,
        pts_voxel_layer=None,
        pts_voxel_encoder=None,
        pts_middle_encoder=None,
        pts_fusion_layer=None,
        img_backbone=None,
        pts_backbone=None,
        img_neck=None,
        pts_neck=None,
        pts_bbox_head=None,
        fcos3d_bbox_head=None,
        img_roi_head=None,
        img_rpn_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        video_test_mode=False,
        num_levels=None,
        num_mono_levels=None,
        mono_loss_weight=1.0,
        frames=(0,),
    ):

        super(BEVFormerV2, self).__init__(
            pts_voxel_layer,
            pts_voxel_encoder,
            pts_middle_encoder,
            pts_fusion_layer,
            img_backbone,
            pts_backbone,
            img_neck,
            pts_neck,
            pts_bbox_head,
            img_roi_head,
            img_rpn_head,
            train_cfg,
            test_cfg,
            pretrained,
        )
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
        )
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False
        self.video_test_mode = video_test_mode

        if isinstance(fcos3d_bbox_head, dict):
            cfg = fcos3d_bbox_head.copy()
            cfg.pop("type", None)
            self.fcos3d_bbox_head = NuscenesDD3D(**cfg)
        self.mono_loss_weight = mono_loss_weight
        self.num_levels = num_levels
        self.num_mono_levels = num_mono_levels
        self.frames = frames

    def extract_img_feat(self, img):
        B = img.size(0)
        if img is not None:
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    def extract_feat(self, img, img_metas, len_queue=None):
        img_feats = self.extract_img_feat(img)
        if (
            "aug_param" in img_metas[0]
            and img_metas[0]["aug_param"]["CropResizeFlipImage_param"][-1] is True
        ):
            img_feats = [
                torch.flip(
                    x,
                    dims=[
                        -1,
                    ],
                )
                for x in img_feats
            ]
        return img_feats

    def forward(self, return_loss=False, **kwargs):
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def obtain_history_bev(self, img_dict, img_metas_dict):
        is_training = self.training
        self.eval()
        prev_bev = OrderedDict({i: None for i in self.frames})
        with torch.no_grad():
            for t in img_dict.keys():
                img = img_dict[t]
                img_metas = [
                    img_metas_dict[t],
                ]
                img_feats = self.extract_feat(img=img, img_metas=img_metas)
                if self.num_levels:
                    img_feats = img_feats[: self.num_levels]
                bev = self.pts_bbox_head(img_feats, img_metas, None, only_bev=True)
                prev_bev[t] = bev.detach()
        if is_training:
            self.train()
        return list(prev_bev.values())

    def forward_test(self, img_metas, img=None, **kwargs):

        img = [img] if img is None else img
        new_prev_bev, bbox_results = self.simple_test(
            img_metas[0], img[0], prev_bev=None, **kwargs
        )
        return bbox_results

    def simple_test_pts(self, x, img_metas, prev_bev=None, rescale=False):
        outs = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev)

        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return outs["bev_embed"], bbox_results

    def simple_test(self, img_metas, img=None, prev_bev=None, rescale=False, **kwargs):
        img_metas = OrderedDict(sorted(img_metas[0].items()))
        img_dict = {}
        for ind, t in enumerate(img_metas.keys()):
            img_dict[t] = img[:, ind, ...]
        img = img_dict[0]
        img_dict.pop(0)

        prev_img_metas = copy.deepcopy(img_metas)
        prev_bev = self.obtain_history_bev(img_dict, prev_img_metas)

        img_metas = [
            img_metas[0],
        ]
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        if self.num_levels:
            img_feats = img_feats[: self.num_levels]

        bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, bbox_pts = self.simple_test_pts(
            img_feats, img_metas, prev_bev, rescale=rescale
        )
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict["pts_bbox"] = pts_bbox
        return new_prev_bev, bbox_list


class ModulatedDeformConv2dPackCPU(nn.Module):
    """Standalone CPU-only DCNv2 using torchvision's deform_conv2d.

    Note: torchvision's deform_conv2d currently supports groups=1 only.
    """

    _version = 2

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]],
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        deform_groups: int = 1,
        bias: Union[bool, str] = True,
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
        self.transposed = False
        self.output_padding = _single(0)

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, *self.kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.deform_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True,
        )

        self.init_weights()

    def init_weights(self) -> None:
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()
        if hasattr(self, "conv_offset"):
            self.conv_offset.weight.data.zero_()
            self.conv_offset.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        out = self.conv_offset(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        return torchvision.ops.deform_conv2d(
            input=x,
            offset=offset,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask,
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            if (
                prefix + "conv_offset.weight" not in state_dict
                and prefix[:-1] + "_offset.weight" in state_dict
            ):
                state_dict[prefix + "conv_offset.weight"] = state_dict.pop(
                    prefix[:-1] + "_offset.weight"
                )
            if (
                prefix + "conv_offset.bias" not in state_dict
                and prefix[:-1] + "_offset.bias" in state_dict
            ):
                state_dict[prefix + "conv_offset.bias"] = state_dict.pop(
                    prefix[:-1] + "_offset.bias"
                )

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


def build_norm_hardcoded(cfg, num_features, postfix=None):
    cfg = cfg.copy()
    layer_type = cfg.pop("type")
    requires_grad = cfg.pop("requires_grad", None)
    if layer_type in ("LN", "LayerNorm"):
        eps = cfg.pop("eps", 1e-5)
        elementwise_affine = cfg.pop("elementwise_affine", True)
        module = nn.LayerNorm(
            num_features, eps=eps, elementwise_affine=elementwise_affine
        )
        abbr = "ln"
    elif layer_type in ("BN", "BN2d"):
        module = nn.BatchNorm2d(num_features, **cfg)
        abbr = "bn"
    elif layer_type == "SyncBN":
        module = nn.SyncBatchNorm(num_features, **cfg)
        abbr = "bn"
    if requires_grad is not None:
        for p in module.parameters():
            p.requires_grad = requires_grad
    if postfix is None:
        return module
    name = f"{abbr}{postfix}"
    return name, module


# ============================================================================
# BACKBONE
# ============================================================================


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
            norm_layer = build_norm_hardcoded(norm_cfg, planes * block.expansion)
            if isinstance(norm_layer, tuple):
                norm_layer = norm_layer[1]
            downsample.extend(
                [
                    nn.Conv2d(
                        inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=conv_stride,
                        bias=False,
                    ),
                    norm_layer,
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
            in_channels=inplanes,
            out_channels=planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.add_module(self.norm1_name, norm1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

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

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
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
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

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
        if self.with_dcn:
            fallback_on_stride = dcn.pop("fallback_on_stride", False)
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
        else:
            self.conv2 = build_conv_layer(
                dcn,
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

    def forward_plugin(self, x, plugin_names):
        out = x
        for name in plugin_names:
            out = getattr(self, name)(out)
        return out

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

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
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
                elif block is Bottleneck:
                    block_init_cfg = dict(
                        type="Constant", val=0, override=dict(name="norm3")
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
            if plugins is not None:
                stage_plugins = self.make_stage_plugins(plugins, i)
            else:
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
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
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
        if self.deep_stem:
            x = self.stem(x)
        else:
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


def kaiming_init(
    module, a=0, mode="fan_out", nonlinearity="relu", bias=0, distribution="normal"
):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity
        )
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class ConvModule(nn.Module):
    _abbr_ = "conv_block"

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias="auto",
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=dict(type="ReLU"),
        inplace=True,
        with_spectral_norm=False,
        padding_mode="zeros",
        order=("conv", "norm", "act"),
    ):
        super(ConvModule, self).__init__()
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
        if self.with_activation:
            act_cfg_ = act_cfg.copy()
            if act_cfg_["type"] not in [
                "Tanh",
                "PReLU",
                "Sigmoid",
                "HSigmoid",
                "Swish",
            ]:
                act_cfg_.setdefault("inplace", inplace)
            cfg = act_cfg.copy()
            act_type = cfg.pop("type")
            self.activate = ACTIVATION_MAP[act_type](**cfg)

        self.init_weights()

    def init_weights(self):
        if not hasattr(self.conv, "init_weights"):
            nonlinearity = "relu"
            a = 0
            kaiming_init(self.conv, a=a, nonlinearity=nonlinearity)

    def forward(self, x, activate=True, norm=True):
        for layer in self.order:
            if layer == "conv":
                x = self.conv(x)
        return x


# ============================================================================
# NECK
# ============================================================================


class FPN(BaseModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_outs,
        start_level=0,
        end_level=-1,
        add_extra_convs=False,
        relu_before_extra_convs=False,
        no_norm_on_lateral=False,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=None,
        upsample_cfg=dict(mode="nearest"),
        init_cfg=dict(type="Xavier", layer="Conv2d", distribution="uniform"),
    ):
        super(FPN, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins
        else:
            self.backbone_end_level = end_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        if isinstance(add_extra_convs, str):
            assert add_extra_convs in ("on_input", "on_lateral", "on_output")
        elif add_extra_convs:  # True
            self.add_extra_convs = "on_input"

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False,
            )
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False,
            )
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False,
                )
                self.fpn_convs.append(extra_fpn_conv)

    def forward(self, inputs):
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += F.interpolate(
                laterals[i], size=prev_shape, **self.upsample_cfg
            )

        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]
        if self.num_outs > len(outs):
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            else:
                extra_source = outs[-1]
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
        return tuple(outs)


def digit_version(version_str: str, length: int = 4):
    version = parse(version_str)
    release = list(version.release)
    release = release[:length]
    if len(release) < length:
        release = release + [0] * (length - len(release))
    release.extend([0, 0])
    return tuple(release)


class ConfigDict(Dict):
    def __getattr__(self, name):
        try:
            value = super(ConfigDict, self).__getattr__(name)
        except Exception as e:
            ex = e
        else:
            return value
        raise ex


def xavier_init(module, gain=1, bias=0, distribution="normal"):
    if hasattr(module, "weight") and module.weight is not None:
        if distribution == "uniform":
            nn.init.xavier_uniform_(module.weight, gain=gain)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


# ============================================================================
# ATTENTION
# ============================================================================


def multi_scale_deformable_attn_pytorch(
    value, value_spatial_shapes, sampling_locations, attention_weights
):
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        value_l_ = (
            value_list[level]
            .flatten(2)
            .transpose(1, 2)
            .reshape(bs * num_heads, embed_dims, H_, W_)
        )
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampling_value_list.append(sampling_value_l_)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(bs, num_heads * embed_dims, num_queries)
    )
    return output.transpose(1, 2).contiguous()


class CustomMSDeformableAttention(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        num_heads=8,
        num_levels=4,
        num_points=4,
        im2col_step=64,
        dropout=0.1,
        batch_first=False,
        norm_cfg=None,
        init_cfg=None,
    ):
        super().__init__(init_cfg)
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2
        )
        self.attention_weights = nn.Linear(
            embed_dims, num_heads * num_levels * num_points
        )
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    def init_weights(self):
        constant_init(self.sampling_offsets, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.num_heads, 1, 1, 2)
            .repeat(1, self.num_levels, self.num_points, 1)
        )
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0.0, bias=0.0)
        xavier_init(self.value_proj, distribution="uniform", bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)
        self._is_init = True

    def forward(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        key_padding_mask=None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        flag="decoder",
        **kwargs,
    ):

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        value = self.value_proj(value)
        value = value.view(bs, num_value, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points
        )
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1
            )
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        output = multi_scale_deformable_attn_pytorch(
            value, spatial_shapes, sampling_locations, attention_weights
        )

        output = self.output_proj(output)

        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity


class FFN(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        feedforward_channels=1024,
        num_fcs=2,
        act_cfg=dict(type="ReLU", inplace=True),
        ffn_drop=0.0,
        dropout_layer=None,
        add_identity=True,
        init_cfg=None,
        **kwargs,
    ):
        super(FFN, self).__init__(init_cfg)
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        cfg = act_cfg.copy()
        act_type = cfg.pop("type")
        self.activate = ACTIVATION_MAP[act_type](**cfg)

        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                Sequential(
                    Linear(in_channels, feedforward_channels),
                    self.activate,
                    nn.Dropout(ffn_drop),
                )
            )
            in_channels = feedforward_channels
        layers.append(Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = Sequential(*layers)
        self.dropout_layer = torch.nn.Identity()
        self.add_identity = add_identity

    def forward(self, x, identity=None):
        out = self.layers(x)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)


class SpatialCrossAttention(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        num_cams=6,
        pc_range=None,
        dropout=0.1,
        init_cfg=None,
        batch_first=False,
        deformable_attention=dict(
            type="MSDeformableAttention3D", embed_dims=256, num_levels=4
        ),
        **kwargs,
    ):
        super(SpatialCrossAttention, self).__init__(init_cfg)

        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range
        cfg = deformable_attention.copy()
        attn_type = cfg.pop("type")
        self.deformable_attention = globals()[attn_type](**cfg)
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.batch_first = batch_first
        self.init_weight()

    def init_weight(self):
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)

    def forward(
        self,
        query,
        key,
        value,
        residual=None,
        query_pos=None,
        key_padding_mask=None,
        reference_points=None,
        spatial_shapes=None,
        reference_points_cam=None,
        bev_mask=None,
        level_start_index=None,
        flag="encoder",
        **kwargs,
    ):

        if residual is None:
            inp_residual = query
            slots = torch.zeros_like(query)

        bs, num_query, _ = query.size()

        D = reference_points_cam.size(3)
        indexes = []
        for i, mask_per_img in enumerate(bev_mask):
            index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
            indexes.append(index_query_per_img)
        max_len = max([len(each) for each in indexes])

        queries_rebatch = query.new_zeros([bs, self.num_cams, max_len, self.embed_dims])
        reference_points_rebatch = reference_points_cam.new_zeros(
            [bs, self.num_cams, max_len, D, 2]
        )

        for j in range(bs):
            for i, reference_points_per_img in enumerate(reference_points_cam):
                index_query_per_img = indexes[i]
                queries_rebatch[j, i, : len(index_query_per_img)] = query[
                    j, index_query_per_img
                ]
                reference_points_rebatch[
                    j, i, : len(index_query_per_img)
                ] = reference_points_per_img[j, index_query_per_img]

        num_cams, l, bs, embed_dims = key.shape

        key = key.permute(2, 0, 1, 3).reshape(bs * self.num_cams, l, self.embed_dims)
        value = value.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims
        )

        queries = self.deformable_attention(
            query=queries_rebatch.view(bs * self.num_cams, max_len, self.embed_dims),
            key=key,
            value=value,
            reference_points=reference_points_rebatch.view(
                bs * self.num_cams, max_len, D, 2
            ),
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        ).view(bs, self.num_cams, max_len, self.embed_dims)
        for j in range(bs):
            for i, index_query_per_img in enumerate(indexes):
                slots[j, index_query_per_img] += queries[
                    j, i, : len(index_query_per_img)
                ]

        count = bev_mask.sum(-1) > 0
        count = count.permute(1, 2, 0).sum(-1)
        count = torch.clamp(count, min=1.0)
        slots = slots / count[..., None]
        slots = self.output_proj(slots)

        return self.dropout(slots) + inp_residual


class MSDeformableAttention3D(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        num_heads=8,
        num_levels=4,
        num_points=8,
        im2col_step=64,
        dropout=0.1,
        batch_first=True,
        norm_cfg=None,
        init_cfg=None,
    ):
        super().__init__(init_cfg)
        self.norm_cfg = norm_cfg
        self.batch_first = batch_first
        self.output_proj = None
        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2
        )
        self.attention_weights = nn.Linear(
            embed_dims, num_heads * num_levels * num_points
        )
        self.value_proj = nn.Linear(embed_dims, embed_dims)

        self.init_weights()

    def init_weights(self):
        constant_init(self.sampling_offsets, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.num_heads, 1, 1, 2)
            .repeat(1, self.num_levels, self.num_points, 1)
        )
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0.0, bias=0.0)
        xavier_init(self.value_proj, distribution="uniform", bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)
        self._is_init = True

    def forward(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        key_padding_mask=None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        **kwargs,
    ):
        if identity is None:
            identity = query

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape

        value = self.value_proj(value)
        value = value.view(bs, num_value, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points
        )

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1
            )
            bs, num_query, num_Z_anchors, xy = reference_points.shape
            reference_points = reference_points[:, :, None, None, None, :, :]
            sampling_offsets = (
                sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
            (
                bs,
                num_query,
                num_heads,
                num_levels,
                num_all_points,
                xy,
            ) = sampling_offsets.shape
            sampling_offsets = sampling_offsets.view(
                bs,
                num_query,
                num_heads,
                num_levels,
                num_all_points // num_Z_anchors,
                num_Z_anchors,
                xy,
            )

            sampling_locations = reference_points + sampling_offsets
            (
                bs,
                num_query,
                num_heads,
                num_levels,
                num_points,
                num_Z_anchors,
                xy,
            ) = sampling_locations.shape

            sampling_locations = sampling_locations.view(
                bs, num_query, num_heads, num_levels, num_all_points, xy
            )

        output = multi_scale_deformable_attn_pytorch(
            value, spatial_shapes, sampling_locations, attention_weights
        )

        return output


class TemporalSelfAttention(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        num_heads=8,
        num_levels=4,
        num_points=4,
        num_bev_queue=2,
        im2col_step=64,
        dropout=0.1,
        batch_first=True,
        norm_cfg=None,
        init_cfg=None,
    ):

        super().__init__(init_cfg)
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_bev_queue = num_bev_queue
        self.sampling_offsets = nn.Linear(
            embed_dims * self.num_bev_queue,
            num_bev_queue * num_heads * num_levels * num_points * 2,
        )
        self.attention_weights = nn.Linear(
            embed_dims * self.num_bev_queue,
            num_bev_queue * num_heads * num_levels * num_points,
        )
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    def init_weights(self):
        constant_init(self.sampling_offsets, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.num_heads, 1, 1, 2)
            .repeat(1, self.num_levels * self.num_bev_queue, self.num_points, 1)
        )

        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0.0, bias=0.0)
        xavier_init(self.value_proj, distribution="uniform", bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)
        self._is_init = True

    def forward(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        key_padding_mask=None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        flag="decoder",
        **kwargs,
    ):

        if value is None:
            bs, len_bev, c = query.shape
            value = torch.stack([query, query], 1).reshape(bs * 2, len_bev, c)

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        bs, num_query, embed_dims = query.shape
        _, num_value, _ = value.shape

        query = torch.cat([value[:bs], query], -1)
        value = self.value_proj(value)
        value = value.reshape(bs * self.num_bev_queue, num_value, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.view(
            bs,
            num_query,
            self.num_heads,
            self.num_bev_queue,
            self.num_levels,
            self.num_points,
            2,
        )
        attention_weights = self.attention_weights(query).view(
            bs,
            num_query,
            self.num_heads,
            self.num_bev_queue,
            self.num_levels * self.num_points,
        )
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(
            bs,
            num_query,
            self.num_heads,
            self.num_bev_queue,
            self.num_levels,
            self.num_points,
        )

        attention_weights = (
            attention_weights.permute(0, 3, 1, 2, 4, 5)
            .reshape(
                bs * self.num_bev_queue,
                num_query,
                self.num_heads,
                self.num_levels,
                self.num_points,
            )
            .contiguous()
        )
        sampling_offsets = sampling_offsets.permute(0, 3, 1, 2, 4, 5, 6).reshape(
            bs * self.num_bev_queue,
            num_query,
            self.num_heads,
            self.num_levels,
            self.num_points,
            2,
        )

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1
            )
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )

        output = multi_scale_deformable_attn_pytorch(
            value, spatial_shapes, sampling_locations, attention_weights
        )
        output = output.permute(1, 2, 0)
        output = output.view(num_query, embed_dims, bs, self.num_bev_queue)
        output = output.mean(-1)

        output = output.permute(2, 0, 1)

        output = self.output_proj(output)

        return self.dropout(output) + identity


class TransformerLayerSequence(BaseModule):
    def __init__(self, transformerlayers=None, num_layers=None, init_cfg=None):
        super(TransformerLayerSequence, self).__init__(init_cfg)
        if isinstance(transformerlayers, dict):
            transformerlayers = [
                copy.deepcopy(transformerlayers) for _ in range(num_layers)
            ]
        self.num_layers = num_layers
        self.layers = ModuleList()
        for i in range(num_layers):
            cfg = transformerlayers[i].copy()
            layer_type = cfg.pop("type")
            self.layers.append(globals()[layer_type](**cfg))
        self.embed_dims = self.layers[0].embed_dims
        self.pre_norm = self.layers[0].pre_norm


class GroupMultiheadAttention(BaseModule):
    def __init__(
        self,
        embed_dims,
        num_heads,
        attn_drop=0.0,
        proj_drop=0.0,
        group=1,
        dropout_layer=dict(type="Dropout", drop_prob=0.0),
        init_cfg=None,
        batch_first=False,
        **kwargs,
    ):
        super().__init__(init_cfg)
        if "dropout" in kwargs:
            attn_drop = kwargs["dropout"]
            dropout_layer["drop_prob"] = kwargs.pop("dropout")

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.group = group
        self.batch_first = batch_first

        self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop, **kwargs)

        self.proj_drop = nn.Dropout(proj_drop)
        if dropout_layer:
            cfg = dropout_layer.copy()
            drop_type = cfg.pop("type")
            if drop_type == "Dropout":
                p = cfg.pop("drop_prob", 0.5)
                self.dropout_layer = nn.Dropout(p=p, **cfg)
            else:
                self.dropout_layer = nn.Identity()

    def forward(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        key_pos=None,
        attn_mask=None,
        key_padding_mask=None,
        **kwargs,
    ):
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        num_queries = query.shape[0]
        bs = query.shape[1]
        if self.training:
            query = torch.cat(query.split(num_queries // self.group, dim=0), dim=1)
            key = torch.cat(key.split(num_queries // self.group, dim=0), dim=1)
            value = torch.cat(value.split(num_queries // self.group, dim=0), dim=1)

        out = self.attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )[0]
        return identity + self.dropout_layer(self.proj_drop(out))


class MultiheadAttention(BaseModule):
    def __init__(
        self,
        embed_dims,
        num_heads,
        attn_drop=0.0,
        proj_drop=0.0,
        dropout_layer=dict(type="Dropout", drop_prob=0.0),
        init_cfg=None,
        batch_first=False,
        **kwargs,
    ):
        super(MultiheadAttention, self).__init__(init_cfg)
        if "dropout" in kwargs:
            attn_drop = kwargs["dropout"]
            dropout_layer["drop_prob"] = kwargs.pop("dropout")

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first

        self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop, **kwargs)

        self.proj_drop = nn.Dropout(proj_drop)
        if dropout_layer:
            cfg = dropout_layer.copy()
            drop_type = cfg.pop("type")
            if drop_type == "Dropout":
                p = cfg.pop("drop_prob", 0.5)
                self.dropout_layer = nn.Dropout(p=p, **cfg)
            else:
                self.dropout_layer = nn.Identity()

    def forward(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        key_pos=None,
        attn_mask=None,
        key_padding_mask=None,
        **kwargs,
    ):
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos
        out = self.attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )[0]

        return identity + self.dropout_layer(self.proj_drop(out))


# ============================================================================
# TRANSFORMER
# ============================================================================


class BaseTransformerLayer(BaseModule):
    def __init__(
        self,
        attn_cfgs=None,
        ffn_cfgs=dict(
            type="FFN",
            embed_dims=256,
            feedforward_channels=1024,
            num_fcs=2,
            ffn_drop=0.0,
            act_cfg=dict(type="ReLU", inplace=True),
        ),
        operation_order=None,
        norm_cfg=dict(type="LN"),
        init_cfg=None,
        batch_first=False,
        **kwargs,
    ):

        deprecated_args = dict(
            feedforward_channels="feedforward_channels",
            ffn_dropout="ffn_drop",
            ffn_num_fcs="num_fcs",
        )
        for ori_name, new_name in deprecated_args.items():
            if ori_name in kwargs:
                ffn_cfgs[new_name] = kwargs[ori_name]

        super(BaseTransformerLayer, self).__init__(init_cfg)

        self.batch_first = batch_first

        num_attn = operation_order.count("self_attn") + operation_order.count(
            "cross_attn"
        )

        self.num_attn = num_attn
        self.operation_order = operation_order
        self.norm_cfg = norm_cfg
        self.pre_norm = operation_order[0] == "norm"
        self.attentions = ModuleList()

        index = 0
        for operation_name in operation_order:
            if operation_name in ["self_attn", "cross_attn"]:
                cfg = attn_cfgs[index].copy()
                attn_type = cfg.pop("type")
                attention = globals()[attn_type](**cfg)
                attention.operation_name = operation_name
                self.attentions.append(attention)
                index += 1

        self.embed_dims = self.attentions[0].embed_dims

        self.ffns = ModuleList()
        num_ffns = operation_order.count("ffn")
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = ConfigDict(ffn_cfgs)
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = [copy.deepcopy(ffn_cfgs) for _ in range(num_ffns)]
        for ffn_index in range(num_ffns):
            if "embed_dims" not in ffn_cfgs[ffn_index]:
                ffn_cfgs["embed_dims"] = self.embed_dims
            _cfg = ffn_cfgs[ffn_index].copy()
            _type = _cfg.pop("type")
            self.ffns.append(globals()[_type](**_cfg))

        self.norms = ModuleList()
        num_norms = operation_order.count("norm")
        for _ in range(num_norms):
            self.norms.append(build_norm_hardcoded(norm_cfg, self.embed_dims))

    def forward(
        self,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        **kwargs,
    ):
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        for layer in self.operation_order:
            if layer == "self_attn":
                temp_key = temp_value = query
                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs,
                )
                attn_index += 1
                identity = query

            elif layer == "norm":
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == "cross_attn":
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs,
                )
                attn_index += 1
                identity = query

            elif layer == "ffn":
                query = self.ffns[ffn_index](query, identity if self.pre_norm else None)
                ffn_index += 1

        return query


class DetrTransformerDecoderLayer(BaseTransformerLayer):
    def __init__(
        self,
        attn_cfgs,
        feedforward_channels,
        ffn_dropout=0.0,
        operation_order=None,
        act_cfg=dict(type="ReLU", inplace=True),
        norm_cfg=dict(type="LN"),
        ffn_num_fcs=2,
        **kwargs,
    ):
        super(DetrTransformerDecoderLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs,
        )


class DetectionTransformerDecoder(TransformerLayerSequence):
    def __init__(self, *args, return_intermediate=False, **kwargs):
        super(DetectionTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

    def forward(
        self,
        query,
        *args,
        reference_points=None,
        reg_branches=None,
        key_padding_mask=None,
        **kwargs,
    ):
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):

            reference_points_input = reference_points[..., :2].unsqueeze(
                2
            )  # BS NUM_QUERY NUM_LEVEL 2
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                key_padding_mask=key_padding_mask,
                **kwargs,
            )
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                tmp = reg_branches[lid](output)
                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(
                    reference_points[..., :2]
                )
                new_reference_points[..., 2:3] = tmp[..., 4:5] + inverse_sigmoid(
                    reference_points[..., 2:3]
                )

                new_reference_points = new_reference_points.sigmoid()

                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


class BEVFormerEncoder(TransformerLayerSequence):
    def __init__(
        self,
        *args,
        pc_range=None,
        num_points_in_pillar=4,
        return_intermediate=False,
        dataset_type="nuscenes",
        **kwargs,
    ):

        super(BEVFormerEncoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

        self.num_points_in_pillar = num_points_in_pillar
        self.pc_range = pc_range

    @staticmethod
    def get_reference_points(
        H,
        W,
        Z=8,
        num_points_in_pillar=4,
        dim="3d",
        bs=1,
        device="cuda",
        dtype=torch.float,
    ):

        if dim == "3d":
            zs = (
                torch.linspace(
                    0.5, Z - 0.5, num_points_in_pillar, dtype=dtype, device=device
                )
                .view(-1, 1, 1)
                .expand(num_points_in_pillar, H, W)
                / Z
            )
            xs = (
                torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device)
                .view(1, 1, W)
                .expand(num_points_in_pillar, H, W)
                / W
            )
            ys = (
                torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device)
                .view(1, H, 1)
                .expand(num_points_in_pillar, H, W)
                / H
            )
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        elif dim == "2d":
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device),
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    def point_sampling(self, reference_points, pc_range, img_metas):
        allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta["lidar2img"])
        lidar2img = np.asarray(lidar2img)
        lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)
        reference_points = reference_points.clone()

        reference_points[..., 0:1] = (
            reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        )
        reference_points[..., 1:2] = (
            reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        )
        reference_points[..., 2:3] = (
            reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
        )

        reference_points = torch.cat(
            (reference_points, torch.ones_like(reference_points[..., :1])), -1
        )

        reference_points = reference_points.permute(1, 0, 2, 3)
        D, B, num_query = reference_points.size()[:3]
        num_cam = lidar2img.size(1)

        reference_points = (
            reference_points.view(D, B, 1, num_query, 4)
            .repeat(1, 1, num_cam, 1, 1)
            .unsqueeze(-1)
        )

        lidar2img = lidar2img.view(1, B, num_cam, 1, 4, 4).repeat(
            D, 1, 1, num_query, 1, 1
        )

        reference_points_cam = torch.matmul(
            lidar2img.to(torch.float32), reference_points.to(torch.float32)
        ).squeeze(-1)
        eps = 1e-5

        bev_mask = reference_points_cam[..., 2:3] > eps
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3],
            torch.ones_like(reference_points_cam[..., 2:3]) * eps,
        )

        reference_points_cam[..., 0] /= img_metas[0]["img_shape"][0][1]
        reference_points_cam[..., 1] /= img_metas[0]["img_shape"][0][0]

        bev_mask = (
            bev_mask
            & (reference_points_cam[..., 1:2] > 0.0)
            & (reference_points_cam[..., 1:2] < 1.0)
            & (reference_points_cam[..., 0:1] < 1.0)
            & (reference_points_cam[..., 0:1] > 0.0)
        )
        bev_mask = torch.nan_to_num(bev_mask)
        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32

        return reference_points_cam, bev_mask

    def forward(
        self,
        bev_query,
        key,
        value,
        *args,
        bev_h=None,
        bev_w=None,
        bev_pos=None,
        spatial_shapes=None,
        level_start_index=None,
        valid_ratios=None,
        prev_bev=None,
        shift=0.0,
        **kwargs,
    ):

        output = bev_query
        intermediate = []
        ref_3d = self.get_reference_points(
            bev_h,
            bev_w,
            self.pc_range[5] - self.pc_range[2],
            self.num_points_in_pillar,
            dim="3d",
            bs=bev_query.size(1),
            device=bev_query.device,
            dtype=bev_query.dtype,
        )
        ref_2d = self.get_reference_points(
            bev_h,
            bev_w,
            dim="2d",
            bs=bev_query.size(1),
            device=bev_query.device,
            dtype=bev_query.dtype,
        )
        reference_points_cam, bev_mask = self.point_sampling(
            ref_3d, self.pc_range, kwargs["img_metas"]
        )
        shift_ref_2d = ref_2d.clone()
        shift_ref_2d += shift[:, None, None, :]

        bev_query = bev_query.permute(1, 0, 2)
        bev_pos = bev_pos.permute(1, 0, 2)
        bs, len_bev, num_bev_level, _ = ref_2d.shape
        hybird_ref_2d = torch.stack([ref_2d, ref_2d], 1).reshape(
            bs * 2, len_bev, num_bev_level, 2
        )
        for lid, layer in enumerate(self.layers):
            output = layer(
                bev_query,
                key,
                value,
                *args,
                bev_pos=bev_pos,
                ref_2d=hybird_ref_2d,
                ref_3d=ref_3d,
                bev_h=bev_h,
                bev_w=bev_w,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cam,
                bev_mask=bev_mask,
                prev_bev=prev_bev,
                **kwargs,
            )

            bev_query = output
        return output


class MyCustomBaseTransformerLayer(BaseModule):
    def __init__(
        self,
        attn_cfgs=None,
        ffn_cfgs=dict(
            type="FFN",
            embed_dims=256,
            feedforward_channels=1024,
            num_fcs=2,
            ffn_drop=0.0,
            act_cfg=dict(type="ReLU", inplace=True),
        ),
        operation_order=None,
        norm_cfg=dict(type="LN"),
        init_cfg=None,
        batch_first=True,
        **kwargs,
    ):

        deprecated_args = dict(
            feedforward_channels="feedforward_channels",
            ffn_dropout="ffn_drop",
            ffn_num_fcs="num_fcs",
        )
        for ori_name, new_name in deprecated_args.items():
            if ori_name in kwargs:
                ffn_cfgs[new_name] = kwargs[ori_name]

        super(MyCustomBaseTransformerLayer, self).__init__(init_cfg)
        self.batch_first = batch_first
        num_attn = operation_order.count("self_attn") + operation_order.count(
            "cross_attn"
        )
        self.num_attn = num_attn
        self.operation_order = operation_order
        self.norm_cfg = norm_cfg
        self.pre_norm = operation_order[0] == "norm"
        self.attentions = ModuleList()

        index = 0
        for operation_name in operation_order:
            if operation_name in ["self_attn", "cross_attn"]:
                attn_cfgs[index]["batch_first"] = self.batch_first
                cfg = attn_cfgs[index].copy()
                attn_type = cfg.pop("type")
                attention = globals()[attn_type](**cfg)
                attention.operation_name = operation_name
                self.attentions.append(attention)
                index += 1

        self.embed_dims = self.attentions[0].embed_dims

        self.ffns = ModuleList()
        num_ffns = operation_order.count("ffn")
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = ConfigDict(ffn_cfgs)
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = [copy.deepcopy(ffn_cfgs) for _ in range(num_ffns)]
        for ffn_index in range(num_ffns):
            if "embed_dims" not in ffn_cfgs[ffn_index]:
                ffn_cfgs["embed_dims"] = self.embed_dims
            _cfg = ffn_cfgs[ffn_index].copy()
            _type = _cfg.pop("type")
            self.ffns.append(globals()[_type](**_cfg))

        self.norms = ModuleList()
        num_norms = operation_order.count("norm")
        for _ in range(num_norms):
            self.norms.append(build_norm_hardcoded(norm_cfg, self.embed_dims))


class BEVFormerLayer(MyCustomBaseTransformerLayer):
    def __init__(
        self,
        attn_cfgs,
        feedforward_channels,
        ffn_dropout=0.0,
        operation_order=None,
        act_cfg=dict(type="ReLU", inplace=True),
        norm_cfg=dict(type="LN"),
        ffn_num_fcs=2,
        **kwargs,
    ):
        super(BEVFormerLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs,
        )

    def forward(
        self,
        query,
        key=None,
        value=None,
        bev_pos=None,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        ref_2d=None,
        ref_3d=None,
        bev_h=None,
        bev_w=None,
        reference_points_cam=None,
        mask=None,
        spatial_shapes=None,
        level_start_index=None,
        prev_bev=None,
        **kwargs,
    ):

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]

        for layer in self.operation_order:
            if layer == "self_attn":
                query = self.attentions[attn_index](
                    query,
                    prev_bev,
                    prev_bev,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_pos=bev_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    reference_points=ref_2d,
                    spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
                    level_start_index=torch.tensor([0], device=query.device),
                    **kwargs,
                )
                attn_index += 1
                identity = query

            elif layer == "norm":
                query = self.norms[norm_index](query)
                norm_index += 1

            if layer == "cross_attn":
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=reference_points_cam,
                    mask=mask,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    **kwargs,
                )
                attn_index += 1
                identity = query

            elif layer == "ffn":
                query = self.ffns[ffn_index](query, identity if self.pre_norm else None)
                ffn_index += 1
        return query


class PerceptionTransformerBEVEncoder(BaseModule):
    def __init__(
        self,
        num_feature_levels=4,
        num_cams=6,
        two_stage_num_proposals=300,
        encoder=None,
        embed_dims=256,
        use_cams_embeds=True,
        rotate_center=[100, 100],
        **kwargs,
    ):
        super(PerceptionTransformerBEVEncoder, self).__init__(**kwargs)
        enc_cfg = encoder.copy()
        enc_type = enc_cfg.pop("type")
        self.encoder = globals()[enc_type](**enc_cfg)
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False

        self.use_cams_embeds = use_cams_embeds

        self.two_stage_num_proposals = two_stage_num_proposals
        self.rotate_center = rotate_center
        self.level_embeds = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims)
        )
        if self.use_cams_embeds:
            self.cams_embeds = nn.Parameter(
                torch.Tensor(self.num_cams, self.embed_dims)
            )

    def forward(
        self,
        mlvl_feats,
        bev_queries,
        bev_h,
        bev_w,
        grid_length=[0.512, 0.512],
        bev_pos=None,
        prev_bev=None,
        **kwargs,
    ):
        bs = mlvl_feats[0].size(0)
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None, None, lvl : lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=bev_pos.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )

        feat_flatten = feat_flatten.permute(
            0, 2, 1, 3
        )  # (num_cam, H*W, bs, embed_dims)

        bev_embed = self.encoder(
            bev_queries,
            feat_flatten,
            feat_flatten,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev=None,
            shift=bev_queries.new_tensor([0, 0]).unsqueeze(0),
            **kwargs,
        )
        prev_bev = bev_embed
        if (
            "aug_param" in kwargs["img_metas"][0]
            and "GlobalRotScaleTransImage_param" in kwargs["img_metas"][0]["aug_param"]
        ):
            rot_angle, scale_ratio, flip_dx, flip_dy, bda_mat, only_gt = kwargs[
                "img_metas"
            ][0]["aug_param"]["GlobalRotScaleTransImage_param"]
            prev_bev = prev_bev.reshape(bs, bev_h, bev_w, -1).permute(
                0, 3, 1, 2
            )  # bchw
        return prev_bev


class BasicBlock_resfusion(BaseModule):
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
        super(BasicBlock_resfusion, self).__init__(init_cfg)

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

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)
        return out


class ResNetFusion(BaseModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        inter_channels,
        num_layer,
        norm_cfg=dict(type="SyncBN"),
        with_cp=False,
    ):
        super(ResNetFusion, self).__init__()
        layers = []
        self.inter_channels = inter_channels
        for i in range(num_layer):
            if i == 0:
                if inter_channels == in_channels:
                    layers.append(
                        BasicBlock_resfusion(
                            in_channels, inter_channels, stride=1, norm_cfg=norm_cfg
                        )
                    )
                else:
                    norm_layer = build_norm_hardcoded(norm_cfg, inter_channels)
                    if isinstance(norm_layer, tuple):
                        norm_layer = norm_layer[1]
                    downsample = nn.Sequential(
                        nn.Conv2d(
                            in_channels,
                            inter_channels,
                            3,
                            stride=1,
                            padding=1,
                            dilation=1,
                            bias=False,
                        ),
                        norm_layer,
                    )
                    layers.append(
                        BasicBlock_resfusion(
                            in_channels,
                            inter_channels,
                            stride=1,
                            norm_cfg=norm_cfg,
                            downsample=downsample,
                        )
                    )
            else:
                layers.append(
                    BasicBlock_resfusion(
                        inter_channels, inter_channels, stride=1, norm_cfg=norm_cfg
                    )
                )
        self.layers = nn.Sequential(*layers)
        self.layer_norm = nn.Sequential(
            nn.Linear(inter_channels, out_channels), nn.LayerNorm(out_channels)
        )
        self.with_cp = with_cp

    def forward(self, x):
        x = torch.cat(x, 1).contiguous()
        for lid, layer in enumerate(self.layers):
            if self.with_cp and x.requires_grad:
                x = checkpoint.checkpoint(layer, x)
            else:
                x = layer(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # nchw -> n(hw)c
        x = self.layer_norm(x)
        return x


class PerceptionTransformerV2(PerceptionTransformerBEVEncoder):
    def __init__(
        self,
        num_feature_levels=4,
        num_cams=6,
        two_stage_num_proposals=300,
        encoder=None,
        embed_dims=256,
        use_cams_embeds=True,
        rotate_center=[100, 100],
        frames=(0,),
        decoder=None,
        num_fusion=3,
        inter_channels=None,
        **kwargs,
    ):
        super(PerceptionTransformerV2, self).__init__(
            num_feature_levels,
            num_cams,
            two_stage_num_proposals,
            encoder,
            embed_dims,
            use_cams_embeds,
            rotate_center,
            **kwargs,
        )
        dec_cfg = decoder.copy()
        dec_type = dec_cfg.pop("type")
        self.decoder = globals()[dec_type](**dec_cfg)
        self.reference_points = nn.Linear(self.embed_dims, 3)
        self.frames = frames
        if len(self.frames) > 1:
            self.fusion = ResNetFusion(
                len(self.frames) * self.embed_dims,
                self.embed_dims,
                inter_channels
                if inter_channels is not None
                else len(self.frames) * self.embed_dims,
                num_fusion,
            )

    def init_weights(self):
        super().init_weights()
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        xavier_init(self.reference_points, distribution="uniform", bias=0.0)

    def get_bev_features(
        self,
        mlvl_feats,
        bev_queries,
        bev_h,
        bev_w,
        grid_length=[0.512, 0.512],
        bev_pos=None,
        prev_bev=None,
        **kwargs,
    ):
        return super().forward(
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            grid_length,
            bev_pos,
            prev_bev,
            **kwargs,
        )

    def forward(
        self,
        mlvl_feats,
        bev_queries,
        object_query_embed,
        bev_h,
        bev_w,
        grid_length=[0.512, 0.512],
        bev_pos=None,
        reg_branches=None,
        cls_branches=None,
        prev_bev=None,
        **kwargs,
    ):
        bev_embed = self.get_bev_features(
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=grid_length,
            bev_pos=bev_pos,
            prev_bev=None,
            **kwargs,
        )  # bev_embed shape: bs, bev_h*bev_w, embed_dims

        if len(self.frames) > 1:
            cur_ind = list(self.frames).index(0)
            prev_bev[cur_ind] = bev_embed

            for i in range(1, cur_ind + 1):
                if prev_bev[cur_ind - i] is None:
                    prev_bev[cur_ind - i] = prev_bev[cur_ind - i + 1].detach()

            for i in range(cur_ind + 1, len(self.frames)):
                if prev_bev[i] is None:
                    prev_bev[i] = prev_bev[i - 1].detach()
            bev_embed = [
                x.reshape(x.shape[0], bev_h, bev_w, x.shape[-1])
                .permute(0, 3, 1, 2)
                .contiguous()
                for x in prev_bev
            ]
            bev_embed = self.fusion(bev_embed)

        bs = mlvl_feats[0].size(0)
        query_pos, query = torch.split(object_query_embed, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_pos)
        reference_points = reference_points.sigmoid()
        init_reference_out = reference_points

        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        bev_embed = bev_embed.permute(1, 0, 2)

        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
            level_start_index=torch.tensor([0], device=query.device),
            **kwargs,
        )

        inter_references_out = inter_references
        return bev_embed, inter_states, init_reference_out, inter_references_out


class PerceptionTransformer(BaseModule):
    def __init__(
        self,
        num_feature_levels=4,
        num_cams=6,
        two_stage_num_proposals=300,
        encoder=None,
        decoder=None,
        embed_dims=256,
        rotate_prev_bev=True,
        use_shift=True,
        use_can_bus=True,
        can_bus_norm=True,
        use_cams_embeds=True,
        rotate_center=[100, 100],
        **kwargs,
    ):
        super(PerceptionTransformer, self).__init__(**kwargs)
        enc_cfg = encoder.copy()
        enc_type = enc_cfg.pop("type")
        self.encoder = globals()[enc_type](**enc_cfg)
        dec_cfg = decoder.copy()
        dec_type = dec_cfg.pop("type")
        self.decoder = globals()[dec_type](**dec_cfg)
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds

        self.two_stage_num_proposals = two_stage_num_proposals
        self.init_layers()
        self.rotate_center = rotate_center

    def init_layers(self):
        self.level_embeds = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims)
        )
        self.cams_embeds = nn.Parameter(torch.Tensor(self.num_cams, self.embed_dims))
        self.reference_points = nn.Linear(self.embed_dims, 3)
        self.can_bus_mlp = nn.Sequential(
            nn.Linear(18, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),
        )
        if self.can_bus_norm:
            self.can_bus_mlp.add_module("norm", nn.LayerNorm(self.embed_dims))

    def get_bev_features(
        self,
        mlvl_feats,
        bev_queries,
        bev_h,
        bev_w,
        grid_length=[0.512, 0.512],
        bev_pos=None,
        prev_bev=None,
        **kwargs,
    ):
        bs = mlvl_feats[0].size(0)
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)

        delta_x = np.array([each["can_bus"][0] for each in kwargs["img_metas"]])
        delta_y = np.array([each["can_bus"][1] for each in kwargs["img_metas"]])
        ego_angle = np.array(
            [each["can_bus"][-2] / np.pi * 180 for each in kwargs["img_metas"]]
        )
        grid_length_y = grid_length[0]
        grid_length_x = grid_length[1]
        translation_length = np.sqrt(delta_x**2 + delta_y**2)
        translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
        bev_angle = ego_angle - translation_angle
        shift_y = (
            translation_length * np.cos(bev_angle / 180 * np.pi) / grid_length_y / bev_h
        )
        shift_x = (
            translation_length * np.sin(bev_angle / 180 * np.pi) / grid_length_x / bev_w
        )
        shift_y = shift_y * self.use_shift
        shift_x = shift_x * self.use_shift
        shift = bev_queries.new_tensor([shift_x, shift_y]).permute(
            1, 0
        )  # xy, bs -> bs, xy

        if prev_bev is not None:
            if prev_bev.shape[1] == bev_h * bev_w:
                prev_bev = prev_bev.permute(1, 0, 2)
            if self.rotate_prev_bev:
                for i in range(bs):
                    rotation_angle = kwargs["img_metas"][i]["can_bus"][-1]
                    tmp_prev_bev = (
                        prev_bev[:, i].reshape(bev_h, bev_w, -1).permute(2, 0, 1)
                    )
                    tmp_prev_bev = rotate(
                        tmp_prev_bev, rotation_angle, center=self.rotate_center
                    )
                    tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(
                        bev_h * bev_w, 1, -1
                    )
                    prev_bev[:, i] = tmp_prev_bev[:, 0]

        can_bus = bev_queries.new_tensor(
            [each["can_bus"] for each in kwargs["img_metas"]]
        )  # [:, :]
        can_bus = self.can_bus_mlp(can_bus)[None, :, :]
        bev_queries = bev_queries + can_bus * self.use_can_bus

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None, None, lvl : lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=bev_pos.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )

        feat_flatten = feat_flatten.permute(
            0, 2, 1, 3
        )  # (num_cam, H*W, bs, embed_dims)
        bev_embed = self.encoder(
            bev_queries,
            feat_flatten,
            feat_flatten,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev=prev_bev,
            shift=shift,
            **kwargs,
        )

        return bev_embed

    def forward(
        self,
        mlvl_feats,
        bev_queries,
        object_query_embed,
        bev_h,
        bev_w,
        grid_length=[0.512, 0.512],
        bev_pos=None,
        reg_branches=None,
        cls_branches=None,
        prev_bev=None,
        **kwargs,
    ):
        bev_embed = self.get_bev_features(
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=grid_length,
            bev_pos=bev_pos,
            prev_bev=prev_bev,
            **kwargs,
        )
        bs = mlvl_feats[0].size(0)
        query_pos, query = torch.split(object_query_embed, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_pos)
        reference_points = reference_points.sigmoid()
        init_reference_out = reference_points

        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        bev_embed = bev_embed.permute(1, 0, 2)
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
            level_start_index=torch.tensor([0], device=query.device),
            **kwargs,
        )

        inter_references_out = inter_references

        return bev_embed, inter_states, init_reference_out, inter_references_out


# ============================================================================
# HEADS
# ============================================================================


class BBoxTestMixin(object):
    def simple_test_bboxes(self, feats, img_metas, rescale=False):
        outs = self.forward(feats)
        results_list = self.get_bboxes(*outs, img_metas, rescale=rescale)
        return results_list


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class Offset(nn.Module):
    def __init__(self, init_value=0.0):
        super(Offset, self).__init__()
        self.bias = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input + self.bias


class ModuleListDial(nn.ModuleList):
    def __init__(self, modules=None):
        super(ModuleListDial, self).__init__(modules)
        self.cur_position = 0

    def forward(self, x):
        result = self[self.cur_position](x)
        self.cur_position += 1
        if self.cur_position >= len(self):
            self.cur_position = 0
        return result


class BaseDenseHead(BaseModule, metaclass=ABCMeta):
    def __init__(self, init_cfg=None):
        super(BaseDenseHead, self).__init__(init_cfg)


class AnchorFreeHead(BaseDenseHead, BBoxTestMixin):
    _version = 1

    def __init__(
        self,
        num_classes,
        in_channels,
        feat_channels=256,
        stacked_convs=4,
        strides=(4, 8, 16, 32, 64),
        dcn_on_last_conv=False,
        conv_bias="auto",
        conv_cfg=None,
        norm_cfg=None,
        train_cfg=None,
        test_cfg=None,
        init_cfg=dict(
            type="Normal",
            layer="Conv2d",
            std=0.01,
            override=dict(type="Normal", name="conv_cls", std=0.01, bias_prob=0.01),
        ),
    ):
        super(AnchorFreeHead, self).__init__(init_cfg)
        self.num_classes = num_classes
        self.cls_out_channels = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.dcn_on_last_conv = dcn_on_last_conv
        self.conv_bias = conv_bias
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg


class LearnedPositionalEncoding(BaseModule):
    def __init__(
        self,
        num_feats,
        row_num_embed=50,
        col_num_embed=50,
        init_cfg=dict(type="Uniform", layer="Embedding"),
    ):
        super(LearnedPositionalEncoding, self).__init__(init_cfg)
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)
        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed

    def forward(self, mask):
        h, w = mask.shape[-2:]
        x = torch.arange(w, device=mask.device)
        y = torch.arange(h, device=mask.device)
        x_embed = self.col_embed(x)
        y_embed = self.row_embed(y)
        pos = (
            torch.cat(
                (
                    x_embed.unsqueeze(0).repeat(h, 1, 1),
                    y_embed.unsqueeze(1).repeat(1, w, 1),
                ),
                dim=-1,
            )
            .permute(2, 0, 1)
            .unsqueeze(0)
            .repeat(mask.shape[0], 1, 1, 1)
        )
        return pos


class FCOS2DHead(nn.Module):
    def __init__(
        self,
        num_classes,
        input_shape,
        num_cls_convs=4,
        num_box_convs=4,
        norm="BN",
        use_deformable=False,
        use_scale=True,
        box2d_scale_init_factor=1.0,
        version="v2",
    ):
        super().__init__()

        self.num_classes = num_classes
        self.in_strides = [shape.stride for shape in input_shape]
        self.num_levels = len(input_shape)
        self.use_scale = use_scale
        self.box2d_scale_init_factor = box2d_scale_init_factor
        self._version = version

        in_channels = [s.channels for s in input_shape]
        in_channels = in_channels[0]

        head_configs = {"cls": num_cls_convs, "box2d": num_box_convs}

        for head_name, num_convs in head_configs.items():
            tower = []
            if self._version == "v2":
                for _ in range(num_convs):
                    if norm in ("BN", "FrozenBN", "SyncBN", "GN"):
                        norm_layer = ModuleListDial(
                            [
                                get_norm(norm, in_channels)
                                for _ in range(self.num_levels)
                            ]
                        )
                    else:
                        norm_layer = get_norm(norm, in_channels)
                    tower.append(
                        Conv2d(
                            in_channels,
                            in_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=norm_layer is None,
                            norm=norm_layer,
                            activation=F.relu,
                        )
                    )
            self.add_module(f"{head_name}_tower", nn.Sequential(*tower))

        self.cls_logits = nn.Conv2d(
            in_channels, self.num_classes, kernel_size=3, stride=1, padding=1
        )
        self.box2d_reg = nn.Conv2d(in_channels, 4, kernel_size=3, stride=1, padding=1)
        self.centerness = nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1)

        if self.use_scale:
            self.scales_box2d_reg = nn.ModuleList(
                [
                    Scale(init_value=stride * self.box2d_scale_init_factor)
                    for stride in self.in_strides
                ]
            )

        self.init_weights()

    def init_weights(self):
        for tower in [self.cls_tower, self.box2d_tower]:
            for l in tower.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.kaiming_normal_(
                        l.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if l.bias is not None:
                        torch.nn.init.constant_(l.bias, 0)

        predictors = [self.cls_logits, self.box2d_reg, self.centerness]

        for modules in predictors:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.kaiming_uniform_(l.weight, a=1)
                    if l.bias is not None:  # depth head may not have bias.
                        torch.nn.init.constant_(l.bias, 0)


class FCOS2DLoss(nn.Module):
    def __init__(
        self,
        num_classes,
        focal_loss_alpha=0.25,
        focal_loss_gamma=2.0,
        loc_loss_type="giou",
    ):
        super().__init__()
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        self.num_classes = num_classes


class FCOS3DHead(nn.Module):
    def __init__(
        self,
        num_classes,
        input_shape,
        num_convs=4,
        norm="BN",
        use_scale=True,
        depth_scale_init_factor=0.3,
        proj_ctr_scale_init_factor=1.0,
        use_per_level_predictors=False,
        class_agnostic=False,
        use_deformable=False,
        mean_depth_per_level=None,
        std_depth_per_level=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_strides = [shape.stride for shape in input_shape]
        self.num_levels = len(input_shape)

        self.use_scale = use_scale
        self.depth_scale_init_factor = depth_scale_init_factor
        self.proj_ctr_scale_init_factor = proj_ctr_scale_init_factor
        self.use_per_level_predictors = use_per_level_predictors

        self.register_buffer("mean_depth_per_level", torch.Tensor(mean_depth_per_level))
        self.register_buffer("std_depth_per_level", torch.Tensor(std_depth_per_level))

        in_channels = [s.channels for s in input_shape]
        in_channels = in_channels[0]

        box3d_tower = []
        for i in range(num_convs):
            if norm in ("BN", "FrozenBN", "SyncBN", "GN"):
                norm_layer = ModuleListDial(
                    [get_norm(norm, in_channels) for _ in range(self.num_levels)]
                )
            box3d_tower.append(
                Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=norm_layer is None,
                    norm=norm_layer,
                    activation=F.relu,
                )
            )
        self.add_module("box3d_tower", nn.Sequential(*box3d_tower))

        num_classes = self.num_classes if not class_agnostic else 1
        num_levels = self.num_levels if use_per_level_predictors else 1

        self.box3d_quat = nn.ModuleList(
            [
                Conv2d(
                    in_channels,
                    4 * num_classes,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                )
                for _ in range(num_levels)
            ]
        )
        self.box3d_ctr = nn.ModuleList(
            [
                Conv2d(
                    in_channels,
                    2 * num_classes,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                )
                for _ in range(num_levels)
            ]
        )
        self.box3d_depth = nn.ModuleList(
            [
                Conv2d(
                    in_channels,
                    1 * num_classes,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=(not self.use_scale),
                )
                for _ in range(num_levels)
            ]
        )
        self.box3d_size = nn.ModuleList(
            [
                Conv2d(
                    in_channels,
                    3 * num_classes,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                )
                for _ in range(num_levels)
            ]
        )
        self.box3d_conf = nn.ModuleList(
            [
                Conv2d(
                    in_channels,
                    1 * num_classes,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                )
                for _ in range(num_levels)
            ]
        )

        if self.use_scale:
            self.scales_proj_ctr = nn.ModuleList(
                [
                    Scale(init_value=stride * self.proj_ctr_scale_init_factor)
                    for stride in self.in_strides
                ]
            )
            self.scales_size = nn.ModuleList(
                [Scale(init_value=1.0) for _ in range(self.num_levels)]
            )
            self.scales_conf = nn.ModuleList(
                [Scale(init_value=1.0) for _ in range(self.num_levels)]
            )

            self.scales_depth = nn.ModuleList(
                [
                    Scale(init_value=sigma * self.depth_scale_init_factor)
                    for sigma in self.std_depth_per_level
                ]
            )
            self.offsets_depth = nn.ModuleList(
                [Offset(init_value=b) for b in self.mean_depth_per_level]
            )

        self._init_weights()

    def _init_weights(self):
        for l in self.box3d_tower.modules():
            if isinstance(l, nn.Conv2d):
                torch.nn.init.kaiming_normal_(
                    l.weight, mode="fan_out", nonlinearity="relu"
                )
                if l.bias is not None:
                    torch.nn.init.constant_(l.bias, 0)

        predictors = [
            self.box3d_quat,
            self.box3d_ctr,
            self.box3d_depth,
            self.box3d_size,
            self.box3d_conf,
        ]

        for modules in predictors:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.kaiming_uniform_(l.weight, a=1)
                    if l.bias is not None:  # depth head may not have bias.
                        torch.nn.init.constant_(l.bias, 0)


class FCOS3DLoss(nn.Module):
    def __init__(
        self,
        num_classes,
        min_depth=0.1,
        max_depth=80.0,
        box3d_loss_weight=2.0,
        conf3d_loss_weight=1.0,
        conf_3d_temperature=1.0,
        smooth_l1_loss_beta=0.05,
        max_loss_per_group=20,
        predict_allocentric_rot=True,
        scale_depth_by_focal_lengths=True,
        scale_depth_by_focal_lengths_factor=500.0,
        class_agnostic=False,
        predict_distance=False,
        canon_box_sizes=None,
    ):
        super().__init__()
        self.canon_box_sizes = canon_box_sizes
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.predict_allocentric_rot = predict_allocentric_rot
        self.scale_depth_by_focal_lengths = scale_depth_by_focal_lengths
        self.scale_depth_by_focal_lengths_factor = scale_depth_by_focal_lengths_factor
        self.predict_distance = predict_distance
        self.box3d_loss_weight = box3d_loss_weight
        self.conf3d_loss_weight = conf3d_loss_weight
        self.conf_3d_temperature = conf_3d_temperature

        self.num_classes = num_classes
        self.class_agnostic = class_agnostic


class DD3D(nn.Module):
    def __init__(
        self,
        num_classes,
        in_channels,
        strides,
        fcos2d_cfg=dict(),
        fcos2d_loss_cfg=dict(),
        fcos3d_cfg=dict(),
        fcos3d_loss_cfg=dict(),
        target_assign_cfg=dict(),
        box3d_on=True,
        feature_locations_offset="none",
    ):
        super().__init__()
        self.backbone_output_shape = [
            ShapeSpec(channels=in_channels, stride=s) for s in strides
        ]

        self.feature_locations_offset = feature_locations_offset

        self.fcos2d_head = FCOS2DHead(
            num_classes=num_classes,
            input_shape=self.backbone_output_shape,
            **fcos2d_cfg,
        )
        self.fcos2d_loss = FCOS2DLoss(num_classes=num_classes, **fcos2d_loss_cfg)

        if box3d_on:
            self.fcos3d_head = FCOS3DHead(
                num_classes=num_classes,
                input_shape=self.backbone_output_shape,
                **fcos3d_cfg,
            )
            self.fcos3d_loss = FCOS3DLoss(num_classes=num_classes, **fcos3d_loss_cfg)
            self.only_box2d = False
        else:
            self.only_box2d = True


class NuscenesDD3D(DD3D):
    def __init__(
        self,
        num_classes,
        in_channels,
        strides,
        fcos2d_cfg=dict(),
        fcos2d_loss_cfg=dict(),
        fcos3d_cfg=dict(),
        fcos3d_loss_cfg=dict(),
        target_assign_cfg=dict(),
        nusc_loss_weight=dict(),
        box3d_on=True,
        feature_locations_offset="none",
    ):
        super().__init__(
            num_classes,
            in_channels,
            strides,
            fcos2d_cfg=fcos2d_cfg,
            fcos2d_loss_cfg=fcos2d_loss_cfg,
            fcos3d_cfg=fcos3d_cfg,
            fcos3d_loss_cfg=fcos3d_loss_cfg,
            target_assign_cfg=target_assign_cfg,
            box3d_on=box3d_on,
            feature_locations_offset=feature_locations_offset,
        )

        self.attr_logits = Conv2d(
            in_channels, 3, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.speed = Conv2d(
            in_channels,
            1,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            activation=F.relu,
        )

        for modules in [self.attr_logits, self.speed]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.kaiming_uniform_(l.weight, a=1)
                    if l.bias is not None:  # depth head may not have bias.
                        torch.nn.init.constant_(l.bias, 0)


class DETRHead(AnchorFreeHead):
    _version = 2

    def __init__(
        self,
        num_classes,
        in_channels,
        num_query=100,
        num_reg_fcs=2,
        transformer=None,
        sync_cls_avg_factor=False,
        positional_encoding=dict(
            type="SinePositionalEncoding", num_feats=128, normalize=True
        ),
        loss_cls=dict(
            type="CrossEntropyLoss",
            bg_cls_weight=0.1,
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=1.0,
        ),
        loss_bbox=dict(type="L1Loss", loss_weight=5.0),
        loss_iou=dict(type="GIoULoss", loss_weight=2.0),
        train_cfg=dict(
            assigner=dict(
                type="HungarianAssigner",
                cls_cost=dict(type="ClassificationCost", weight=1.0),
                reg_cost=dict(type="BBoxL1Cost", weight=5.0),
                iou_cost=dict(type="IoUCost", iou_mode="giou", weight=2.0),
            )
        ),
        test_cfg=dict(max_per_img=100),
        init_cfg=None,
        **kwargs,
    ):
        super(AnchorFreeHead, self).__init__(init_cfg)
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        class_weight = loss_cls.get("class_weight", None)
        if class_weight is not None and (self.__class__ is DETRHead):
            bg_cls_weight = loss_cls.get("bg_cls_weight", class_weight)
            class_weight = torch.ones(num_classes + 1) * class_weight
            class_weight[num_classes] = bg_cls_weight
            loss_cls.update({"class_weight": class_weight})
            if "bg_cls_weight" in loss_cls:
                loss_cls.pop("bg_cls_weight")
            self.bg_cls_weight = bg_cls_weight

        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_reg_fcs = num_reg_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.cls_out_channels = num_classes

        self.act_cfg = transformer.get("act_cfg", dict(type="ReLU", inplace=True))
        cfg = self.act_cfg.copy()
        act_type = cfg.pop("type")
        self.activate = ACTIVATION_MAP[act_type](**cfg)
        cfg = positional_encoding.copy()
        cfg.pop("type", None)
        self.positional_encoding = LearnedPositionalEncoding(**cfg)
        args = transformer.copy()
        args.pop("type", None)
        trans_type = transformer.get("type", "PerceptionTransformer")
        if trans_type == "PerceptionTransformerV2":
            self.transformer = PerceptionTransformerV2(**args)
        else:
            self.transformer = PerceptionTransformer(**args)
        self.embed_dims = self.transformer.embed_dims
        self._init_layers()

    def init_weights(self):
        self.transformer.init_weights()

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)
        if (version is None or version < 2) and self.__class__ is DETRHead:
            convert_dict = {
                ".self_attn.": ".attentions.0.",
                ".ffn.": ".ffns.0.",
                ".multihead_attn.": ".attentions.1.",
                ".decoder.norm.": ".decoder.post_norm.",
            }
            state_dict_keys = list(state_dict.keys())
            for k in state_dict_keys:
                for ori_key, convert_key in convert_dict.items():
                    if ori_key in k:
                        convert_key = k.replace(ori_key, convert_key)
                        state_dict[convert_key] = state_dict[k]
                        del state_dict[k]

        super(AnchorFreeHead, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class BEVFormerHead(DETRHead):
    def __init__(
        self,
        *args,
        with_box_refine=False,
        as_two_stage=False,
        transformer=None,
        bbox_coder=None,
        num_cls_fcs=2,
        code_weights=None,
        bev_h=30,
        bev_w=30,
        **kwargs,
    ):

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer["as_two_stage"] = self.as_two_stage
        self.code_size = 10
        self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        if isinstance(bbox_coder, dict):
            cfg = bbox_coder.copy()
            cfg.pop("type", None)
            self.bbox_coder = NMSFreeCoder(**cfg)
        self.pc_range = self.bbox_coder.pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1
        super(BEVFormerHead, self).__init__(*args, transformer=transformer, **kwargs)
        self.code_weights = nn.Parameter(
            torch.tensor(self.code_weights, requires_grad=False), requires_grad=False
        )

    def _init_layers(self):
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        num_pred = (
            (self.transformer.decoder.num_layers + 1)
            if self.as_two_stage
            else self.transformer.decoder.num_layers
        )

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList([fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList([reg_branch for _ in range(num_pred)])

        if not self.as_two_stage:
            self.bev_embedding = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)
            self.query_embedding = nn.Embedding(self.num_query, self.embed_dims * 2)

    def forward(self, mlvl_feats, img_metas, prev_bev=None, only_bev=False):
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        object_query_embeds = self.query_embedding.weight.to(dtype)
        bev_queries = self.bev_embedding.weight.to(dtype)
        bev_mask = torch.zeros(
            (bs, self.bev_h, self.bev_w), device=bev_queries.device
        ).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)

        if (
            only_bev
        ):  # only use encoder to obtain BEV features, TODO: refine the workaround
            return self.transformer.get_bev_features(
                mlvl_feats,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )
        else:
            outputs = self.transformer(
                mlvl_feats,
                bev_queries,
                object_query_embeds,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
                bev_pos=bev_pos,
                reg_branches=self.reg_branches
                if self.with_box_refine
                else None,  # noqa:E501
                cls_branches=self.cls_branches if self.as_two_stage else None,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )
        bev_embed, hs, init_reference, inter_references = outputs
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])

            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
            tmp[..., 0:1] = (
                tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            )
            tmp[..., 1:2] = (
                tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            )
            tmp[..., 4:5] = (
                tmp[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
            )

            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)

        outs = {
            "bev_embed": bev_embed,
            "all_cls_scores": outputs_classes,
            "all_bbox_preds": outputs_coords,
            "enc_cls_scores": None,
            "enc_bbox_preds": None,
        }
        return outs

    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        preds_dicts = self.bbox_coder.decode(preds_dicts)

        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds["bboxes"]

            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5

            code_size = bboxes.shape[-1]
            bboxes = LiDARInstance3DBoxes(bboxes, code_size)
            scores, labels = preds["scores"], preds["labels"]
            ret_list.append([bboxes, scores, labels])

        return ret_list


class BEVFormerHead_GroupDETR(BEVFormerHead):
    def __init__(self, *args, group_detr=1, **kwargs):
        self.group_detr = group_detr
        kwargs["num_query"] = group_detr * kwargs["num_query"]
        super().__init__(*args, **kwargs)

    def forward(self, mlvl_feats, img_metas, prev_bev=None, only_bev=False):
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        object_query_embeds = self.query_embedding.weight.to(dtype)
        if not self.training:
            object_query_embeds = object_query_embeds[
                : self.num_query // self.group_detr
            ]
        bev_queries = self.bev_embedding.weight.to(dtype)

        bev_mask = torch.zeros(
            (bs, self.bev_h, self.bev_w), device=bev_queries.device
        ).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)
        outputs = self.transformer(
            mlvl_feats,
            bev_queries,
            object_query_embeds,
            self.bev_h,
            self.bev_w,
            grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
            bev_pos=bev_pos,
            reg_branches=self.reg_branches
            if self.with_box_refine
            else None,  # noqa:E501
            cls_branches=self.cls_branches if self.as_two_stage else None,
            img_metas=img_metas,
            prev_bev=prev_bev,
        )

        bev_embed, hs, init_reference, inter_references = outputs
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
            tmp[..., 0:1] = (
                tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            )
            tmp[..., 1:2] = (
                tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            )
            tmp[..., 4:5] = (
                tmp[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
            )
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)

        outs = {
            "bev_embed": bev_embed,
            "all_cls_scores": outputs_classes,
            "all_bbox_preds": outputs_coords,
            "enc_cls_scores": None,
            "enc_bbox_preds": None,
        }

        return outs


def denormalize_bbox(normalized_bboxes, pc_range):
    rot_sine = normalized_bboxes[..., 6:7]

    rot_cosine = normalized_bboxes[..., 7:8]
    rot = torch.atan2(rot_sine, rot_cosine)

    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 4:5]

    w = normalized_bboxes[..., 2:3]
    l = normalized_bboxes[..., 3:4]
    h = normalized_bboxes[..., 5:6]

    w = w.exp()
    l = l.exp()
    h = h.exp()
    if normalized_bboxes.size(-1) > 8:
        vx = normalized_bboxes[:, 8:9]
        vy = normalized_bboxes[:, 9:10]
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot, vx, vy], dim=-1)
    else:
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot], dim=-1)
    return denormalized_bboxes


# ============================================================================
# BBOX CODERS
# ============================================================================


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


class NMSFreeCoder(BaseBBoxCoder):
    def __init__(
        self,
        pc_range,
        voxel_size=None,
        post_center_range=None,
        max_num=100,
        score_threshold=None,
        num_classes=10,
    ):
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes

    def encode(self):
        pass

    def decode_single(self, cls_scores, bbox_preds):
        max_num = self.max_num

        cls_scores = cls_scores.sigmoid()
        scores, indexs = cls_scores.view(-1).topk(max_num)
        labels = indexs % self.num_classes
        bbox_index = indexs // self.num_classes
        bbox_preds = bbox_preds[bbox_index]

        final_box_preds = denormalize_bbox(bbox_preds, self.pc_range)
        final_scores = scores
        final_preds = labels

        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(
                self.post_center_range, device=scores.device
            )
            mask = (final_box_preds[..., :3] >= self.post_center_range[:3]).all(1)
            mask &= (final_box_preds[..., :3] <= self.post_center_range[3:]).all(1)

            boxes3d = final_box_preds[mask]
            scores = final_scores[mask]

            labels = final_preds[mask]
            predictions_dict = {"bboxes": boxes3d, "scores": scores, "labels": labels}

        return predictions_dict

    def decode(self, preds_dicts):
        all_cls_scores = preds_dicts["all_cls_scores"][-1]
        all_bbox_preds = preds_dicts["all_bbox_preds"][-1]

        batch_size = all_cls_scores.size()[0]
        predictions_list = []
        for i in range(batch_size):
            predictions_list.append(
                self.decode_single(all_cls_scores[i], all_bbox_preds[i])
            )
        return predictions_list
