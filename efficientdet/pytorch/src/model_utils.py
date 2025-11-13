# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Efficientdet model and input utils

Apdapted from : https://github.com/rwightman/efficientdet-pytorch
License : https://github.com/rwightman/efficientdet-pytorch/blob/master/LICENSE
"""

import torch
import numpy as np
from copy import deepcopy
from PIL import Image

from omegaconf import OmegaConf
from timm.models import load_checkpoint
from torch.hub import load_state_dict_from_url

from .efficientdet import EfficientDet


# ============================================================================
# INPUT UTILS
# ============================================================================

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
FILL_COLOR = tuple([int(round(255 * x)) for x in IMAGENET_DEFAULT_MEAN])
NORM_MEAN = torch.tensor([x * 255 for x in IMAGENET_DEFAULT_MEAN])
NORM_STD = torch.tensor([x * 255 for x in IMAGENET_DEFAULT_STD])


def resize_and_pad(
    img, target_size=(512, 512), interpolation=Image.BILINEAR, fill_color=FILL_COLOR
):

    width, height = img.size
    img_scale_y = target_size[0] / height
    img_scale_x = target_size[1] / width
    img_scale = min(img_scale_y, img_scale_x)
    scaled_h = int(height * img_scale)
    scaled_w = int(width * img_scale)
    new_img = Image.new("RGB", (target_size[1], target_size[0]), color=fill_color)
    img_resized = img.resize((scaled_w, scaled_h), interpolation)
    new_img.paste(img_resized, (0, 0))

    return new_img


def preprocess_image(img_path, target_size=(512, 512)):

    img = Image.open(img_path).convert("RGB")
    img_padded = resize_and_pad(img, target_size, fill_color=FILL_COLOR)
    np_img = np.array(img_padded, dtype=np.uint8)
    if np_img.ndim < 3:
        np_img = np.expand_dims(np_img, axis=-1)
    np_img = np.moveaxis(np_img, 2, 0)
    img_tensor = torch.from_numpy(np_img).float()
    mean = NORM_MEAN.view(3, 1, 1)
    std = NORM_STD.view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    img_tensor = img_tensor.unsqueeze(0)

    return img_tensor


# ============================================================================
# MODEL UTILS
# ============================================================================


def load_pretrained(model, url, filter_fn=None, strict=True):
    state_dict = load_state_dict_from_url(url, progress=False, map_location="cpu")
    if filter_fn is not None:
        state_dict = filter_fn(state_dict)
    model.load_state_dict(state_dict, strict=strict)


def default_detection_model_configs():
    h = OmegaConf.create()
    h.name = "tf_efficientdet_d1"
    h.backbone_name = "tf_efficientnet_b1"
    h.backbone_args = None
    h.backbone_indices = None
    h.image_size = (640, 640)
    h.num_classes = 90
    h.min_level = 3
    h.max_level = 7
    h.num_levels = h.max_level - h.min_level + 1
    h.num_scales = 3
    h.aspect_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
    h.anchor_scale = 4.0
    h.pad_type = "same"
    h.act_type = "swish"
    h.norm_layer = None
    h.norm_kwargs = dict(eps=0.001, momentum=0.01)
    h.box_class_repeats = 3
    h.fpn_cell_repeats = 3
    h.fpn_channels = 88
    h.separable_conv = True
    h.apply_resample_bn = True
    h.conv_bn_relu_pattern = False
    h.downsample_type = "max"
    h.upsample_type = "nearest"
    h.redundant_bias = True
    h.head_bn_level_first = False
    h.head_act_type = None
    h.fpn_name = None
    h.fpn_config = None
    h.fpn_drop_path_rate = 0.0
    h.alpha = 0.25
    h.gamma = 1.5
    h.label_smoothing = 0.0
    h.legacy_focal = False
    h.jit_loss = False
    h.delta = 0.1
    h.box_loss_weight = 50.0
    h.soft_nms = False
    h.max_detection_points = 5000
    h.max_det_per_image = 100

    return h


efficientdet_model_param_dict = dict(
    tf_efficientdet_d0=dict(
        name="tf_efficientdet_d0",
        backbone_name="tf_efficientnet_b0",
        image_size=(512, 512),
        fpn_channels=64,
        fpn_cell_repeats=3,
        box_class_repeats=3,
        backbone_args=dict(drop_path_rate=0.2),
        url="https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d0_34-f153e0cf.pth",
    ),
    tf_efficientdet_d1=dict(
        name="tf_efficientdet_d1",
        backbone_name="tf_efficientnet_b1",
        image_size=(640, 640),
        fpn_channels=88,
        fpn_cell_repeats=4,
        box_class_repeats=3,
        backbone_args=dict(drop_path_rate=0.2),
        url="https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d1_40-a30f94af.pth",
    ),
    tf_efficientdet_d2=dict(
        name="tf_efficientdet_d2",
        backbone_name="tf_efficientnet_b2",
        image_size=(768, 768),
        fpn_channels=112,
        fpn_cell_repeats=5,
        box_class_repeats=3,
        backbone_args=dict(drop_path_rate=0.2),
        url="https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d2_43-8107aa99.pth",
    ),
    tf_efficientdet_d3=dict(
        name="tf_efficientdet_d3",
        backbone_name="tf_efficientnet_b3",
        image_size=(896, 896),
        fpn_channels=160,
        fpn_cell_repeats=6,
        box_class_repeats=4,
        backbone_args=dict(drop_path_rate=0.2),
        url="https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d3_47-0b525f35.pth",
    ),
    tf_efficientdet_d4=dict(
        name="tf_efficientdet_d4",
        backbone_name="tf_efficientnet_b4",
        image_size=(1024, 1024),
        fpn_channels=224,
        fpn_cell_repeats=7,
        box_class_repeats=4,
        backbone_args=dict(drop_path_rate=0.2),
        url="https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d4_49-f56376d9.pth",
    ),
    tf_efficientdet_d5=dict(
        name="tf_efficientdet_d5",
        backbone_name="tf_efficientnet_b5",
        image_size=(1280, 1280),
        fpn_channels=288,
        fpn_cell_repeats=7,
        box_class_repeats=4,
        backbone_args=dict(drop_path_rate=0.2),
        url="https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d5_51-c79f9be6.pth",
    ),
    tf_efficientdet_d6=dict(
        name="tf_efficientdet_d6",
        backbone_name="tf_efficientnet_b6",
        image_size=(1280, 1280),
        fpn_channels=384,
        fpn_cell_repeats=8,
        box_class_repeats=5,
        fpn_name="bifpn_sum",
        backbone_args=dict(drop_path_rate=0.2),
        url="https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d6_52-4eda3773.pth",
    ),
    tf_efficientdet_d7=dict(
        name="tf_efficientdet_d7",
        backbone_name="tf_efficientnet_b6",
        image_size=(1536, 1536),
        fpn_channels=384,
        fpn_cell_repeats=8,
        box_class_repeats=5,
        anchor_scale=5.0,
        fpn_name="bifpn_sum",
        backbone_args=dict(drop_path_rate=0.2),
        url="https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d7_53-6d1d7a95.pth",
    ),
    tf_efficientdet_d7x=dict(
        name="tf_efficientdet_d7x",
        backbone_name="tf_efficientnet_b7",
        image_size=(1536, 1536),
        fpn_channels=384,
        fpn_cell_repeats=8,
        box_class_repeats=5,
        anchor_scale=4.0,
        max_level=8,
        fpn_name="bifpn_sum",
        backbone_args=dict(drop_path_rate=0.2),
        url="https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d7x-f390b87c.pth",
    ),
)


def get_efficientdet_config(model_name="tf_efficientdet_d1"):
    h = default_detection_model_configs()
    h.update(efficientdet_model_param_dict[model_name])
    h.num_levels = h.max_level - h.min_level + 1
    h = deepcopy(h)
    return h


def create_model_from_config(
    config, pretrained=False, checkpoint_path="", checkpoint_ema=False, **kwargs
):
    pretrained_backbone = kwargs.pop("pretrained_backbone", True)
    if pretrained or checkpoint_path:
        pretrained_backbone = False

    model = EfficientDet(config, pretrained_backbone=pretrained_backbone, **kwargs)

    if pretrained:
        load_pretrained(model, config.url)

    if checkpoint_path:
        load_checkpoint(model, checkpoint_path, use_ema=checkpoint_ema)

    return model


def create_model(
    model_name, pretrained=False, checkpoint_path="", checkpoint_ema=False, **kwargs
):
    config = get_efficientdet_config(model_name)
    return create_model_from_config(
        config,
        pretrained=pretrained,
        checkpoint_path=checkpoint_path,
        checkpoint_ema=checkpoint_ema,
        **kwargs
    )
