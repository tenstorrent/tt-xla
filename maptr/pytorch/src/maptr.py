# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
MAPTR model implementation

Apdapted from: https://github.com/hustvl/MapTR.git

MIT License

Copyright (c) 2022 Hust Vision Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

# ============================================================================
# IMPORTS
# ============================================================================

import copy
import math
import sys
from collections import OrderedDict
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from torch.nn import init
from torch.nn.init import normal_
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter

from mmcv import ConfigDict
from mmcv.cnn import (
    CONV_LAYERS,
    Linear,
    MODELS as MMCV_MODELS,
    build_conv_layer,
    build_norm_layer,
    constant_init,
    xavier_init,
)
from mmcv.cnn.bricks.registry import (
    ATTENTION,
    TRANSFORMER_LAYER,
    TRANSFORMER_LAYER_SEQUENCE,
)
from mmcv.cnn.bricks.transformer import (
    TransformerLayerSequence,
    build_attention,
    build_feedforward_network,
    build_transformer_layer_sequence,
)
from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
from mmcv.runner.base_module import BaseModule, ModuleList
from mmcv.utils import Registry

from mmdet.core.bbox import BaseBBoxCoder, build_bbox_coder
from mmdet.core.bbox.builder import BBOX_CODERS
from mmdet.core.bbox.transforms import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from mmdet.models import DETECTORS, HEADS
from mmdet.models.backbones.resnet import BasicBlock
from mmdet.models.builder import BACKBONES, NECKS
from mmdet.models.dense_heads import DETRHead
from mmdet.models.detectors import BaseDetector
from mmdet.models.utils.builder import TRANSFORMER
from mmdet.models.utils.transformer import inverse_sigmoid

from third_party.tt_forge_models.pointpillars.pytorch.src.utils import hard_voxelize

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


class _Voxelization(nn.Module):
    def forward(
        points,
        voxel_size,
        coors_range,
        max_points=35,
        max_voxels=20000,
        deterministic=True,
    ):

        voxels = points.new_zeros(size=(max_voxels, max_points, points.size(1)))
        coors = points.new_zeros(size=(max_voxels, 3), dtype=torch.int)
        num_points_per_voxel = points.new_zeros(size=(max_voxels,), dtype=torch.int)
        voxel_num = hard_voxelize(
            points,
            voxels,
            coors,
            num_points_per_voxel,
            voxel_size,
            coors_range,
            max_points,
            max_voxels,
            3,
        )
        voxels_out = voxels[:voxel_num]
        coors_out = coors[:voxel_num].flip(-1)
        num_points_per_voxel_out = num_points_per_voxel[:voxel_num]
        return voxels_out, coors_out, num_points_per_voxel_out


voxelization = _Voxelization.forward


class Voxelization(nn.Module):
    def __init__(
        self,
        voxel_size,
        point_cloud_range,
        max_num_points,
        max_voxels=20000,
        deterministic=True,
    ):

        super(Voxelization, self).__init__()
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.max_num_points = max_num_points
        if isinstance(max_voxels, tuple):
            self.max_voxels = max_voxels
        else:
            self.max_voxels = _pair(max_voxels)
        self.deterministic = deterministic

        point_cloud_range = torch.tensor(point_cloud_range, dtype=torch.float32)
        voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
        grid_size = (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
        grid_size = torch.round(grid_size).long()
        input_feat_shape = grid_size[:2]
        self.grid_size = grid_size
        self.pcd_shape = [*input_feat_shape, 1]

    def forward(self, input):
        max_voxels = self.max_voxels[1]

        return voxelization(
            input,
            self.voxel_size,
            self.point_cloud_range,
            self.max_num_points,
            max_voxels,
            self.deterministic,
        )


"""
CPU Implementation of Quickcumsum
"""


def Quickcumsum_cpu(feats, coords, ranks, B, D, H, W):
    """
    - Processes intervals of points that map to same BEV location
    - For each interval, sums features and stores at BEV position
    - Uses indexing: out[b_idx][d_idx][h_idx][w_idx][c_idx]
    """
    N, C = feats.shape

    # Convert float dimensions to int
    B = int(B)
    D = int(D)
    H = int(H)
    W = int(W)

    # Sort by ranks
    indices = ranks.argsort()
    feats_sorted = feats[indices]
    coords_sorted = coords[indices]  # coords are [h_idx, w_idx, d_idx, b_idx]
    ranks_sorted = ranks[indices]

    # Find interval boundaries
    kept = torch.ones(N, device=feats.device, dtype=torch.bool)
    kept[1:] = ranks_sorted[1:] != ranks_sorted[:-1]
    interval_starts = torch.where(kept)[0].int()
    interval_lengths = torch.zeros_like(interval_starts)
    interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
    interval_lengths[-1] = N - interval_starts[-1]

    n_intervals = interval_lengths.shape[0]

    # Initialize output tensor [B, D, H, W, C]
    output = torch.zeros((B, D, H, W, C), dtype=feats.dtype, device=feats.device)

    for index in range(n_intervals):
        interval_start = interval_starts[index].item()
        interval_length = interval_lengths[index].item()

        # Get geometry features for this interval
        cur_geom_feats = coords_sorted[interval_start]  # [h_idx, w_idx, d_idx, b_idx]

        # Extract indices
        h_idx = cur_geom_feats[0].item()  # cur_geom_feats[0]
        w_idx = cur_geom_feats[1].item()  # cur_geom_feats[1]
        d_idx = cur_geom_feats[2].item()  # cur_geom_feats[2]
        b_idx = cur_geom_feats[3].item()  # cur_geom_feats[3]

        # Process each channel
        for cur_c in range(C):
            psum = 0.0
            for i in range(interval_length):
                point_idx = interval_start + i
                psum += feats_sorted[point_idx, cur_c].item()
            output[b_idx, d_idx, h_idx, w_idx, cur_c] = psum

    return output


def bev_pool(feats, coords, B, D, H, W):
    assert feats.shape[0] == coords.shape[0]

    ranks = (
        coords[:, 0] * (W * D * B)
        + coords[:, 1] * (D * B)
        + coords[:, 2] * B
        + coords[:, 3]
    )
    indices = ranks.argsort()
    feats, coords, ranks = feats[indices], coords[indices], ranks[indices]

    x = Quickcumsum_cpu(feats, coords, ranks, B, D, H, W)
    x = x.permute(0, 4, 1, 2, 3).contiguous()
    return x


def denormalize_2d_bbox(bboxes, pc_range):
    bboxes = bbox_cxcywh_to_xyxy(bboxes)
    bboxes[..., 0::2] = bboxes[..., 0::2] * (pc_range[3] - pc_range[0]) + pc_range[0]
    bboxes[..., 1::2] = bboxes[..., 1::2] * (pc_range[4] - pc_range[1]) + pc_range[1]
    return bboxes


def denormalize_2d_pts(pts, pc_range):
    new_pts = pts.clone()
    new_pts[..., 0:1] = pts[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
    new_pts[..., 1:2] = pts[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
    return new_pts


# ============================================================================
# MODEL CONFIG
# ============================================================================

point_cloud_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
voxel_size = [0.15, 0.15, 4]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)

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
map_classes = ["divider", "ped_crossing", "boundary"]
fixed_ptsnum_per_gt_line = 20
fixed_ptsnum_per_pred_line = 20
eval_use_same_gt_sample_num_flag = True
num_map_classes = len(map_classes)
input_modality = dict(
    use_lidar=False, use_camera=True, use_radar=False, use_map=False, use_external=True
)
_dim_ = 256
_pos_dim_ = _dim_ // 2
_ffn_dim_ = _dim_ * 2
_num_levels_ = 1
bev_h_ = 200
bev_w_ = 100
queue_length = 1
dbound = [1.0, 35.0, 0.5]

lidar_point_cloud_range = [-15.0, -30.0, -5.0, 15.0, 30.0, 3.0]
lidar_encoder_cfg = dict(
    voxelize=dict(
        max_num_points=10,
        point_cloud_range=lidar_point_cloud_range,
        voxel_size=voxel_size,
        max_voxels=[90000, 120000],
    ),
    backbone=dict(
        type="SparseEncoder",
        in_channels=5,
        sparse_shape=[300, 600, 41],
        output_channels=128,
        order=("conv", "norm", "act"),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
        encoder_paddings=([0, 0, 1], [0, 0, 1], [0, 0, [1, 1, 0]], [0, 0]),
        block_type="basicblock",
    ),
)

fuser_cfg = dict(
    type="ConvFuser",
    in_channels=[_dim_, 256],
    out_channels=_dim_,
)

lss_cfg = dict(
    type="LSSTransform",
    in_channels=_dim_,
    out_channels=_dim_,
    feat_down_sample=32,
    pc_range=point_cloud_range,
    voxel_size=voxel_size,
    dbound=dbound,
    downsample=2,
)

gkt_cfg = dict(
    type="GeometrySptialCrossAttention",
    pc_range=point_cloud_range,
    attention=dict(
        type="GeometryKernelAttention",
        embed_dims=_dim_,
        num_heads=4,
        dilation=1,
        kernel_size=(3, 5),
        num_levels=_num_levels_,
    ),
    embed_dims=_dim_,
)

model_cfg = dict(
    type="MapTR",
    use_grid_mask=True,
    video_test_mode=False,
    pretrained=dict(img=None),
    img_backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=1,
        norm_cfg=dict(type="BN", requires_grad=False),
        norm_eval=True,
        style="pytorch",
    ),
    img_neck=dict(
        type="FPN",
        in_channels=[2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs="on_output",
        num_outs=_num_levels_,
        relu_before_extra_convs=True,
    ),
    pts_bbox_head=dict(
        type="MapTRHead",
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=900,
        num_vec=50,
        num_pts_per_vec=fixed_ptsnum_per_pred_line,
        num_pts_per_gt_vec=fixed_ptsnum_per_gt_line,
        dir_interval=1,
        query_embed_type="instance_pts",
        transform_method="minmax",
        gt_shift_pts_pattern="v2",
        num_classes=num_map_classes,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        code_size=2,
        code_weights=[1.0, 1.0, 1.0, 1.0],
        transformer=dict(
            type="MapTRPerceptionTransformer",
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=_dim_,
            encoder=dict(
                type="BEVFormerEncoder",
                num_layers=1,
                pc_range=point_cloud_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type="BEVFormerLayer",
                    attn_cfgs=[
                        dict(
                            type="TemporalSelfAttention", embed_dims=_dim_, num_levels=1
                        ),
                        dict(
                            type="SpatialCrossAttention",
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type="MSDeformableAttention3D",
                                embed_dims=_dim_,
                                num_points=8,
                                num_levels=_num_levels_,
                            ),
                            embed_dims=_dim_,
                        ),
                    ],
                    feedforward_channels=_ffn_dim_,
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
                type="MapTRDecoder",
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type="DetrTransformerDecoderLayer",
                    attn_cfgs=[
                        dict(
                            type="MultiheadAttention",
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1,
                        ),
                        dict(
                            type="CustomMSDeformableAttention",
                            embed_dims=_dim_,
                            num_levels=1,
                        ),
                    ],
                    feedforward_channels=_ffn_dim_,
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
            type="MapTRNMSFreeCoder",
            post_center_range=[-20, -35, -20, -35, 20, 35, 20, 35],
            pc_range=point_cloud_range,
            max_num=50,
            voxel_size=voxel_size,
            num_classes=num_map_classes,
        ),
        positional_encoding=dict(
            type="LearnedPositionalEncoding",
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
        ),
        loss_cls=dict(
            type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0
        ),
    ),
)

# ============================================================================
# BUILDERS
# ============================================================================


def build_backbone(cfg):
    return BACKBONES.build(cfg)


def build_neck(cfg):
    return NECKS.build(cfg)


def build_head(cfg):
    return HEADS.build(cfg)


MODELS = Registry("models", parent=MMCV_MODELS)
MIDDLE_ENCODERS = MODELS


def build_middle_encoder(cfg):
    return MIDDLE_ENCODERS.build(cfg)


FUSERS = Registry("fusers")


def build_fuser(cfg):
    return FUSERS.build(cfg)


def build_detector(cfg, train_cfg=None, test_cfg=None):
    return DETECTORS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg)
    )


def build_model(cfg, train_cfg=None, test_cfg=None):
    return build_detector(cfg, train_cfg=train_cfg, test_cfg=test_cfg)


# ============================================================================
# BBOX CODERS
# ============================================================================


@BBOX_CODERS.register_module()
class MapTRNMSFreeCoder(BaseBBoxCoder):
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

    def decode_single(self, cls_scores, bbox_preds, pts_preds):
        max_num = self.max_num

        cls_scores = cls_scores.sigmoid()
        scores, indexs = cls_scores.view(-1).topk(max_num)
        labels = indexs % self.num_classes
        bbox_index = indexs // self.num_classes
        bbox_preds = bbox_preds[bbox_index]
        pts_preds = pts_preds[bbox_index]

        final_box_preds = denormalize_2d_bbox(bbox_preds, self.pc_range)
        final_pts_preds = denormalize_2d_pts(pts_preds, self.pc_range)
        final_scores = scores
        final_preds = labels

        self.post_center_range = torch.tensor(
            self.post_center_range, device=scores.device
        )
        mask = (final_box_preds[..., :4] >= self.post_center_range[:4]).all(1)
        mask &= (final_box_preds[..., :4] <= self.post_center_range[4:]).all(1)

        boxes3d = final_box_preds[mask]
        scores = final_scores[mask]
        pts = final_pts_preds[mask]
        labels = final_preds[mask]
        predictions_dict = {
            "bboxes": boxes3d,
            "scores": scores,
            "labels": labels,
            "pts": pts,
        }
        return predictions_dict

    def decode(self, preds_dicts):

        all_cls_scores = preds_dicts["all_cls_scores"][-1]
        all_bbox_preds = preds_dicts["all_bbox_preds"][-1]
        all_pts_preds = preds_dicts["all_pts_preds"][-1]
        batch_size = all_cls_scores.size()[0]
        predictions_list = []
        for i in range(batch_size):
            predictions_list.append(
                self.decode_single(
                    all_cls_scores[i], all_bbox_preds[i], all_pts_preds[i]
                )
            )
        return predictions_list


# ============================================================================
# SparseConv
# ============================================================================


class SparseModule(nn.Module):
    pass


class SparseBasicBlock(BasicBlock, SparseModule):

    expansion = 1

    def __init__(
        self, inplanes, planes, stride=1, downsample=None, conv_cfg=None, norm_cfg=None
    ):
        SparseModule.__init__(self)
        BasicBlock.__init__(
            self,
            inplanes,
            planes,
            stride=stride,
            downsample=downsample,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
        )

    def forward(self, x):
        identity = x.features

        assert x.features.dim() == 2, f"x.features.dim()={x.features.dim()}"

        out = self.conv1(x)
        out.features = self.norm1(out.features)
        out.features = self.relu(out.features)

        out = self.conv2(out)
        out.features = self.norm2(out.features)

        if self.downsample is not None:
            identity = self.downsample(x)

        out.features += identity
        out.features = self.relu(out.features)

        return out


def _calculate_fan_in_and_fan_out_hwio(tensor):
    dimensions = tensor.ndimension()

    if dimensions == 2:
        fan_in = tensor.size(-2)
        fan_out = tensor.size(-1)
    else:
        num_input_fmaps = tensor.size(-2)
        num_output_fmaps = tensor.size(-1)
        receptive_field_size = 1
        if tensor.dim() > 2:
            receptive_field_size = tensor[..., 0, 0].numel()
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def get_deconv_output_size(
    input_size, kernel_size, stride, padding, dilation, output_padding
):
    ndim = len(input_size)
    output_size = []
    for i in range(ndim):
        if kernel_size[i] == -1:
            raise ValueError("deconv don't support kernel_size < 0")
        size = (
            (input_size[i] - 1) * stride[i]
            - 2 * padding[i]
            + kernel_size[i]
            + output_padding[i]
        )
        output_size.append(size)
    return output_size


def get_conv_output_size(input_size, kernel_size, stride, padding, dilation):
    ndim = len(input_size)
    output_size = []
    for i in range(ndim):
        size = (
            input_size[i] + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1
        ) // stride[i] + 1
        if kernel_size[i] == -1:
            output_size.append(1)
        else:
            output_size.append(size)
    return output_size


def get_valid_out_pos_3d(
    input_pos: torch.Tensor,
    kernel_size: List[int],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    out_spatial_shape: List[int],
) -> Tuple[torch.Tensor, int]:
    NDim = 3
    lowers = torch.zeros(NDim, dtype=torch.int32)
    uppers = torch.zeros(NDim, dtype=torch.int32)
    counter = torch.zeros(NDim, dtype=torch.int32)
    counter_size = torch.zeros(NDim, dtype=torch.int32)

    # Calculate bounds
    for i in range(NDim):
        lowers[i] = (
            input_pos[i]
            - (kernel_size[i] - 1) * dilation[i]
            - 1
            + stride[i]
            + padding[i]
        ) // stride[i]
        uppers[i] = (input_pos[i] + padding[i]) // stride[i]

    # Calculate counter sizes
    num_points = 1
    for i in range(NDim):
        counter_size[i] = (uppers[i] - lowers[i]) // dilation[i] + 1
        num_points *= counter_size[i].item()

    # Initialize counter
    counter.zero_()

    # Generate valid points
    valid_points = []
    point_counter = 0

    for i in range(num_points):
        valid = True
        m = 1
        offset = 0
        point = torch.zeros(NDim + 1, dtype=torch.int32)

        # Process dimensions in reverse order
        for j in range(NDim - 1, -1, -1):
            val = uppers[j] - counter[j] * dilation[j]
            point[j] = val

            if val < 0 or val > out_spatial_shape[j] - 1:
                valid = False

            offset += m * (input_pos[j] - val * stride[j] + padding[j]) // dilation[j]
            m *= kernel_size[j]

        point[NDim] = offset

        if valid:
            valid_points.append(point.clone())
            point_counter += 1

        # Update counter
        counter[NDim - 1] += 1
        for c in range(NDim - 1, -1, -1):
            if counter[c] == counter_size[c] and c > 0:
                counter[c - 1] += 1
                counter[c] = 0

    if valid_points:
        return torch.stack(valid_points), point_counter
    else:
        return torch.empty((0, NDim + 1), dtype=torch.int32), point_counter


def row_array_idx_3d(point: torch.Tensor, spatial_shape: List[int]) -> int:
    return (
        point[0] * spatial_shape[1] * spatial_shape[2]
        + point[1] * spatial_shape[2]
        + point[2]
    ).item()


def get_indice_pairs_3d_cpu(
    indices: torch.Tensor,
    batch_size: int,
    out_shape: List[int],
    spatial_shape: List[int],
    ksize: List[int],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    out_padding: List[int],
    subm: int,
    transpose: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Python implementation of getIndicePairs function for both SubManifold and regular convolution
    """

    num_act_in = indices.shape[0]

    if num_act_in == 0:
        kernel_volume = torch.tensor(ksize).prod().item()
        indice_pairs = torch.full((kernel_volume, 2, 1000), -1, dtype=torch.int32)
        out_indices = torch.zeros((0, 4), dtype=torch.int32)
        indice_num = torch.zeros(kernel_volume, dtype=torch.int32)
        return out_indices, indice_pairs, indice_num

    # Calculate spatial volume
    spatial_volume = torch.tensor(out_shape).prod().item()
    kernel_volume = torch.tensor(ksize).prod().item()

    # Initialize grids
    total_grid_size = batch_size * spatial_volume
    grids_out = torch.full((total_grid_size,), -1, dtype=torch.int32)

    # Initialize output structures
    indice_num = torch.zeros(kernel_volume, dtype=torch.int32)
    max_indices = num_act_in  # Use same size as input for consistent shape
    indice_pairs = torch.full((kernel_volume, 2, max_indices), -1, dtype=torch.int32)

    if subm == 1:
        # SubM convolution
        # Populate grids with input indices
        for j in range(num_act_in):
            batch_idx = indices[j, 0].item()
            spatial_coords = indices[j, 1:4]
            index = (
                row_array_idx_3d(spatial_coords, out_shape) + spatial_volume * batch_idx
            )
            grids_out[index] = j

        # Process each input sequentially
        for j in range(num_act_in):
            batch_idx = indices[j, 0].item()
            input_pos = indices[j, 1:4]

            # Get valid output positions
            valid_points, num_valid = get_valid_out_pos_3d(
                input_pos, ksize, stride, padding, dilation, out_shape
            )

            # Process each valid point
            for i in range(num_valid):
                point = valid_points[i]
                offset = point[3].item()  # kernel offset
                out_coords = point[:3]  # spatial coordinates

                # Calculate output index
                index = (
                    row_array_idx_3d(out_coords, out_shape) + spatial_volume * batch_idx
                )

                if grids_out[index] > -1:
                    current_slot = indice_num[offset].item()
                    indice_pairs[offset, 0, current_slot] = j
                    indice_pairs[offset, 1, current_slot] = grids_out[index]
                    indice_num[offset] += 1

        # Return original indices for SubM
        out_indices = indices.int()

    else:
        # Regular convolution (subm=0)
        out_indices_list = []
        num_act_out = 0

        for j in range(num_act_in):
            batch_idx = indices[j, 0].item()
            input_pos = indices[j, 1:4]

            # Get valid output positions for this input
            valid_points, num_valid = get_valid_out_pos_3d(
                input_pos, ksize, stride, padding, dilation, out_shape
            )

            # Process each valid point
            for i in range(num_valid):
                point = valid_points[i]
                offset = point[3].item()  # kernel offset
                out_coords = point[:3]  # spatial coordinates

                # Calculate grid index
                grid_idx = (
                    row_array_idx_3d(out_coords, out_shape) + spatial_volume * batch_idx
                )

                # Check if this output position is new
                if grids_out[grid_idx] == -1:
                    # New output position - add to output indices
                    out_indices_list.append(
                        [
                            batch_idx,
                            out_coords[0].item(),
                            out_coords[1].item(),
                            out_coords[2].item(),
                        ]
                    )
                    grids_out[grid_idx] = num_act_out
                    num_act_out += 1

                # Add indice pair
                current_slot = indice_num[offset].item()
                if current_slot < max_indices:
                    indice_pairs[offset, 0, current_slot] = j
                    indice_pairs[offset, 1, current_slot] = grids_out[grid_idx]
                    indice_num[offset] += 1

        # Convert output indices list to tensor
        if out_indices_list:
            out_indices = torch.tensor(out_indices_list, dtype=torch.int32)
        else:
            out_indices = torch.zeros((0, 4), dtype=torch.int32)

    return out_indices, indice_pairs, indice_num


def get_indice_conv_cpu_fp32(
    features: torch.Tensor,
    filters: torch.Tensor,
    indice_pairs: torch.Tensor,
    indice_pair_num: torch.Tensor,
    num_activate_out: int,
    inverse: int = 0,
    subm: int = 0,
) -> torch.Tensor:

    """
    Python implementation of sparse_conv_ext.indice_conv_fp32

    Performs sparse convolution using indice pairs that define the mapping between
    input and output features.

    Args:
        features: Input features [N, C_in] where N is number of active input points
        filters: Convolution weights [kernel_volume, C_in, C_out]
        indice_pairs: Index pairs [kernel_volume, 2, max_pairs] mapping input->output indices
        indice_pair_num: Number of valid pairs per kernel position [kernel_volume]
        num_activate_out: Number of output active points
        inverse: Whether this is inverse/transpose convolution (0=normal, 1=inverse)
        subm: Whether this is submanifold convolution (0=normal, 1=subm)

    Returns:
        output: Output features [num_activate_out, C_out]
    """

    # Get dimensions
    device = features.device
    dtype = features.dtype
    kernel_volume = indice_pairs.size(0)
    num_in_planes = features.size(1)
    num_out_planes = filters.size(-1)  # filters shape: [kernel_volume, C_in, C_out]

    # Move indice_pair_num to CPU for processing
    indice_pair_num_cpu = indice_pair_num.cpu()

    # Find the kernel position with maximum number of pairs
    indice_pair_max_size_iter = torch.argmax(indice_pair_num_cpu)
    indice_pair_max_offset = indice_pair_max_size_iter.item()
    indice_pair_max_size = indice_pair_num_cpu[indice_pair_max_offset].item()

    # Initialize output tensor
    output = torch.zeros(num_activate_out, num_out_planes, dtype=dtype, device=device)

    # Handle edge case where no pairs exist
    if indice_pair_max_size <= 0:
        return output

    # Reshape filters to [kernel_volume, C_in, C_out]
    filters = filters.view(-1, num_in_planes, num_out_planes)

    # Handle submanifold convolution center position
    # In SubM conv, the center kernel position doesn't need gather/scatter operations
    if subm == 1:
        # Direct matrix multiplication for center position
        output = torch.mm(features, filters[indice_pair_max_offset])

    # Process each kernel position
    for i in range(kernel_volume):
        n_hot = indice_pair_num_cpu[i].item()

        # Skip empty kernel positions or center position in SubM
        if n_hot <= 0 or (subm == 1 and i == indice_pair_max_offset):
            continue

        # GATHER operation: collect input features based on indice pairs
        if inverse == 0:
            # Normal convolution: gather using input indices
            input_indices = indice_pairs[i, 0, :n_hot]  # First column: input indices
        else:
            # Inverse convolution: gather using output indices (swapped)
            input_indices = indice_pairs[i, 1, :n_hot]  # Second column as input indices

        # Perform the gather operation
        input_buffer_blob = torch.index_select(features, 0, input_indices)

        # GEMM operation: matrix multiplication
        output_buffer_blob = torch.mm(input_buffer_blob, filters[i])

        # SCATTER-ADD operation: accumulate results to output positions
        if inverse == 0:
            # Normal convolution: scatter using output indices
            output_indices = indice_pairs[i, 1, :n_hot]  # Second column: output indices
        else:
            # Inverse convolution: scatter using input indices (swapped)
            output_indices = indice_pairs[
                i, 0, :n_hot
            ]  # First column as output indices

        # Perform scatter-add operation
        for j in range(n_hot):
            out_idx = output_indices[j].item()
            output[out_idx] += output_buffer_blob[j]

    return output


def get_indice_pairs(
    indices,
    batch_size,
    spatial_shape,
    ksize=3,
    stride=1,
    padding=0,
    dilation=1,
    out_padding=0,
    subm=False,
    transpose=False,
    grid=None,
):
    ndim = indices.shape[1] - 1
    if not isinstance(ksize, (list, tuple)):
        ksize = [ksize] * ndim
    if not isinstance(stride, (list, tuple)):
        stride = [stride] * ndim
    if not isinstance(padding, (list, tuple)):
        padding = [padding] * ndim
    if not isinstance(dilation, (list, tuple)):
        dilation = [dilation] * ndim
    if not isinstance(out_padding, (list, tuple)):
        out_padding = [out_padding] * ndim

    if not subm:
        if transpose:
            out_shape = get_deconv_output_size(
                spatial_shape, ksize, stride, padding, dilation, out_padding
            )
        else:
            out_shape = get_conv_output_size(
                spatial_shape, ksize, stride, padding, dilation
            )

    else:
        out_shape = spatial_shape
    if grid is None:

        op = get_indice_pairs_3d_cpu(
            indices,
            batch_size,
            out_shape,
            spatial_shape,
            ksize,
            stride,
            padding,
            dilation,
            out_padding,
            int(subm),
            int(transpose),
        )

        return op


def indice_conv(
    features,
    filters,
    indice_pairs,
    indice_pair_num,
    num_activate_out,
    inverse=False,
    subm=False,
):

    op = get_indice_conv_cpu_fp32(
        features,
        filters,
        indice_pairs,
        indice_pair_num,
        num_activate_out,
        int(inverse),
        int(subm),
    )

    return op


def indice_subm_conv(
    features,
    filters,
    indice_pairs,
    indice_pair_num,
    num_activate_out,
    inverse=False,
    subm=True,
):

    op = get_indice_conv_cpu_fp32(
        features,
        filters,
        indice_pairs,
        indice_pair_num,
        num_activate_out,
        int(inverse),
        int(subm),
    )

    return op


class SparseConvolution(SparseModule):
    def __init__(
        self,
        ndim,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        subm=False,
        output_padding=0,
        transposed=False,
        inverse=False,
        indice_key=None,
        fused_bn=False,
    ):
        super(SparseConvolution, self).__init__()
        assert groups == 1
        if not isinstance(kernel_size, (list, tuple)):
            kernel_size = [kernel_size] * ndim
        if not isinstance(stride, (list, tuple)):
            stride = [stride] * ndim
        if not isinstance(padding, (list, tuple)):
            padding = [padding] * ndim
        if not isinstance(dilation, (list, tuple)):
            dilation = [dilation] * ndim
        if not isinstance(output_padding, (list, tuple)):
            output_padding = [output_padding] * ndim

        for d, s in zip(dilation, stride):
            assert any([s == 1, d == 1]), "don't support this."

        self.ndim = ndim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.conv1x1 = np.prod(kernel_size) == 1
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.inverse = inverse
        self.output_padding = output_padding
        self.groups = groups
        self.subm = subm
        self.indice_key = indice_key
        self.fused_bn = fused_bn

        self.weight = Parameter(torch.Tensor(*kernel_size, in_channels, out_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = _calculate_fan_in_and_fan_out_hwio(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        assert isinstance(input, SparseConvTensor)
        features = input.features
        device = features.device
        indices = input.indices
        spatial_shape = input.spatial_shape
        batch_size = input.batch_size
        if not self.subm:
            if self.transposed:
                out_spatial_shape = get_deconv_output_size(
                    spatial_shape,
                    self.kernel_size,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.output_padding,
                )
            else:
                out_spatial_shape = get_conv_output_size(
                    spatial_shape,
                    self.kernel_size,
                    self.stride,
                    self.padding,
                    self.dilation,
                )

        else:
            out_spatial_shape = spatial_shape
        if self.conv1x1:
            features = torch.mm(
                input.features, self.weight.view(self.in_channels, self.out_channels)
            )
            if self.bias is not None:
                features += self.bias
            out_tensor = SparseConvTensor(
                features, input.indices, input.spatial_shape, input.batch_size
            )
            out_tensor.indice_dict = input.indice_dict
            out_tensor.grid = input.grid
            return out_tensor
        datas = input.find_indice_pair(self.indice_key)
        if self.inverse:
            assert datas is not None and self.indice_key is not None
            _, outids, indice_pairs, indice_pair_num, out_spatial_shape = datas
            assert indice_pairs.shape[0] == np.prod(
                self.kernel_size
            ), "inverse conv must have same kernel size as its couple conv"
        else:
            if self.indice_key is not None and datas is not None:
                outids, _, indice_pairs, indice_pair_num, _ = datas
            else:
                outids, indice_pairs, indice_pair_num = get_indice_pairs(
                    indices,
                    batch_size,
                    spatial_shape,
                    self.kernel_size,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.output_padding,
                    self.subm,
                    self.transposed,
                    grid=input.grid,
                )
                input.indice_dict[self.indice_key] = (
                    outids,
                    indices,
                    indice_pairs,
                    indice_pair_num,
                    spatial_shape,
                )
        if self.subm:
            out_features = indice_subm_conv(
                features,
                self.weight,
                indice_pairs.to(device),
                indice_pair_num,
                outids.shape[0],
            )
        else:
            if not self.inverse:
                out_features = indice_conv(
                    features,
                    self.weight,
                    indice_pairs.to(device),
                    indice_pair_num,
                    outids.shape[0],
                )

        if self.bias is not None:
            out_features += self.bias
        out_tensor = SparseConvTensor(
            out_features, outids, out_spatial_shape, batch_size
        )
        out_tensor.indice_dict = input.indice_dict
        out_tensor.grid = input.grid
        return out_tensor


@CONV_LAYERS.register_module(force=True)
class SubMConv3d(SparseConvolution):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        indice_key=None,
    ):
        super(SubMConv3d, self).__init__(
            3,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            True,
            indice_key=indice_key,
        )


@CONV_LAYERS.register_module(force=True)
class SparseConv3d(SparseConvolution):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        indice_key=None,
    ):
        super(SparseConv3d, self).__init__(
            3,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            indice_key=indice_key,
        )


@CONV_LAYERS.register_module(force=True)
class SubMConv3d(SparseConvolution):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        indice_key=None,
    ):
        super(SubMConv3d, self).__init__(
            3,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            True,
            indice_key=indice_key,
        )


def make_sparse_convmodule(
    in_channels,
    out_channels,
    kernel_size,
    indice_key,
    stride=1,
    padding=0,
    conv_type="SubMConv3d",
    norm_cfg=None,
    order=("conv", "norm", "act"),
):

    assert isinstance(order, tuple) and len(order) <= 3
    assert set(order) | {"conv", "norm", "act"} == {"conv", "norm", "act"}

    conv_cfg = dict(type=conv_type, indice_key=indice_key)

    layers = list()
    for layer in order:
        if layer == "conv":
            if conv_type not in [
                "SparseInverseConv3d",
                "SparseInverseConv2d",
                "SparseInverseConv1d",
            ]:
                layers.append(
                    build_conv_layer(
                        conv_cfg,
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride=stride,
                        padding=padding,
                        bias=False,
                    )
                )
            else:
                layers.append(
                    build_conv_layer(
                        conv_cfg, in_channels, out_channels, kernel_size, bias=False
                    )
                )
        elif layer == "norm":
            layers.append(build_norm_layer(norm_cfg, out_channels)[1])
        elif layer == "act":
            layers.append(nn.ReLU(inplace=True))

    layers = SparseSequential(*layers)
    return layers


def scatter_nd(indices, updates, shape):
    ret = torch.zeros(*shape, dtype=updates.dtype, device=updates.device)
    ndim = indices.shape[-1]
    output_shape = list(indices.shape[:-1]) + shape[indices.shape[-1] :]
    flatted_indices = indices.view(-1, ndim)
    slices = [flatted_indices[:, i] for i in range(ndim)]
    slices += [Ellipsis]
    ret[slices] = updates.view(*output_shape)
    return ret


class SparseConvTensor:
    def __init__(self, features, indices, spatial_shape, batch_size, grid=None):
        self.features = features
        self.indices = indices
        if self.indices.dtype != torch.int32:
            self.indices.int()
        self.spatial_shape = spatial_shape
        self.batch_size = batch_size
        self.indice_dict = {}
        self.grid = grid

    @property
    def spatial_size(self):
        return np.prod(self.spatial_shape)

    def find_indice_pair(self, key):
        if key is None:
            return None
        if key in self.indice_dict:
            return self.indice_dict[key]
        return None

    def dense(self, channels_first=True):
        output_shape = (
            [self.batch_size] + list(self.spatial_shape) + [self.features.shape[1]]
        )
        res = scatter_nd(self.indices.long(), self.features, output_shape)
        if not channels_first:
            return res
        ndim = len(self.spatial_shape)
        trans_params = list(range(0, ndim + 1))
        trans_params.insert(1, ndim + 1)
        return res.permute(*trans_params).contiguous()

    @property
    def sparity(self):
        return self.indices.shape[0] / np.prod(self.spatial_shape) / self.batch_size


def is_spconv_module(module):
    spconv_modules = (SparseModule,)
    return isinstance(module, spconv_modules)


def is_sparse_conv(module):
    return isinstance(module, SparseConvolution)


class SparseSequential(SparseModule):
    def __init__(self, *args, **kwargs):
        super(SparseSequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        for name, module in kwargs.items():
            if sys.version_info < (3, 6):
                raise ValueError("kwargs only supported in py36+")
            if name in self._modules:
                raise ValueError("name exists.")
            self.add_module(name, module)
        self._sparity_dict = {}

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError("index {} is out of range".format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    @property
    def sparity_dict(self):
        return self._sparity_dict

    def add(self, module, name=None):
        if name is None:
            name = str(len(self._modules))
            if name in self._modules:
                raise KeyError("name exists")
        self.add_module(name, module)

    def forward(self, input):
        for k, module in self._modules.items():
            if is_spconv_module(module):
                assert isinstance(input, SparseConvTensor)
                self._sparity_dict[k] = input.sparity
                input = module(input)
            else:
                if isinstance(input, SparseConvTensor):
                    if input.indices.shape[0] != 0:
                        input.features = module(input.features)
                else:
                    input = module(input)

        return input

    def fused(self):
        mods = [v for k, v in self._modules.items()]
        fused_mods = []
        idx = 0
        while idx < len(mods):
            if is_sparse_conv(mods[idx]):
                if idx < len(mods) - 1 and isinstance(mods[idx + 1], nn.BatchNorm1d):
                    new_module = SparseConvolution(
                        ndim=mods[idx].ndim,
                        in_channels=mods[idx].in_channels,
                        out_channels=mods[idx].out_channels,
                        kernel_size=mods[idx].kernel_size,
                        stride=mods[idx].stride,
                        padding=mods[idx].padding,
                        dilation=mods[idx].dilation,
                        groups=mods[idx].groups,
                        bias=True,
                        subm=mods[idx].subm,
                        output_padding=mods[idx].output_padding,
                        transposed=mods[idx].transposed,
                        inverse=mods[idx].inverse,
                        indice_key=mods[idx].indice_key,
                        fused_bn=True,
                    )
                    new_module.load_state_dict(mods[idx].state_dict(), False)
                    new_module.to(mods[idx].weight.device)
                    conv = new_module
                    bn = mods[idx + 1]
                    conv.bias.data.zero_()
                    conv.weight.data[:] = (
                        conv.weight.data
                        * bn.weight.data
                        / (torch.sqrt(bn.running_var) + bn.eps)
                    )
                    conv.bias.data[:] = (
                        conv.bias.data - bn.running_mean
                    ) * bn.weight.data / (
                        torch.sqrt(bn.running_var) + bn.eps
                    ) + bn.bias.data
                    fused_mods.append(conv)
                    idx += 2
                else:
                    fused_mods.append(mods[idx])
                    idx += 1
            else:
                fused_mods.append(mods[idx])
                idx += 1
        return SparseSequential(*fused_mods)


# ============================================================================
# MIDDLE ENCODERS
# ============================================================================


@MIDDLE_ENCODERS.register_module()
class SparseEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        sparse_shape,
        order=("conv", "norm", "act"),
        norm_cfg=dict(type="BN1d", eps=1e-3, momentum=0.01),
        base_channels=16,
        output_channels=128,
        encoder_channels=((16,), (32, 32, 32), (64, 64, 64), (64, 64, 64)),
        encoder_paddings=((1,), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1)),
        block_type="conv_module",
    ):
        super().__init__()
        assert block_type in ["conv_module", "basicblock"]
        self.sparse_shape = sparse_shape
        self.in_channels = in_channels
        self.order = order
        self.base_channels = base_channels
        self.output_channels = output_channels
        self.encoder_channels = encoder_channels
        self.encoder_paddings = encoder_paddings
        self.stage_num = len(self.encoder_channels)
        self.fp16_enabled = False

        assert isinstance(order, (list, tuple)) and len(order) == 3
        assert set(order) == {"conv", "norm", "act"}

        if self.order[0] != "conv":
            self.conv_input = make_sparse_convmodule(
                in_channels,
                self.base_channels,
                3,
                norm_cfg=norm_cfg,
                padding=1,
                indice_key="subm1",
                conv_type="SubMConv3d",
                order=("conv",),
            )
        else:
            self.conv_input = make_sparse_convmodule(
                in_channels,
                self.base_channels,
                3,
                norm_cfg=norm_cfg,
                padding=1,
                indice_key="subm1",
                conv_type="SubMConv3d",
            )

        encoder_out_channels = self.make_encoder_layers(
            make_sparse_convmodule, norm_cfg, self.base_channels, block_type=block_type
        )

        self.conv_out = make_sparse_convmodule(
            encoder_out_channels,
            self.output_channels,
            kernel_size=(1, 1, 3),
            stride=(1, 1, 2),
            norm_cfg=norm_cfg,
            padding=0,
            indice_key="spconv_down2",
            conv_type="SparseConv3d",
        )

    def forward(self, voxel_features, coors, batch_size, **kwargs):
        coors = coors.int()
        input_sp_tensor = SparseConvTensor(
            voxel_features, coors, self.sparse_shape, batch_size
        )
        x = self.conv_input(input_sp_tensor)

        encode_features = []
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
            encode_features.append(x)

        out = self.conv_out(encode_features[-1])
        spatial_features = out.dense()
        N, C, H, W, D = spatial_features.shape
        spatial_features = spatial_features.permute(0, 1, 4, 2, 3).contiguous()
        spatial_features = spatial_features.view(N, C * D, H, W)

        return spatial_features

    def make_encoder_layers(
        self,
        make_block,
        norm_cfg,
        in_channels,
        block_type="conv_module",
        conv_cfg=dict(type="SubMConv3d"),
    ):
        assert block_type in ["conv_module", "basicblock"]
        self.encoder_layers = SparseSequential()

        for i, blocks in enumerate(self.encoder_channels):
            blocks_list = []
            for j, out_channels in enumerate(tuple(blocks)):
                padding = tuple(self.encoder_paddings[i])[j]
                if i != 0 and j == 0 and block_type == "conv_module":
                    blocks_list.append(
                        make_block(
                            in_channels,
                            out_channels,
                            3,
                            norm_cfg=norm_cfg,
                            stride=2,
                            padding=padding,
                            indice_key=f"spconv{i + 1}",
                            conv_type="SparseConv3d",
                        )
                    )
                elif block_type == "basicblock":
                    if j == len(blocks) - 1 and i != len(self.encoder_channels) - 1:
                        blocks_list.append(
                            make_block(
                                in_channels,
                                out_channels,
                                3,
                                norm_cfg=norm_cfg,
                                stride=2,
                                padding=padding,
                                indice_key=f"spconv{i + 1}",
                                conv_type="SparseConv3d",
                            )
                        )
                    else:
                        blocks_list.append(
                            SparseBasicBlock(
                                out_channels,
                                out_channels,
                                norm_cfg=norm_cfg,
                                conv_cfg=conv_cfg,
                            )
                        )
                else:
                    blocks_list.append(
                        make_block(
                            in_channels,
                            out_channels,
                            3,
                            norm_cfg=norm_cfg,
                            padding=padding,
                            indice_key=f"subm{i + 1}",
                            conv_type="SubMConv3d",
                        )
                    )
                in_channels = out_channels
            stage_name = f"encoder_layer{i + 1}"
            stage_layers = SparseSequential(*blocks_list)
            self.encoder_layers.add_module(stage_name, stage_layers)
        return out_channels


# ============================================================================
# FUSERS
# ============================================================================


@FUSERS.register_module()
class ConvFuser(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        super().__init__(
            nn.Conv2d(sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        return super().forward(torch.cat(inputs, dim=1))


# ============================================================================
# ATTENTION
# ============================================================================


"""
CPU Implementation of Geometric Kernel Attention
"""


def geometry_kernel_attention_cpu(
    value,
    spatial_shapes,
    level_start_index,
    sampling_locations,
    attention_weights,
    im2col_step=64,
):
    """
    Args:
        value: Input feature values (batch, spatial_size, num_heads, channels)
        spatial_shapes: Spatial shapes of feature levels (num_levels, 2)
        level_start_index: Start indices for each level (num_levels,)
        sampling_locations: Sampling coordinates (batch, num_query, num_heads, num_levels, num_points, 2)
        attention_weights: Attention weights (batch, num_query, num_heads, num_levels, num_points)
        im2col_step: Batch processing step

    Returns:
        output: Attended features (batch, num_query, num_heads * channels)
    """
    batch_size, spatial_size, num_heads, channels = value.shape
    _, num_query, _, num_levels, num_points, _ = sampling_locations.shape

    # Initialize output
    output = torch.zeros(
        batch_size,
        num_query,
        num_heads,
        channels,
        dtype=value.dtype,
        device=value.device,
    )

    # Process each level
    for level in range(num_levels):
        level_start_idx = level_start_index[level].item()
        level_height, level_width = spatial_shapes[level]
        level_height, level_width = level_height.item(), level_width.item()

        # Get sampling locations for this level
        level_locs = sampling_locations[:, :, :, level, :, :]  # (B, Q, H, P, 2)

        # Clip coordinates
        level_locs_x = torch.clamp(level_locs[:, :, :, :, 0], 0, level_width - 1)
        level_locs_y = torch.clamp(level_locs[:, :, :, :, 1], 0, level_height - 1)

        # Convert to integer coordinates
        if torch.is_floating_point(level_locs_x):
            level_locs_x = torch.round(level_locs_x).long()
        if torch.is_floating_point(level_locs_y):
            level_locs_y = torch.round(level_locs_y).long()

        # Calculate spatial indices
        spatial_indices = level_start_idx + level_locs_y * level_width + level_locs_x

        # Get attention weights for this level
        level_weights = attention_weights[:, :, :, level, :]  # (B, Q, H, P)

        # Sample values using advanced indexing
        # handle the batch dimension
        for b in range(batch_size):
            for h in range(num_heads):
                # Get indices for this batch and head
                batch_spatial_indices = spatial_indices[b, :, h, :]  # (Q, P)
                batch_weights = level_weights[b, :, h, :]  # (Q, P)

                # Sample values
                sampled_values = value[b, batch_spatial_indices, h, :]  # (Q, P, C)

                # Apply attention weights and sum over points
                weighted_values = sampled_values * batch_weights.unsqueeze(
                    -1
                )  # (Q, P, C)
                output[b, :, h, :] += weighted_values.sum(dim=1)  # (Q, C)

    # Reshape output
    output = output.view(batch_size, num_query, num_heads * channels)

    return output


@ATTENTION.register_module()
class GeometrySptialCrossAttention(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        num_cams=6,
        pc_range=None,
        dropout=0.1,
        init_cfg=None,
        batch_first=False,
        attention=dict(type="MSDeformableAttention3D", embed_dims=256, num_levels=4),
        **kwargs,
    ):
        super(GeometrySptialCrossAttention, self).__init__(init_cfg)

        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.attention = build_attention(attention)
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

        if key is None:
            key = query
        if value is None:
            value = key
        if residual is None:
            inp_residual = query
            slots = torch.zeros_like(query)
        if query_pos is not None:
            query = query + query_pos

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

        queries = self.attention(
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


@ATTENTION.register_module()
class GeometryKernelAttention(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        num_heads=8,
        num_levels=4,
        num_points=4,
        kernel_size=(3, 3),
        dilation=1,
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
        self.fp16_enabled = False
        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.num_points = kernel_size[0] * kernel_size[1]

        self.attention_weights = nn.Linear(
            embed_dims, num_levels * self.num_points * self.num_heads
        )
        self.value_proj = nn.Linear(embed_dims, embed_dims)

        grid_h, grid_w = kernel_size
        y = (torch.arange(grid_h) - grid_h // 2) * dilation
        x = (torch.arange(grid_w) - grid_w // 2) * dilation
        offsets = (
            torch.stack(torch.meshgrid(x, y))
            .permute(1, 2, 0)
            .reshape(grid_h * grid_w, 2)
        )
        self.register_buffer("grid_offsets", offsets, persistent=False)
        self.init_weights()

    def init_weights(self):
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

        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        value = value.view(bs, num_value, self.num_heads, -1)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )

        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points
        )

        if reference_points.shape[-1] == 2:
            with torch.no_grad():
                offset_normalizer = torch.stack(
                    [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1
                )

                bs, num_query, num_Z_anchors, xy = reference_points.shape
                offsets = self.grid_offsets[None, None, None, None]
                reference_points = (
                    reference_points[:, :, :, None, :] * offset_normalizer
                )
                sampling_locations = (
                    (reference_points[:, :, :, :, None, :] + offsets).round().long()
                )
            (
                bs,
                num_query,
                num_heads,
                num_levels,
                num_all_points,
                xy,
            ) = sampling_locations.shape

        output = geometry_kernel_attention_cpu(
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations.contiguous(),
            attention_weights,
            self.im2col_step,
        )
        if not self.batch_first:
            output = output.permute(1, 0, 2)
        return output


@ATTENTION.register_module()
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
        self.fp16_enabled = False

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

        if value is None:
            value = query

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

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


@ATTENTION.register_module()
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
        self.fp16_enabled = False
        self.deformable_attention = build_attention(deformable_attention)
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

        if key is None:
            key = query
        if value is None:
            value = key
        if residual is None:
            inp_residual = query
            slots = torch.zeros_like(query)
        if query_pos is not None:
            query = query + query_pos

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


@ATTENTION.register_module()
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
        self.fp16_enabled = False

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

        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

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


@ATTENTION.register_module()
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
        self.fp16_enabled = False

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
            assert self.batch_first
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


# ============================================================================
# TRANSFORMER LAYER
# ============================================================================


@TRANSFORMER_LAYER.register_module()
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
        if isinstance(attn_cfgs, dict):
            attn_cfgs = [copy.deepcopy(attn_cfgs) for _ in range(num_attn)]

        self.num_attn = num_attn
        self.operation_order = operation_order
        self.norm_cfg = norm_cfg
        self.pre_norm = operation_order[0] == "norm"
        self.attentions = ModuleList()

        index = 0
        for operation_name in operation_order:
            if operation_name in ["self_attn", "cross_attn"]:
                attn_cfgs[index]["batch_first"] = self.batch_first
                attention = build_attention(attn_cfgs[index])
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
            assert ffn_cfgs[ffn_index]["embed_dims"] == self.embed_dims

            self.ffns.append(build_feedforward_network(ffn_cfgs[ffn_index]))

        self.norms = ModuleList()
        num_norms = operation_order.count("norm")
        for _ in range(num_norms):
            self.norms.append(build_norm_layer(norm_cfg, self.embed_dims)[1])


@TRANSFORMER_LAYER.register_module()
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
        self.fp16_enabled = False
        assert len(operation_order) == 6
        assert set(operation_order) == set(["self_attn", "norm", "cross_attn", "ffn"])

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

            elif layer == "cross_attn":
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


# ============================================================================
# TRANSFORMER LAYER SEQUENCES
# ============================================================================


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.Tensor(
        [int((row[1] - row[0]) / row[2]) for row in [xbound, ybound, zbound]]
    )
    return dx, bx, nx


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class BaseTransform(BaseModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        feat_down_sample,
        pc_range,
        voxel_size,
        dbound,
    ):
        super(BaseTransform, self).__init__()
        self.in_channels = in_channels
        self.feat_down_sample = feat_down_sample
        self.xbound = [pc_range[0], pc_range[3], voxel_size[0]]
        self.ybound = [pc_range[1], pc_range[4], voxel_size[1]]
        self.zbound = [pc_range[2], pc_range[5], voxel_size[2]]
        self.dbound = dbound

        dx, bx, nx = gen_dx_bx(self.xbound, self.ybound, self.zbound)
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.C = out_channels
        self.frustum = None
        self.D = int((dbound[1] - dbound[0]) / dbound[2])
        self.fp16_enabled = False

    def create_frustum(self, fH, fW, img_metas):
        iH = img_metas[0]["img_shape"][0][0]
        iW = img_metas[0]["img_shape"][0][1]
        assert iH // self.feat_down_sample == fH
        ds = (
            torch.arange(*self.dbound, dtype=torch.float)
            .view(-1, 1, 1)
            .expand(-1, fH, fW)
        )
        D, _, _ = ds.shape

        xs = (
            torch.linspace(0, iW - 1, fW, dtype=torch.float)
            .view(1, 1, fW)
            .expand(D, fH, fW)
        )
        ys = (
            torch.linspace(0, iH - 1, fH, dtype=torch.float)
            .view(1, fH, 1)
            .expand(D, fH, fW)
        )

        frustum = torch.stack((xs, ys, ds), -1)
        return frustum

    def get_geometry_v1(
        self,
        fH,
        fW,
        rots,
        trans,
        intrins,
        post_rots,
        post_trans,
        lidar2ego_rots,
        lidar2ego_trans,
        img_metas,
        **kwargs,
    ):
        B, N, _ = trans.shape
        device = trans.device
        if self.frustum == None:
            self.frustum = self.create_frustum(fH, fW, img_metas)
            self.frustum = self.frustum.to(device)

        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = (
            torch.inverse(post_rots)
            .view(B, N, 1, 1, 1, 3, 3)
            .matmul(points.unsqueeze(-1))
        )
        points = torch.cat(
            (
                points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                points[:, :, :, :, :, 2:3],
            ),
            5,
        )
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)
        points -= lidar2ego_trans.view(B, 1, 1, 1, 1, 3)
        points = (
            torch.inverse(lidar2ego_rots)
            .view(B, 1, 1, 1, 1, 3, 3)
            .matmul(points.unsqueeze(-1))
            .squeeze(-1)
        )
        return points

    def bev_pool(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W
        x = x.reshape(Nprime, C)
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat(
            [
                torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long)
                for ix in range(B)
            ]
        )
        geom_feats = torch.cat((geom_feats, batch_ix), 1)
        kept = (
            (geom_feats[:, 0] >= 0)
            & (geom_feats[:, 0] < self.nx[0])
            & (geom_feats[:, 1] >= 0)
            & (geom_feats[:, 1] < self.nx[1])
            & (geom_feats[:, 2] >= 0)
            & (geom_feats[:, 2] < self.nx[2])
        )
        x = x[kept]
        geom_feats = geom_feats[kept]

        x = bev_pool(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])
        final = torch.cat(x.unbind(dim=2), 1)

        return final

    def forward(self, images, img_metas):
        B, N, C, fH, fW = images.shape
        lidar2img = []
        camera2ego = []
        camera_intrinsics = []
        img_aug_matrix = []
        lidar2ego = []

        for img_meta in img_metas:
            lidar2img.append(img_meta["lidar2img"])
            camera2ego.append(img_meta["camera2ego"])
            camera_intrinsics.append(img_meta["camera_intrinsics"])
            img_aug_matrix.append(img_meta["img_aug_matrix"])
            lidar2ego.append(img_meta["lidar2ego"])
        lidar2img = np.asarray(lidar2img)
        lidar2img = images.new_tensor(lidar2img)  # (B, N, 4, 4)
        camera2ego = np.asarray(camera2ego)
        camera2ego = images.new_tensor(camera2ego)  # (B, N, 4, 4)
        camera_intrinsics = np.asarray(camera_intrinsics)
        camera_intrinsics = images.new_tensor(camera_intrinsics)  # (B, N, 4, 4)
        img_aug_matrix = np.asarray(img_aug_matrix)
        img_aug_matrix = images.new_tensor(img_aug_matrix)  # (B, N, 4, 4)
        lidar2ego = np.asarray(lidar2ego)
        lidar2ego = images.new_tensor(lidar2ego)  # (B, N, 4, 4)

        rots = camera2ego[..., :3, :3]
        trans = camera2ego[..., :3, 3]
        intrins = camera_intrinsics[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        lidar2ego_rots = lidar2ego[..., :3, :3]
        lidar2ego_trans = lidar2ego[..., :3, 3]

        geom = self.get_geometry_v1(
            fH,
            fW,
            rots,
            trans,
            intrins,
            post_rots,
            post_trans,
            lidar2ego_rots,
            lidar2ego_trans,
            img_metas,
        )
        x = self.get_cam_feats(images)
        x = self.bev_pool(geom, x)
        x = x.permute(0, 1, 3, 2).contiguous()

        return x


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class LSSTransform(BaseTransform):
    def __init__(
        self,
        in_channels,
        out_channels,
        feat_down_sample,
        pc_range,
        voxel_size,
        dbound,
        downsample=1,
    ):
        super(LSSTransform, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            feat_down_sample=feat_down_sample,
            pc_range=pc_range,
            voxel_size=voxel_size,
            dbound=dbound,
        )
        self.depthnet = nn.Conv2d(in_channels, int(self.D + self.C), 1)
        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    3,
                    stride=downsample,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )

    def get_cam_feats(self, x):
        B, N, C, fH, fW = x.shape

        x = x.view(B * N, C, fH, fW)

        x = self.depthnet(x)
        depth = x[:, : self.D].softmax(dim=1)
        x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2)

        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)
        return x

    def forward(self, images, img_metas):
        x = super().forward(images, img_metas)
        x = self.downsample(x)
        return x


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class MapTRDecoder(TransformerLayerSequence):
    def __init__(self, *args, return_intermediate=False, **kwargs):
        super(MapTRDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.fp16_enabled = False

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

            reference_points_input = reference_points[..., :2].unsqueeze(2)
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

                assert reference_points.shape[-1] == 2

                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(
                    reference_points[..., :2]
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


@TRANSFORMER_LAYER_SEQUENCE.register_module()
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
        self.fp16_enabled = False

    @staticmethod
    def get_reference_points(
        H,
        W,
        Z=8,
        num_points_in_pillar=4,
        dim="3d",
        bs=1,
        device="cpu",
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

        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta["lidar2img"])
        lidar2img = np.asarray(lidar2img)
        lidar2img = reference_points.new_tensor(lidar2img)
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
            if self.return_intermediate:
                intermediate.append(output)

        return output


# ============================================================================
# TRANSFORMER
# ============================================================================


@TRANSFORMER.register_module()
class MapTRPerceptionTransformer(BaseModule):
    def __init__(
        self,
        num_feature_levels=4,
        num_cams=6,
        two_stage_num_proposals=300,
        fuser=None,
        encoder=None,
        decoder=None,
        embed_dims=256,
        rotate_prev_bev=True,
        use_shift=True,
        use_can_bus=True,
        len_can_bus=18,
        can_bus_norm=True,
        use_cams_embeds=True,
        rotate_center=[100, 100],
        modality="vision",
        **kwargs,
    ):
        super(MapTRPerceptionTransformer, self).__init__(**kwargs)
        if modality == "fusion":
            self.fuser = build_fuser(fuser)
        self.use_attn_bev = encoder["type"] == "BEVFormerEncoder"
        self.encoder = build_transformer_layer_sequence(encoder)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False

        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.len_can_bus = len_can_bus
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
        self.reference_points = nn.Linear(self.embed_dims, 2)
        self.can_bus_mlp = nn.Sequential(
            nn.Linear(self.len_can_bus, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),
        )
        if self.can_bus_norm:
            self.can_bus_mlp.add_module("norm", nn.LayerNorm(self.embed_dims))

    def init_weights(self):

        for m in self.modules():
            if (
                isinstance(m, MSDeformableAttention3D)
                or isinstance(m, TemporalSelfAttention)
                or isinstance(m, CustomMSDeformableAttention)
            ):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        normal_(self.level_embeds)
        normal_(self.cams_embeds)
        xavier_init(self.reference_points, distribution="uniform", bias=0.0)
        xavier_init(self.can_bus_mlp, distribution="uniform", bias=0.0)

    def attn_bev_encode(
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
        shift = bev_queries.new_tensor([shift_x, shift_y]).permute(1, 0)

        can_bus = bev_queries.new_tensor(
            [each["can_bus"] for each in kwargs["img_metas"]]
        )
        can_bus = self.can_bus_mlp(can_bus[:, : self.len_can_bus])[None, :, :]
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

        feat_flatten = feat_flatten.permute(0, 2, 1, 3)

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

    def lss_bev_encode(self, mlvl_feats, prev_bev=None, **kwargs):
        assert (
            len(mlvl_feats) == 1
        ), "Currently we only support single level feat in LSS"
        images = mlvl_feats[0]
        img_metas = kwargs["img_metas"]
        bev_embed = self.encoder(images, img_metas)
        bs, c, _, _ = bev_embed.shape
        bev_embed = bev_embed.view(bs, c, -1).permute(0, 2, 1).contiguous()

        return bev_embed

    def get_bev_features(
        self,
        mlvl_feats,
        lidar_feat,
        bev_queries,
        bev_h,
        bev_w,
        grid_length=[0.512, 0.512],
        bev_pos=None,
        prev_bev=None,
        **kwargs,
    ):
        if self.use_attn_bev:
            bev_embed = self.attn_bev_encode(
                mlvl_feats,
                bev_queries,
                bev_h,
                bev_w,
                grid_length=grid_length,
                bev_pos=bev_pos,
                prev_bev=prev_bev,
                **kwargs,
            )
        else:
            bev_embed = self.lss_bev_encode(mlvl_feats, prev_bev=prev_bev, **kwargs)

        if lidar_feat is not None:
            bs = mlvl_feats[0].size(0)
            bev_embed = (
                bev_embed.view(bs, bev_h, bev_w, -1).permute(0, 3, 1, 2).contiguous()
            )
            lidar_feat = lidar_feat.permute(0, 1, 3, 2).contiguous()  # B C H W
            lidar_feat = nn.functional.interpolate(
                lidar_feat, size=(bev_h, bev_w), mode="bicubic", align_corners=False
            )
            fused_bev = self.fuser([bev_embed, lidar_feat])
            fused_bev = fused_bev.flatten(2).permute(0, 2, 1).contiguous()
            bev_embed = fused_bev

        return bev_embed

    def forward(
        self,
        mlvl_feats,
        lidar_feat,
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
            lidar_feat,
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
@HEADS.register_module()
class MapTRHead(DETRHead):
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
        num_vec=20,
        num_pts_per_vec=2,
        num_pts_per_gt_vec=2,
        query_embed_type="all_pts",
        transform_method="minmax",
        gt_shift_pts_pattern="v0",
        dir_interval=1,
        **kwargs,
    ):

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False

        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        self.bev_encoder_type = transformer.encoder.type
        if self.as_two_stage:
            transformer["as_two_stage"] = self.as_two_stage
        if "code_size" in kwargs:
            self.code_size = kwargs["code_size"]

        if code_weights is not None:
            self.code_weights = code_weights

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1

        self.query_embed_type = query_embed_type
        self.transform_method = transform_method
        self.gt_shift_pts_pattern = gt_shift_pts_pattern
        num_query = num_vec * num_pts_per_vec
        self.num_query = num_query
        self.num_vec = num_vec
        self.num_pts_per_vec = num_pts_per_vec
        self.num_pts_per_gt_vec = num_pts_per_gt_vec
        self.dir_interval = dir_interval

        super(MapTRHead, self).__init__(*args, transformer=transformer, **kwargs)
        self.code_weights = nn.Parameter(
            torch.tensor(self.code_weights, requires_grad=False), requires_grad=False
        )
        self._init_layers()

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

        if not self.as_two_stage:
            if self.bev_encoder_type == "BEVFormerEncoder":
                self.bev_embedding = nn.Embedding(
                    self.bev_h * self.bev_w, self.embed_dims
                )
            else:
                self.bev_embedding = None
            if self.query_embed_type == "instance_pts":
                self.query_embedding = None
                self.instance_embedding = nn.Embedding(
                    self.num_vec, self.embed_dims * 2
                )
                self.pts_embedding = nn.Embedding(
                    self.num_pts_per_vec, self.embed_dims * 2
                )

    def init_weights(self):
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)

    def forward(self, mlvl_feats, lidar_feat, img_metas, prev_bev=None, only_bev=False):
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype

        if self.query_embed_type == "instance_pts":
            pts_embeds = self.pts_embedding.weight.unsqueeze(0)
            instance_embeds = self.instance_embedding.weight.unsqueeze(1)
            object_query_embeds = (pts_embeds + instance_embeds).flatten(0, 1).to(dtype)

        if self.bev_embedding is not None:
            bev_queries = self.bev_embedding.weight.to(dtype)

            bev_mask = torch.zeros(
                (bs, self.bev_h, self.bev_w), device=bev_queries.device
            ).to(dtype)
            bev_pos = self.positional_encoding(bev_mask).to(dtype)
        else:
            bev_queries = None
            bev_mask = None
            bev_pos = None

        outputs = self.transformer(
            mlvl_feats,
            lidar_feat,
            bev_queries,
            object_query_embeds,
            self.bev_h,
            self.bev_w,
            grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
            bev_pos=bev_pos,
            reg_branches=self.reg_branches if self.with_box_refine else None,
            cls_branches=self.cls_branches if self.as_two_stage else None,
            img_metas=img_metas,
            prev_bev=prev_bev,
        )

        bev_embed, hs, init_reference, inter_references = outputs
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        outputs_pts_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](
                hs[lvl].view(bs, self.num_vec, self.num_pts_per_vec, -1).mean(2)
            )
            tmp = self.reg_branches[lvl](hs[lvl])

            tmp[..., 0:2] += reference[..., 0:2]
            tmp = tmp.sigmoid()
            outputs_coord, outputs_pts_coord = self.transform_box(tmp)
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_pts_coords.append(outputs_pts_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        outputs_pts_coords = torch.stack(outputs_pts_coords)
        outs = {
            "bev_embed": bev_embed,
            "all_cls_scores": outputs_classes,
            "all_bbox_preds": outputs_coords,
            "all_pts_preds": outputs_pts_coords,
            "enc_cls_scores": None,
            "enc_bbox_preds": None,
            "enc_pts_preds": None,
        }

        return outs

    def transform_box(self, pts, y_first=False):

        pts_reshape = pts.view(pts.shape[0], self.num_vec, self.num_pts_per_vec, 2)
        pts_y = pts_reshape[:, :, :, 0] if y_first else pts_reshape[:, :, :, 1]
        pts_x = pts_reshape[:, :, :, 1] if y_first else pts_reshape[:, :, :, 0]
        if self.transform_method == "minmax":
            xmin = pts_x.min(dim=2, keepdim=True)[0]
            xmax = pts_x.max(dim=2, keepdim=True)[0]
            ymin = pts_y.min(dim=2, keepdim=True)[0]
            ymax = pts_y.max(dim=2, keepdim=True)[0]
            bbox = torch.cat([xmin, ymin, xmax, ymax], dim=2)
            bbox = bbox_xyxy_to_cxcywh(bbox)

        return bbox, pts_reshape

    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        preds_dicts = self.bbox_coder.decode(preds_dicts)

        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds["bboxes"]
            scores = preds["scores"]
            labels = preds["labels"]
            pts = preds["pts"]

            ret_list.append([bboxes, scores, labels, pts])

        return ret_list


# ============================================================================
# DETECTORS
# ============================================================================


@DETECTORS.register_module()
class MVXTwoStageDetector(BaseDetector):
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
            self.pts_bbox_head = build_head(pts_bbox_head)

        if img_backbone:
            self.img_backbone = build_backbone(img_backbone)
        if img_neck is not None:
            self.img_neck = build_neck(img_neck)

        self.test_cfg = test_cfg

        img_pretrained = pretrained.get("img", None)
        pts_pretrained = pretrained.get("pts", None)

        if self.with_img_backbone:
            if img_pretrained is not None:

                self.img_backbone.init_cfg = dict(
                    type="Pretrained", checkpoint=img_pretrained
                )
        if self.with_img_roi_head:
            if img_pretrained is not None:

                self.img_roi_head.init_cfg = dict(
                    type="Pretrained", checkpoint=img_pretrained
                )

        if self.with_pts_backbone:
            if pts_pretrained is not None:

                self.pts_backbone.init_cfg = dict(
                    type="Pretrained", checkpoint=pts_pretrained
                )

    @property
    def with_img_backbone(self):
        return hasattr(self, "img_backbone") and self.img_backbone is not None

    @property
    def with_pts_backbone(self):
        return hasattr(self, "pts_backbone") and self.pts_backbone is not None

    @property
    def with_img_neck(self):
        return hasattr(self, "img_neck") and self.img_neck is not None

    @property
    def with_img_roi_head(self):
        return hasattr(self, "img_roi_head") and self.img_roi_head is not None

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        pass


@DETECTORS.register_module()
class MapTR(MVXTwoStageDetector):
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
        modality="vision",
        lidar_encoder=None,
    ):

        super(MapTR, self).__init__(
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
        self.fp16_enabled = False
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            "prev_bev": None,
            "scene_token": None,
            "prev_pos": 0,
            "prev_angle": 0,
        }
        self.modality = modality
        if self.modality == "fusion" and lidar_encoder is not None:
            if lidar_encoder["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**lidar_encoder["voxelize"])
            self.lidar_modal_extractor = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": build_middle_encoder(lidar_encoder["backbone"]),
                }
            )
            self.voxelize_reduce = lidar_encoder.get("voxelize_reduce", True)

    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.lidar_modal_extractor["voxelize"](res)
            if len(ret) == 3:
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
                    -1, 1
                )
                feats = feats.contiguous()

        return feats, coords, sizes

    def extract_lidar_feat(self, points):
        feats, coords, sizes = self.voxelize(points)
        batch_size = coords[-1, 0] + 1
        lidar_feat = self.lidar_modal_extractor["backbone"](
            feats, coords, batch_size, sizes=sizes
        )
        return lidar_feat

    def extract_img_feat(self, img, img_metas, len_queue=None):
        B = img.size(0)
        if img is not None:
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            img_feats = self.img_backbone(img)

        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    def extract_feat(self, img, img_metas=None, len_queue=None):
        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        return img_feats

    def forward(self, **kwargs):
        return self.forward_test(**kwargs)

    def forward_test(self, img_metas, img=None, points=None, **kwargs):
        img = [img] if img is None else img
        points = [points] if points is None else points

        if img_metas[0][0]["scene_token"] != self.prev_frame_info["scene_token"]:
            self.prev_frame_info["prev_bev"] = None

        self.prev_frame_info["scene_token"] = img_metas[0][0]["scene_token"]

        if not self.video_test_mode:
            self.prev_frame_info["prev_bev"] = None

        tmp_pos = copy.deepcopy(img_metas[0][0]["can_bus"][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]["can_bus"][-1])

        img_metas[0][0]["can_bus"][-1] = 0
        img_metas[0][0]["can_bus"][:3] = 0

        new_prev_bev, bbox_results = self.simple_test(
            img_metas[0],
            img[0],
            points[0],
            prev_bev=self.prev_frame_info["prev_bev"],
            **kwargs,
        )
        self.prev_frame_info["prev_pos"] = tmp_pos
        self.prev_frame_info["prev_angle"] = tmp_angle
        self.prev_frame_info["prev_bev"] = new_prev_bev
        return bbox_results

    def pred2result(self, bboxes, scores, labels, pts, attrs=None):

        result_dict = dict(
            boxes_3d=bboxes,
            scores_3d=scores,
            labels_3d=labels,
            pts_3d=pts,
        )

        return result_dict

    def simple_test_pts(self, x, lidar_feat, img_metas, prev_bev=None, rescale=False):
        outs = self.pts_bbox_head(x, lidar_feat, img_metas, prev_bev=prev_bev)

        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=rescale)

        bbox_results = [
            self.pred2result(bboxes, scores, labels, pts)
            for bboxes, scores, labels, pts in bbox_list
        ]
        return outs["bev_embed"], bbox_results

    def simple_test(
        self, img_metas, img=None, points=None, prev_bev=None, rescale=False, **kwargs
    ):
        lidar_feat = None
        if self.modality == "fusion":
            lidar_feat = self.extract_lidar_feat(points)
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, bbox_pts = self.simple_test_pts(
            img_feats, lidar_feat, img_metas, prev_bev, rescale=rescale
        )
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict["pts_bbox"] = pts_bbox
        return new_prev_bev, bbox_list
