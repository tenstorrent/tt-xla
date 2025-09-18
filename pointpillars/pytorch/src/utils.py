# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from shapely.geometry import Polygon
import os


def dynamic_voxelize(
    points, coors, voxel_size, coors_range, grid_size, num_points, num_features, NDim
):
    ndim_minus_1 = NDim - 1

    for i in range(num_points):
        failed = False
        coor = [0] * NDim

        for j in range(NDim):
            c = int(torch.floor((points[i, j] - coors_range[j]) / voxel_size[j]))
            # necessary to rm points out of range
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[ndim_minus_1 - j] = c

        for k in range(NDim):
            if failed:
                coors[i, k] = -1
            else:
                coors[i, k] = coor[k]


def hard_voxelize_(
    points,
    voxels,
    coors,
    num_points_per_voxel,
    coor_to_voxelidx,
    voxel_size,
    coors_range,
    grid_size,
    max_points,
    max_voxels,
    num_points,
    num_features,
    NDim,
):
    """
    Args:
        points: (num_points, num_features) input points tensor
        voxels: (max_voxels, max_points, num_features) pre-allocated output tensor
        coors: (max_voxels, NDim) pre-allocated coordinates tensor
        num_points_per_voxel: (max_voxels,) pre-allocated count tensor
        coor_to_voxelidx: (grid_size[2], grid_size[1], grid_size[0]) mapping tensor
        voxel_size: list of voxel dimensions
        coors_range: list of coordinate ranges
        grid_size: list of grid dimensions
        max_points: maximum points per voxel
        max_voxels: maximum number of voxels
        num_points: number of input points
        num_features: number of features per point
        NDim: number of dimensions (3)

    Returns:
        voxel_num: number of voxels created
    """
    # declare a temp coors - exactly as C++ version
    temp_coors = torch.zeros(
        (num_points, NDim), dtype=torch.int32, device=points.device
    )

    # First use dynamic voxelization to get coors, then check max points/voxels constraints
    dynamic_voxelize(
        points,
        temp_coors,
        voxel_size,
        coors_range,
        grid_size,
        num_points,
        num_features,
        NDim,
    )

    voxel_num = 0

    for i in range(num_points):
        # Skip invalid points
        if temp_coors[i, 0] == -1:
            continue

        voxelidx = coor_to_voxelidx[
            temp_coors[i, 0], temp_coors[i, 1], temp_coors[i, 2]
        ]

        # record voxel
        if voxelidx == -1:
            voxelidx = voxel_num
            if max_voxels != -1 and voxel_num >= max_voxels:
                continue
            voxel_num += 1

            coor_to_voxelidx[
                temp_coors[i, 0], temp_coors[i, 1], temp_coors[i, 2]
            ] = voxelidx

            for k in range(NDim):
                coors[voxelidx, k] = temp_coors[i, k]

        # put points into voxel
        num = num_points_per_voxel[voxelidx]
        if max_points == -1 or num < max_points:
            for k in range(num_features):
                voxels[voxelidx, num, k] = points[i, k]
            num_points_per_voxel[voxelidx] += 1

    return voxel_num


def hard_voxelize(
    points,
    voxels,
    coors,
    num_points_per_voxel,
    voxel_size,
    coors_range,
    max_points,
    max_voxels,
    NDim=3,
):
    # Calculate grid size exactly as C++ version
    grid_size = []
    num_points = points.size(0)
    num_features = points.size(1)

    for i in range(NDim):
        grid_size.append(
            round((coors_range[NDim + i] - coors_range[i]) / voxel_size[i])
        )

    # Create coor_to_voxelidx exactly as C++ version
    coor_to_voxelidx = -torch.ones(
        (grid_size[2], grid_size[1], grid_size[0]),
        dtype=torch.int32,
        device=points.device,
    )

    # Call the kernel with exact same parameters
    voxel_num = hard_voxelize_(
        points,
        voxels,
        coors,
        num_points_per_voxel,
        coor_to_voxelidx,
        voxel_size,
        coors_range,
        grid_size,
        max_points,
        max_voxels,
        num_points,
        num_features,
        NDim,
    )

    return voxel_num


class Voxelization(nn.Module):
    def __init__(
        self,
        voxel_size,
        point_cloud_range,
        max_num_points,
        max_voxels,
        deterministic=True,
    ):
        """
        Args:
            voxel_size (list): list [x, y, z] size of three dimension
            point_cloud_range (list):
                [x_min, y_min, z_min, x_max, y_max, z_max]
            max_num_points (int): max number of points per voxel
            max_voxels (tuple): max number of voxels in
                (training, testing) time
        """
        super(Voxelization, self).__init__()
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.max_num_points = max_num_points
        self.max_voxels = max_voxels
        self.deterministic = deterministic

        point_cloud_range = torch.tensor(point_cloud_range, dtype=torch.float32)
        voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
        grid_size = (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
        grid_size = torch.round(grid_size).long()
        input_feat_shape = grid_size[:2]
        self.grid_size = grid_size
        self.pcd_shape = [*input_feat_shape, 1][::-1]

    def _voxelize_forward(
        self,
        points,
        voxel_size,
        coors_range,
        max_points=35,
        max_voxels=20000,
        deterministic=True,
    ):
        """convert kitti points(N, >=3) to voxels.
        Args:
            points: [N, ndim] float tensor. points[:, :3] contain xyz points
                and points[:, 3:] contain other information like reflectivity
            voxel_size: [3] list/tuple or array, float. xyz, indicate voxel
                size
            coors_range: [6] list/tuple or array, float. indicate voxel
                range. format: xyzxyz, minmax
            max_points: int. indicate maximum points contained in a voxel. if
                max_points=-1, it means using dynamic_voxelize
            max_voxels: int. indicate maximum voxels this function create.
                for second, 20000 is a good choice. Users should shuffle points
                before call this function because max_voxels may drop points.
        Returns:
            voxels: [M, max_points, ndim] float tensor. only contain points
                    and returned when max_points != -1.
            coordinates: [M, 3] int32 tensor, always returned.
            num_points_per_voxel: [M] int32 tensor. Only returned when
                max_points != -1.
        """
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

    def forward(self, input):
        """
        input: shape=(N, c)
        """
        if self.training:
            max_voxels = self.max_voxels[0]
        else:
            max_voxels = self.max_voxels[1]

        return self._voxelize_forward(
            input,
            self.voxel_size,
            self.point_cloud_range,
            self.max_num_points,
            max_voxels,
            self.deterministic,
        )


class Anchors:
    def __init__(self, ranges, sizes, rotations):
        assert len(ranges) == len(sizes)
        self.ranges = ranges
        self.sizes = sizes
        self.rotations = rotations

    def get_anchors(self, feature_map_size, anchor_range, anchor_size, rotations):
        """
        feature_map_size: (y_l, x_l)
        anchor_range: [x1, y1, z1, x2, y2, z2]
        anchor_size: [w, l, h]
        rotations: [0, 1.57]
        return: shape=(y_l, x_l, 2, 7)
        """
        device = feature_map_size.device
        x_centers = torch.linspace(
            anchor_range[0], anchor_range[3], feature_map_size[1] + 1, device=device
        )
        y_centers = torch.linspace(
            anchor_range[1], anchor_range[4], feature_map_size[0] + 1, device=device
        )
        z_centers = torch.linspace(
            anchor_range[2], anchor_range[5], 1 + 1, device=device
        )

        x_shift = (x_centers[1] - x_centers[0]) / 2
        y_shift = (y_centers[1] - y_centers[0]) / 2
        z_shift = (z_centers[1] - z_centers[0]) / 2
        x_centers = (
            x_centers[: feature_map_size[1]] + x_shift
        )  # (feature_map_size[1], )
        y_centers = (
            y_centers[: feature_map_size[0]] + y_shift
        )  # (feature_map_size[0], )
        z_centers = z_centers[:1] + z_shift  # (1, )

        # [feature_map_size[1], feature_map_size[0], 1, 2] * 4
        meshgrids = torch.meshgrid(x_centers, y_centers, z_centers, rotations)
        meshgrids = list(meshgrids)
        for i in range(len(meshgrids)):
            meshgrids[i] = meshgrids[i][
                ..., None
            ]  # [feature_map_size[1], feature_map_size[0], 1, 2, 1]

        anchor_size = anchor_size[None, None, None, None, :]
        repeat_shape = [feature_map_size[1], feature_map_size[0], 1, len(rotations), 1]
        anchor_size = anchor_size.repeat(
            repeat_shape
        )  # [feature_map_size[1], feature_map_size[0], 1, 2, 3]
        meshgrids.insert(3, anchor_size)
        anchors = (
            torch.cat(meshgrids, dim=-1).permute(2, 1, 0, 3, 4).contiguous()
        )  # [1, feature_map_size[0], feature_map_size[1], 2, 7]
        return anchors.squeeze(0)

    def get_multi_anchors(self, feature_map_size):
        """
        feature_map_size: (y_l, x_l)
        ranges: [[x1, y1, z1, x2, y2, z2], [x1, y1, z1, x2, y2, z2], [x1, y1, z1, x2, y2, z2]]
        sizes: [[w, l, h], [w, l, h], [w, l, h]]
        rotations: [0, 1.57]
        return: shape=(y_l, x_l, 3, 2, 7)
        """
        device = feature_map_size.device
        ranges = torch.tensor(self.ranges, device=device)
        sizes = torch.tensor(self.sizes, device=device)
        rotations = torch.tensor(self.rotations, device=device)
        multi_anchors = []
        for i in range(len(ranges)):
            anchors = self.get_anchors(
                feature_map_size=feature_map_size,
                anchor_range=ranges[i],
                anchor_size=sizes[i],
                rotations=rotations,
            )
            multi_anchors.append(anchors[:, :, None, :, :])
        multi_anchors = torch.cat(multi_anchors, dim=2)

        return multi_anchors


def bboxes2deltas(bboxes, anchors):
    """
    bboxes: (M, 7), (x, y, z, w, l, h, theta)
    anchors: (M, 7)
    return: (M, 7)
    """
    da = torch.sqrt(anchors[:, 3] ** 2 + anchors[:, 4] ** 2)

    dx = (bboxes[:, 0] - anchors[:, 0]) / da
    dy = (bboxes[:, 1] - anchors[:, 1]) / da

    zb = bboxes[:, 2] + bboxes[:, 5] / 2  # bottom center
    za = anchors[:, 2] + anchors[:, 5] / 2  # bottom center
    dz = (zb - za) / anchors[:, 5]  # bottom center

    dw = torch.log(bboxes[:, 3] / anchors[:, 3])
    dl = torch.log(bboxes[:, 4] / anchors[:, 4])
    dh = torch.log(bboxes[:, 5] / anchors[:, 5])
    dtheta = bboxes[:, 6] - anchors[:, 6]

    deltas = torch.stack([dx, dy, dz, dw, dl, dh, dtheta], dim=1)
    return deltas


def iou2d(bboxes1, bboxes2, metric=0):
    """
    bboxes1: (n, 4), (x1, y1, x2, y2)
    bboxes2: (m, 4), (x1, y1, x2, y2)
    return: (n, m)
    """
    bboxes_x1 = torch.maximum(bboxes1[:, 0][:, None], bboxes2[:, 0][None, :])  # (n, m)
    bboxes_y1 = torch.maximum(bboxes1[:, 1][:, None], bboxes2[:, 1][None, :])  # (n, m)
    bboxes_x2 = torch.minimum(bboxes1[:, 2][:, None], bboxes2[:, 2][None, :])
    bboxes_y2 = torch.minimum(bboxes1[:, 3][:, None], bboxes2[:, 3][None, :])

    bboxes_w = torch.clamp(bboxes_x2 - bboxes_x1, min=0)
    bboxes_h = torch.clamp(bboxes_y2 - bboxes_y1, min=0)

    iou_area = bboxes_w * bboxes_h  # (n, m)

    bboxes1_wh = bboxes1[:, 2:] - bboxes1[:, :2]
    area1 = bboxes1_wh[:, 0] * bboxes1_wh[:, 1]  # (n, )
    bboxes2_wh = bboxes2[:, 2:] - bboxes2[:, :2]
    area2 = bboxes2_wh[:, 0] * bboxes2_wh[:, 1]  # (m, )
    if metric == 0:
        iou = iou_area / (area1[:, None] + area2[None, :] - iou_area + 1e-8)
    elif metric == 1:
        iou = iou_area / (area1[:, None] + 1e-8)
    return iou


def nearest_bev(bboxes):
    """
    bboxes: (n, 7), (x, y, z, w, l, h, theta)
    return: (n, 4), (x1, y1, x2, y2)
    """
    bboxes_bev = copy.deepcopy(bboxes[:, [0, 1, 3, 4]])
    bboxes_angle = limit_period(bboxes[:, 6].cpu(), offset=0.5, period=np.pi).to(
        bboxes_bev
    )
    bboxes_bev = torch.where(
        torch.abs(bboxes_angle[:, None]) > np.pi / 4,
        bboxes_bev[:, [0, 1, 3, 2]],
        bboxes_bev,
    )

    bboxes_xy = bboxes_bev[:, :2]
    bboxes_wl = bboxes_bev[:, 2:]
    bboxes_bev_x1y1x2y2 = torch.cat(
        [bboxes_xy - bboxes_wl / 2, bboxes_xy + bboxes_wl / 2], dim=-1
    )
    return bboxes_bev_x1y1x2y2


def iou2d_nearest(bboxes1, bboxes2):
    """
    bboxes1: (n, 7), (x, y, z, w, l, h, theta)
    bboxes2: (m, 7),
    return: (n, m)
    """
    bboxes1_bev = nearest_bev(bboxes1)
    bboxes2_bev = nearest_bev(bboxes2)
    iou = iou2d(bboxes1_bev, bboxes2_bev)
    return iou


def limit_period(val, offset=0.5, period=np.pi):
    """
    val: array or float
    offset: float
    period: float
    return: Value in the range of [-offset * period, (1-offset) * period]
    """
    limited_val = val - np.floor(val / period + offset) * period
    return limited_val


def anchors2bboxes(anchors, deltas):
    """
    anchors: (M, 7),  (x, y, z, w, l, h, theta)
    deltas: (M, 7)
    return: (M, 7)
    """
    da = torch.sqrt(anchors[:, 3] ** 2 + anchors[:, 4] ** 2)
    x = deltas[:, 0] * da + anchors[:, 0]
    y = deltas[:, 1] * da + anchors[:, 1]
    z = deltas[:, 2] * anchors[:, 5] + anchors[:, 2] + anchors[:, 5] / 2

    w = anchors[:, 3] * torch.exp(deltas[:, 3])
    l = anchors[:, 4] * torch.exp(deltas[:, 4])
    h = anchors[:, 5] * torch.exp(deltas[:, 5])

    z = z - h / 2

    theta = anchors[:, 6] + deltas[:, 6]

    bboxes = torch.stack([x, y, z, w, l, h, theta], dim=1)
    return bboxes


def anchor_target(
    batched_anchors, batched_gt_bboxes, batched_gt_labels, assigners, nclasses
):
    """
    batched_anchors: [(y_l, x_l, 3, 2, 7), (y_l, x_l, 3, 2, 7), ... ]
    batched_gt_bboxes: [(n1, 7), (n2, 7), ...]
    batched_gt_labels: [(n1, ), (n2, ), ...]
    return:
           dict = {batched_anchors_labels: (bs, n_anchors),
                   batched_labels_weights: (bs, n_anchors),
                   batched_anchors_reg: (bs, n_anchors, 7),
                   batched_reg_weights: (bs, n_anchors),
                   batched_anchors_dir: (bs, n_anchors),
                   batched_dir_weights: (bs, n_anchors)}
    """
    assert len(batched_anchors) == len(batched_gt_bboxes) == len(batched_gt_labels)
    batch_size = len(batched_anchors)
    n_assigners = len(assigners)
    batched_labels, batched_label_weights = [], []
    batched_bbox_reg, batched_bbox_reg_weights = [], []
    batched_dir_labels, batched_dir_labels_weights = [], []
    for i in range(batch_size):
        anchors = batched_anchors[i]
        gt_bboxes, gt_labels = batched_gt_bboxes[i], batched_gt_labels[i]
        # what we want to get next ?
        # 1. identify positive anchors and negative anchors  -> cls
        # 2. identify the regresstion values  -> reg
        # 3. indentify the direction  -> dir_cls
        multi_labels, multi_label_weights = [], []
        multi_bbox_reg, multi_bbox_reg_weights = [], []
        multi_dir_labels, multi_dir_labels_weights = [], []
        d1, d2, d3, d4, d5 = anchors.size()
        for j in range(n_assigners):  # multi anchors
            assigner = assigners[j]
            pos_iou_thr, neg_iou_thr, min_iou_thr = (
                assigner["pos_iou_thr"],
                assigner["neg_iou_thr"],
                assigner["min_iou_thr"],
            )
            cur_anchors = anchors[:, :, j, :, :].reshape(-1, 7)
            overlaps = iou2d_nearest(gt_bboxes, cur_anchors)
            max_overlaps, max_overlaps_idx = torch.max(overlaps, dim=0)
            gt_max_overlaps, _ = torch.max(overlaps, dim=1)

            assigned_gt_inds = -torch.ones_like(cur_anchors[:, 0], dtype=torch.long)
            # a. negative anchors
            assigned_gt_inds[max_overlaps < neg_iou_thr] = 0

            # b. positive anchors
            # rule 1
            assigned_gt_inds[max_overlaps >= pos_iou_thr] = (
                max_overlaps_idx[max_overlaps >= pos_iou_thr] + 1
            )

            # rule 2
            # support one bbox to multi anchors, only if the anchors are with the highest iou.
            # rule2 may modify the labels generated by rule 1
            for i in range(len(gt_bboxes)):
                if gt_max_overlaps[i] >= min_iou_thr:
                    assigned_gt_inds[overlaps[i] == gt_max_overlaps[i]] = i + 1

            pos_flag = assigned_gt_inds > 0
            neg_flag = assigned_gt_inds == 0
            # 1. anchor labels
            assigned_gt_labels = (
                torch.zeros_like(cur_anchors[:, 0], dtype=torch.long) + nclasses
            )  # -1 is not optimal, for some bboxes are with labels -1
            assigned_gt_labels[pos_flag] = gt_labels[
                assigned_gt_inds[pos_flag] - 1
            ].long()
            assigned_gt_labels_weights = torch.zeros_like(cur_anchors[:, 0])
            assigned_gt_labels_weights[pos_flag] = 1
            assigned_gt_labels_weights[neg_flag] = 1

            # 2. anchor regression
            assigned_gt_reg_weights = torch.zeros_like(cur_anchors[:, 0])
            assigned_gt_reg_weights[pos_flag] = 1

            assigned_gt_reg = torch.zeros_like(cur_anchors)
            positive_anchors = cur_anchors[pos_flag]
            corr_gt_bboxes = gt_bboxes[assigned_gt_inds[pos_flag] - 1]
            assigned_gt_reg[pos_flag] = bboxes2deltas(corr_gt_bboxes, positive_anchors)

            # 3. anchor direction
            assigned_gt_dir_weights = torch.zeros_like(cur_anchors[:, 0])
            assigned_gt_dir_weights[pos_flag] = 1

            assigned_gt_dir = torch.zeros_like(cur_anchors[:, 0], dtype=torch.long)
            dir_cls_targets = limit_period(corr_gt_bboxes[:, 6].cpu(), 0, 2 * np.pi).to(
                corr_gt_bboxes
            )
            dir_cls_targets = torch.floor(dir_cls_targets / np.pi).long()
            assigned_gt_dir[pos_flag] = torch.clamp(dir_cls_targets, min=0, max=1)

            multi_labels.append(assigned_gt_labels.reshape(d1, d2, 1, d4))
            multi_label_weights.append(
                assigned_gt_labels_weights.reshape(d1, d2, 1, d4)
            )
            multi_bbox_reg.append(assigned_gt_reg.reshape(d1, d2, 1, d4, -1))
            multi_bbox_reg_weights.append(
                assigned_gt_reg_weights.reshape(d1, d2, 1, d4)
            )
            multi_dir_labels.append(assigned_gt_dir.reshape(d1, d2, 1, d4))
            multi_dir_labels_weights.append(
                assigned_gt_dir_weights.reshape(d1, d2, 1, d4)
            )

        multi_labels = torch.cat(multi_labels, dim=-2).reshape(-1)
        multi_label_weights = torch.cat(multi_label_weights, dim=-2).reshape(-1)
        multi_bbox_reg = torch.cat(multi_bbox_reg, dim=-3).reshape(-1, d5)
        multi_bbox_reg_weights = torch.cat(multi_bbox_reg_weights, dim=-2).reshape(-1)
        multi_dir_labels = torch.cat(multi_dir_labels, dim=-2).reshape(-1)
        multi_dir_labels_weights = torch.cat(multi_dir_labels_weights, dim=-2).reshape(
            -1
        )

        batched_labels.append(multi_labels)
        batched_label_weights.append(multi_label_weights)
        batched_bbox_reg.append(multi_bbox_reg)
        batched_bbox_reg_weights.append(multi_bbox_reg_weights)
        batched_dir_labels.append(multi_dir_labels)
        batched_dir_labels_weights.append(multi_dir_labels_weights)

    rt_dict = dict(
        batched_labels=torch.stack(batched_labels, 0),  # (bs, y_l * x_l * 3 * 2)
        batched_label_weights=torch.stack(
            batched_label_weights, 0
        ),  # (bs, y_l * x_l * 3 * 2)
        batched_bbox_reg=torch.stack(batched_bbox_reg, 0),  # (bs, y_l * x_l * 3 * 2, 7)
        batched_bbox_reg_weights=torch.stack(
            batched_bbox_reg_weights, 0
        ),  # (bs, y_l * x_l * 3 * 2)
        batched_dir_labels=torch.stack(
            batched_dir_labels, 0
        ),  # (bs, y_l * x_l * 3 * 2)
        batched_dir_labels_weights=torch.stack(
            batched_dir_labels_weights, 0
        ),  # (bs, y_l * x_l * 3 * 2)
    )

    return rt_dict


def to_center_format(box):
    x_min, y_min, x_max, y_max, angle = box
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    w = x_max - x_min
    h = y_max - y_min
    return np.array([x_center, y_center, w, h, angle])


def compute_iou_bev(box_a, box_b):

    x_a, y_a, w_a, h_a, angle_a = box_a
    x_b, y_b, w_b, h_b, angle_b = box_b

    def get_corners(x, y, w, h, angle):
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        w_half, h_half = w / 2, h / 2
        corners = np.array(
            [
                [
                    x - w_half * cos_a + h_half * sin_a,
                    y - w_half * sin_a - h_half * cos_a,
                ],  # Top-left
                [
                    x + w_half * cos_a + h_half * sin_a,
                    y + w_half * sin_a - h_half * cos_a,
                ],  # Top-right
                [
                    x + w_half * cos_a - h_half * sin_a,
                    y + w_half * sin_a + h_half * cos_a,
                ],  # Bottom-right
                [
                    x - w_half * cos_a - h_half * sin_a,
                    y - w_half * sin_a + h_half * cos_a,
                ],  # Bottom-left
            ]
        )

        return corners

    corners_a = get_corners(x_a, y_a, w_a, h_a, angle_a)
    corners_b = get_corners(x_b, y_b, w_b, h_b, angle_b)

    polygon_a = Polygon(corners_a)
    polygon_b = Polygon(corners_b)

    if not polygon_a.is_valid or not polygon_b.is_valid:
        return 0.0
    intersection_area = polygon_a.intersection(polygon_b).area
    union_area = polygon_a.area + polygon_b.area - intersection_area
    iou = intersection_area / union_area if union_area > 0 else 0.0
    return iou


def nms_cpu(boxes, scores, thres=0.4, pre_maxsize=None, post_maxsize=None):
    is_torch = isinstance(boxes, torch.Tensor)
    if is_torch:
        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()

    order = np.argsort(-scores)
    if pre_maxsize is not None:
        order = order[:pre_maxsize]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ious = np.array(
            [
                compute_iou_bev(to_center_format(boxes[i]), to_center_format(boxes[j]))
                for j in order[1:]
            ]
        )

        remaining = np.where(ious < thres)[0]
        order = order[remaining + 1]

    keep = np.array(keep, dtype=np.int64).tolist()
    if post_maxsize is not None:
        keep = keep[:post_maxsize]
    return keep


def read_points(file_path, dim=4):
    suffix = os.path.splitext(file_path)[1]
    assert suffix in [".bin", ".ply"]
    if suffix == ".bin":
        return np.fromfile(file_path, dtype=np.float32).reshape(-1, dim)
    else:
        raise NotImplementedError


def point_range_filter(pts, point_range=[0, -39.68, -3, 69.12, 39.68, 1]):
    """
    data_dict: dict(pts, gt_bboxes_3d, gt_labels, gt_names, difficulty)
    point_range: [x1, y1, z1, x2, y2, z2]
    """
    flag_x_low = pts[:, 0] > point_range[0]
    flag_y_low = pts[:, 1] > point_range[1]
    flag_z_low = pts[:, 2] > point_range[2]
    flag_x_high = pts[:, 0] < point_range[3]
    flag_y_high = pts[:, 1] < point_range[4]
    flag_z_high = pts[:, 2] < point_range[5]
    keep_mask = (
        flag_x_low & flag_y_low & flag_z_low & flag_x_high & flag_y_high & flag_z_high
    )
    pts = pts[keep_mask]
    return pts


def keep_bbox_from_lidar_range(result, pcd_limit_range):
    """
    result: dict(lidar_bboxes, labels, scores, bboxes2d, camera_bboxes)
    pcd_limit_range: []
    return: dict(lidar_bboxes, labels, scores, bboxes2d, camera_bboxes)
    """

    lidar_bboxes, labels, scores = (
        result["lidar_bboxes"],
        result["labels"],
        result["scores"],
    )
    if "bboxes2d" not in result:
        result["bboxes2d"] = np.zeros_like(lidar_bboxes[:, :4])
    if "camera_bboxes" not in result:
        result["camera_bboxes"] = np.zeros_like(lidar_bboxes)
    bboxes2d, camera_bboxes = result["bboxes2d"], result["camera_bboxes"]
    flag1 = lidar_bboxes[:, :3] > pcd_limit_range[:3][None, :]  # (n, 3)
    flag2 = lidar_bboxes[:, :3] < pcd_limit_range[3:][None, :]  # (n, 3)
    keep_flag = np.all(flag1, axis=-1) & np.all(flag2, axis=-1)

    result = {
        "lidar_bboxes": lidar_bboxes[keep_flag],
        "labels": labels[keep_flag],
        "scores": scores[keep_flag],
        "bboxes2d": bboxes2d[keep_flag],
        "camera_bboxes": camera_bboxes[keep_flag],
    }
    return result
