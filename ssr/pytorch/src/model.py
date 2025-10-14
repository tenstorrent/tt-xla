# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SSR model implementation

Apdapted from: https://github.com/PeidongLi/SSR.git

Apache-2.0 License: https://github.com/PeidongLi/SSR/blob/main/LICENSE
"""
import copy
import math
import torch
import numpy as np
import cv2
from skimage.draw import polygon
from third_party.tt_forge_models.bevdepth.pytorch.src.model import (
    BaseModule,
    Sequential,
    ConvModule,
    BasicBlock,
    Bottleneck,
)
from third_party.tt_forge_models.bevformer.pytorch.src.model import (
    ModuleList,
    Linear,
    xavier_init,
    constant_init,
    ConfigDict,
    GridMask,
)
import torch.nn.functional as F
import torch.nn as nn
from abc import ABCMeta, abstractmethod

ego_width, ego_length = 1.85, 4.084
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
num_classes = len(class_names)

img_backbone = {
    "type": "ResNet",
    "depth": 50,
    "num_stages": 4,
    "out_indices": (3,),
    "frozen_stages": 1,
    "norm_cfg": {"type": "BN", "requires_grad": False},
    "norm_eval": True,
    "style": "pytorch",
}
img_neck = {
    "type": "FPN",
    "in_channels": [2048],
    "out_channels": 256,
    "start_level": 0,
    "add_extra_convs": "on_output",
    "num_outs": 1,
    "relu_before_extra_convs": True,
}
pts_bbox_head = {
    "type": "SSRHead",
    "map_thresh": 0.5,
    "dis_thresh": 0.2,
    "pe_normalization": True,
    "tot_epoch": 12,
    "use_traj_lr_warmup": False,
    "query_thresh": 0.0,
    "query_use_fix_pad": False,
    "ego_his_encoder": None,
    "ego_lcf_feat_idx": None,
    "valid_fut_ts": 6,
    "latent_decoder": {
        "type": "CustomTransformerDecoder",
        "num_layers": 3,
        "return_intermediate": False,
        "transformerlayers": {
            "type": "BaseTransformerLayer",
            "attn_cfgs": [
                {"type": "MultiheadAttention", "embed_dims": 256, "num_heads": 8}
            ],
            "feedforward_channels": 512,
            "operation_order": ("self_attn", "norm", "ffn", "norm"),
        },
    },
    "way_decoder": {
        "type": "CustomTransformerDecoder",
        "num_layers": 1,
        "return_intermediate": False,
        "transformerlayers": {
            "type": "BaseTransformerLayer",
            "attn_cfgs": [
                {"type": "MultiheadAttention", "embed_dims": 256, "num_heads": 8}
            ],
            "feedforward_channels": 512,
            "operation_order": ("cross_attn", "norm", "ffn", "norm"),
        },
    },
    "use_pe": True,
    "bev_h": 100,
    "bev_w": 100,
    "num_query": 300,
    "num_classes": num_classes,
    "in_channels": 256,
    "sync_cls_avg_factor": True,
    "with_box_refine": True,
    "as_two_stage": False,
    "map_num_vec": 100,
    "map_num_classes": 3,
    "map_num_pts_per_vec": 20,
    "map_num_pts_per_gt_vec": 20,
    "map_query_embed_type": "instance_pts",
    "map_transform_method": "minmax",
    "map_gt_shift_pts_pattern": "v2",
    "map_dir_interval": 1,
    "map_code_size": 2,
    "map_code_weights": [1.0, 1.0, 1.0, 1.0],
    "transformer": {
        "type": "SSRPerceptionTransformer",
        "map_num_vec": 100,
        "map_num_pts_per_vec": 20,
        "rotate_prev_bev": False,
        "use_shift": True,
        "use_can_bus": True,
        "embed_dims": 256,
        "encoder": {
            "type": "BEVFormerEncoder",
            "num_layers": 3,
            "pc_range": [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
            "num_points_in_pillar": 4,
            "return_intermediate": False,
            "transformerlayers": {
                "type": "BEVFormerLayer",
                "attn_cfgs": [
                    {
                        "type": "TemporalSelfAttention",
                        "embed_dims": 256,
                        "num_levels": 1,
                    },
                    {
                        "type": "SpatialCrossAttention",
                        "pc_range": [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
                        "deformable_attention": {
                            "type": "MSDeformableAttention3D",
                            "embed_dims": 256,
                            "num_points": 8,
                            "num_levels": 1,
                        },
                        "embed_dims": 256,
                    },
                ],
                "feedforward_channels": 512,
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
        "type": "CustomNMSFreeCoder",
        "post_center_range": [-20, -35, -10.0, 20, 35, 10.0],
        "pc_range": [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
        "max_num": 100,
        "voxel_size": [0.15, 0.15, 4],
        "num_classes": num_classes,
    },
    "map_bbox_coder": {
        "type": "MapNMSFreeCoder",
        "post_center_range": [-20, -35, -20, -35, 20, 35, 20, 35],
        "pc_range": [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
        "max_num": 50,
        "voxel_size": [0.15, 0.15, 4],
        "num_classes": 3,
    },
    "positional_encoding": {
        "type": "LearnedPositionalEncoding",
        "num_feats": 128,
        "row_num_embed": 100,
        "col_num_embed": 100,
    },
    "loss_cls": {
        "type": "FocalLoss",
        "use_sigmoid": True,
        "gamma": 2.0,
        "alpha": 0.25,
        "loss_weight": 2.0,
    },
    "loss_bbox": {"type": "L1Loss", "loss_weight": 0.25},
    "loss_traj": {"type": "L1Loss", "loss_weight": 0.2},
    "loss_traj_cls": {
        "type": "FocalLoss",
        "use_sigmoid": True,
        "gamma": 2.0,
        "alpha": 0.25,
        "loss_weight": 0.2,
    },
    "loss_iou": {"type": "GIoULoss", "loss_weight": 0.0},
    "loss_map_cls": {
        "type": "FocalLoss",
        "use_sigmoid": True,
        "gamma": 2.0,
        "alpha": 0.25,
        "loss_weight": 2.0,
    },
    "loss_map_bbox": {"type": "L1Loss", "loss_weight": 0.0},
    "loss_map_iou": {"type": "GIoULoss", "loss_weight": 0.0},
    "loss_map_pts": {"type": "PtsL1Loss", "loss_weight": 1.0},
    "loss_map_dir": {"type": "PtsDirCosLoss", "loss_weight": 0.005},
    "loss_plan_reg": {"type": "L1Loss", "loss_weight": 1.0},
    "loss_plan_bound": {
        "type": "PlanMapBoundLoss",
        "loss_weight": 1.0,
        "dis_thresh": 1.0,
    },
    "loss_plan_col": {"type": "PlanCollisionLoss", "loss_weight": 1.0},
    "loss_plan_dir": {"type": "PlanMapDirectionLoss", "loss_weight": 0.5},
}
latent_world_model = {
    "type": "CustomTransformerDecoder",
    "num_layers": 2,
    "return_intermediate": False,
    "transformerlayers": {
        "type": "BaseTransformerLayer",
        "attn_cfgs": [
            {"type": "MultiheadAttention", "embed_dims": 256, "num_heads": 8}
        ],
        "feedforward_channels": 512,
        "operation_order": ("self_attn", "norm", "ffn", "norm"),
    },
}

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


def build_norm_hardcoded(cfg, num_features, postfix=None):
    cfg = cfg.copy()
    layer_type = cfg.pop("type")
    requires_grad = cfg.pop("requires_grad", None)
    if layer_type in ("LN", "LayerNorm"):
        module = nn.LayerNorm(num_features, **cfg)
        abbr = "ln"
    elif layer_type in ("GN", "GroupNorm"):
        num_groups = cfg.pop("num_groups")
        module = nn.GroupNorm(num_groups=num_groups, num_channels=num_features, **cfg)
        abbr = "gn"
    else:
        module = nn.BatchNorm2d(num_features, **cfg)
        abbr = "bn"
    if requires_grad is not None:
        for p in module.parameters():
            p.requires_grad = requires_grad
    if postfix is None:
        postfix = ""
    name = f"{abbr}{postfix}"
    return name, module


class PlanningMetric:
    def __init__(self):
        super().__init__()
        self.X_BOUND = [-50.0, 50.0, 0.5]  # Forward
        self.Y_BOUND = [-50.0, 50.0, 0.5]  # Sides
        self.Z_BOUND = [-10.0, 10.0, 20.0]  # Height
        dx, bx, _ = self.gen_dx_bx(self.X_BOUND, self.Y_BOUND, self.Z_BOUND)
        self.dx, self.bx = dx[:2], bx[:2]

        (
            bev_resolution,
            bev_start_position,
            bev_dimension,
        ) = self.calculate_birds_eye_view_parameters(
            self.X_BOUND, self.Y_BOUND, self.Z_BOUND
        )
        self.bev_resolution = bev_resolution.numpy()
        self.bev_start_position = bev_start_position.numpy()
        self.bev_dimension = bev_dimension.numpy()

        self.W = ego_width
        self.H = ego_length

        self.category_index = {
            "human": [2, 3, 4, 5, 6, 7, 8],
            "vehicle": [14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        }

    def gen_dx_bx(self, xbound, ybound, zbound):
        dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
        bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
        nx = torch.LongTensor(
            [(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]]
        )

        return dx, bx, nx

    def calculate_birds_eye_view_parameters(self, x_bounds, y_bounds, z_bounds):
        bev_resolution = torch.tensor(
            [row[2] for row in [x_bounds, y_bounds, z_bounds]]
        )
        bev_start_position = torch.tensor(
            [row[0] + row[2] / 2.0 for row in [x_bounds, y_bounds, z_bounds]]
        )
        bev_dimension = torch.tensor(
            [(row[1] - row[0]) / row[2] for row in [x_bounds, y_bounds, z_bounds]],
            dtype=torch.long,
        )

        return bev_resolution, bev_start_position, bev_dimension

    def get_label(
        self,
        gt_agent_boxes,
        gt_agent_feats,
        gt_map_boxes,
        gt_map_labels,
    ):
        segmentation_np, pedestrian_np = self.get_birds_eye_view_label(
            gt_agent_boxes, gt_agent_feats
        )
        segmentation = torch.from_numpy(segmentation_np).long().unsqueeze(0)
        pedestrian = torch.from_numpy(pedestrian_np).long().unsqueeze(0)

        segmentation_plus = (
            segmentation.squeeze(0).permute(1, 2, 0).cpu().clone().numpy()
        )
        segmentation_plus *= 0  # only consider boudnary, temporal
        map_gt_bboxes_3d = gt_map_boxes[gt_map_labels == 2]
        map_gt_bboxes_3d = (map_gt_bboxes_3d - self.bx.cpu().numpy()) / (
            self.dx.cpu().numpy()
        )
        a = segmentation_plus[:, :, :3].copy()
        a = np.ascontiguousarray(a, dtype=np.uint8)
        b = segmentation_plus[:, :, :3].copy()
        b = np.ascontiguousarray(a, dtype=np.uint8)
        for line in map_gt_bboxes_3d:
            line = line.clip(0, 999).numpy().astype(np.int32)
            for i, corner in enumerate(line[:-1]):
                a = cv2.line(
                    a, tuple(line[i]), tuple(line[i + 1]), color=(1, 1, 1), thickness=1
                )
                b = cv2.line(
                    b, tuple(line[i]), tuple(line[i + 1]), color=(1, 1, 1), thickness=1
                )
        segmentation_plus = (
            torch.cat([torch.tensor(a), torch.tensor(b)], -1)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )

        return segmentation, pedestrian, segmentation_plus

    def get_birds_eye_view_label(self, gt_agent_boxes, gt_agent_feats):
        T = 6
        segmentation = np.zeros((T, self.bev_dimension[0], self.bev_dimension[1]))
        pedestrian = np.zeros((T, self.bev_dimension[0], self.bev_dimension[1]))
        agent_num = gt_agent_feats.shape[1]

        gt_agent_boxes = gt_agent_boxes.tensor.cpu().numpy()  # (N, 9)
        gt_agent_feats = gt_agent_feats.cpu().numpy()

        gt_agent_fut_trajs = gt_agent_feats[..., : T * 2].reshape(-1, 6, 2)
        gt_agent_fut_mask = gt_agent_feats[..., T * 2 : T * 3].reshape(-1, 6)
        gt_agent_fut_yaw = gt_agent_feats[..., T * 3 + 10 : T * 4 + 10].reshape(
            -1, 6, 1
        )
        gt_agent_fut_trajs = np.cumsum(gt_agent_fut_trajs, axis=1)
        gt_agent_fut_yaw = np.cumsum(gt_agent_fut_yaw, axis=1)

        gt_agent_boxes[:, 6:7] = -1 * (
            gt_agent_boxes[:, 6:7] + np.pi / 2
        )  # NOTE: convert yaw to lidar frame
        gt_agent_fut_trajs = gt_agent_fut_trajs + gt_agent_boxes[:, np.newaxis, 0:2]
        gt_agent_fut_yaw = gt_agent_fut_yaw + gt_agent_boxes[:, np.newaxis, 6:7]

        for t in range(T):
            for i in range(agent_num):
                if gt_agent_fut_mask[i][t] == 1:
                    category_index = int(gt_agent_feats[0, i][27])
                    agent_length, agent_width = (
                        gt_agent_boxes[i][4],
                        gt_agent_boxes[i][3],
                    )
                    x_a = gt_agent_fut_trajs[i, t, 0]
                    y_a = gt_agent_fut_trajs[i, t, 1]
                    yaw_a = gt_agent_fut_yaw[i, t, 0]
                    param = [x_a, y_a, yaw_a, agent_length, agent_width]
                    if category_index in self.category_index["vehicle"]:
                        poly_region = self._get_poly_region_in_image(param)
                        cv2.fillPoly(segmentation[t], [poly_region], 1.0)
                    if category_index in self.category_index["human"]:
                        poly_region = self._get_poly_region_in_image(param)
                        cv2.fillPoly(pedestrian[t], [poly_region], 1.0)
        return segmentation, pedestrian

    def _get_poly_region_in_image(self, param):
        lidar2cv_rot = np.array([[1, 0], [0, -1]])
        x_a, y_a, yaw_a, agent_length, agent_width = param
        trans_a = np.array([[x_a, y_a]]).T
        rot_mat_a = np.array(
            [[np.cos(yaw_a), -np.sin(yaw_a)], [np.sin(yaw_a), np.cos(yaw_a)]]
        )
        agent_corner = np.array(
            [
                [
                    agent_length / 2,
                    -agent_length / 2,
                    -agent_length / 2,
                    agent_length / 2,
                ],
                [agent_width / 2, agent_width / 2, -agent_width / 2, -agent_width / 2],
            ]
        )  # (2,4)
        agent_corner_lidar = np.matmul(rot_mat_a, agent_corner) + trans_a  # (2,4)
        agent_corner_cv2 = (
            np.matmul(lidar2cv_rot, agent_corner_lidar)
            - self.bev_start_position[:2, None]
            + self.bev_resolution[:2, None] / 2.0
        ).T / self.bev_resolution[
            :2
        ]  # (4,2)
        agent_corner_cv2 = np.round(agent_corner_cv2).astype(np.int32)

        return agent_corner_cv2

    def evaluate_single_coll(self, traj, segmentation, input_gt):
        pts = np.array(
            [
                [-self.H / 2.0 + 0.5, self.W / 2.0],
                [self.H / 2.0 + 0.5, self.W / 2.0],
                [self.H / 2.0 + 0.5, -self.W / 2.0],
                [-self.H / 2.0 + 0.5, -self.W / 2.0],
            ]
        )
        pts = (pts - self.bx.cpu().numpy()) / (self.dx.cpu().numpy())
        pts[:, [0, 1]] = pts[:, [1, 0]]
        rr, cc = polygon(pts[:, 1], pts[:, 0])
        rc = np.concatenate([rr[:, None], cc[:, None]], axis=-1)

        n_future, _ = traj.shape
        trajs = traj.view(n_future, 1, 2)
        trajs_ = copy.deepcopy(trajs)
        trajs_[:, :, [0, 1]] = trajs_[:, :, [1, 0]]  # can also change original tensor
        trajs_ = trajs_ / self.dx.to(trajs.device)
        trajs_ = trajs_.cpu().numpy() + rc  # (n_future, 32, 2)

        r = (self.bev_dimension[0] - trajs_[:, :, 0]).astype(np.int32)
        r = np.clip(r, 0, self.bev_dimension[0] - 1)

        c = trajs_[:, :, 1].astype(np.int32)
        c = np.clip(c, 0, self.bev_dimension[1] - 1)

        collision = np.full(n_future, False)
        for t in range(n_future):
            rr = r[t]
            cc = c[t]
            I = np.logical_and(
                np.logical_and(rr >= 0, rr < self.bev_dimension[0]),
                np.logical_and(cc >= 0, cc < self.bev_dimension[1]),
            )
            collision[t] = np.any(segmentation[t, rr[I], cc[I]].cpu().numpy())

        return torch.from_numpy(collision).to(device=traj.device)

    def evaluate_coll(self, trajs, gt_trajs, segmentation):
        B, n_future, _ = trajs.shape

        obj_coll_sum = torch.zeros(n_future, device=segmentation.device)
        obj_box_coll_sum = torch.zeros(n_future, device=segmentation.device)

        for i in range(B):
            gt_box_coll = self.evaluate_single_coll(
                gt_trajs[i], segmentation[i], input_gt=True
            )

            xx, yy = trajs[i, :, 0], trajs[i, :, 1]
            xi = ((-self.bx[0] / 2 - yy) / self.dx[0]).long()
            yi = ((-self.bx[1] / 2 + xx) / self.dx[1]).long()

            m1 = torch.logical_and(
                torch.logical_and(xi >= 0, xi < self.bev_dimension[0]),
                torch.logical_and(yi >= 0, yi < self.bev_dimension[1]),
            ).to(gt_box_coll.device)
            m1 = torch.logical_and(m1, torch.logical_not(gt_box_coll))

            ti = torch.arange(n_future)
            m1 = m1.cpu()
            obj_coll_sum[ti[m1]] += segmentation[i, ti[m1], xi[m1], yi[m1]].long()

            m2 = torch.logical_not(gt_box_coll)
            m2 = m2.cpu()
            box_coll = self.evaluate_single_coll(
                trajs[i], segmentation[i], input_gt=False
            ).to(ti.device)
            obj_box_coll_sum[ti[m2]] += (box_coll[ti[m2]]).long()

        return obj_coll_sum, obj_box_coll_sum

    def compute_L2(self, trajs, gt_trajs):
        pred_len = trajs.shape[0]
        ade = float(
            sum(
                torch.sqrt(
                    (trajs[i, 0] - gt_trajs[i, 0]) ** 2
                    + (trajs[i, 1] - gt_trajs[i, 1]) ** 2
                )
                for i in range(pred_len)
            )
            / pred_len
        )

        return ade

    def compute_L2_stp3(self, trajs, gt_trajs):
        ade = float(
            torch.sqrt(
                (trajs[-1, 0] - gt_trajs[-1, 0]) ** 2
                + (trajs[-1, 1] - gt_trajs[-1, 1]) ** 2
            )
        )
        return ade


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
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
        self.backbone_end_level = self.num_ins
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
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

    def forward(self, inputs):
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=prev_shape, **self.upsample_cfg
            )
        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]
        return tuple(outs)


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


class MLN(nn.Module):
    def __init__(self, c_dim, f_dim=256, use_ln=True):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.use_ln = use_ln

        self.reduce = nn.Sequential(
            nn.Linear(c_dim, f_dim),
            nn.ReLU(),
        )
        self.gamma = nn.Linear(f_dim, f_dim)
        self.beta = nn.Linear(f_dim, f_dim)
        self.ln = nn.LayerNorm(f_dim, elementwise_affine=False)
        self.init_weight()

    def init_weight(self):
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, c):
        x = self.ln(x)
        c = self.reduce(c)
        gamma = self.gamma(c)
        beta = self.beta(c)
        out = gamma * x + beta

        return out


class TokenLearnerV11(nn.Module):
    def __init__(self, num_tokens, in_channels, bottleneck_dim=64, dropout_rate=0.0):
        super(TokenLearnerV11, self).__init__()
        self.num_tokens = num_tokens
        self.bottleneck_dim = bottleneck_dim
        self.dropout_rate = dropout_rate
        self.layer_norm = nn.GroupNorm(1, in_channels, eps=1e-6)
        self.mlp = MlpBlock(
            input_dim=in_channels,
            mlp_dim=self.bottleneck_dim,
            output_dim=self.num_tokens,
            dropout_rate=self.dropout_rate,
        )

    def forward(self, inputs, deterministic=True):
        selected = self.mlp(self.layer_norm(inputs.permute(0, 2, 1)).permute(0, 2, 1))
        selected = selected.view(inputs.shape[0], self.num_tokens, -1).softmax(dim=-1)
        feat = inputs.view(inputs.shape[0], -1, inputs.shape[-1])
        outputs = torch.einsum("bsi,bic->bsc", selected, feat)
        return outputs, selected


class SELayer(nn.Module):
    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.mlp_reduce = nn.Linear(channels, channels)
        self.act1 = act_layer()
        self.mlp_expand = nn.Linear(channels, channels)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.mlp_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.mlp_expand(x_se)
        return x * self.gate(x_se)


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
        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return output


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
        if isinstance(deformable_attention, dict):
            cfg = deformable_attention.copy()
            cfg.pop("type", None)
            self.deformable_attention = MSDeformableAttention3D(**cfg)
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

        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)

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


class BaseBBoxCoder(metaclass=ABCMeta):
    """Base bounding box coder."""

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def encode(self, bboxes, gt_bboxes):
        """Encode deltas between bboxes and ground truth boxes."""

    @abstractmethod
    def decode(self, bboxes, bboxes_pred):
        """Decode the predicted bboxes according to prediction and base
        boxes."""


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
                if isinstance(attn_cfgs[index], dict):
                    attention_type = attn_cfgs[index].pop("type")
                    cfg = attn_cfgs[index].copy()
                    cfg.pop("type", None)
                    if attention_type == "TemporalSelfAttention":
                        attention = TemporalSelfAttention(**cfg)
                    elif attention_type == "SpatialCrossAttention":
                        attention = SpatialCrossAttention(**cfg)

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
            if isinstance(ffn_cfgs[ffn_index], dict):
                cfg = ffn_cfgs[ffn_index].copy()
                cfg.pop("type", None)
                self.ffns.append(FFN(**cfg))

        self.norms = ModuleList()
        num_norms = operation_order.count("norm")
        for _ in range(num_norms):
            self.norms.append(build_norm_hardcoded(norm_cfg, self.embed_dims)[1])

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
        super().__init__(init_cfg)
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        if isinstance(act_cfg, dict):
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
        if dropout_layer:
            cfg = dropout_layer.copy()
            drop_type = cfg.pop("type")
            if drop_type == "Dropout":
                p = cfg.pop("drop_prob", 0.0)
                self.dropout_layer = nn.Dropout(p=p, **cfg)
        else:
            self.dropout_layer = nn.Identity()
        self.add_identity = add_identity

    def forward(self, x, identity=None):
        out = self.layers(x)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)


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

        super().__init__(init_cfg)

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
                if isinstance(attn_cfgs[index], dict):
                    cfg = attn_cfgs[index].copy()
                    cfg.pop("type", None)
                    attention = MultiheadAttention(**cfg)
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
            if isinstance(ffn_cfgs[ffn_index], dict):
                cfg = ffn_cfgs[ffn_index].copy()
                cfg.pop("type", None)
                self.ffns.append(FFN(**cfg))

        self.norms = ModuleList()
        num_norms = operation_order.count("norm")
        for _ in range(num_norms):
            self.norms.append(build_norm_hardcoded(norm_cfg, self.embed_dims)[1])

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
        super().__init__(init_cfg)

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first

        self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop, **kwargs)

        self.proj_drop = nn.Dropout(proj_drop)
        if dropout_layer:
            cfg = dropout_layer.copy()
            drop_type = cfg.pop("type")
            if drop_type == "Dropout":
                p = cfg.pop("drop_prob", 0.0)
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


class TransformerLayerSequence(BaseModule):
    def __init__(self, transformerlayers=None, num_layers=None, init_cfg=None):
        super().__init__(init_cfg)
        if isinstance(transformerlayers, dict):
            transformerlayers = [
                copy.deepcopy(transformerlayers) for _ in range(num_layers)
            ]
        self.num_layers = num_layers
        self.layers = ModuleList()
        for i in range(num_layers):
            layer_cfg = transformerlayers[i]
            if isinstance(layer_cfg, dict):
                layer_type = layer_cfg.get("type", "BaseTransformerLayer")
                cfg = layer_cfg.copy()
                cfg.pop("type", None)
                if layer_type == "BEVFormerLayer":
                    layer = BEVFormerLayer(**cfg)
                elif layer_type == "BaseTransformerLayer":
                    layer = BaseTransformerLayer(**cfg)
            self.layers.append(layer)
        self.embed_dims = self.layers[0].embed_dims
        self.pre_norm = self.layers[0].pre_norm

    def forward(
        self,
        query,
        key,
        value,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        **kwargs,
    ):
        for layer in self.layers:
            query = layer(
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                **kwargs,
            )
        return query


class CustomTransformerDecoder(TransformerLayerSequence):
    def __init__(self, *args, return_intermediate=False, **kwargs):
        super(CustomTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.fp16_enabled = False

    def forward(
        self,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        key_padding_mask=None,
        *args,
        **kwargs,
    ):
        for lid, layer in enumerate(self.layers):
            query = layer(
                query=query,
                key=key,
                value=value,
                query_pos=query_pos,
                key_pos=key_pos,
                attn_masks=attn_masks,
                key_padding_mask=key_padding_mask,
                *args,
                **kwargs,
            )

        return query


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

        shift_ref_2d = ref_2d
        shift_ref_2d += shift[:, None, None, :]

        bev_query = bev_query.permute(1, 0, 2)
        bev_pos = bev_pos.permute(1, 0, 2)
        bs, len_bev, num_bev_level, _ = ref_2d.shape
        if prev_bev is not None:
            prev_bev = prev_bev.permute(1, 0, 2)
            prev_bev = torch.stack([prev_bev, bev_query], 1).reshape(
                bs * 2, len_bev, -1
            )
            hybird_ref_2d = torch.stack([shift_ref_2d, ref_2d], 1).reshape(
                bs * 2, len_bev, num_bev_level, 2
            )
        else:
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


class SSRPerceptionTransformer(BaseModule):
    def __init__(
        self,
        num_feature_levels=4,
        num_cams=6,
        two_stage_num_proposals=300,
        encoder=None,
        decoder=None,
        map_decoder=None,
        embed_dims=256,
        rotate_prev_bev=True,
        use_shift=True,
        use_can_bus=True,
        can_bus_norm=True,
        use_cams_embeds=True,
        rotate_center=[100, 100],
        map_num_vec=50,
        map_num_pts_per_vec=10,
        **kwargs,
    ):
        super(SSRPerceptionTransformer, self).__init__(**kwargs)
        if isinstance(encoder, dict):
            cfg = encoder.copy()
            cfg.pop("type", None)
            self.encoder = BEVFormerEncoder(**cfg)

        self.decoder = None
        self.map_decoder = None

        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False
        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds
        self.two_stage_num_proposals = two_stage_num_proposals
        self.rotate_center = rotate_center
        self.map_num_vec = map_num_vec
        self.map_num_pts_per_vec = map_num_pts_per_vec
        self.init_layers()

    def init_layers(self):
        self.level_embeds = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims)
        )
        self.cams_embeds = nn.Parameter(torch.Tensor(self.num_cams, self.embed_dims))
        self.reference_points = nn.Linear(self.embed_dims, 3)
        self.map_reference_points = nn.Linear(self.embed_dims, 2)
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


class CustomNMSFreeCoder(BaseBBoxCoder):
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

    def decode(self, preds_dicts):
        all_cls_scores = preds_dicts["all_cls_scores"][-1]
        all_bbox_preds = preds_dicts["all_bbox_preds"][-1]
        all_traj_preds = preds_dicts["all_traj_preds"][-1]

        batch_size = all_cls_scores.size()[0]
        predictions_list = []
        for i in range(batch_size):
            predictions_list.append(
                self.decode_single(
                    all_cls_scores[i], all_bbox_preds[i], all_traj_preds[i]
                )
            )
        return predictions_list


class BaseDenseHead(BaseModule, metaclass=ABCMeta):
    def __init__(self, init_cfg=None):
        super(BaseDenseHead, self).__init__(init_cfg)


class BBoxTestMixin(object):
    def simple_test_bboxes(self, feats, img_metas, rescale=False):
        outs = self.forward(feats)
        results_list = self.get_bboxes(*outs, img_metas=img_metas, rescale=rescale)
        return results_list


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
        loss_cls=dict(
            type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0
        ),
        loss_bbox=dict(type="IoULoss", loss_weight=1.0),
        bbox_coder=dict(type="DistancePointBBoxCoder"),
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
        self.use_sigmoid_cls = loss_cls.get("use_sigmoid", False)
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.dcn_on_last_conv = dcn_on_last_conv
        self.conv_bias = conv_bias
        self.num_base_priors = self.prior_generator.num_base_priors[0]

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False

        self._init_layers()


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
            self.bg_cls_weight = bg_cls_weight
        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_reg_fcs = num_reg_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.cls_out_channels = num_classes

        self.act_cfg = transformer.get("act_cfg", dict(type="ReLU", inplace=True))
        if isinstance(self.act_cfg, dict):
            cfg = self.act_cfg.copy()
            act_type = cfg.pop("type")
            self.activate = ACTIVATION_MAP[act_type](**cfg)

        if isinstance(positional_encoding, dict):
            cfg = positional_encoding.copy()
            cfg.pop("type", None)
            self.positional_encoding = LearnedPositionalEncoding(**cfg)
        if isinstance(transformer, dict):
            cfg = transformer.copy()
            cfg.pop("type", None)
            self.transformer = SSRPerceptionTransformer(**cfg)
        self.embed_dims = self.transformer.embed_dims
        self._init_layers()

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
        super(AnchorFreeHead, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class SSRHead(DETRHead):
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
        fut_ts=6,
        fut_mode=6,
        loss_traj=dict(type="L1Loss", loss_weight=0.25),
        loss_traj_cls=dict(
            type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=0.8
        ),
        map_bbox_coder=None,
        map_num_query=900,
        map_num_classes=3,
        map_num_vec=20,
        map_num_pts_per_vec=2,
        map_num_pts_per_gt_vec=2,
        map_query_embed_type="all_pts",
        map_transform_method="minmax",
        map_gt_shift_pts_pattern="v0",
        map_dir_interval=1,
        map_code_size=None,
        map_code_weights=None,
        loss_map_cls=dict(
            type="CrossEntropyLoss",
            bg_cls_weight=0.1,
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=1.0,
        ),
        loss_map_bbox=dict(type="L1Loss", loss_weight=5.0),
        loss_map_iou=dict(type="GIoULoss", loss_weight=2.0),
        loss_map_pts=dict(
            type="ChamferDistance", loss_src_weight=1.0, loss_dst_weight=1.0
        ),
        loss_map_dir=dict(type="PtsDirCosLoss", loss_weight=2.0),
        num_scenes=16,
        latent_decoder=None,
        way_decoder=None,
        ego_fut_mode=3,
        loss_plan_reg=dict(type="L1Loss", loss_weight=0.25),
        ego_lcf_feat_idx=None,
        valid_fut_ts=6,
        **kwargs,
    ):

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode

        self.latent_decoder = latent_decoder
        self.way_decoder = way_decoder

        self.ego_fut_mode = ego_fut_mode
        self.ego_lcf_feat_idx = ego_lcf_feat_idx
        self.valid_fut_ts = valid_fut_ts
        self.num_scenes = num_scenes
        self.traj_num_cls = 1
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        self.code_size = 10
        self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
        self.map_code_size = map_code_size
        self.map_code_weights = map_code_weights
        if isinstance(bbox_coder, dict):
            cfg = bbox_coder.copy()
            cfg.pop("type", None)
            self.bbox_coder = CustomNMSFreeCoder(**cfg)
        self.pc_range = self.bbox_coder.pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]

        self.map_query_embed_type = map_query_embed_type
        self.map_num_vec = map_num_vec
        self.map_num_pts_per_vec = map_num_pts_per_vec
        self.map_cls_out_channels = map_num_classes

        super(SSRHead, self).__init__(*args, transformer=transformer, **kwargs)
        self.code_weights = nn.Parameter(
            torch.tensor(self.code_weights, requires_grad=False), requires_grad=False
        )
        self.map_code_weights = nn.Parameter(
            torch.tensor(self.map_code_weights, requires_grad=False),
            requires_grad=False,
        )

    def _init_layers(self):
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        cls_branch = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        traj_branch = []
        for _ in range(self.num_reg_fcs):
            traj_branch.append(Linear(self.embed_dims * 2, self.embed_dims * 2))
            traj_branch.append(nn.ReLU())
        traj_branch.append(Linear(self.embed_dims * 2, self.fut_ts * 2))
        traj_branch = nn.Sequential(*traj_branch)

        traj_cls_branch = []
        for _ in range(self.num_reg_fcs):
            traj_cls_branch.append(Linear(self.embed_dims * 2, self.embed_dims * 2))
            traj_cls_branch.append(nn.LayerNorm(self.embed_dims * 2))
            traj_cls_branch.append(nn.ReLU(inplace=True))
        traj_cls_branch.append(Linear(self.embed_dims * 2, self.traj_num_cls))
        traj_cls_branch = nn.Sequential(*traj_cls_branch)

        map_cls_branch = []
        for _ in range(self.num_reg_fcs):
            map_cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            map_cls_branch.append(nn.LayerNorm(self.embed_dims))
            map_cls_branch.append(nn.ReLU(inplace=True))
        map_cls_branch.append(Linear(self.embed_dims, self.map_cls_out_channels))
        map_cls_branch = nn.Sequential(*map_cls_branch)

        map_reg_branch = []
        for _ in range(self.num_reg_fcs):
            map_reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            map_reg_branch.append(nn.ReLU())
        map_reg_branch.append(Linear(self.embed_dims, self.map_code_size))
        map_reg_branch = nn.Sequential(*map_reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        num_decoder_layers = 1
        num_map_decoder_layers = 1
        if self.transformer.decoder is not None:
            num_decoder_layers = self.transformer.decoder.num_layers
        if self.transformer.map_decoder is not None:
            num_map_decoder_layers = self.transformer.map_decoder.num_layers
        num_motion_decoder_layers = 1
        num_pred = (num_decoder_layers + 1) if self.as_two_stage else num_decoder_layers
        motion_num_pred = (
            (num_motion_decoder_layers + 1)
            if self.as_two_stage
            else num_motion_decoder_layers
        )
        map_num_pred = (
            (num_map_decoder_layers + 1)
            if self.as_two_stage
            else num_map_decoder_layers
        )

        if self.with_box_refine:
            self.cls_branches = _get_clones(cls_branch, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
            self.traj_branches = _get_clones(traj_branch, motion_num_pred)
            self.traj_cls_branches = _get_clones(traj_cls_branch, motion_num_pred)
            self.map_cls_branches = _get_clones(map_cls_branch, map_num_pred)
            self.map_reg_branches = _get_clones(map_reg_branch, map_num_pred)
        if not self.as_two_stage:
            self.bev_embedding = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)
            self.query_embedding = nn.Embedding(self.num_query, self.embed_dims * 2)
            self.map_query_embedding = None
            self.map_instance_embedding = nn.Embedding(
                self.map_num_vec, self.embed_dims * 2
            )
            self.map_pts_embedding = nn.Embedding(
                self.map_num_pts_per_vec, self.embed_dims * 2
            )

        self.ego_query = nn.Embedding(1, self.embed_dims)

        ego_fut_decoder = []
        ego_fut_dec_in_dim = (
            self.embed_dims + len(self.ego_lcf_feat_idx)
            if self.ego_lcf_feat_idx is not None
            else self.embed_dims
        )
        for _ in range(self.num_reg_fcs):
            ego_fut_decoder.append(Linear(ego_fut_dec_in_dim, ego_fut_dec_in_dim))
            ego_fut_decoder.append(nn.ReLU())
        ego_fut_decoder.append(Linear(ego_fut_dec_in_dim, 2))
        self.ego_fut_decoder = nn.Sequential(*ego_fut_decoder)
        self.navi_embedding = nn.Embedding(3, self.embed_dims)
        self.navi_se = SELayer(self.embed_dims)

        self.way_point = nn.Embedding(
            self.ego_fut_mode * self.fut_ts, self.embed_dims * 2
        )
        self.tokenlearner = TokenLearnerV11(self.num_scenes, self.embed_dims * 2)
        if isinstance(self.latent_decoder, dict):
            cfg = self.latent_decoder.copy()
            cfg.pop("type", None)
            self.latent_decoder = CustomTransformerDecoder(**cfg)
        if isinstance(self.way_decoder, dict):
            cfg = self.way_decoder.copy()
            cfg.pop("type", None)
            self.way_decoder = CustomTransformerDecoder(**cfg)

        self.action_mln = MLN(self.fut_ts * 2)
        self.pos_mln = MLN(self.fut_ts * 2)

    def forward(
        self,
        mlvl_feats,
        img_metas,
        prev_bev=None,
        only_bev=False,
        ego_his_trajs=None,
        ego_lcf_feat=None,
        cmd=None,
    ):
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype

        bev_queries = self.bev_embedding.weight.to(dtype)

        bev_mask = torch.zeros(
            (bs, self.bev_h, self.bev_w), device=bev_queries.device
        ).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)

        bev_embed = self.transformer.get_bev_features(
            mlvl_feats,
            bev_queries,
            self.bev_h,
            self.bev_w,
            grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
            bev_pos=bev_pos,
            img_metas=img_metas,
            prev_bev=prev_bev,
        )
        if only_bev:
            return bev_embed

        pos_embd = bev_pos.flatten(2).permute(0, 2, 1)
        cmd = cmd[0, 0, 0]
        cmd_idx = torch.nonzero(cmd)[0, 0]
        navi_embed = self.navi_embedding.weight[cmd_idx][None, None]
        bev_navi_embed = self.navi_se(bev_embed, navi_embed)
        bev_query = torch.cat((bev_navi_embed, pos_embd), -1)

        learned_latent_query, selected = self.tokenlearner(bev_query)
        learned_latent_query = learned_latent_query.permute(1, 0, 2)
        latent_query, latent_pos = torch.split(
            learned_latent_query, self.embed_dims, dim=2
        )

        latent_query = self.latent_decoder(
            query=latent_query,
            key=latent_query,
            value=latent_query,
            query_pos=latent_pos,
            key_pos=latent_pos,
        )

        way_point = self.way_point.weight.to(dtype)
        wp_pos, way_point = torch.split(way_point, self.embed_dims, dim=1)

        wp_pos = wp_pos.unsqueeze(0).expand(bs, -1, -1)
        way_point = way_point.unsqueeze(0).expand(bs, -1, -1)
        wp_pos = wp_pos.permute(1, 0, 2)
        way_point = way_point.permute(1, 0, 2)

        way_point = self.way_decoder(
            query=way_point,
            key=latent_query,
            value=latent_query,
            query_pos=wp_pos,
            key_pos=latent_pos,
        )

        outputs_ego_trajs = self.ego_fut_decoder(way_point)
        outputs_ego_trajs = outputs_ego_trajs.permute(1, 0, 2).view(
            bs, self.ego_fut_mode, self.fut_ts, 2
        )
        outputs_ego_trajs_fut = outputs_ego_trajs[:, cmd_idx, ...]
        wp_vector = outputs_ego_trajs_fut.reshape(-1)

        wp_vector = wp_vector.unsqueeze(0).unsqueeze(0)

        act_query = self.action_mln(latent_query, wp_vector)

        outs = {
            "bev_embed": bev_embed,
            "scene_query": latent_query,
            "act_query": act_query,
            "ego_fut_preds": outputs_ego_trajs,
        }

        return outs


class MlpBlock(nn.Module):
    def __init__(self, input_dim, mlp_dim, output_dim, dropout_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class TokenFuser(nn.Module):
    def __init__(
        self,
        num_tokens,
        in_channels,
        use_normalization=True,
        bottleneck_dim=64,
        dropout_rate=0.0,
    ):
        super().__init__()
        self.use_normalization = use_normalization
        self.bottleneck_dim = bottleneck_dim
        self.dropout_rate = dropout_rate
        self.norm = nn.GroupNorm(1, in_channels, eps=1e-6)
        self.norm1 = (
            nn.GroupNorm(1, in_channels, eps=1e-6) if use_normalization else None
        )
        self.norm2 = (
            nn.GroupNorm(1, in_channels, eps=1e-6) if use_normalization else None
        )
        self.dense = nn.Linear(in_features=num_tokens, out_features=bottleneck_dim)
        self.mlp = MlpBlock(
            input_dim=in_channels,
            mlp_dim=bottleneck_dim,
            output_dim=bottleneck_dim,
            dropout_rate=dropout_rate,
        )
        self.dropout = nn.Dropout(dropout_rate)


class BaseDetector(BaseModule, metaclass=ABCMeta):
    def __init__(self, init_cfg=None):
        super(BaseDetector, self).__init__(init_cfg)
        self.fp16_enabled = False

    def val_step(self, data, optimizer=None):
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)
        outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(data["img_metas"]))
        return outputs


class Base3DDetector(BaseDetector):
    def forward(self, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)


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
            if isinstance(pts_bbox_head, dict):
                cfg = pts_bbox_head.copy()
                cfg.pop("type", None)
                self.pts_bbox_head = SSRHead(**cfg)

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

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if pretrained is None:
            img_pretrained = None
            pts_pretrained = None
        elif isinstance(pretrained, dict):
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


class SSR(MVXTwoStageDetector):
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
        latent_world_model=None,
        img_roi_head=None,
        img_rpn_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        video_test_mode=False,
        fut_ts=6,
        fut_mode=6,
        loss_bev=None,
    ):

        super(SSR, self).__init__(
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
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.valid_fut_ts = pts_bbox_head["valid_fut_ts"]

        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            "prev_bev": None,
            "scene_token": None,
            "prev_pos": 0,
            "prev_angle": 0,
        }

        self.planning_metric = None
        self.embed_dims = 256
        self.latent_world_model = latent_world_model
        self.tokenfuser = TokenFuser(16, 256)
        if self.latent_world_model is not None:
            if isinstance(self.latent_world_model, dict):
                cfg = self.latent_world_model.copy()
                cfg.pop("type", None)
                self.latent_world_model = CustomTransformerDecoder(**cfg)
            for p in self.latent_world_model.parameters():
                if p.dim() > 1:
                    torch.nn.init.xavier_uniform_(p)

    def extract_img_feat(self, img, img_metas, len_queue=None):
        B = img.size(0)
        if img is not None:
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            if self.use_grid_mask:
                img = self.grid_mask(img)
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

    def forward(self, return_loss=False, **kwargs):
        return self.forward_test(**kwargs)

    def forward_test(
        self,
        img_metas,
        gt_bboxes_3d,
        gt_labels_3d,
        map_gt_bboxes_3d,
        map_gt_labels_3d,
        img=None,
        ego_his_trajs=None,
        ego_fut_trajs=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        gt_attr_labels=None,
        **kwargs,
    ):
        img = [img] if img is None else img

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
            img_metas=img_metas[0],
            img=img[0],
            prev_bev=self.prev_frame_info["prev_bev"],
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            map_gt_bboxes_3d=map_gt_bboxes_3d,
            map_gt_labels_3d=map_gt_labels_3d,
            ego_his_trajs=ego_his_trajs[0],
            ego_fut_trajs=ego_fut_trajs[0],
            ego_fut_cmd=ego_fut_cmd[0],
            ego_lcf_feat=ego_lcf_feat[0],
            gt_attr_labels=gt_attr_labels,
            **kwargs,
        )
        self.prev_frame_info["prev_pos"] = tmp_pos
        self.prev_frame_info["prev_bev"] = new_prev_bev
        self.prev_frame_info["prev_angle"] = tmp_angle

        return bbox_results

    def simple_test(
        self,
        img_metas,
        gt_bboxes_3d,
        gt_labels_3d,
        map_gt_bboxes_3d,
        map_gt_labels_3d,
        img=None,
        prev_bev=None,
        points=None,
        rescale=False,
        ego_his_trajs=None,
        ego_fut_trajs=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        gt_attr_labels=None,
        **kwargs,
    ):
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, bbox_pts, metric_dict = self.simple_test_pts(
            img_feats,
            img_metas,
            gt_bboxes_3d,
            gt_labels_3d,
            map_gt_bboxes_3d,
            map_gt_labels_3d,
            prev_bev,
            rescale=rescale,
            start=None,
            ego_his_trajs=ego_his_trajs,
            ego_fut_trajs=ego_fut_trajs,
            ego_fut_cmd=ego_fut_cmd,
            ego_lcf_feat=ego_lcf_feat,
            gt_attr_labels=gt_attr_labels,
        )
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict["pts_bbox"] = pts_bbox
            result_dict["metric_results"] = metric_dict

        return new_prev_bev, bbox_list

    def simple_test_pts(
        self,
        x,
        img_metas,
        gt_bboxes_3d,
        gt_labels_3d,
        map_gt_bboxes_3d,
        map_gt_labels_3d,
        prev_bev=None,
        rescale=False,
        start=None,
        ego_his_trajs=None,
        ego_fut_trajs=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        gt_attr_labels=None,
    ):

        outs = self.pts_bbox_head(
            x,
            img_metas,
            prev_bev=prev_bev,
            cmd=ego_fut_cmd,
            ego_his_trajs=ego_his_trajs,
            ego_lcf_feat=ego_lcf_feat,
        )

        bbox_results = []
        for i in range(len(outs["ego_fut_preds"])):
            bbox_result = dict()
            bbox_result["ego_fut_preds"] = outs["ego_fut_preds"][i].cpu()
            bbox_result["ego_fut_cmd"] = ego_fut_cmd.cpu()
            bbox_results.append(bbox_result)

        with torch.no_grad():
            gt_bbox = gt_bboxes_3d[0][0]
            gt_map_bbox = map_gt_bboxes_3d[0]
            gt_label = gt_labels_3d[0][0].to("cpu")
            gt_map_label = map_gt_labels_3d[0].to("cpu")
            gt_attr_label = gt_attr_labels[0][0].to("cpu")

            metric_dict = {}
            ego_fut_preds = bbox_result["ego_fut_preds"]
            ego_fut_trajs = ego_fut_trajs[0, 0]

            ego_fut_cmd = ego_fut_cmd[0, 0, 0]
            ego_fut_cmd_idx = torch.nonzero(ego_fut_cmd)[0, 0]

            ego_fut_pred = ego_fut_preds[ego_fut_cmd_idx]
            ego_fut_pred = ego_fut_pred.cumsum(dim=-2)
            ego_fut_trajs = ego_fut_trajs.cumsum(dim=-2)

            metric_dict_planner_stp3 = self.compute_planner_metric_stp3(
                pred_ego_fut_trajs=ego_fut_pred[None],
                gt_ego_fut_trajs=ego_fut_trajs[None],
                gt_agent_boxes=gt_bbox,
                gt_agent_feats=gt_attr_label.unsqueeze(0),
                gt_map_boxes=gt_map_bbox,
                gt_map_labels=gt_map_label,
            )
            metric_dict.update(metric_dict_planner_stp3)

        return outs["bev_embed"], bbox_results, metric_dict

    def compute_planner_metric_stp3(
        self,
        pred_ego_fut_trajs,
        gt_ego_fut_trajs,
        gt_agent_boxes,
        gt_agent_feats,
        gt_map_boxes,
        gt_map_labels,
    ):
        metric_dict = {
            "plan_L2_1s": 0,
            "plan_L2_2s": 0,
            "plan_L2_3s": 0,
            "plan_obj_col_1s": 0,
            "plan_obj_col_2s": 0,
            "plan_obj_col_3s": 0,
            "plan_obj_box_col_1s": 0,
            "plan_obj_box_col_2s": 0,
            "plan_obj_box_col_3s": 0,
        }
        future_second = 3
        if self.planning_metric is None:
            self.planning_metric = PlanningMetric()
        segmentation, pedestrian, segmentation_plus = self.planning_metric.get_label(
            gt_agent_boxes, gt_agent_feats, gt_map_boxes, gt_map_labels
        )
        occupancy = torch.logical_or(segmentation, pedestrian)

        for i in range(future_second):
            cur_time = (i + 1) * 2
            traj_L2 = self.planning_metric.compute_L2(
                pred_ego_fut_trajs[0, :cur_time].detach().to(gt_ego_fut_trajs.device),
                gt_ego_fut_trajs[0, :cur_time],
            )
            traj_L2_stp3 = self.planning_metric.compute_L2_stp3(
                pred_ego_fut_trajs[0, :cur_time].detach().to(gt_ego_fut_trajs.device),
                gt_ego_fut_trajs[0, :cur_time],
            )
            obj_coll, obj_box_coll = self.planning_metric.evaluate_coll(
                pred_ego_fut_trajs[:, :cur_time].detach(),
                gt_ego_fut_trajs[:, :cur_time],
                occupancy,
            )
            metric_dict["plan_L2_{}s".format(i + 1)] = traj_L2
            metric_dict["plan_obj_col_{}s".format(i + 1)] = obj_coll.mean().item()
            metric_dict[
                "plan_obj_box_col_{}s".format(i + 1)
            ] = obj_box_coll.mean().item()

            metric_dict["plan_L2_stp3_{}s".format(i + 1)] = traj_L2_stp3
            metric_dict["plan_obj_col_stp3_{}s".format(i + 1)] = obj_coll[-1].item()
            metric_dict["plan_obj_box_col_stp3_{}s".format(i + 1)] = obj_box_coll[
                -1
            ].item()

        return metric_dict
