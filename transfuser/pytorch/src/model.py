# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from third_party.tt_forge_models.transfuser.pytorch.src.transfuser import (
    TransfuserBackbone,
)
from six.moves import zip
import torch.nn.functional as F
import math


def get_lidar_to_bevimage_transform():
    T = torch.tensor([[0, -1, 16], [-1, 0, 32], [0, 0, 1]], dtype=torch.float32)
    T[:2, :] *= 8

    return T


def gather_feat(feat, ind, mask=None):
    """Gather feature according to index.

    Args:
        feat (Tensor): Target feature map.
        ind (Tensor): Target coord index.
        mask (Tensor | None): Mask of feature map. Default: None.

    Returns:
        feat (Tensor): Gathered feature.
    """
    dim = feat.size(2)
    ind = ind.unsqueeze(2).repeat(1, 1, dim)
    feat = feat.gather(1, ind)
    return feat


def transpose_and_gather_feat(feat, ind):
    """Transpose and gather feature according to index.

    Args:
        feat (Tensor): Target feature map.
        ind (Tensor): Target coord index.

    Returns:
        feat (Tensor): Transposed and gathered feature.
    """
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = gather_feat(feat, ind)
    return feat


def get_local_maximum(heat, kernel=3):
    """Extract local maximum pixel with given kernel.

    Args:
        heat (Tensor): Target heatmap.
        kernel (int): Kernel size of max pooling. Default: 3.

    Returns:
        heat (Tensor): A heatmap where local maximum pixels maintain its
            own value and other positions are 0.
    """
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heat, kernel, stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def get_topk_from_heatmap(scores, k=20):
    """Get top k positions from heatmap.

    Args:
        scores (Tensor): Target heatmap with shape
            [batch, num_classes, height, width].
        k (int): Target number. Default: 20.

    Returns:
        tuple[torch.Tensor]: Scores, indexes, categories and coords of
            topk keypoint. Containing following Tensors:

        - topk_scores (Tensor): Max scores of each topk keypoint.
        - topk_inds (Tensor): Indexes of each topk keypoint.
        - topk_clses (Tensor): Categories of each topk keypoint.
        - topk_ys (Tensor): Y-coord of each topk keypoint.
        - topk_xs (Tensor): X-coord of each topk keypoint.
    """
    batch, _, height, width = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), k)
    topk_clses = topk_inds // (height * width)
    topk_inds = topk_inds % (height * width)
    topk_ys = topk_inds // width
    topk_xs = (topk_inds % width).int().float()

    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs


class LidarCenterNetHead(nn.Module):
    """Objects as Points Head. CenterHead use center_point to indicate object's
    position. Paper link <https://arxiv.org/abs/1904.07850>

    Args:
        in_channel (int): Number of channel in the input feature map.
        feat_channel (int): Number of channel in the intermediate feature map.
        num_classes (int): Number of categories excluding the background
            category.
    """

    def __init__(self, in_channel=64, feat_channel=64, num_classes=1):
        super().__init__()
        self.num_classes = num_classes
        self.heatmap_head = self._build_head(in_channel, feat_channel, num_classes)
        self.wh_head = self._build_head(in_channel, feat_channel, 2)
        self.offset_head = self._build_head(in_channel, feat_channel, 2)
        self.num_dir_bins = 12
        self.yaw_class_head = self._build_head(
            in_channel, feat_channel, self.num_dir_bins
        )
        self.yaw_res_head = self._build_head(in_channel, feat_channel, 1)
        self.velocity_head = self._build_head(in_channel, feat_channel, 1)
        self.brake_head = self._build_head(in_channel, feat_channel, 2)

    def _build_head(self, in_channel, feat_channel, out_channel):
        """Build head for each branch."""
        layer = nn.Sequential(
            nn.Conv2d(in_channel, feat_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channel, out_channel, kernel_size=1),
        )
        return layer

    def forward(self, feats):
        """Forward features. Notice CenterNet head does not use FPN.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            center_heatmap_preds (List[Tensor]): center predict heatmaps for
                all levels, the channels number is num_classes.
            wh_preds (List[Tensor]): wh predicts for all levels, the channels
                number is 2.
            offset_preds (List[Tensor]): offset predicts for all levels, the
               channels number is 2.
        """
        return self.forward_single(feats)

    def forward_single(self, feat):
        """Forward feature of a single level.

        Args:
            feat (Tensor): Feature of a single level.

        Returns:
            center_heatmap_pred (Tensor): center predict heatmaps, the
               channels number is num_classes.
            wh_pred (Tensor): wh predicts, the channels number is 2.
            offset_pred (Tensor): offset predicts, the channels number is 2.
        """
        center_heatmap_pred = self.heatmap_head(feat).sigmoid()
        wh_pred = self.wh_head(feat)
        offset_pred = self.offset_head(feat)
        yaw_class_pred = self.yaw_class_head(feat)
        yaw_res_pred = self.yaw_res_head(feat)
        velocity_pred = self.velocity_head(feat)
        brake_pred = self.brake_head(feat)
        return (
            center_heatmap_pred,
            wh_pred,
            offset_pred,
            yaw_class_pred,
            yaw_res_pred,
            velocity_pred,
            brake_pred,
        )

    def class2angle(self, angle_cls, angle_res, limit_period=True):
        """Inverse function to angle2class.
        Args:
            angle_cls (torch.Tensor): Angle class to decode.
            angle_res (torch.Tensor): Angle residual to decode.
            limit_period (bool): Whether to limit angle to [-pi, pi].
        Returns:
            torch.Tensor: Angle decoded from angle_cls and angle_res.
        """
        angle_per_class = 2 * math.pi / float(self.num_dir_bins)
        angle_center = angle_cls.float() * angle_per_class
        angle = angle_center + angle_res
        if limit_period:
            angle = torch.where(angle > math.pi, angle - 2 * math.pi, angle)
        return angle

    def get_bboxes(
        self,
        center_heatmap_preds,
        wh_preds,
        offset_preds,
        yaw_class_preds,
        yaw_res_preds,
        velocity_preds,
        brake_preds,
    ):
        """Transform network output for a batch into bbox predictions.

        Args:
            center_heatmap_preds (list[Tensor]): center predict heatmaps for
                all levels with shape (B, num_classes, H, W).
            wh_preds (list[Tensor]): wh predicts for all levels with
                shape (B, 2, H, W).
            offset_preds (list[Tensor]): offset predicts for all levels
                with shape (B, 2, H, W).

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where 5 represent
                (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.
        """
        assert len(center_heatmap_preds) == len(wh_preds) == len(offset_preds) == 1

        batch_det_bboxes, batch_labels = self.decode_heatmap(
            center_heatmap_preds,
            wh_preds,
            offset_preds,
            yaw_class_preds,
            yaw_res_preds,
            velocity_preds,
            brake_preds,
            k=100,
            kernel=3,
        )

        det_results = [tuple(bs) for bs in zip(batch_det_bboxes, batch_labels)]
        return det_results

    def decode_heatmap(
        self,
        center_heatmap_pred,
        wh_pred,
        offset_pred,
        yaw_class_pred,
        yaw_res_pred,
        velocity_pred,
        brake_pred,
        k=100,
        kernel=3,
    ):
        """Transform outputs into detections raw bbox prediction.

        Args:
            center_heatmap_pred (Tensor): center predict heatmap,
               shape (B, num_classes, H, W).
            wh_pred (Tensor): wh predict, shape (B, 2, H, W).
            offset_pred (Tensor): offset predict, shape (B, 2, H, W).
            k (int): Get top k center keypoints from heatmap. Default 100.
            kernel (int): Max pooling kernel for extract local maximum pixels.
               Default 3.

        Returns:
            tuple[torch.Tensor]: Decoded output of CenterNetHead, containing
               the following Tensors:

              - batch_bboxes (Tensor): Coords of each box with shape (B, k, 5)
              - batch_topk_labels (Tensor): Categories of each box with \
                  shape (B, k)
        """

        center_heatmap_pred = get_local_maximum(center_heatmap_pred, kernel=3)
        *batch_dets, topk_ys, topk_xs = get_topk_from_heatmap(
            center_heatmap_pred, k=100
        )
        batch_scores, batch_index, batch_topk_labels = batch_dets

        wh = transpose_and_gather_feat(wh_pred, batch_index)
        offset = transpose_and_gather_feat(offset_pred, batch_index)
        yaw_class = transpose_and_gather_feat(yaw_class_pred, batch_index)
        yaw_res = transpose_and_gather_feat(yaw_res_pred, batch_index)
        velocity = transpose_and_gather_feat(velocity_pred, batch_index)
        brake = transpose_and_gather_feat(brake_pred, batch_index)
        brake = torch.argmax(brake, -1)
        velocity = velocity[..., 0]

        yaw_class = torch.argmax(yaw_class, -1)
        yaw = self.class2angle(yaw_class, yaw_res.squeeze(2))

        topk_xs = topk_xs + offset[..., 0]
        topk_ys = topk_ys + offset[..., 1]

        ratio = 4.0

        batch_bboxes = torch.stack(
            [topk_xs, topk_ys, wh[..., 0], wh[..., 1], yaw, velocity, brake], dim=2
        )
        batch_bboxes = torch.cat((batch_bboxes, batch_scores[..., None]), dim=-1)
        batch_bboxes[:, :, :4] *= ratio

        return batch_bboxes, batch_topk_labels


class LidarCenterNet(nn.Module):
    """
    Encoder network for LiDAR input list
    Args:
        in_channels: input channels
    """

    def __init__(
        self,
        image_architecture="resnet34",
        lidar_architecture="resnet18",
        use_velocity=True,
    ):
        super().__init__()
        self.pred_len = 4
        self.use_target_point_image = True
        self.gru_concat_target_point = True
        self._model = TransfuserBackbone()
        channel = 64
        self.head = LidarCenterNetHead()
        self.join = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )

        gru_concat_target_point = True
        gru_hidden_size = 64
        self.decoder = nn.GRUCell(
            input_size=4 if gru_concat_target_point else 2, hidden_size=gru_hidden_size
        )
        self.output = nn.Linear(gru_hidden_size, 3)

    def forward_gru(self, z, target_point):
        z = self.join(z)
        output_wp = list()
        x = torch.zeros(size=(z.shape[0], 2), dtype=z.dtype).to(z.device)
        target_point = target_point.clone()
        target_point[:, 1] *= -1

        for _ in range(self.pred_len):
            if self.gru_concat_target_point:
                x_in = torch.cat([x, target_point], dim=1)

            z = self.decoder(x_in, z)
            dx = self.output(z)
            x = dx[:, :2] + x
            output_wp.append(x[:, :2])

        pred_wp = torch.stack(output_wp, dim=1)
        pred_wp[:, :, 0] = pred_wp[:, :, 0] - 1.3
        pred_brake = None
        steer = None
        throttle = None
        brake = None

        return pred_wp, pred_brake, steer, throttle, brake

    def forward(
        self,
        rgb=None,
        lidar_bev=None,
        target_point=None,
        target_point_image=None,
        ego_vel=None,
    ):
        if self.use_target_point_image:
            lidar_bev = torch.cat((lidar_bev, target_point_image), dim=1)
        features, image_features_grid, fused_features = self._model(
            rgb, lidar_bev, ego_vel
        )

        pred_wp, _, _, _, _ = self.forward_gru(fused_features, target_point)

        preds = self.head(features[0])
        results = self.head.get_bboxes(
            preds[0], preds[1], preds[2], preds[3], preds[4], preds[5], preds[6]
        )
        bboxes, _ = results[0]
        rotated_bboxes = []
        for bbox in bboxes.detach().cpu():
            bbox = self.get_bbox_local_metric(bbox)
            rotated_bboxes.append(bbox)
        return pred_wp, rotated_bboxes

    def get_bbox_local_metric(self, bbox):
        x, y, w, h, yaw, speed, brake, confidence = bbox
        w = w / 2 / 8
        h = h / 2 / 8

        T = get_lidar_to_bevimage_transform()
        T_inv = torch.linalg.inv(T)
        center = torch.stack([x, y, torch.tensor(1.0, dtype=torch.float32)])
        center_old_coordinate_sys = T_inv @ center
        center_old_coordinate_sys = center_old_coordinate_sys + torch.stack(
            [
                torch.tensor(1.3, dtype=torch.float32),
                torch.tensor(0.0, dtype=torch.float32),
                torch.tensor(2.5, dtype=torch.float32),
            ]
        )
        center_old_coordinate_sys[1] = -center_old_coordinate_sys[1]

        bbox = torch.stack(
            [
                torch.stack([-h, -w, torch.tensor(1.0, dtype=torch.float32)]),
                torch.stack([-h, w, torch.tensor(1.0, dtype=torch.float32)]),
                torch.stack([h, w, torch.tensor(1.0, dtype=torch.float32)]),
                torch.stack([h, -w, torch.tensor(1.0, dtype=torch.float32)]),
                torch.stack(
                    [
                        torch.tensor(0.0, dtype=torch.float32),
                        torch.tensor(0.0, dtype=torch.float32),
                        torch.tensor(1.0, dtype=torch.float32),
                    ]
                ),
                torch.stack(
                    [
                        torch.tensor(0.0, dtype=torch.float32),
                        h * speed * 0.5,
                        torch.tensor(1.0, dtype=torch.float32),
                    ]
                ),
            ],
            dim=0,
        )

        R = torch.stack(
            [
                torch.stack(
                    [
                        torch.cos(yaw),
                        -torch.sin(yaw),
                        torch.tensor(0.0, dtype=torch.float32),
                    ]
                ),
                torch.stack(
                    [
                        torch.sin(yaw),
                        torch.cos(yaw),
                        torch.tensor(0.0, dtype=torch.float32),
                    ]
                ),
                torch.stack(
                    [
                        torch.tensor(0.0, dtype=torch.float32),
                        torch.tensor(0.0, dtype=torch.float32),
                        torch.tensor(1.0, dtype=torch.float32),
                    ]
                ),
            ],
            dim=0,
        )

        for point_index in range(bbox.shape[0]):
            bbox[point_index] = R @ bbox[point_index]
            bbox[point_index] = bbox[point_index] + torch.stack(
                [
                    center_old_coordinate_sys[0],
                    center_old_coordinate_sys[1],
                    torch.tensor(0.0, dtype=torch.float32),
                ]
            )

        return bbox, brake, confidence
