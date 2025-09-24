# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch


def bbox_cxcywh_to_xyxy(bbox):
    """Convert bbox coordinates from (cx, cy, w, h) to (x1, y1, x2, y2).

    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    cx, cy, w, h = bbox.split((1, 1, 1, 1), dim=-1)
    bbox_new = [(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)]
    return torch.cat(bbox_new, dim=-1)


def denormalize_2d_pts(pts, pc_range):
    new_pts = pts.clone()
    new_pts[..., 0:1] = pts[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
    new_pts[..., 1:2] = pts[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
    return new_pts


def denormalize_2d_bbox(bboxes, pc_range):

    bboxes = bbox_cxcywh_to_xyxy(bboxes)
    bboxes[..., 0::2] = bboxes[..., 0::2] * (pc_range[3] - pc_range[0]) + pc_range[0]
    bboxes[..., 1::2] = bboxes[..., 1::2] * (pc_range[4] - pc_range[1]) + pc_range[1]

    return bboxes


class CustomNMSFreeCoder:
    """Bbox coder for NMS-free detector (for 3D objects).
    Args:
        pc_range (list[float]): Range of point cloud, length 6 [x_min, y_min, z_min, x_max, y_max, z_max].
        post_center_range (list[float]): Limit of the center, length 6. Default: None.
        max_num (int): Max number to be kept. Default: 100.
        score_threshold (float): Threshold to filter boxes based on score. Default: None.
        num_classes (int): Number of classes. Default: 10
    """

    def __init__(
        self, pc_range, post_center_range, max_num, score_threshold, num_classes
    ):
        self.pc_range = pc_range
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes

    def encode(self):
        pass

    def decode_single(self, cls_scores, bbox_preds, traj_preds):
        """Decode bboxes.
        Args:
            cls_scores (Tensor): Shape [num_query, cls_out_channels].
            bbox_preds (Tensor): Shape [num_query, 9 or 10].
            traj_preds (Tensor): Shape [num_query, ...].
        Returns:
            dict: Decoded boxes.
        """
        num_query = bbox_preds.size(0)
        max_num = min(self.max_num, num_query)

        cls_scores = cls_scores.sigmoid()
        scores, indexs = cls_scores.view(-1).topk(max_num)

        labels = indexs % self.num_classes
        bbox_index = indexs // self.num_classes
        bbox_index = bbox_index.clamp(max=num_query - 1)

        bbox_preds = bbox_preds[bbox_index]
        traj_preds = traj_preds[bbox_index]

        final_box_preds = denormalize_bbox(bbox_preds, self.pc_range)

        final_scores = scores
        final_preds = labels
        final_traj_preds = traj_preds

        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold
            tmp_score = self.score_threshold
            while thresh_mask.sum() == 0:
                tmp_score *= 0.9
                if tmp_score < 0.01:
                    thresh_mask = final_scores > -1
                    break
                thresh_mask = final_scores >= tmp_score

        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(self.post_center_range)
            mask = (final_box_preds[..., :3] >= self.post_center_range[:3]).all(1)
            mask &= (final_box_preds[..., :3] <= self.post_center_range[3:]).all(1)

            if self.score_threshold:
                mask &= thresh_mask

            boxes3d = final_box_preds[mask]
            scores = final_scores[mask]
            labels = final_preds[mask]
            trajs = final_traj_preds[mask]

            predictions_dict = {
                "bboxes": boxes3d,
                "scores": scores,
                "labels": labels,
                "trajs": trajs,
            }
        else:
            raise NotImplementedError(
                "Need to reorganize output as a batch, only "
                "support post_center_range is not None for now!"
            )
        return predictions_dict

    def decode(self, preds_dicts):
        """Decode bboxes.
        Args:
            preds_dicts (dict): Contains 'all_cls_scores', 'all_bbox_preds', 'all_traj_preds'.
        Returns:
            list[dict]: Decoded boxes.
        """

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


def denormalize_bbox(normalized_bboxes, pc_range):
    # rotation
    rot_sine = normalized_bboxes[..., 6:7]

    rot_cosine = normalized_bboxes[..., 7:8]
    rot = torch.atan2(rot_sine, rot_cosine)

    # center in the bev
    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 4:5]

    # size
    w = normalized_bboxes[..., 2:3]
    l = normalized_bboxes[..., 3:4]
    h = normalized_bboxes[..., 5:6]

    w = w.exp()
    l = l.exp()
    h = h.exp()
    if normalized_bboxes.size(-1) > 8:
        # velocity
        vx = normalized_bboxes[:, 8:9]
        vy = normalized_bboxes[:, 9:10]
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot, vx, vy], dim=-1)
    else:
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot], dim=-1)
    return denormalized_bboxes


class MapNMSFreeCoder:
    """Bbox coder for NMS-free detector.
    Args:
        pc_range (list[float]): Range of point cloud, length 6 [x_min, y_min, z_min, x_max, y_max, z_max].
        post_center_range (list[float]): Limit of the center, length 8 [x1_min, y1_min, x2_min, y2_min, x1_max, y1_max, x2_max, y2_max].
            Default: None.
        max_num (int): Max number to be kept. Default: 100.
        score_threshold (float): Threshold to filter boxes based on score. Default: None.
        num_classes (int): Number of classes. Default: 10
    """

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
        """Decode bboxes.
        Args:
            cls_scores (Tensor): Outputs from the classification head, \
                shape [num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            bbox_preds (Tensor): Outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l). \
                Shape [num_query, 4].
            pts_preds (Tensor):
                Shape [num_query, fixed_num_pts, 2]
        Returns:
            dict: Decoded boxes.
        """
        max_num = self.max_num

        cls_scores = cls_scores.sigmoid()
        scores, indexs = cls_scores.view(-1).topk(max_num)
        labels = indexs % self.num_classes
        bbox_index = indexs // self.num_classes
        bbox_preds = bbox_preds[bbox_index]
        pts_preds = pts_preds[bbox_index]

        final_box_preds = denormalize_2d_bbox(bbox_preds, self.pc_range)
        final_pts_preds = denormalize_2d_pts(pts_preds, self.pc_range)  # num_q,num_p,2
        final_scores = scores
        final_preds = labels

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold
            tmp_score = self.score_threshold
            while thresh_mask.sum() == 0:
                tmp_score *= 0.9
                if tmp_score < 0.01:
                    thresh_mask = final_scores > -1
                    break
                thresh_mask = final_scores >= tmp_score

        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(self.post_center_range)
            mask = (final_box_preds[..., :4] >= self.post_center_range[:4]).all(1)
            mask &= (final_box_preds[..., :4] <= self.post_center_range[4:]).all(1)

            if self.score_threshold:
                mask &= thresh_mask

            boxes3d = final_box_preds[mask]
            scores = final_scores[mask]
            pts = final_pts_preds[mask]
            labels = final_preds[mask]
            predictions_dict = {
                "map_bboxes": boxes3d,
                "map_scores": scores,
                "map_labels": labels,
                "map_pts": pts,
            }

        else:
            raise NotImplementedError(
                "Need to reorganize output as a batch, only "
                "support post_center_range is not None for now!"
            )
        return predictions_dict

    def decode(self, preds_dicts):
        """Decode bboxes.
        Args:
            preds_dicts (dict): Contains 'map_all_cls_scores', 'map_all_bbox_preds', 'map_all_pts_preds'.
        Returns:
            list[dict]: Decoded boxes.
        """

        all_cls_scores = preds_dicts["map_all_cls_scores"][-1]
        all_bbox_preds = preds_dicts["map_all_bbox_preds"][-1]
        all_pts_preds = preds_dicts["map_all_pts_preds"][-1]
        batch_size = all_cls_scores.size()[0]
        predictions_list = []
        for i in range(batch_size):
            predictions_list.append(
                self.decode_single(
                    all_cls_scores[i], all_bbox_preds[i], all_pts_preds[i]
                )
            )
        return predictions_list
