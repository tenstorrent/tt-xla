# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from PIL import Image
import cv2
from skimage.draw import polygon
import numpy as np
import torch.nn.functional as F

ego_width, ego_length = 1.85, 4.084
import copy


def multi_scale_deformable_attn_pytorch(
    value, value_spatial_shapes, sampling_locations, attention_weights
):
    """CPU version of multi-scale deformable attention.

    Args:
        value (Tensor): The value has shape
            (bs, num_keys, mum_heads, embed_dims//num_heads)
        value_spatial_shapes (Tensor): Spatial shape of
            each feature map, has shape (num_levels, 2),
            last dimension 2 represent (h, w)
        sampling_locations (Tensor): The location of sampling points,
            has shape
            (bs ,num_queries, num_heads, num_levels, num_points, 2),
            the last dimension 2 represent (x, y).
        attention_weights (Tensor): The weight of sampling points used
            when calculate the attention, has shape
            (bs ,num_queries, num_heads, num_levels, num_points),

    Returns:
        Tensor: has shape (bs, num_queries, embed_dims)
    """

    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        # bs, H_*W_, num_heads, embed_dims ->
        # bs, H_*W_, num_heads*embed_dims ->
        # bs, num_heads*embed_dims, H_*W_ ->
        # bs*num_heads, embed_dims, H_, W_
        value_l_ = (
            value_list[level]
            .flatten(2)
            .transpose(1, 2)
            .reshape(bs * num_heads, embed_dims, H_, W_)
        )
        # bs, num_queries, num_heads, num_points, 2 ->
        # bs, num_heads, num_queries, num_points, 2 ->
        # bs*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        # bs*num_heads, embed_dims, num_queries, num_points
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampling_value_list.append(sampling_value_l_)
    # (bs, num_queries, num_heads, num_levels, num_points) ->
    # (bs, num_heads, num_queries, num_levels, num_points) ->
    # (bs, num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(bs, num_heads * embed_dims, num_queries)
    )
    return output.transpose(1, 2).contiguous()


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_unit, verbose=False):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_unit), nn.LayerNorm(hidden_unit), nn.ReLU()
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class LaneNet(nn.Module):
    def __init__(self, in_channels, hidden_unit, num_subgraph_layers):
        super(LaneNet, self).__init__()
        self.num_subgraph_layers = num_subgraph_layers
        self.layer_seq = nn.Sequential()
        for i in range(num_subgraph_layers):
            self.layer_seq.add_module(f"lmlp_{i}", MLP(in_channels, hidden_unit))
            in_channels = hidden_unit * 2

    def forward(self, pts_lane_feats):
        """
            Extract lane_feature from vectorized lane representation

        Args:
            pts_lane_feats: [batch size, max_pnum, pts, D]

        Returns:
            inst_lane_feats: [batch size, max_pnum, D]
        """
        x = pts_lane_feats
        for name, layer in self.layer_seq.named_modules():
            if isinstance(layer, MLP):
                # x [bs,max_lane_num,9,dim]
                x = layer(x)
                x_max = torch.max(x, -2)[0]
                x_max = x_max.unsqueeze(2).repeat(1, 1, x.shape[2], 1)
                x = torch.cat([x, x_max], dim=-1)
        x_max = torch.max(x, -2)[0]
        return x_max


class LearnedPositionalEncoding(nn.Module):
    """Position embedding with learnable embedding weights.

    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. The final returned dimension for
            each position is 2 times of this value.
        row_num_embed (int, optional): The dictionary size of row embeddings.
            Default 50.
        col_num_embed (int, optional): The dictionary size of col embeddings.
            Default 50.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(
        self,
        num_feats=128,
        row_num_embed=100,
        col_num_embed=100,
        init_cfg=dict(type="Uniform", layer="Embedding"),
    ):
        super().__init__()
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)
        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed

    def forward(self, mask):
        """Forward function for `LearnedPositionalEncoding`.

        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].

        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        h, w = mask.shape[-2:]
        x = torch.arange(w)
        y = torch.arange(h)
        x = x.to(self.col_embed.weight.device)
        y = y.to(self.row_embed.weight.device)
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

    def __repr__(self):
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f"(num_feats={self.num_feats}, "
        repr_str += f"row_num_embed={self.row_num_embed}, "
        repr_str += f"col_num_embed={self.col_num_embed})"
        return repr_str


def bbox_xyxy_to_cxcywh(bbox):
    """Convert bbox coordinates from (x1, y1, x2, y2) to (cx, cy, w, h).

    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    x1, y1, x2, y2 = bbox.split((1, 1, 1, 1), dim=-1)
    bbox_new = [(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)]
    return torch.cat(bbox_new, dim=-1)


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
        """
        Parameters
        ----------
            x_bounds: Forward direction in the ego-car.
            y_bounds: Sides
            z_bounds: Height

        Returns
        -------
            bev_resolution: Bird's-eye view bev_resolution
            bev_start_position Bird's-eye view first element
            bev_dimension Bird's-eye view tensor spatial dimension
        """
        bev_resolution = torch.tensor(
            [row[2] for row in [x_bounds, y_bounds, z_bounds]]
        )
        bev_start_position = torch.tensor(
            [row[0] + row[2] / 2.0 for row in [x_bounds, y_bounds, z_bounds]]
        )
        bev_dimension = torch.tensor(
            [(row[1] - row[0]) / row[2] for row in [x_bounds, y_bounds, z_bounds]]
        )

        return bev_resolution, bev_start_position, bev_dimension

    def get_label(self, gt_agent_boxes, gt_agent_feats):
        segmentation_np, pedestrian_np = self.get_birds_eye_view_label(
            gt_agent_boxes, gt_agent_feats
        )
        segmentation = torch.from_numpy(segmentation_np).long().unsqueeze(0)
        pedestrian = torch.from_numpy(pedestrian_np).long().unsqueeze(0)

        return segmentation, pedestrian

    def get_birds_eye_view_label(self, gt_agent_boxes, gt_agent_feats):
        """
        gt_agent_boxes (LiDARInstance3DBoxes): list of GT Bboxs.
            dim 9 = (x,y,z)+(w,l,h)+yaw+(vx,vy)
        gt_agent_feats: (B, A, 34)
            dim 34 = fut_traj(6*2) + fut_mask(6) + goal(1) + lcf_feat(9) + fut_yaw(6)
            lcf_feat (x, y, yaw, vx, vy, width, length, height, type)
        ego_lcf_feats: (B, 9)
            dim 8 = (vx, vy, ax, ay, w, length, width, vel, steer)
        """
        T = 6
        segmentation = np.zeros(
            (int(T), int(self.bev_dimension[0]), int(self.bev_dimension[1]))
        )
        pedestrian = np.zeros(
            (int(T), int(self.bev_dimension[0]), int(self.bev_dimension[1]))
        )
        agent_num = gt_agent_feats.shape[1]

        gt_agent_boxes = gt_agent_boxes.tensor.numpy()  # (N, 9)
        gt_agent_feats = gt_agent_feats.numpy()

        gt_agent_fut_trajs = gt_agent_feats[..., : T * 2].reshape(-1, 6, 2)
        gt_agent_fut_mask = gt_agent_feats[..., T * 2 : T * 3].reshape(-1, 6)
        # gt_agent_lcf_feat = gt_agent_feats[..., T*3+1:T*3+10].reshape(-1, 9)
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
                    # Filter out all non vehicle instances
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
        # convert to cv frame
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
        """
        traj: torch.Tensor (n_future, 2)
            自车lidar系为轨迹参考系
                ^ y
                |
                |
                0------->
                        x
        segmentation: torch.Tensor (n_future, 200, 200)
        """
        pts = np.array(
            [
                [-self.H / 2.0 + 0.5, self.W / 2.0],
                [self.H / 2.0 + 0.5, self.W / 2.0],
                [self.H / 2.0 + 0.5, -self.W / 2.0],
                [-self.H / 2.0 + 0.5, -self.W / 2.0],
            ]
        )
        pts = (pts - self.bx.numpy()) / (self.dx.numpy())
        pts[:, [0, 1]] = pts[:, [1, 0]]
        rr, cc = polygon(pts[:, 1], pts[:, 0])
        rc = np.concatenate([rr[:, None], cc[:, None]], axis=-1)

        n_future, _ = traj.shape
        trajs = traj.view(n_future, 1, 2)
        # 轨迹坐标系转换为:
        #  ^ x
        #  |
        #  |
        #  0-------> y
        trajs_ = copy.deepcopy(trajs)
        trajs_[:, :, [0, 1]] = trajs_[:, :, [1, 0]]  # can also change original tensor
        trajs_ = trajs_ / self.dx
        trajs_ = trajs_.numpy() + rc  # (n_future, 32, 2)

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
            collision[t] = np.any(segmentation[t, rr[I], cc[I]].numpy())

        return torch.from_numpy(collision)

    def evaluate_coll(self, trajs, gt_trajs, segmentation):
        """
        trajs: torch.Tensor (B, n_future, 2)
            自车lidar系为轨迹参考系
            ^ y
            |
            |
            0------->
                    x
        gt_trajs: torch.Tensor (B, n_future, 2)
        segmentation: torch.Tensor (B, n_future, 200, 200)

        """
        B, n_future, _ = trajs.shape
        # trajs = trajs * torch.tensor([-1, 1], device=trajs.device)
        # gt_trajs = gt_trajs * torch.tensor([-1, 1], device=gt_trajs.device)
        obj_coll_sum = torch.zeros(n_future)
        obj_box_coll_sum = torch.zeros(n_future)

        for i in range(B):
            gt_box_coll = self.evaluate_single_coll(
                gt_trajs[i], segmentation[i], input_gt=True
            )

            xx, yy = trajs[i, :, 0], trajs[i, :, 1]
            # lidar系下的轨迹转换到图片坐标系下
            xi = ((-self.bx[0] / 2 - yy) / self.dx[0]).long()
            yi = ((-self.bx[1] / 2 + xx) / self.dx[1]).long()

            m1 = torch.logical_and(
                torch.logical_and(xi >= 0, xi < self.bev_dimension[0]),
                torch.logical_and(yi >= 0, yi < self.bev_dimension[1]),
            )
            m1 = torch.logical_and(m1, torch.logical_not(gt_box_coll))

            ti = torch.arange(n_future)
            obj_coll_sum[ti[m1]] += segmentation[i, ti[m1], xi[m1], yi[m1]].long()

            m2 = torch.logical_not(gt_box_coll)
            box_coll = self.evaluate_single_coll(
                trajs[i], segmentation[i], input_gt=False
            )
            obj_box_coll_sum[ti[m2]] += (box_coll[ti[m2]]).long()

        return obj_coll_sum, obj_box_coll_sum

    def compute_L2(self, trajs, gt_trajs):
        """
        trajs: torch.Tensor (n_future, 2)
        gt_trajs: torch.Tensor (n_future, 2)
        """
        # return torch.sqrt(((trajs[:, :, :2] - gt_trajs[:, :, :2]) ** 2).sum(dim=-1))
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


def bbox3d2result(bboxes, scores, labels, attrs=None):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (torch.Tensor): Bounding boxes with shape of (n, 5).
        labels (torch.Tensor): Labels with shape of (n, ).
        scores (torch.Tensor): Scores with shape of (n, ).
        attrs (torch.Tensor, optional): Attributes with shape of (n, ). \
            Defaults to None.

    Returns:
        dict[str, torch.Tensor]: Bounding box results in cpu mode.

            - boxes_3d (torch.Tensor): 3D boxes.
            - scores (torch.Tensor): Prediction scores.
            - labels_3d (torch.Tensor): Box labels.
            - attrs_3d (torch.Tensor, optional): Box attributes.
    """
    result_dict = dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)

    if attrs is not None:
        result_dict["attrs_3d"] = attrs

    return result_dict


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

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * epoch / max_epoch  # + 1.#0.5

    # @auto_fp16()
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
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + self.l, hh)
                mask[s:t, :] *= 0
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + self.l, ww)
                mask[:, s:t] *= 0

        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[
            (hh - h) // 2 : (hh - h) // 2 + h, (ww - w) // 2 : (ww - w) // 2 + w
        ]

        mask = torch.from_numpy(mask)
        if self.mode == 1:
            mask = 1 - mask
        mask = mask.expand_as(x)
        if self.offset:
            offset = torch.from_numpy(2 * (np.random.rand(h, w) - 0.5))
            x = x * mask + offset * (1 - mask)
        else:
            x = x * mask

        return x.view(n, c, h, w)


class ConvModule(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding=0):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channel, out_channel, kernel_size, stride=1, padding=padding
        )

    def forward(self, x):
        return self.conv(x)


class FPN(nn.Module):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(mode='nearest')`
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(
        self,
        in_channels=[2048],
        out_channels=256,
        num_outs=1,
        start_level=0,
        end_level=-1,
        add_extra_convs="on_output",
        relu_before_extra_convs=True,
        no_norm_on_lateral=False,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=None,
        upsample_cfg=dict(mode="nearest"),
        init_cfg=dict(type="Xavier", layer="Conv2d", distribution="uniform"),
    ):
        super().__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        self.backbone_end_level = self.num_ins
        assert num_outs >= self.num_ins - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        assert add_extra_convs in ("on_input", "on_lateral", "on_output")
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(in_channels[i], out_channels, 1)
            fpn_conv = ConvModule(out_channels, out_channels, 3, padding=1)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == "on_input":
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
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
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            if "scale_factor" in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg
                )

        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]
        return tuple(outs)


class ResLayer(nn.Sequential):
    """ResLayer to build ResNet style backbone.

    Args:
        block (nn.Module): block used to build ResLayer.
        inplanes (int): inplanes of block.
        planes (int): planes of block.
        num_blocks (int): number of blocks.
        stride (int): stride of the first block. Default: 1
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        downsample_first (bool): Downsample at the first block or last block.
            False for Hourglass, True for ResNet. Default: True
    """

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
            if avg_down:
                conv_stride = 1
                downsample.append(
                    nn.AvgPool2d(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False,
                    )
                )
            downsample.extend(
                [
                    torch.nn.Conv2d(
                        inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=conv_stride,
                        bias=False,
                    ),
                    torch.nn.BatchNorm2d(planes * block.expansion),
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

        else:  # downsample_first=False is for HourglassModule
            for _ in range(num_blocks - 1):
                layers.append(
                    block(
                        inplanes=inplanes,
                        planes=inplanes,
                        stride=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        **kwargs,
                    )
                )
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
        super(ResLayer, self).__init__(*layers)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes=1,
        planes=1,
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
        """Bottleneck block for ResNet.

        If style is "pytorch", the stride-two layer is the 3x3 conv layer, if
        it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        # super(Bottleneck, self).__init__(init_cfg)
        super().__init__()
        assert style in ["pytorch", "caffe"]
        assert dcn is None or isinstance(dcn, dict)
        assert plugins is None or isinstance(plugins, list)
        # print("self.inplanes",inplanes)
        # print("self.planes",planes)

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
        self.conv1_stride = 1
        self.conv2_stride = stride
        self.norm1_name, norm1 = "bn1", torch.nn.BatchNorm2d(
            planes, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.norm2_name, norm2 = "bn2", torch.nn.BatchNorm2d(
            planes, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.norm3_name, norm3 = "bn3", torch.nn.BatchNorm2d(
            planes * self.expansion,
            eps=1e-05,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
        )

        self.conv1 = torch.nn.Conv2d(
            inplanes, planes, kernel_size=1, stride=self.conv1_stride, bias=False
        )

        self.add_module(self.norm1_name, norm1)
        fallback_on_stride = False
        if self.with_dcn:
            fallback_on_stride = dcn.pop("fallback_on_stride", False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = torch.nn.Conv2d(
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False,
            )

        self.add_module(self.norm2_name, norm2)
        self.conv3 = torch.nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        """nn.Module: normalization layer after the third convolution layer"""
        return getattr(self, self.norm3_name)

    def forward(self, x):
        """Forward function."""

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


class ResNet(nn.Module):
    """ResNet backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        stem_channels (int | None): Number of stem channels. If not specified,
            it will be the same as `base_channels`. Default: None.
        base_channels (int): Number of base channels of res layer. Default: 64.
        in_channels (int): Number of input image channels. Default: 3.
        num_stages (int): Resnet stages. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        norm_cfg (dict): Dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        plugins (list[dict]): List of plugins for stages, each dict contains:

            - cfg (dict, required): Cfg dict to build plugin.
            - position (str, required): Position inside block to insert
              plugin, options are 'after_conv1', 'after_conv2', 'after_conv3'.
            - stages (tuple[bool], optional): Stages to apply plugin, length
              should be same as 'num_stages'.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity.
        pretrained (str, optional): model pretrained path. Default: None
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None

    Example:
        >>> from mmdet.models import ResNet
        >>> import torch
        >>> self = ResNet(depth=18)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 64, 8, 8)
        (1, 128, 4, 4)
        (1, 256, 2, 2)
        (1, 512, 1, 1)
    """

    arch_settings = {
        50: (Bottleneck, (3, 4, 6, 3)),
    }

    def __init__(
        self,
        depth=50,
        in_channels=3,
        stem_channels=None,
        base_channels=64,
        num_stages=4,
        strides=(1, 2, 2, 2),
        dilations=(1, 1, 1, 1),
        out_indices=(3,),
        style="pytorch",
        deep_stem=False,
        avg_down=False,
        frozen_stages=1,
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
        super().__init__()
        self.zero_init_residual = zero_init_residual
        if depth not in self.arch_settings:
            raise KeyError(f"invalid depth {depth} for resnet")

        block_init_cfg = None
        assert not (
            init_cfg and pretrained
        ), "init_cfg and pretrained cannot be setting at the same time"

        self.depth = depth
        if stem_channels is None:
            stem_channels = base_channels
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
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
        if dcn is not None:
            assert len(stage_with_dcn) == num_stages
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
        """Pack all blocks in a stage into a ``ResLayer``."""
        return ResLayer(**kwargs)

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, stem_channels):
        self.conv1 = torch.nn.Conv2d(
            3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.norm1_name, norm1 = "bn1", torch.nn.BatchNorm2d(
            64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
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
        """Forward function."""
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
