# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch.nn as nn
import torch
import copy
import pickle
import math
from einops import rearrange
import warnings
import torch.nn.functional as F
from third_party.tt_forge_models.uniad.pytorch.src.transformer import (
    FFN,
    multi_scale_deformable_attn_pytorch,
)


def bivariate_gaussian_activation(ip):
    """
    Activation function to output parameters of bivariate Gaussian distribution.

    Args:
        ip (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Output tensor containing the parameters of the bivariate Gaussian distribution.
    """
    mu_x = ip[..., 0:1]
    mu_y = ip[..., 1:2]
    sig_x = ip[..., 2:3]
    sig_y = ip[..., 3:4]
    rho = ip[..., 4:5]
    sig_x = torch.exp(sig_x)
    sig_y = torch.exp(sig_y)
    rho = torch.tanh(rho)
    out = torch.cat([mu_x, mu_y, sig_x, sig_y, rho], dim=-1)
    return out


def norm_points(pos, pc_range):
    """
    Normalize the end points of a given position tensor.

    Args:
        pos (torch.Tensor): Input position tensor.
        pc_range (List[float]): Point cloud range.

    Returns:
        torch.Tensor: Normalized end points tensor.
    """
    x_norm = (pos[..., 0] - pc_range[0]) / (pc_range[3] - pc_range[0])
    y_norm = (pos[..., 1] - pc_range[1]) / (pc_range[4] - pc_range[1])
    return torch.stack([x_norm, y_norm], dim=-1)


def pos2posemb2d(pos, num_pos_feats=128, temperature=10000):
    """
    Convert 2D position into positional embeddings.

    Args:
        pos (torch.Tensor): Input 2D position tensor.
        num_pos_feats (int, optional): Number of positional features. Default is 128.
        temperature (int, optional): Temperature factor for positional embeddings. Default is 10000.

    Returns:
        torch.Tensor: Positional embeddings tensor.
    """
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_x = torch.stack(
        (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1
    ).flatten(-2)
    pos_y = torch.stack(
        (pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1
    ).flatten(-2)
    posemb = torch.cat((pos_y, pos_x), dim=-1)
    return posemb


def rot_2d(yaw):
    """
    Compute 2D rotation matrix for a given yaw angle tensor.

    Args:
        yaw (torch.Tensor): Input yaw angle tensor.

    Returns:
        torch.Tensor: 2D rotation matrix tensor.
    """
    sy, cy = torch.sin(yaw), torch.cos(yaw)
    out = torch.stack([torch.stack([cy, -sy]), torch.stack([sy, cy])]).permute(
        [2, 0, 1]
    )
    return out


def anchor_coordinate_transform(
    anchors, bbox_results, with_translation_transform=True, with_rotation_transform=True
):
    """
    Transform anchor coordinates with respect to detected bounding boxes in the batch.

    Args:
        anchors (torch.Tensor): A tensor containing the k-means anchor values.
        bbox_results (List[Tuple[torch.Tensor]]): A list of tuples containing the bounding box results for each image in the batch.
        with_translate (bool, optional): Whether to perform translation transformation. Defaults to True.
        with_rot (bool, optional): Whether to perform rotation transformation. Defaults to True.

    Returns:
        torch.Tensor: A tensor containing the transformed anchor coordinates.
    """
    batch_size = len(bbox_results)
    batched_anchors = []
    transformed_anchors = anchors[None, ...]
    for i in range(batch_size):
        bboxes, scores, labels, bbox_index, mask = bbox_results[i]
        yaw = bboxes.yaw.to(transformed_anchors)
        bbox_centers = bboxes.gravity_center.to(transformed_anchors)
        if with_rotation_transform:
            angle = yaw - 3.1415953
            rot_yaw = rot_2d(angle)
            rot_yaw = rot_yaw[:, None, None, :, :]
            transformed_anchors = rearrange(
                transformed_anchors, "b g m t c -> b g m c t"
            )
            transformed_anchors = torch.matmul(rot_yaw, transformed_anchors)
            transformed_anchors = rearrange(
                transformed_anchors, "b g m c t -> b g m t c"
            )
        if with_translation_transform:
            transformed_anchors = (
                bbox_centers[:, None, None, None, :2] + transformed_anchors
            )
        batched_anchors.append(transformed_anchors)
    return torch.stack(batched_anchors)


def trajectory_coordinate_transform(
    trajectory,
    bbox_results,
    with_translation_transform=True,
    with_rotation_transform=True,
):
    """
    Transform trajectory coordinates with respect to detected bounding boxes in the batch.
    Args:
        trajectory (torch.Tensor): predicted trajectory.
        bbox_results (List[Tuple[torch.Tensor]]): A list of tuples containing the bounding box results for each image in the batch.
        with_translate (bool, optional): Whether to perform translation transformation. Defaults to True.
        with_rot (bool, optional): Whether to perform rotation transformation. Defaults to True.

    Returns:
        torch.Tensor: A tensor containing the transformed trajectory coordinates.
    """
    batch_size = len(bbox_results)
    batched_trajectories = []
    for i in range(batch_size):
        bboxes, scores, labels, bbox_index, mask = bbox_results[i]
        yaw = bboxes.yaw.to(trajectory)
        bbox_centers = bboxes.gravity_center.to(trajectory)
        transformed_trajectory = trajectory[i, ...]
        if with_rotation_transform:

            angle = -(yaw - 3.1415953)
            rot_yaw = rot_2d(angle)
            rot_yaw = rot_yaw[:, None, None, :, :]
            transformed_trajectory = rearrange(
                transformed_trajectory, "a g p t c -> a g p c t"
            )
            transformed_trajectory = torch.matmul(rot_yaw, transformed_trajectory)
            transformed_trajectory = rearrange(
                transformed_trajectory, "a g p c t -> a g p t c"
            )
        if with_translation_transform:
            transformed_trajectory = (
                bbox_centers[:, None, None, None, :2] + transformed_trajectory
            )
        batched_trajectories.append(transformed_trajectory)
    return torch.stack(batched_trajectories)


class BaseMotionHead(nn.Module):
    def __init__(self, *args, **kwargs):
        super(BaseMotionHead, self).__init__()
        pass

    def _load_anchors(self, anchor_info_path):
        """
        Load the anchor information from a file.

        Args:
            anchor_info_path (str): The path to the file containing the anchor information.

        Returns:
            None
        """
        anchor_infos = pickle.load(open(anchor_info_path, "rb"))
        self.kmeans_anchors = torch.stack(
            [torch.from_numpy(a) for a in anchor_infos["anchors_all"]]
        )

    def _build_layers(self, transformerlayers, det_layer_num):
        """
        Build the layers of the motion prediction module.

        Args:
            transformerlayers (dict): A dictionary containing the parameters for the transformer layers.
            det_layer_num (int): The number of detection layers.

        Returns:
            None
        """
        self.learnable_motion_query_embedding = nn.Embedding(
            self.num_anchor * self.num_anchor_group, self.embed_dims
        )
        self.motionformer = MotionTransformerDecoder()

        self.layer_track_query_fuser = nn.Sequential(
            nn.Linear(self.embed_dims * det_layer_num, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )

        self.agent_level_embedding_layer = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims * 2),
            nn.ReLU(),
            nn.Linear(self.embed_dims * 2, self.embed_dims),
        )
        self.scene_level_ego_embedding_layer = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims * 2),
            nn.ReLU(),
            nn.Linear(self.embed_dims * 2, self.embed_dims),
        )
        self.scene_level_offset_embedding_layer = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims * 2),
            nn.ReLU(),
            nn.Linear(self.embed_dims * 2, self.embed_dims),
        )
        self.boxes_query_embedding_layer = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims * 2),
            nn.ReLU(),
            nn.Linear(self.embed_dims * 2, self.embed_dims),
        )

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        traj_cls_branch = []
        traj_cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
        traj_cls_branch.append(nn.LayerNorm(self.embed_dims))
        traj_cls_branch.append(nn.ReLU(inplace=True))
        for _ in range(self.num_reg_fcs - 1):
            traj_cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            traj_cls_branch.append(nn.LayerNorm(self.embed_dims))
            traj_cls_branch.append(nn.ReLU(inplace=True))
        traj_cls_branch.append(nn.Linear(self.embed_dims, 1))
        traj_cls_branch = nn.Sequential(*traj_cls_branch)

        traj_reg_branch = []
        traj_reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
        traj_reg_branch.append(nn.ReLU())
        for _ in range(self.num_reg_fcs - 1):
            traj_reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            traj_reg_branch.append(nn.ReLU())
        traj_reg_branch.append(nn.Linear(self.embed_dims, self.predict_steps * 5))
        traj_reg_branch = nn.Sequential(*traj_reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        num_pred = self.motionformer.num_layers
        self.traj_cls_branches = _get_clones(traj_cls_branch, num_pred)
        self.traj_reg_branches = _get_clones(traj_reg_branch, num_pred)

    def _extract_tracking_centers(self, bbox_results, bev_range):
        """
        extract the bboxes centers and normized according to the bev range

        Args:
            bbox_results (List[Tuple[torch.Tensor]]): A list of tuples containing the bounding box results for each image in the batch.
            bev_range (List[float]): A list of float values representing the bird's eye view range.

        Returns:
            torch.Tensor: A tensor representing normized centers of the detection bounding boxes.
        """
        boxes, scores, labels, bbox_index, mask = bbox_results[0]
        batch_size = len(bbox_results)
        det_bbox_posembed = []
        for i in range(batch_size):
            bboxes, scores, labels, bbox_index, mask = bbox_results[i]
            if bboxes is None:
                return torch.empty(0, 2), torch.empty(0, dtype=torch.bool)
            xy = bboxes.gravity_center[:, :2]
            x_norm = (xy[:, 0] - bev_range[0]) / (bev_range[3] - bev_range[0])
            y_norm = (xy[:, 1] - bev_range[1]) / (bev_range[4] - bev_range[1])
            det_bbox_posembed.append(
                torch.cat([x_norm[:, None], y_norm[:, None]], dim=-1)
            )
        return torch.stack(det_bbox_posembed)


class MotionTransformerDecoder(nn.Module):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(
        self,
        pc_range=None,
        embed_dims=256,
        transformerlayers=None,
        num_layers=3,
        **kwargs,
    ):
        super(MotionTransformerDecoder, self).__init__()
        self.pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        self.embed_dims = 256
        self.num_layers = 3
        self.intention_interaction_layers = IntentionInteraction()
        self.track_agent_interaction_layers = nn.ModuleList(
            [TrackAgentInteraction() for i in range(self.num_layers)]
        )
        self.map_interaction_layers = nn.ModuleList(
            [MapInteraction() for i in range(self.num_layers)]
        )
        self.bev_interaction_layers = nn.ModuleList(
            [
                MotionTransformerAttentionLayer(
                    attn_cfgs=[MotionDeformableAttention()],
                )
                for _ in range(self.num_layers)
            ]
        )

        self.static_dynamic_fuser = nn.Sequential(
            nn.Linear(self.embed_dims * 2, self.embed_dims * 2),
            nn.ReLU(),
            nn.Linear(self.embed_dims * 2, self.embed_dims),
        )
        self.dynamic_embed_fuser = nn.Sequential(
            nn.Linear(self.embed_dims * 3, self.embed_dims * 2),
            nn.ReLU(),
            nn.Linear(self.embed_dims * 2, self.embed_dims),
        )
        self.in_query_fuser = nn.Sequential(
            nn.Linear(self.embed_dims * 2, self.embed_dims * 2),
            nn.ReLU(),
            nn.Linear(self.embed_dims * 2, self.embed_dims),
        )
        self.out_query_fuser = nn.Sequential(
            nn.Linear(self.embed_dims * 4, self.embed_dims * 2),
            nn.ReLU(),
            nn.Linear(self.embed_dims * 2, self.embed_dims),
        )

    def forward(
        self,
        track_query,
        lane_query,
        track_query_pos=None,
        lane_query_pos=None,
        track_bbox_results=None,
        bev_embed=None,
        reference_trajs=None,
        traj_reg_branches=None,
        agent_level_embedding=None,
        scene_level_ego_embedding=None,
        scene_level_offset_embedding=None,
        learnable_embed=None,
        agent_level_embedding_layer=None,
        scene_level_ego_embedding_layer=None,
        scene_level_offset_embedding_layer=None,
        **kwargs,
    ):
        """Forward function for `MotionTransformerDecoder`.
        Args:
            agent_query (B, A, D)
            map_query (B, M, D)
            map_query_pos (B, G, D)
            static_intention_embed (B, A, P, D)
            offset_query_embed (B, A, P, D)
            global_intention_embed (B, A, P, D)
            learnable_intention_embed (B, A, P, D)
            det_query_pos (B, A, D)
        Returns:
            None
        """
        intermediate = []
        intermediate_reference_trajs = []

        B, _, P, D = agent_level_embedding.shape
        track_query_bc = track_query.unsqueeze(2).expand(-1, -1, P, -1)
        track_query_pos_bc = track_query_pos.unsqueeze(2).expand(-1, -1, P, -1)

        agent_level_embedding = self.intention_interaction_layers(agent_level_embedding)
        static_intention_embed = (
            agent_level_embedding + scene_level_offset_embedding + learnable_embed
        )
        reference_trajs_input = reference_trajs.unsqueeze(4).detach()

        query_embed = torch.zeros_like(static_intention_embed)
        for lid in range(self.num_layers):
            dynamic_query_embed = self.dynamic_embed_fuser(
                torch.cat(
                    [
                        agent_level_embedding,
                        scene_level_offset_embedding,
                        scene_level_ego_embedding,
                    ],
                    dim=-1,
                )
            )

            query_embed_intention = self.static_dynamic_fuser(
                torch.cat([static_intention_embed, dynamic_query_embed], dim=-1)
            )

            query_embed = self.in_query_fuser(
                torch.cat([query_embed, query_embed_intention], dim=-1)
            )

            track_query_embed = self.track_agent_interaction_layers[lid](
                query_embed,
                track_query,
                query_pos=track_query_pos_bc,
                key_pos=track_query_pos,
            )

            map_query_embed = self.map_interaction_layers[lid](
                query_embed,
                lane_query,
                query_pos=track_query_pos_bc,
                key_pos=lane_query_pos,
            )

            bev_query_embed = self.bev_interaction_layers[lid](
                query_embed,
                value=bev_embed,
                query_pos=track_query_pos_bc,
                bbox_results=track_bbox_results,
                reference_trajs=reference_trajs_input,
                **kwargs,
            )

            query_embed = [
                track_query_embed,
                map_query_embed,
                bev_query_embed,
                track_query_bc + track_query_pos_bc,
            ]
            query_embed = torch.cat(query_embed, dim=-1)
            query_embed = self.out_query_fuser(query_embed)

            if traj_reg_branches is not None:
                tmp = traj_reg_branches[lid](query_embed)
                bs, n_agent, n_modes, n_steps, _ = reference_trajs.shape
                tmp = tmp.view(bs, n_agent, n_modes, n_steps, -1)

                tmp[..., :2] = torch.cumsum(tmp[..., :2], dim=3)
                new_reference_trajs = torch.zeros_like(reference_trajs)
                new_reference_trajs = tmp[..., :2]
                reference_trajs = new_reference_trajs.detach()
                reference_trajs_input = reference_trajs.unsqueeze(4)

                ep_offset_embed = reference_trajs.detach()
                ep_ego_embed = (
                    trajectory_coordinate_transform(
                        reference_trajs.unsqueeze(2),
                        track_bbox_results,
                        with_translation_transform=True,
                        with_rotation_transform=False,
                    )
                    .squeeze(2)
                    .detach()
                )
                ep_agent_embed = (
                    trajectory_coordinate_transform(
                        reference_trajs.unsqueeze(2),
                        track_bbox_results,
                        with_translation_transform=False,
                        with_rotation_transform=True,
                    )
                    .squeeze(2)
                    .detach()
                )

                agent_level_embedding = agent_level_embedding_layer(
                    pos2posemb2d(norm_points(ep_agent_embed[..., -1, :], self.pc_range))
                )
                scene_level_ego_embedding = scene_level_ego_embedding_layer(
                    pos2posemb2d(norm_points(ep_ego_embed[..., -1, :], self.pc_range))
                )
                scene_level_offset_embedding = scene_level_offset_embedding_layer(
                    pos2posemb2d(
                        norm_points(ep_offset_embed[..., -1, :], self.pc_range)
                    )
                )

                intermediate.append(query_embed)
                intermediate_reference_trajs.append(reference_trajs)

        return torch.stack(intermediate), torch.stack(intermediate_reference_trajs)


class TrackAgentInteraction(nn.Module):
    """
    Modeling the interaction between the agents
    """

    def __init__(
        self,
        embed_dims=256,
        num_heads=8,
        dropout=0.1,
        batch_first=True,
        norm_cfg=None,
    ):
        super().__init__()
        self.batch_first = batch_first
        self.interaction_transformer = nn.TransformerDecoderLayer(
            d_model=embed_dims,
            nhead=num_heads,
            dropout=dropout,
            dim_feedforward=embed_dims * 2,
            batch_first=batch_first,
        )

    def forward(self, query, key, query_pos=None, key_pos=None):
        """
        query: context query (B, A, P, D)
        query_pos: mode pos embedding (B, A, P, D)
        key: (B, A, D)
        key_pos: (B, A, D)
        """
        B, A, P, D = query.shape
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos
        mem = key.expand(B * A, -1, -1)
        query = torch.flatten(query, start_dim=0, end_dim=1)
        query = self.interaction_transformer(query, mem)
        query = query.view(B, A, P, D)
        return query


class MapInteraction(nn.Module):
    """
    Modeling the interaction between the agent and the map
    """

    def __init__(
        self,
        embed_dims=256,
        num_heads=8,
        dropout=0.1,
        batch_first=True,
        norm_cfg=None,
    ):
        super().__init__()

        self.batch_first = batch_first
        self.interaction_transformer = nn.TransformerDecoderLayer(
            d_model=embed_dims,
            nhead=num_heads,
            dropout=dropout,
            dim_feedforward=embed_dims * 2,
            batch_first=batch_first,
        )

    def forward(self, query, key, query_pos=None, key_pos=None):
        """
        x: context query (B, A, P, D)
        query_pos: mode pos embedding (B, A, P, D)
        """
        B, A, P, D = query.shape
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        query = torch.flatten(query, start_dim=0, end_dim=1)
        mem = key.expand(B * A, -1, -1)
        query = self.interaction_transformer(query, mem)
        query = query.view(B, A, P, D)
        return query


class IntentionInteraction(nn.Module):
    """
    Modeling the interaction between anchors
    """

    def __init__(
        self,
        embed_dims=256,
        num_heads=8,
        dropout=0.1,
        batch_first=True,
        norm_cfg=None,
    ):
        super().__init__()
        self.batch_first = batch_first
        self.interaction_transformer = nn.TransformerEncoderLayer(
            d_model=embed_dims,
            nhead=num_heads,
            dropout=dropout,
            dim_feedforward=embed_dims * 2,
            batch_first=batch_first,
        )

    def forward(self, query):
        B, A, P, D = query.shape
        rebatch_x = torch.flatten(query, start_dim=0, end_dim=1)
        rebatch_x = self.interaction_transformer(rebatch_x)
        out = rebatch_x.view(B, A, P, D)
        return out


class MotionTransformerAttentionLayer(nn.Module):
    """Base `TransformerLayer` for vision transformer.
    It can be built from `mmcv.ConfigDict` and support more flexible
    customization, for example, using any number of `FFN or LN ` and
    use different kinds of `attention` by specifying a list of `ConfigDict`
    named `attn_cfgs`. It is worth mentioning that it supports `prenorm`
    when you specifying `norm` as the first element of `operation_order`.
    More details about the `prenorm`: `On Layer Normalization in the
    Transformer Architecture <https://arxiv.org/abs/2002.04745>`_ .
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict` | None )):
            Configs for `self_attention` or `cross_attention` modules,
            The order of the configs in the list should be consistent with
            corresponding attentions in operation_order.
            If it is a dict, all of the attention modules in operation_order
            will be built with this config. Default: None.
        ffn_cfgs (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict` | None )):
            Configs for FFN, The order of the configs in the list should be
            consistent with corresponding ffn in operation_order.
            If it is a dict, all of the attention modules in operation_order
            will be built with this config.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Support `prenorm` when you specifying first element as `norm`.
            Default：None.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        batch_first (bool): Key, Query and Value are shape
            of (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
    """

    def __init__(
        self,
        attn_cfgs=None,
        ffn_cfgs=dict(
            type="FFN",
            embed_dims=256,
            feedforward_channels=512,
            num_fcs=2,
            ffn_drop=0.1,
            act_cfg=dict(type="ReLU", inplace=True),
        ),
        operation_order=("cross_attn", "norm", "ffn", "norm"),
        norm_cfg=dict(type="LN"),
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
                warnings.warn(
                    f"The arguments `{ori_name}` in BaseTransformerLayer "
                    f"has been deprecated, now you should set `{new_name}` "
                    f"and other FFN related arguments "
                    f"to a dict named `ffn_cfgs`. ",
                    DeprecationWarning,
                )
                ffn_cfgs[new_name] = kwargs[ori_name]

        super().__init__()

        self.batch_first = batch_first

        num_attn = operation_order.count("self_attn") + operation_order.count(
            "cross_attn"
        )
        if isinstance(attn_cfgs, dict):
            attn_cfgs = [copy.deepcopy(attn_cfgs) for _ in range(num_attn)]
        else:
            assert num_attn == len(attn_cfgs), (
                f"The length "
                f"of attn_cfg {num_attn} is "
                f"not consistent with the number of attention"
                f"in operation_order {operation_order}."
            )

        self.num_attn = num_attn
        self.operation_order = operation_order
        self.norm_cfg = norm_cfg
        self.pre_norm = operation_order[0] == "norm"
        self.attentions = nn.ModuleList()

        index = 0
        for operation_name in operation_order:
            if operation_name in ["self_attn", "cross_attn"]:
                attention = MotionDeformableAttention()
                attention.operation_name = operation_name
                self.attentions.append(attention)
                index += 1

        self.embed_dims = self.attentions[0].embed_dims

        self.ffns = nn.ModuleList()
        self.ffns.append(
            FFN(
                embed_dims=256,
                feedforward_channels=512,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type="ReLU", inplace=True),
            )
        )
        self.norms = nn.ModuleList()
        num_norms = operation_order.count("norm")
        for _ in range(num_norms):
            self.norms.append(nn.LayerNorm(self.embed_dims))

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
        """Forward function for `TransformerDecoderLayer`.
        **kwargs contains some specific arguments of attentions.
        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.
        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
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


class MotionDeformableAttention(nn.Module):
    """An attention module used in Deformable-Detr.

    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.

    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
    """

    def __init__(
        self,
        embed_dims=256,
        num_heads=8,
        num_levels=1,
        num_points=4,
        num_steps=12,
        sample_index=-1,
        im2col_step=64,
        dropout=0.1,
        bev_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        voxel_size=[0.2, 0.2, 8],
        batch_first=True,
        norm_cfg=None,
    ):
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_heads, "
                f"but got {embed_dims} and {num_heads}"
            )
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.fp16_enabled = False
        self.bev_range = bev_range

        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    "invalid input for _is_power_of_2: {} (type: {})".format(n, type(n))
                )
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                "MultiScaleDeformAttention to make "
                "the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation."
            )

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_steps = num_steps
        self.sample_index = sample_index
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_steps * num_levels * num_points * 2
        )
        self.attention_weights = nn.Linear(
            embed_dims, num_heads * num_steps * num_levels * num_points
        )
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Sequential(
            nn.Linear(num_steps * embed_dims, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        key_padding_mask=None,
        spatial_shapes=None,
        level_start_index=None,
        bbox_results=None,
        reference_trajs=None,
        flag="decoder",
        **kwargs,
    ):
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].

        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        bs, num_agent, num_mode, _ = query.shape
        num_query = num_agent * num_mode
        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        query = torch.flatten(query, start_dim=1, end_dim=2)

        value = value.permute(1, 0, 2)
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs,
            num_query,
            self.num_heads,
            self.num_steps,
            self.num_levels,
            self.num_points,
            2,
        )
        attention_weights = self.attention_weights(query).view(
            bs,
            num_query,
            self.num_heads,
            self.num_steps,
            self.num_levels * self.num_points,
        )
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(
            bs,
            num_query,
            self.num_heads,
            self.num_steps,
            self.num_levels,
            self.num_points,
        )

        reference_trajs = reference_trajs[:, :, :, [self.sample_index], :, :]
        reference_trajs_ego = self.agent_coords_to_ego_coords(
            copy.deepcopy(reference_trajs), bbox_results
        ).detach()
        reference_trajs_ego = torch.flatten(reference_trajs_ego, start_dim=1, end_dim=2)
        reference_trajs_ego = reference_trajs_ego[:, :, None, :, :, None, :]
        reference_trajs_ego[..., 0] -= self.bev_range[0]
        reference_trajs_ego[..., 1] -= self.bev_range[1]
        reference_trajs_ego[..., 0] /= self.bev_range[3] - self.bev_range[0]
        reference_trajs_ego[..., 1] /= self.bev_range[4] - self.bev_range[1]
        offset_normalizer = torch.stack(
            [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1
        )
        sampling_locations = (
            reference_trajs_ego
            + sampling_offsets / offset_normalizer[None, None, None, None, :, None, :]
        )

        sampling_locations = rearrange(
            sampling_locations, "bs nq nh ns nl np c -> bs nq ns nh nl np c"
        )
        attention_weights = rearrange(
            attention_weights, "bs nq nh ns nl np -> bs nq ns nh nl np"
        )
        sampling_locations = sampling_locations.reshape(
            bs,
            num_query * self.num_steps,
            self.num_heads,
            self.num_levels,
            self.num_points,
            2,
        )
        attention_weights = attention_weights.reshape(
            bs,
            num_query * self.num_steps,
            self.num_heads,
            self.num_levels,
            self.num_points,
        )
        output = multi_scale_deformable_attn_pytorch(
            value, spatial_shapes, sampling_locations, attention_weights
        )
        output = output.view(bs, num_query, self.num_steps, -1)
        output = torch.flatten(output, start_dim=2, end_dim=3)
        output = self.output_proj(output)
        output = output.view(bs, num_agent, num_mode, -1)
        return self.dropout(output) + identity

    def agent_coords_to_ego_coords(self, reference_trajs, bbox_results):
        batch_size = len(bbox_results)
        reference_trajs_ego = []
        for i in range(batch_size):
            boxes_3d, scores, labels, bbox_index, mask = bbox_results[i]
            det_centers = boxes_3d.gravity_center.to(reference_trajs)
            batch_reference_trajs = reference_trajs[i]
            batch_reference_trajs += det_centers[:, None, None, None, :2]
            reference_trajs_ego.append(batch_reference_trajs)
        return torch.stack(reference_trajs_ego)
