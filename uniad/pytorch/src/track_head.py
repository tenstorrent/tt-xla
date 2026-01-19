# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import copy
import torch
import torch.nn as nn

from third_party.tt_forge_models.uniad.pytorch.src.track_utils import *


class BEVFormerTrackHead(DETRHead):
    """Head of Detr3D.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
           bev_h=200,
    """

    def __init__(
        self,
        *args,
        with_box_refine=True,
        num_cls_fcs=2,
        code_weights=None,
        as_two_stage=False,
        bbox_coder=DETRTrack3DCoder(),
        bev_h=200,
        bev_w=200,
        fut_steps=4,
        in_channels=256,
        num_classes=10,
        num_query=900,
        past_steps=4,
        positional_encoding=LearnedPositionalEncoding(),
        sync_cls_avg_factor=True,
        transformer=PerceptionTransformer(),
        **kwargs
    ):

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False

        self.with_box_refine = with_box_refine

        assert as_two_stage is False, "as_two_stage is not supported yet."
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer["as_two_stage"] = self.as_two_stage

        self.code_size = 10
        self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        self.bbox_coder = DETRTrack3DCoder()
        self.pc_range = self.bbox_coder.pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1
        self.past_steps = past_steps
        self.fut_steps = fut_steps
        super(BEVFormerTrackHead, self).__init__(
            *args, transformer=transformer, **kwargs
        )
        self.code_weights = nn.Parameter(
            torch.tensor(self.code_weights, requires_grad=False), requires_grad=False
        )

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(nn.Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(nn.Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        past_traj_reg_branch = []
        for _ in range(self.num_reg_fcs):
            past_traj_reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            past_traj_reg_branch.append(nn.ReLU())
        past_traj_reg_branch.append(
            nn.Linear(self.embed_dims, (self.past_steps + self.fut_steps) * 2)
        )
        past_traj_reg_branch = nn.Sequential(*past_traj_reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        num_pred = (
            (self.transformer.decoder.num_layers + 1)
            if self.as_two_stage
            else self.transformer.decoder.num_layers
        )

        self.cls_branches = _get_clones(fc_cls, num_pred)
        self.reg_branches = _get_clones(reg_branch, num_pred)
        self.past_traj_reg_branches = _get_clones(past_traj_reg_branch, num_pred)
        if not self.as_two_stage:
            self.bev_embedding = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)

    def get_bev_features(self, mlvl_feats, img_metas, prev_bev=None):
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        bev_queries = self.bev_embedding.weight.to(dtype)

        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w)).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)
        bev_embed = self.transformer.get_bev_features(
            mlvl_feats,
            bev_queries,
            self.bev_h,
            self.bev_w,
            grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
            bev_pos=bev_pos,
            prev_bev=prev_bev,
            img_metas=img_metas,
        )
        return bev_embed, bev_pos

    def get_detections(
        self,
        bev_embed,
        object_query_embeds=None,
        ref_points=None,
        img_metas=None,
    ):
        hs, init_reference, inter_references = self.transformer.get_states_and_refs(
            bev_embed,
            object_query_embeds,
            self.bev_h,
            self.bev_w,
            reference_points=ref_points,
            reg_branches=self.reg_branches if self.with_box_refine else None,
            cls_branches=self.cls_branches if self.as_two_stage else None,
            img_metas=img_metas,
        )
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        outputs_trajs = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = ref_points.sigmoid()
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])
            outputs_past_traj = self.past_traj_reg_branches[lvl](hs[lvl]).view(
                tmp.shape[0], -1, self.past_steps + self.fut_steps, 2
            )
            assert reference.shape[-1] == 3
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()

            last_ref_points = torch.cat(
                [tmp[..., 0:2], tmp[..., 4:5]],
                dim=-1,
            )

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
            outputs_trajs.append(outputs_past_traj)
        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        outputs_trajs = torch.stack(outputs_trajs)
        last_ref_points = inverse_sigmoid(last_ref_points)
        outs = {
            "all_cls_scores": outputs_classes,
            "all_bbox_preds": outputs_coords,
            "all_past_traj_preds": outputs_trajs,
            "enc_cls_scores": None,
            "enc_bbox_preds": None,
            "last_ref_points": last_ref_points,
            "query_feats": hs,
        }
        return outs
