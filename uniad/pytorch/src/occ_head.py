# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import copy
from third_party.tt_forge_models.uniad.pytorch.src.occ_utils import *


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class OccHead(nn.Module):
    def __init__(
        self,
        receptive_field=3,
        n_future=4,
        spatial_extent=(50, 50),
        ignore_index=255,
        grid_conf={
            "xbound": [-50.0, 50.0, 0.5],
            "ybound": [-50.0, 50.0, 0.5],
            "zbound": [-10.0, 10.0, 20.0],
        },
        bev_size=(200, 200),
        bev_emb_dim=256,
        bev_proj_dim=256,
        bev_proj_nlayers=4,
        query_dim=256,
        query_mlp_layers=3,
        detach_query_pos=True,
        temporal_mlp_layer=2,
        transformer_decoder=DetrTransformerDecoder(),
        attn_mask_thresh=0.3,
        sample_ignore_mode="all_valid",
        pan_eval=True,
        test_seg_thresh: float = 0.1,
        test_with_track_score=True,
    ):
        super().__init__()
        self.receptive_field = receptive_field
        self.n_future = n_future
        self.spatial_extent = spatial_extent
        self.ignore_index = ignore_index

        bevformer_bev_conf = {
            "xbound": [-51.2, 51.2, 0.512],
            "ybound": [-51.2, 51.2, 0.512],
            "zbound": [-10.0, 10.0, 20.0],
        }
        self.bev_sampler = BevFeatureSlicer(bevformer_bev_conf, grid_conf)

        self.bev_size = bev_size
        self.bev_proj_dim = bev_proj_dim

        self.bev_light_proj = SimpleConv2d(
            in_channels=bev_emb_dim,
            conv_channels=bev_emb_dim,
            out_channels=bev_proj_dim,
            num_conv=bev_proj_nlayers,
        )

        self.base_downscale = nn.Sequential(
            Bottleneck(in_channels=bev_proj_dim, downsample=True),
            Bottleneck(in_channels=bev_proj_dim, downsample=True),
        )

        self.n_future_blocks = self.n_future + 1

        self.attn_mask_thresh = attn_mask_thresh

        self.num_trans_layers = 5
        assert self.num_trans_layers % self.n_future_blocks == 0

        self.num_heads = 8
        self.transformer_decoder = DetrTransformerDecoder()

        temporal_mlp = MLP(
            query_dim, query_dim, bev_proj_dim, num_layers=temporal_mlp_layer
        )
        self.temporal_mlps = _get_clones(temporal_mlp, self.n_future_blocks)

        downscale_conv = Bottleneck(in_channels=bev_proj_dim, downsample=True)
        self.downscale_convs = _get_clones(downscale_conv, self.n_future_blocks)

        upsample_add = UpsamplingAdd(
            in_channels=bev_proj_dim, out_channels=bev_proj_dim
        )
        self.upsample_adds = _get_clones(upsample_add, self.n_future_blocks)

        self.dense_decoder = CVT_Decoder(
            dim=bev_proj_dim,
            blocks=[bev_proj_dim, bev_proj_dim],
        )

        self.mode_fuser = nn.Sequential(
            nn.Linear(query_dim, bev_proj_dim),
            nn.LayerNorm(bev_proj_dim),
            nn.ReLU(inplace=True),
        )
        self.multi_query_fuser = nn.Sequential(
            nn.Linear(query_dim * 3, query_dim * 2),
            nn.LayerNorm(query_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(query_dim * 2, bev_proj_dim),
        )

        self.detach_query_pos = detach_query_pos

        self.query_to_occ_feat = MLP(
            query_dim, query_dim, bev_proj_dim, num_layers=query_mlp_layers
        )
        self.temporal_mlp_for_mask = copy.deepcopy(self.query_to_occ_feat)

        self.sample_ignore_mode = sample_ignore_mode
        assert self.sample_ignore_mode in ["all_valid", "past_valid", "none"]

        self.pan_eval = pan_eval
        self.test_seg_thresh = test_seg_thresh

        self.test_with_track_score = test_with_track_score

    def get_attn_mask(self, state, ins_query):
        ins_embed = self.temporal_mlp_for_mask(ins_query)
        mask_pred = torch.einsum("bqc,bchw->bqhw", ins_embed, state)
        attn_mask = mask_pred.sigmoid() < self.attn_mask_thresh
        attn_mask = (
            rearrange(attn_mask, "b q h w -> b (h w) q")
            .unsqueeze(1)
            .repeat(1, self.num_heads, 1, 1)
            .flatten(0, 1)
        )
        attn_mask = attn_mask.detach()

        attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

        upsampled_mask_pred = F.interpolate(
            mask_pred, self.bev_size, mode="bilinear", align_corners=False
        )

        return attn_mask, upsampled_mask_pred, ins_embed

    def forward(self, x, ins_query):
        base_state = rearrange(x, "(h w) b d -> b d h w", h=self.bev_size[0])

        base_state = self.bev_sampler(base_state)
        base_state = self.bev_light_proj(base_state)
        base_state = self.base_downscale(base_state)
        base_ins_query = ins_query

        last_state = base_state
        last_ins_query = base_ins_query
        future_states = []
        mask_preds = []
        temporal_query = []
        temporal_embed_for_mask_attn = []
        n_trans_layer_each_block = self.num_trans_layers // self.n_future_blocks
        assert n_trans_layer_each_block >= 1

        for i in range(self.n_future_blocks):
            cur_state = self.downscale_convs[i](last_state)

            cur_ins_query = self.temporal_mlps[i](last_ins_query)
            temporal_query.append(cur_ins_query)

            attn_mask, mask_pred, cur_ins_emb_for_mask_attn = self.get_attn_mask(
                cur_state, cur_ins_query
            )
            attn_masks = [None, attn_mask]

            mask_preds.append(mask_pred)
            temporal_embed_for_mask_attn.append(cur_ins_emb_for_mask_attn)

            cur_state = rearrange(cur_state, "b c h w -> (h w) b c")
            cur_ins_query = rearrange(cur_ins_query, "b q c -> q b c")

            for j in range(n_trans_layer_each_block):
                trans_layer_ind = i * n_trans_layer_each_block + j
                trans_layer = self.transformer_decoder.layers[trans_layer_ind]
                cur_state = trans_layer(
                    query=cur_state,
                    key=cur_ins_query,
                    value=cur_ins_query,
                    query_pos=None,
                    key_pos=None,
                    attn_masks=attn_masks,
                    query_key_padding_mask=None,
                    key_padding_mask=None,
                )

            cur_state = rearrange(
                cur_state, "(h w) b c -> b c h w", h=self.bev_size[0] // 8
            )

            cur_state = self.upsample_adds[i](cur_state, last_state)

            future_states.append(cur_state)
            last_state = cur_state

        future_states = torch.stack(future_states, dim=1)
        temporal_query = torch.stack(temporal_query, dim=1)
        mask_preds = torch.stack(mask_preds, dim=2)
        ins_query = torch.stack(temporal_embed_for_mask_attn, dim=1)

        future_states = self.dense_decoder(future_states)
        ins_occ_query = self.query_to_occ_feat(ins_query)
        ins_occ_logits = torch.einsum("btqc,btchw->bqthw", ins_occ_query, future_states)

        return mask_preds, ins_occ_logits

    def merge_queries(self, outs_dict, detach_query_pos=True):
        ins_query = outs_dict.get("traj_query", None)
        track_query = outs_dict["track_query"]
        track_query_pos = outs_dict["track_query_pos"]

        if detach_query_pos:
            track_query_pos = track_query_pos.detach()

        ins_query = ins_query[-1]
        ins_query = self.mode_fuser(ins_query).max(2)[0]
        ins_query = self.multi_query_fuser(
            torch.cat([ins_query, track_query, track_query_pos], dim=-1)
        )

        return ins_query

    def forward_test(
        self,
        bev_feat,
        outs_dict,
        no_query=False,
        gt_segmentation=None,
        gt_instance=None,
        gt_img_is_valid=None,
    ):
        gt_segmentation, gt_instance, gt_img_is_valid = self.get_occ_labels(
            gt_segmentation, gt_instance, gt_img_is_valid
        )

        out_dict = dict()
        out_dict["seg_gt"] = gt_segmentation[:, : 1 + self.n_future]
        out_dict["ins_seg_gt"] = self.get_ins_seg_gt(
            gt_instance[:, : 1 + self.n_future]
        )
        if no_query:
            out_dict["seg_out"] = torch.zeros_like(out_dict["seg_gt"]).long()
            out_dict["ins_seg_out"] = torch.zeros_like(out_dict["ins_seg_gt"]).long()
            return out_dict

        ins_query = self.merge_queries(outs_dict, self.detach_query_pos)

        _, pred_ins_logits = self(bev_feat, ins_query=ins_query)

        out_dict["pred_ins_logits"] = pred_ins_logits

        pred_ins_logits = pred_ins_logits[:, :, : 1 + self.n_future]
        pred_ins_sigmoid = pred_ins_logits.sigmoid()

        if self.test_with_track_score:
            track_scores = outs_dict["track_scores"].to(pred_ins_sigmoid)
            track_scores = track_scores[:, :, None, None, None]
            pred_ins_sigmoid = pred_ins_sigmoid * track_scores

        out_dict["pred_ins_sigmoid"] = pred_ins_sigmoid
        pred_seg_scores = pred_ins_sigmoid.max(1)[0]
        seg_out = (pred_seg_scores > self.test_seg_thresh).long().unsqueeze(2)
        out_dict["seg_out"] = seg_out
        if self.pan_eval:
            pred_consistent_instance_seg = (
                predict_instance_segmentation_and_trajectories(
                    seg_out, pred_ins_sigmoid
                )
            )

            out_dict["ins_seg_out"] = pred_consistent_instance_seg

        return out_dict

    def get_ins_seg_gt(self, gt_instance):
        ins_gt_old = gt_instance
        ins_gt_new = torch.zeros_like(ins_gt_old).to(ins_gt_old)
        ins_inds_unique = torch.unique(ins_gt_old)
        new_id = 1
        for uni_id in ins_inds_unique:
            if uni_id.item() in [0, self.ignore_index]:
                continue
            ins_gt_new[ins_gt_old == uni_id] = new_id
            new_id += 1
        return ins_gt_new

    def get_occ_labels(self, gt_segmentation, gt_instance, gt_img_is_valid):
        if not self.training:
            gt_segmentation = gt_segmentation[0]
            gt_instance = gt_instance[0]
            gt_img_is_valid = gt_img_is_valid[0]

        gt_segmentation = gt_segmentation[0][:, : self.n_future + 1].long().unsqueeze(2)
        gt_instance = gt_instance[0][:, : self.n_future + 1].long()
        gt_img_is_valid = gt_img_is_valid[:, : self.receptive_field + self.n_future]
        return gt_segmentation, gt_instance, gt_img_is_valid
