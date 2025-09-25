# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from third_party.tt_forge_models.uniad.pytorch.src.panseg_utils import *
from third_party.tt_forge_models.uniad.pytorch.src.utils import *
from third_party.tt_forge_models.uniad.pytorch.src.transformer import *
from abc import ABCMeta
from functools import partial
from typing import Optional
import copy
from torch import Tensor


class SegDETRHead(nn.Module, metaclass=ABCMeta):
    """Implements the DETR transformer head.

    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.

    Args:
        num_classes (int): Number of categories excluding the background.
        in_channels (int): Number of channels in the input feature map.
        num_query (int): Number of query in Transformer.
        num_reg_fcs (int, optional): Number of fully-connected layers used in
            `FFN`, which is then used for the regression head. Default 2.
        transformer (obj:`mmcv.ConfigDict`|dict): Config for transformer.
            Default: None.
        sync_cls_avg_factor (bool): Whether to sync the avg_factor of
            all ranks. Default to False.
        positional_encoding (obj:`mmcv.ConfigDict`|dict):
            Config for position encoding.
        test_cfg (obj:`mmcv.ConfigDict`|dict): Testing config of
            transformer head.
    """

    _version = 2

    def __init__(
        self,
        bev_h=200,
        bev_w=200,
        canvas_size=(200, 200),
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        num_classes=4,
        num_things_classes=3,
        num_stuff_classes=1,
        in_channels=2048,
        num_query=300,
        num_reg_fcs=2,
        sync_cls_avg_factor=True,
        as_two_stage=False,
        with_box_refine=True,
        transformer=SegDeformableTransformer(),
        positional_encoding=SinePositionalEncoding(),
        test_cfg=dict(max_per_img=100),
        thing_transformer_head=dict(
            type="SegMaskHead", d_model=256, nhead=8, num_decoder_layers=4
        ),
        stuff_transformer_head=dict(
            type="SegMaskHead",
            d_model=256,
            nhead=8,
            num_decoder_layers=6,
            self_attn=True,
        ),
    ):

        super(SegDETRHead, self).__init__()
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        class_weight = None
        self.num_query = 300
        self.num_classes = 4
        self.num_things_classes = 3
        self.num_stuff_classes = 1
        self.in_channels = 2048
        self.num_reg_fcs = num_reg_fcs
        self.test_cfg = test_cfg
        self.fp16_enabled = False

        self.cls_out_channels = num_things_classes

        self.positional_encoding = SinePositionalEncoding(
            num_feats=128, normalize=True, offset=-0.5
        )
        self.transformer = SegDeformableTransformer()

        self.embed_dims = self.transformer.embed_dims
        num_feats = positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, (
            "embed_dims should"
            f" be exactly 2 times of num_feats. Found {self.embed_dims}"
            f" and {num_feats}."
        )
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the transformer head."""
        self.input_proj = nn.Conv2d(self.in_channels, self.embed_dims, kernel_size=1)
        self.fc_cls = nn.Linear(self.embed_dims, self.cls_out_channels)
        self.reg_ffn = FFN(
            self.embed_dims,
            self.embed_dims,
            self.num_reg_fcs,
            self.act_cfg,
            dropout=0.0,
            add_residual=False,
        )
        self.fc_reg = nn.Linear(self.embed_dims, 4)
        self.query_embedding = nn.Embedding(self.num_query, self.embed_dims)

    def forward(self, feats, img_metas):
        """Forward function.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[list[Tensor], list[Tensor]]: Outputs for all scale levels.

                - all_cls_scores_list (list[Tensor]): Classification scores \
                    for each scale level. Each is a 4D-tensor with shape \
                    [nb_dec, bs, num_query, cls_out_channels]. Note \
                    `cls_out_channels` should includes background.
                - all_bbox_preds_list (list[Tensor]): Sigmoid regression \
                    outputs for each scale level. Each is a 4D-tensor with \
                    normalized coordinate format (cx, cy, w, h) and shape \
                    [nb_dec, bs, num_query, 4].
        """
        num_levels = len(feats)
        img_metas_list = [img_metas for _ in range(num_levels)]
        return multi_apply(self.forward_single, feats, img_metas_list)

    def get_targets(
        self,
        cls_scores_list,
        bbox_preds_list,
        gt_bboxes_list,
        gt_labels_list,
        img_metas,
        gt_bboxes_ignore_list=None,
    ):
        """"Compute regression and classification targets for a batch image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.

        Returns:
            tuple: a tuple containing the following targets.

                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert (
            gt_bboxes_ignore_list is None
        ), "Only supports for gt_bboxes_ignore setting to None."
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [gt_bboxes_ignore_list for _ in range(num_imgs)]

        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            pos_inds_list,
            neg_inds_list,
        ) = multi_apply(
            self._get_target_single,
            cls_scores_list,
            bbox_preds_list,
            gt_bboxes_list,
            gt_labels_list,
            img_metas,
            gt_bboxes_ignore_list,
        )
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg,
        )

    def get_bboxes(
        self, all_cls_scores_list, all_bbox_preds_list, img_metas, rescale=False
    ):
        """Transform network outputs for a batch into bbox predictions.

        Args:
            all_cls_scores_list (list[Tensor]): Classification outputs
                for each feature level. Each is a 4D-tensor with shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds_list (list[Tensor]): Sigmoid regression
                outputs for each feature level. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                [nb_dec, bs, num_query, 4].
            img_metas (list[dict]): Meta information of each image.
            rescale (bool, optional): If True, return boxes in original
                image space. Default False.

        Returns:
            list[list[Tensor, Tensor]]: Each item in result_list is 2-tuple. \
                The first item is an (n, 5) tensor, where the first 4 columns \
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the \
                5-th column is a score between 0 and 1. The second item is a \
                (n,) tensor where each item is the predicted class label of \
                the corresponding box.
        """

        cls_scores = all_cls_scores_list[-1][-1]
        bbox_preds = all_bbox_preds_list[-1][-1]

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            img_shape = img_metas[img_id]["img_shape"]
            scale_factor = img_metas[img_id]["scale_factor"]
            proposals = self._get_bboxes_single(
                cls_score, bbox_pred, img_shape, scale_factor, rescale
            )
            result_list.append(proposals)

        return result_list


class PansegformerHead(SegDETRHead):
    """
    Head of Panoptic SegFormer

    Code is modified from the `official github repo
    <https://github.com/open-mmlab/mmdetection>`_.

    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
    """

    def __init__(
        self,
        *args,
        bev_h=200,
        bev_w=200,
        canvas_size=(200, 200),
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        num_classes=4,
        num_things_classes=3,
        num_stuff_classes=1,
        in_channels=2048,
        num_query=300,
        num_reg_fcs=2,
        sync_cls_avg_factor=True,
        as_two_stage=False,
        with_box_refine=True,
        transformer=SegDeformableTransformer(),
        positional_encoding=SinePositionalEncoding(),
        test_cfg=dict(max_per_img=100),
        thing_transformer_head=dict(
            type="SegMaskHead", d_model=256, nhead=8, num_decoder_layers=4
        ),
        stuff_transformer_head=dict(
            type="SegMaskHead",
            d_model=256,
            nhead=8,
            num_decoder_layers=6,
            self_attn=True,
        ),
        quality_threshold_things=0.25,
        quality_threshold_stuff=0.25,
        overlap_threshold_things=0.4,
        overlap_threshold_stuff=0.2,
        **kwargs,
    ):
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.canvas_size = canvas_size
        self.pc_range = pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]

        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        self.quality_threshold_things = 0.1
        self.quality_threshold_stuff = quality_threshold_stuff
        self.overlap_threshold_things = overlap_threshold_things
        self.overlap_threshold_stuff = overlap_threshold_stuff
        self.fp16_enabled = False

        if self.as_two_stage:
            transformer["as_two_stage"] = self.as_two_stage
        self.num_dec_things = 4
        self.num_dec_stuff = 6
        super(PansegformerHead, self).__init__(*args, transformer=transformer, **kwargs)

        self.things_mask_head = SegMaskHead(d_model=256, nhead=8, num_decoder_layers=4)

        self.stuff_mask_head = SegMaskHead(
            d_model=256, nhead=8, num_decoder_layers=6, self_attn=True
        )
        self.count = 0

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        if not self.as_two_stage:
            self.bev_embedding = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)

        fc_cls = nn.Linear(self.embed_dims, self.cls_out_channels)
        fc_cls_stuff = nn.Linear(self.embed_dims, 1)
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(nn.Linear(self.embed_dims, 4))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        num_pred = (
            (self.transformer.decoder.num_layers + 1)
            if self.as_two_stage
            else self.transformer.decoder.num_layers
        )

        self.cls_branches = _get_clones(fc_cls, num_pred)
        self.reg_branches = _get_clones(reg_branch, num_pred)

        if not self.as_two_stage:
            self.query_embedding = nn.Embedding(self.num_query, self.embed_dims * 2)
        self.stuff_query = nn.Embedding(self.num_stuff_classes, self.embed_dims * 2)
        self.reg_branches2 = _get_clones(reg_branch, self.num_dec_things)
        self.cls_thing_branches = _get_clones(fc_cls, self.num_dec_things)
        self.cls_stuff_branches = _get_clones(fc_cls_stuff, self.num_dec_stuff)

    def forward(self, bev_embed):
        """Forward function.

        Args:
            bev_embed (tuple[Tensor]): Features from the upstream
                network, each is a 4D-tensor with shape
                (N, C, H, W).
            img_metas (list[dict]): List of image information.

        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, h). \
                Shape [nb_dec, bs, num_query, 4].
            enc_outputs_class (Tensor): The score of each point on encode \
                feature map, has shape (N, h*w, num_class). Only when \
                as_two_stage is True it would be returned, otherwise \
                `None` would be returned.
            enc_outputs_coord (Tensor): The proposal generate from the \
                encode feature map, has shape (N, h*w, 4). Only when \
                as_two_stage is True it would be returned, otherwise \
                `None` would be returned.
        """
        _, bs, _ = bev_embed.shape

        mlvl_feats = [
            torch.reshape(bev_embed, (bs, self.bev_h, self.bev_w, -1)).permute(
                0, 3, 1, 2
            )
        ]
        img_masks = mlvl_feats[0].new_zeros((bs, self.bev_h, self.bev_w))

        hw_lvl = [feat_lvl.shape[-2:] for feat_lvl in mlvl_feats]
        mlvl_masks = []
        mlvl_positional_encodings = []
        for feat in mlvl_feats:
            mlvl_masks.append(
                F.interpolate(img_masks[None], size=feat.shape[-2:])
                .to(torch.bool)
                .squeeze(0)
            )
            mlvl_positional_encodings.append(self.positional_encoding(mlvl_masks[-1]))

        query_embeds = None
        if not self.as_two_stage:
            query_embeds = self.query_embedding.weight
        (
            (memory, memory_pos, memory_mask, query_pos),
            hs,
            init_reference,
            inter_references,
            enc_outputs_class,
            enc_outputs_coord,
        ) = self.transformer(
            mlvl_feats,
            mlvl_masks,
            query_embeds,
            mlvl_positional_encodings,
            reg_branches=self.reg_branches if self.with_box_refine else None,
            cls_branches=self.cls_branches if self.as_two_stage else None,
        )

        memory = memory.permute(1, 0, 2)

        query = hs[-1].permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        memory_pos = memory_pos.permute(1, 0, 2)

        args_tuple = [memory, memory_mask, memory_pos, query, None, query_pos, hw_lvl]

        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])

            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)

        outs = {
            "bev_embed": None if self.as_two_stage else bev_embed,
            "outputs_classes": outputs_classes,
            "outputs_coords": outputs_coords,
            "enc_outputs_class": enc_outputs_class if self.as_two_stage else None,
            "enc_outputs_coord": enc_outputs_coord.sigmoid()
            if self.as_two_stage
            else None,
            "args_tuple": args_tuple,
            "reference": reference,
        }

        return outs

    def filter_query(
        self,
        cls_scores_list,
        bbox_preds_list,
        gt_bboxes_list,
        gt_labels_list,
        img_metas,
        gt_bboxes_ignore_list=None,
    ):
        """
        This function aims to using the cost from the location decoder to filter out low-quality queries.
        """
        assert (
            gt_bboxes_ignore_list is None
        ), "Only supports for gt_bboxes_ignore setting to None."
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [gt_bboxes_ignore_list for _ in range(num_imgs)]

        (
            pos_inds_mask_list,
            neg_inds_mask_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            pos_inds_list,
            neg_inds_list,
        ) = multi_apply(
            self._filter_query_single,
            cls_scores_list,
            bbox_preds_list,
            gt_bboxes_list,
            gt_labels_list,
            img_metas,
            gt_bboxes_ignore_list,
        )
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))

        return (
            pos_inds_mask_list,
            neg_inds_mask_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg,
            pos_inds_list,
            neg_inds_list,
        )

    def get_targets_with_mask(
        self,
        cls_scores_list,
        bbox_preds_list,
        masks_preds_list_thing,
        gt_bboxes_list,
        gt_labels_list,
        gt_masks_list,
        img_metas,
        gt_bboxes_ignore_list=None,
    ):
        """ "Compute regression and classification targets for a batch image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            masks_preds_list_thing  (list[Tensor]):
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        """
        assert (
            gt_bboxes_ignore_list is None
        ), "Only supports for gt_bboxes_ignore setting to None."
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [gt_bboxes_ignore_list for _ in range(num_imgs)]

        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            mask_targets_list,
            mask_weights_list,
            pos_inds_list,
            neg_inds_list,
        ) = multi_apply(
            self._get_target_single_with_mask,
            cls_scores_list,
            bbox_preds_list,
            masks_preds_list_thing,
            gt_bboxes_list,
            gt_labels_list,
            gt_masks_list,
            img_metas,
            gt_bboxes_ignore_list,
        )
        num_total_pos_thing = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg_thing = sum((inds.numel() for inds in neg_inds_list))
        return (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            mask_targets_list,
            mask_weights_list,
            num_total_pos_thing,
            num_total_neg_thing,
            pos_inds_list,
        )

    def _get_target_single_with_mask(
        self,
        cls_score,
        bbox_pred,
        masks_preds_things,
        gt_bboxes,
        gt_labels,
        gt_masks,
        img_meta,
        gt_bboxes_ignore=None,
    ):
        """ """

        num_bboxes = bbox_pred.size(0)

        gt_masks = gt_masks.float()

        assign_result = self.assigner_with_mask.assign(
            bbox_pred,
            cls_score,
            masks_preds_things,
            gt_bboxes,
            gt_labels,
            gt_masks,
            img_meta,
            gt_bboxes_ignore,
        )
        sampling_result = self.sampler_with_mask.sample(
            assign_result, bbox_pred, gt_bboxes, gt_masks
        )
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        labels = gt_bboxes.new_full(
            (num_bboxes,), self.num_things_classes, dtype=torch.long
        )
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        bbox_targets = torch.zeros_like(bbox_pred)
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0
        img_h, img_w, _ = img_meta["img_shape"]

        factor = bbox_pred.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)
        pos_gt_bboxes_normalized = sampling_result.pos_gt_bboxes / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        bbox_targets[pos_inds] = pos_gt_bboxes_targets

        mask_weights = masks_preds_things.new_zeros(num_bboxes)
        mask_weights[pos_inds] = 1.0
        pos_gt_masks = sampling_result.pos_gt_masks
        _, w, h = pos_gt_masks.shape
        mask_target = masks_preds_things.new_zeros([num_bboxes, w, h])
        mask_target[pos_inds] = pos_gt_masks

        return (
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            mask_target,
            mask_weights,
            pos_inds,
            neg_inds,
        )

    def forward_test(
        self,
        pts_feats=None,
        gt_lane_labels=None,
        gt_lane_masks=None,
        img_metas=None,
        rescale=False,
    ):
        bbox_list = [dict() for i in range(len(img_metas))]

        pred_seg_dict = self(pts_feats)
        results = self.get_bboxes(
            pred_seg_dict["outputs_classes"],
            pred_seg_dict["outputs_coords"],
            pred_seg_dict["enc_outputs_class"],
            pred_seg_dict["enc_outputs_coord"],
            pred_seg_dict["args_tuple"],
            pred_seg_dict["reference"],
            img_metas,
            rescale=rescale,
        )

        with torch.no_grad():
            drivable_pred = results[0]["drivable"]
            drivable_gt = gt_lane_masks[0][0, -1]

            drivable_iou, drivable_intersection, drivable_union = IOU(
                drivable_pred.view(1, -1), drivable_gt.view(1, -1)
            )

            lane_pred = results[0]["lane"]
            lanes_pred = (results[0]["lane"].sum(0) > 0).int()
            lanes_gt = (gt_lane_masks[0][0][:-1].sum(0) > 0).int()
            lanes_iou, lanes_intersection, lanes_union = IOU(
                lanes_pred.view(1, -1), lanes_gt.view(1, -1)
            )

            divider_gt = (
                gt_lane_masks[0][0][gt_lane_labels[0][0] == 0].sum(0) > 0
            ).int()
            crossing_gt = (
                gt_lane_masks[0][0][gt_lane_labels[0][0] == 1].sum(0) > 0
            ).int()
            contour_gt = (
                gt_lane_masks[0][0][gt_lane_labels[0][0] == 2].sum(0) > 0
            ).int()
            divider_iou, divider_intersection, divider_union = IOU(
                lane_pred[0].view(1, -1), divider_gt.view(1, -1)
            )
            crossing_iou, crossing_intersection, crossing_union = IOU(
                lane_pred[1].view(1, -1), crossing_gt.view(1, -1)
            )
            contour_iou, contour_intersection, contour_union = IOU(
                lane_pred[2].view(1, -1), contour_gt.view(1, -1)
            )

            ret_iou = {
                "drivable_intersection": drivable_intersection,
                "drivable_union": drivable_union,
                "lanes_intersection": lanes_intersection,
                "lanes_union": lanes_union,
                "divider_intersection": divider_intersection,
                "divider_union": divider_union,
                "crossing_intersection": crossing_intersection,
                "crossing_union": crossing_union,
                "contour_intersection": contour_intersection,
                "contour_union": contour_union,
                "drivable_iou": drivable_iou,
                "lanes_iou": lanes_iou,
                "divider_iou": divider_iou,
                "crossing_iou": crossing_iou,
                "contour_iou": contour_iou,
            }
        for result_dict, pts_bbox in zip(bbox_list, results):
            result_dict["pts_bbox"] = pts_bbox
            result_dict["ret_iou"] = ret_iou
            result_dict["args_tuple"] = pred_seg_dict["args_tuple"]
        return bbox_list

    def _get_bboxes_single(
        self, cls_score, bbox_pred, img_shape, scale_factor, rescale=False
    ):
        """ """
        assert len(cls_score) == len(bbox_pred)
        max_per_img = self.test_cfg.get("max_per_img", self.num_query)

        cls_score = cls_score.sigmoid()
        scores, indexes = cls_score.view(-1).topk(max_per_img)
        det_labels = indexes % self.num_things_classes
        bbox_index = indexes // self.num_things_classes
        bbox_pred = bbox_pred[bbox_index]

        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        if rescale:
            det_bboxes /= det_bboxes.new_tensor(scale_factor)
        det_bboxes = torch.cat((det_bboxes, scores.unsqueeze(1)), -1)

        return bbox_index, det_bboxes, det_labels

    def get_bboxes(
        self,
        all_cls_scores,
        all_bbox_preds,
        enc_cls_scores,
        enc_bbox_preds,
        args_tuple,
        reference,
        img_metas,
        rescale=False,
    ):
        """ """
        cls_scores = all_cls_scores[-1]
        bbox_preds = all_bbox_preds[-1]
        memory, memory_mask, memory_pos, query, _, query_pos, hw_lvl = args_tuple

        seg_list = []
        stuff_score_list = []
        panoptic_list = []
        bbox_list = []
        labels_list = []
        drivable_list = []
        lane_list = []
        lane_score_list = []
        score_list = []
        for img_id in range(len(img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]

            img_shape = (self.canvas_size[0], self.canvas_size[1], 3)
            ori_shape = (self.canvas_size[0], self.canvas_size[1], 3)
            scale_factor = 1

            index, bbox, labels = self._get_bboxes_single(
                cls_score, bbox_pred, img_shape, scale_factor, rescale
            )

            i = img_id
            thing_query = query[i : i + 1, index, :]
            thing_query_pos = query_pos[i : i + 1, index, :]
            joint_query = torch.cat(
                [thing_query, self.stuff_query.weight[None, :, : self.embed_dims]], 1
            )

            stuff_query_pos = self.stuff_query.weight[None, :, self.embed_dims :]

            mask_things, mask_inter_things, query_inter_things = self.things_mask_head(
                memory[i : i + 1],
                memory_mask[i : i + 1],
                None,
                joint_query[:, : -self.num_stuff_classes],
                None,
                None,
                hw_lvl=hw_lvl,
            )
            mask_stuff, mask_inter_stuff, query_inter_stuff = self.stuff_mask_head(
                memory[i : i + 1],
                memory_mask[i : i + 1],
                None,
                joint_query[:, -self.num_stuff_classes :],
                None,
                stuff_query_pos,
                hw_lvl=hw_lvl,
            )

            attn_map = torch.cat([mask_things, mask_stuff], 1)
            attn_map = attn_map.squeeze(-1)

            stuff_query = query_inter_stuff[-1]
            scores_stuff = (
                self.cls_stuff_branches[-1](stuff_query).sigmoid().reshape(-1)
            )

            mask_pred = attn_map.reshape(-1, *hw_lvl[0])

            mask_pred = F.interpolate(
                mask_pred.unsqueeze(0), size=ori_shape[:2], mode="bilinear"
            ).squeeze(0)

            masks_all = mask_pred
            score_list.append(masks_all)
            drivable_list.append(masks_all[-1] > 0.5)
            masks_all = masks_all[: -self.num_stuff_classes]
            seg_all = masks_all > 0.5
            sum_seg_all = seg_all.sum((1, 2)).float() + 1

            scores_all = bbox[:, -1]
            bboxes_all = bbox
            labels_all = labels

            seg_scores = (masks_all * seg_all.float()).sum((1, 2)) / sum_seg_all
            scores_all *= seg_scores**2

            scores_all, index = torch.sort(scores_all, descending=True)

            masks_all = masks_all[index]
            labels_all = labels_all[index]
            bboxes_all = bboxes_all[index]
            seg_all = seg_all[index]

            bboxes_all[:, -1] = scores_all

            things_selected = labels_all < self.num_things_classes
            stuff_selected = labels_all >= self.num_things_classes
            bbox_th = bboxes_all[things_selected][:100]
            labels_th = labels_all[things_selected][:100]
            seg_th = seg_all[things_selected][:100]
            labels_st = labels_all[stuff_selected]
            scores_st = scores_all[stuff_selected]
            masks_st = masks_all[stuff_selected]

            stuff_score_list.append(scores_st)

            results = torch.zeros((2, *mask_pred.shape[-2:])).to(torch.long)
            id_unique = 1
            lane = torch.zeros((self.num_things_classes, *mask_pred.shape[-2:])).to(
                torch.long
            )
            lane_score = torch.zeros(
                (self.num_things_classes, *mask_pred.shape[-2:])
            ).to(mask_pred.dtype)
            for i, scores in enumerate(scores_all):
                if (
                    labels_all[i] < self.num_things_classes
                    and scores < self.quality_threshold_things
                ):
                    continue
                elif (
                    labels_all[i] >= self.num_things_classes
                    and scores < self.quality_threshold_stuff
                ):
                    continue
                _mask = masks_all[i] > 0.5
                mask_area = _mask.sum().item()
                intersect = _mask & (results[0] > 0)
                intersect_area = intersect.sum().item()
                if labels_all[i] < self.num_things_classes:
                    if (
                        mask_area == 0
                        or (intersect_area * 1.0 / mask_area)
                        > self.overlap_threshold_things
                    ):
                        continue
                if intersect_area > 0:
                    _mask = _mask & (results[0] == 0)
                results[0, _mask] = labels_all[i]
                if labels_all[i] < self.num_things_classes:
                    lane[labels_all[i], _mask] = 1
                    lane_score[labels_all[i], _mask] = masks_all[i][_mask]
                    results[1, _mask] = id_unique
                    id_unique += 1

            panoptic_list.append((results.permute(1, 2, 0).numpy(), ori_shape))

            bbox_list.append(bbox_th)
            labels_list.append(labels_th)
            seg_list.append(seg_th)
            lane_list.append(lane)
            lane_score_list.append(lane_score)
        results = []
        for i in range(len(img_metas)):
            results.append(
                {
                    "bbox": bbox_list[i],
                    "segm": seg_list[i],
                    "labels": labels_list[i],
                    "panoptic": panoptic_list[i],
                    "drivable": drivable_list[i],
                    "score_list": score_list[i],
                    "lane": lane_list[i],
                    "lane_score": lane_score_list[i],
                    "stuff_score_list": stuff_score_list[i],
                }
            )
        return results


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        self.fp16_enabled = False
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)

        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SelfAttention(nn.Module):
    def __init__(
        self,
        cfg,
        dim,
        num_heads=2,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.fp16_enabled = False
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
            .contiguous()
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Attention(nn.Module):
    def __init__(
        self,
        cfg,
        dim,
        num_heads=2,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.fp16_enabled = False
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.linear_l1 = nn.Sequential(
            nn.Linear(self.num_heads, self.num_heads),
            nn.ReLU(),
        )
        self.linear = nn.Sequential(
            nn.Linear(self.num_heads, 1),
            nn.ReLU(),
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, query, key, value, key_padding_mask, hw_lvl):
        B, N, C = query.shape
        _, L, _ = key.shape
        q = (
            self.q(query)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        k = (
            self.k(key)
            .reshape(B, L, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        v = (
            self.v(value)
            .reshape(B, L, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        attn = (q @ k.transpose(-2, -1).contiguous()) * self.scale

        attn = attn.permute(0, 2, 3, 1)

        new_feats = self.linear_l1(attn)
        mask = self.linear(new_feats)

        attn = attn.permute(0, 3, 1, 2)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).contiguous().reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, mask


class AttentionTail(nn.Module):
    def __init__(
        self,
        cfg,
        dim,
        num_heads=2,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.fp16_enabled = False
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)

        self.linear_l1 = nn.Sequential(
            nn.Linear(self.num_heads, self.num_heads),
            nn.ReLU(),
        )

        self.linear = nn.Sequential(
            nn.Linear(self.num_heads, 1),
            nn.ReLU(),
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, query, key, key_padding_mask, hw_lvl=None):
        B, N, C = query.shape
        _, L, _ = key.shape
        q = (
            self.q(query)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        k = (
            self.k(key)
            .reshape(B, L, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        attn = (q @ k.transpose(-2, -1).contiguous()) * self.scale

        attn = attn.permute(0, 2, 3, 1)

        new_feats = self.linear_l1(attn)
        mask = self.linear(new_feats)
        return mask


class Block(nn.Module):
    def __init__(
        self,
        cfg,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        self_attn=False,
    ):
        super().__init__()
        self.fp16_enabled = False
        self.head_norm1 = norm_layer(dim)
        self.self_attn = self_attn
        self.attn = Attention(
            cfg,
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.head_norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        if self.self_attn:
            self.self_attention = SelfAttention(
                cfg,
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
            self.norm3 = norm_layer(dim)

    def forward(self, query, key, value, key_padding_mask=None, hw_lvl=None):
        if self.self_attn:
            query = query + self.drop_path(self.self_attention(query))
            query = self.norm3(query)
        x, mask = self.attn(query, key, value, key_padding_mask, hw_lvl=hw_lvl)
        query = query + self.drop_path(x)
        query = self.head_norm1(query)

        query = query + self.drop_path(self.mlp(query))
        query = self.head_norm2(query)
        return query, mask


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-53296self.num_heads956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class SegMaskHead(nn.Module):
    def __init__(
        self,
        cfg=None,
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=6,
        dim_feedforward=64,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
        self_attn=False,
    ):
        super().__init__()

        self.fp16_enabled = False
        mlp_ratio = 4
        qkv_bias = True
        qk_scale = None
        drop_rate = 0
        attn_drop_rate = 0

        norm_layer = None
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = None
        act_layer = act_layer or nn.GELU
        block = Block(
            cfg,
            dim=d_model,
            num_heads=nhead,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=0,
            norm_layer=norm_layer,
            act_layer=act_layer,
            self_attn=self_attn,
        )
        self.blocks = _get_clones(block, num_decoder_layers)
        self.attnen = AttentionTail(
            cfg,
            d_model,
            num_heads=nhead,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop_rate,
            proj_drop=0,
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        if pos is None:
            return tensor
        else:
            return tensor + pos

    def forward(
        self,
        memory,
        mask_memory,
        pos_memory,
        query_embed,
        mask_query,
        pos_query,
        hw_lvl,
    ):
        if mask_memory is not None and isinstance(mask_memory, torch.Tensor):
            mask_memory = mask_memory.to(torch.bool)
        masks = []
        inter_query = []
        for i, block in enumerate(self.blocks):
            query_embed, mask = block(
                self.with_pos_embed(query_embed, pos_query),
                self.with_pos_embed(memory, pos_memory),
                memory,
                key_padding_mask=mask_memory,
                hw_lvl=hw_lvl,
            )
            masks.append(mask)
            inter_query.append(query_embed)
        attn = self.attnen(
            self.with_pos_embed(query_embed, pos_query),
            self.with_pos_embed(memory, pos_memory),
            key_padding_mask=mask_memory,
            hw_lvl=hw_lvl,
        )
        return attn, masks, inter_query
