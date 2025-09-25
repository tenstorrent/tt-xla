# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import copy
from third_party.tt_forge_models.uniad.pytorch.src.uniad_track import UniADTrack
from third_party.tt_forge_models.uniad.pytorch.src.planning_head import (
    PlanningHeadSingleMode,
)
from third_party.tt_forge_models.uniad.pytorch.src.motion_head import MotionHead
from third_party.tt_forge_models.uniad.pytorch.src.occ_head import OccHead
from third_party.tt_forge_models.uniad.pytorch.src.panseg_head import PansegformerHead
from third_party.tt_forge_models.uniad.pytorch.src.utils import *
from third_party.tt_forge_models.uniad.pytorch.src.track_head import BEVFormerTrackHead


class UniAD(UniADTrack):
    """
    UniAD: Unifying Detection, Tracking, Segmentation, Motion Forecasting, Occupancy Prediction and Planning for Autonomous Driving
    """

    def __init__(
        self,
        seg_head=True,
        motion_head=True,
        occ_head=True,
        planning_head=True,
        task_loss_weight=dict(track=1.0, map=1.0, motion=1.0, occ=1.0, planning=1.0),
        filter_score_thresh=0.35,
        freeze_bev_encoder=True,
        freeze_bn=True,
        freeze_img_backbone=True,
        freeze_img_neck=True,
        gt_iou_threshold=0.3,
        img_backbone=ResNet(),
        img_neck=FPN(),
        num_classes=10,
        num_query=900,
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        pretrained=None,
        pts_bbox_head=BEVFormerTrackHead(),
        queue_length=3,
        score_thresh=0.4,
        test_cfg=None,
        use_grid_mask=True,
        vehicle_id_list=[0, 1, 2, 3, 4, 6, 7],
        video_test_mode=True,
    ):

        super(UniAD, self).__init__(
            filter_score_thresh=filter_score_thresh,
            freeze_bev_encoder=freeze_bev_encoder,
            freeze_bn=freeze_bn,
            freeze_img_backbone=freeze_img_backbone,
            freeze_img_neck=freeze_img_neck,
            gt_iou_threshold=gt_iou_threshold,
            img_backbone=img_backbone,
            img_neck=img_neck,
            num_classes=num_classes,
            num_query=num_query,
            pc_range=pc_range,
            pretrained=pretrained,
            pts_bbox_head=pts_bbox_head,
            queue_length=queue_length,
            score_thresh=score_thresh,
            test_cfg=test_cfg,
            use_grid_mask=use_grid_mask,
            vehicle_id_list=vehicle_id_list,
            video_test_mode=video_test_mode,
        )
        if seg_head:
            self.seg_head = PansegformerHead()

        if occ_head:
            self.occ_head = OccHead()

        if motion_head:
            self.motion_head = MotionHead()

        if planning_head:
            self.planning_head = PlanningHeadSingleMode()

        self.task_loss_weight = task_loss_weight
        assert set(task_loss_weight.keys()) == {
            "track",
            "occ",
            "motion",
            "map",
            "planning",
        }

    @property
    def with_planning_head(self):
        return hasattr(self, "planning_head") and self.planning_head is not None

    @property
    def with_occ_head(self):
        return hasattr(self, "occ_head") and self.occ_head is not None

    @property
    def with_motion_head(self):
        return hasattr(self, "motion_head") and self.motion_head is not None

    @property
    def with_seg_head(self):
        return hasattr(self, "seg_head") and self.seg_head is not None

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                kwargs[k] = v.to("cpu")
            elif isinstance(v, list):
                kwargs[k] = [
                    x.to("cpu") if isinstance(x, torch.Tensor) else x for x in v
                ]
        return self.forward_test(**kwargs)

    def loss_weighted_and_prefixed(self, loss_dict, prefix=""):
        loss_factor = self.task_loss_weight[prefix]
        loss_dict = {f"{prefix}.{k}": v * loss_factor for k, v in loss_dict.items()}
        return loss_dict

    def forward_test(
        self,
        img=None,
        img_metas=None,
        l2g_t=None,
        l2g_r_mat=None,
        timestamp=None,
        gt_lane_labels=None,
        gt_lane_masks=None,
        rescale=False,
        sdc_planning=None,
        sdc_planning_mask=None,
        command=None,
        gt_segmentation=None,
        gt_instance=None,
        gt_occ_img_is_valid=None,
        **kwargs,
    ):
        """Test function"""
        for var, name in [(img_metas, "img_metas")]:
            if not isinstance(var, list):
                raise TypeError("{} must be a list, but got {}".format(name, type(var)))
        img = [img] if img is None else img

        if img_metas[0][0]["scene_token"] != self.prev_frame_info["scene_token"]:
            self.prev_frame_info["prev_bev"] = None
        self.prev_frame_info["scene_token"] = img_metas[0][0]["scene_token"]

        self.prev_frame_info["prev_bev"] = None

        tmp_pos = copy.deepcopy(img_metas[0][0]["can_bus"][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]["can_bus"][-1])
        img_metas[0][0]["can_bus"][:3] -= self.prev_frame_info["prev_pos"]
        img_metas[0][0]["can_bus"][-1] -= self.prev_frame_info["prev_angle"]
        self.prev_frame_info["prev_pos"] = tmp_pos
        self.prev_frame_info["prev_angle"] = tmp_angle

        img = img[0]
        img_metas = img_metas[0]
        timestamp = timestamp[0] if timestamp is not None else None

        result = [dict() for i in range(len(img_metas))]
        result_track = self.simple_test_track(
            img, l2g_t, l2g_r_mat, img_metas, timestamp
        )
        result_track[0] = self.upsample_bev_if_tiny(result_track[0])
        bev_embed = result_track[0]["bev_embed"]

        if self.with_seg_head:
            result_seg = self.seg_head.forward_test(
                bev_embed, gt_lane_labels, gt_lane_masks, img_metas, rescale
            )

        if self.with_motion_head:
            result_motion, outs_motion = self.motion_head.forward_test(
                bev_embed, outs_track=result_track[0], outs_seg=result_seg[0]
            )
            outs_motion["bev_pos"] = result_track[0]["bev_pos"]

        outs_occ = dict()
        if self.with_occ_head:
            occ_no_query = outs_motion["track_query"].shape[1] == 0
            outs_occ = self.occ_head.forward_test(
                bev_embed,
                outs_motion,
                no_query=occ_no_query,
                gt_segmentation=gt_segmentation,
                gt_instance=gt_instance,
                gt_img_is_valid=gt_occ_img_is_valid,
            )
            result[0]["occ"] = outs_occ

        if self.with_planning_head:
            planning_gt = dict(
                segmentation=gt_segmentation,
                sdc_planning=sdc_planning,
                sdc_planning_mask=sdc_planning_mask,
                command=command,
            )
            result_planning = self.planning_head.forward_test(
                bev_embed, outs_motion, outs_occ, command
            )
            result[0]["planning"] = dict(
                planning_gt=planning_gt,
                result_planning=result_planning,
            )

        pop_track_list = [
            "prev_bev",
            "bev_pos",
            "bev_embed",
            "track_query_embeddings",
            "sdc_embedding",
        ]
        result_track[0] = pop_elem_in_result(result_track[0], pop_track_list)

        if self.with_seg_head:
            result_seg[0] = pop_elem_in_result(
                result_seg[0], pop_list=["pts_bbox", "args_tuple"]
            )
        if self.with_motion_head:
            result_motion[0] = pop_elem_in_result(result_motion[0])
        if self.with_occ_head:
            result[0]["occ"] = pop_elem_in_result(
                result[0]["occ"],
                pop_list=[
                    "seg_out_mask",
                    "flow_out",
                    "future_states_occ",
                    "pred_ins_masks",
                    "pred_raw_occ",
                    "pred_ins_logits",
                    "pred_ins_sigmoid",
                ],
            )

        for i, res in enumerate(result):
            res["token"] = img_metas[i]["sample_idx"]
            res.update(result_track[i])
            if self.with_motion_head:
                res.update(result_motion[i])
            if self.with_seg_head:
                res.update(result_seg[i])

        return result


def pop_elem_in_result(task_result: dict, pop_list: list = None):
    all_keys = list(task_result.keys())
    for k in all_keys:
        if k.endswith("query") or k.endswith("query_pos") or k.endswith("embedding"):
            task_result.pop(k)

    if pop_list is not None:
        for pop_k in pop_list:
            task_result.pop(pop_k, None)
    return task_result
