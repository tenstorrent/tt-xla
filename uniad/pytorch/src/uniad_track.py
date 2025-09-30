# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
import math
import warnings
from einops import rearrange
from abc import abstractmethod, ABCMeta
from third_party.tt_forge_models.uniad.pytorch.src.utils import FPN, ResNet
from third_party.tt_forge_models.uniad.pytorch.src.track_head import BEVFormerTrackHead
from third_party.tt_forge_models.uniad.pytorch.src.track_utils import *


class BaseDetector(nn.Module, metaclass=ABCMeta):
    """Base class for detectors."""

    def __init__(self):
        super(BaseDetector, self).__init__()
        self.fp16_enabled = False

    @property
    def with_neck(self):
        """bool: whether the detector has a neck"""
        return hasattr(self, "neck") and self.neck is not None

    @property
    def with_shared_head(self):
        """bool: whether the detector has a shared head in the RoI Head"""
        return hasattr(self, "roi_head") and self.roi_head.with_shared_head

    @property
    def with_bbox(self):
        """bool: whether the detector has a bbox head"""
        return (hasattr(self, "roi_head") and self.roi_head.with_bbox) or (
            hasattr(self, "bbox_head") and self.bbox_head is not None
        )

    @property
    def with_mask(self):
        """bool: whether the detector has a mask head"""
        return (hasattr(self, "roi_head") and self.roi_head.with_mask) or (
            hasattr(self, "mask_head") and self.mask_head is not None
        )

    def extract_feat(self, imgs):
        """Extract features from images."""
        pass

    def extract_feats(self, imgs):
        """Extract features from multiple images.

        Args:
            imgs (list[torch.Tensor]): A list of images. The images are
                augmented from the same image but in different ways.

        Returns:
            list[torch.Tensor]: Features of different images
        """
        assert isinstance(imgs, list)
        return [self.extract_feat(img) for img in imgs]

    def simple_test(self, img, img_metas, **kwargs):
        pass

    @abstractmethod
    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        pass

    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(imgs, "imgs"), (img_metas, "img_metas")]:
            if not isinstance(var, list):
                raise TypeError(f"{name} must be a list, but got {type(var)}")

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(
                f"num of augmentations ({len(imgs)}) "
                f"!= num of image meta ({len(img_metas)})"
            )

        for img, img_meta in zip(imgs, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]["batch_input_shape"] = tuple(img.size()[-2:])

    def forward(self, img, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.
        """

        return self.forward_test(img, img_metas, **kwargs)


class Base3DDetector(BaseDetector):
    """Base class for detectors."""

    def forward_test(self, points, img_metas, img=None, **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        for var, name in [(points, "points"), (img_metas, "img_metas")]:
            if not isinstance(var, list):
                raise TypeError("{} must be a list, but got {}".format(name, type(var)))

        num_augs = len(points)
        if num_augs != len(img_metas):
            raise ValueError(
                "num of augmentations ({}) != num of image meta ({})".format(
                    len(points), len(img_metas)
                )
            )

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

        return self.forward_test(**kwargs)


class MVXTwoStageDetector(Base3DDetector):
    """Base class of Multi-modality VoxelNet."""

    def __init__(
        self,
        pts_voxel_layer=None,
        pts_voxel_encoder=None,
        pts_middle_encoder=None,
        pts_fusion_layer=None,
        img_backbone=True,
        pts_backbone=None,
        img_neck=True,
        pts_neck=None,
        pts_bbox_head=True,
        img_roi_head=None,
        img_rpn_head=None,
        test_cfg=None,
        pretrained=None,
    ):
        super(MVXTwoStageDetector, self).__init__()

        if pts_bbox_head:
            self.pts_bbox_head = BEVFormerTrackHead()
        if img_backbone:
            self.img_backbone = ResNet()
        if img_neck is not None:
            self.img_neck = FPN()

        self.test_cfg = test_cfg

    @property
    def with_img_backbone(self):
        """bool: Whether the detector has a 2D image backbone."""
        return hasattr(self, "img_backbone") and self.img_backbone is not None

    @property
    def with_pts_backbone(self):
        """bool: Whether the detector has a 3D backbone."""
        return hasattr(self, "pts_backbone") and self.pts_backbone is not None

    @property
    def with_img_neck(self):
        """bool: Whether the detector has a neck in image branch."""
        return hasattr(self, "img_neck") and self.img_neck is not None

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        img_feats, pts_feats = self.extract_feats(points, img_metas, imgs)

        bbox_list = dict()

        return [bbox_list]

    def extract_feats(self, points, img_metas, imgs=None):
        """Extract point and image features of multiple samples."""
        if imgs is None:
            imgs = [None] * len(img_metas)
        img_feats, pts_feats = multi_apply(self.extract_feat, points, imgs, img_metas)
        return img_feats, pts_feats


class UniADTrack(MVXTwoStageDetector):
    """UniAD tracking part"""

    def __init__(
        self,
        use_grid_mask=True,
        img_backbone=True,
        img_neck=True,
        pts_bbox_head=True,
        test_cfg=None,
        pretrained=None,
        video_test_mode=False,
        qim_args=None,
        mem_args=None,
        bbox_coder=DETRTrack3DCoder(),
        pc_range=None,
        embed_dims=256,
        num_query=900,
        num_classes=10,
        vehicle_id_list=None,
        score_thresh=0.2,
        filter_score_thresh=0.1,
        miss_tolerance=5,
        gt_iou_threshold=0.0,
        freeze_img_backbone=False,
        freeze_img_neck=False,
        freeze_bn=False,
        freeze_bev_encoder=False,
        queue_length=3,
    ):
        super(UniADTrack, self).__init__(
            img_backbone=img_backbone,
            img_neck=img_neck,
            pts_bbox_head=pts_bbox_head,
            test_cfg=test_cfg,
            pretrained=pretrained,
        )

        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
        )
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False
        self.embed_dims = embed_dims
        self.num_query = num_query
        self.num_classes = num_classes
        self.vehicle_id_list = vehicle_id_list
        self.pc_range = pc_range
        self.queue_length = queue_length
        if freeze_img_backbone:
            if freeze_bn:
                self.img_backbone.eval()
            for param in self.img_backbone.parameters():
                param.requires_grad = False

        if freeze_img_neck:
            if freeze_bn:
                self.img_neck.eval()
            for param in self.img_neck.parameters():
                param.requires_grad = False
        self.prev_frame_info = {
            "prev_bev": None,
            "scene_token": None,
            "prev_pos": 0,
            "prev_angle": 0,
        }
        self.query_embedding = nn.Embedding(self.num_query + 1, self.embed_dims * 2)
        self.reference_points = nn.Linear(self.embed_dims, 3)

        self.track_base = RuntimeTrackerBase(
            score_thresh=score_thresh,
            filter_score_thresh=filter_score_thresh,
            miss_tolerance=miss_tolerance,
        )

        self.query_interact = QueryInteractionModule(
            qim_args,
            dim_in=embed_dims,
            hidden_dim=embed_dims,
            dim_out=embed_dims,
        )

        self.bbox_coder = DETRTrack3DCoder()

        self.memory_bank = MemoryBank(
            mem_args,
            dim_in=embed_dims,
            hidden_dim=embed_dims,
            dim_out=embed_dims,
        )
        self.mem_bank_len = (
            0 if self.memory_bank is None else self.memory_bank.max_his_length
        )

        self.test_track_instances = None
        self.l2g_r_mat = None
        self.l2g_t = None
        self.gt_iou_threshold = gt_iou_threshold
        self.bev_h, self.bev_w = 200, 200
        self.freeze_bev_encoder = freeze_bev_encoder

    def extract_img_feat(self, img, len_queue=None):
        """Extract features of images."""
        if img is None:
            return None
        assert img[0].dim() == 5
        B, N, C, H, W = img[0].size()
        img = img[0].reshape(B * N, C, H, W)
        if self.use_grid_mask:
            img = self.grid_mask(img)
        img_feats = self.img_backbone(img)
        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            _, c, h, w = img_feat.size()
            img_feat_reshaped = img_feat.view(B, N, c, h, w)
            img_feats_reshaped.append(img_feat_reshaped)
        return img_feats_reshaped

    def _generate_empty_tracks(self):
        num_queries, dim = self.query_embedding.weight.shape
        query = self.query_embedding.weight
        track_instances = Instances((1, 1), query=query)
        track_instances.ref_pts = self.reference_points(query[..., : dim // 2])
        pred_boxes_init = torch.zeros((len(track_instances), 10), dtype=torch.float)
        track_instances.query = query
        track_instances.output_embedding = torch.zeros((num_queries, dim >> 1))
        track_instances.obj_idxes = torch.full(
            (len(track_instances),), -1, dtype=torch.long
        )
        track_instances.matched_gt_idxes = torch.full(
            (len(track_instances),), -1, dtype=torch.long
        )
        track_instances.disappear_time = torch.zeros(
            (len(track_instances),), dtype=torch.long
        )
        track_instances.iou = torch.zeros((len(track_instances),), dtype=torch.float)
        track_instances.scores = torch.zeros((len(track_instances),), dtype=torch.float)
        track_instances.track_scores = torch.zeros(
            (len(track_instances),), dtype=torch.float
        )
        track_instances.pred_boxes = pred_boxes_init
        track_instances.pred_logits = torch.zeros(
            (len(track_instances), self.num_classes), dtype=torch.float
        )
        mem_bank_len = self.mem_bank_len
        track_instances.mem_bank = torch.zeros(
            (len(track_instances), mem_bank_len, dim // 2), dtype=torch.float32
        )
        track_instances.mem_padding_mask = torch.ones(
            (len(track_instances), mem_bank_len), dtype=torch.bool
        )
        track_instances.save_period = torch.zeros(
            (len(track_instances),), dtype=torch.float32
        )
        track_instances = track_instances
        return track_instances

    def get_bevs(
        self, imgs, img_metas, prev_img=None, prev_img_metas=None, prev_bev=None
    ):
        if prev_img is not None and prev_img_metas is not None:
            assert prev_bev is None
            if isinstance(prev_img, torch.Tensor) and prev_img.dim() == 4:
                prev_img = [prev_img.unsqueeze(0)]
            prev_bev = self.get_history_bev(prev_img, prev_img_metas)

        if isinstance(imgs, torch.Tensor) and imgs.dim() == 5:
            imgs = [imgs]
        img_feats = self.extract_img_feat(img=imgs)
        if self.freeze_bev_encoder:
            with torch.no_grad():
                bev_embed, bev_pos = self.pts_bbox_head.get_bev_features(
                    mlvl_feats=img_feats, img_metas=img_metas, prev_bev=prev_bev
                )

        if bev_embed.shape[1] == self.bev_h * self.bev_w:
            bev_embed = bev_embed.permute(1, 0, 2)

        return bev_embed, bev_pos

    def select_active_track_query(
        self, track_instances, active_index, img_metas, with_mask=True
    ):
        result_dict = self._track_instances2results(
            track_instances[active_index], img_metas, with_mask=with_mask
        )
        result_dict["track_query_embeddings"] = track_instances.output_embedding[
            active_index
        ][result_dict["bbox_index"]][result_dict["mask"]]
        result_dict["track_query_matched_idxes"] = track_instances.matched_gt_idxes[
            active_index
        ][result_dict["bbox_index"]][result_dict["mask"]]
        return result_dict

    def select_sdc_track_query(self, sdc_instance, img_metas):
        out = dict()
        result_dict = self._track_instances2results(
            sdc_instance, img_metas, with_mask=False
        )
        out["sdc_boxes_3d"] = result_dict["boxes_3d"]
        out["sdc_scores_3d"] = result_dict["scores_3d"]
        out["sdc_track_scores"] = result_dict["track_scores"]
        out["sdc_track_bbox_results"] = result_dict["track_bbox_results"]
        out["sdc_embedding"] = sdc_instance.output_embedding[0]
        return out

    def upsample_bev_if_tiny(self, outs_track):
        if outs_track["bev_embed"].size(0) == 100 * 100:
            bev_embed = outs_track["bev_embed"]
            dim, _, _ = bev_embed.size()
            w = h = int(math.sqrt(dim))
            assert h == w == 100

            bev_embed = rearrange(bev_embed, "(h w) b c -> b c h w", h=h, w=w)
            bev_embed = nn.Upsample(scale_factor=2)(bev_embed)
            bev_embed = rearrange(bev_embed, "b c h w -> (h w) b c")
            outs_track["bev_embed"] = bev_embed

            prev_bev = outs_track.get("prev_bev", None)

            bev_pos = outs_track["bev_pos"]
            bev_pos = nn.Upsample(scale_factor=2)(bev_pos)
            outs_track["bev_pos"] = bev_pos
        return outs_track

    def _forward_single_frame_inference(
        self,
        img,
        img_metas,
        track_instances,
        prev_bev=None,
        l2g_r1=None,
        l2g_t1=None,
        l2g_r2=None,
        l2g_t2=None,
        time_delta=None,
    ):
        """
        img: B, num_cam, C, H, W = img.shape
        """

        """ velo update """
        active_inst = track_instances[track_instances.obj_idxes >= 0]
        other_inst = track_instances[track_instances.obj_idxes < 0]

        track_instances = Instances.cat([other_inst, active_inst])
        bev_embed, bev_pos = self.get_bevs(img, img_metas, prev_bev=prev_bev)

        det_output = self.pts_bbox_head.get_detections(
            bev_embed,
            object_query_embeds=track_instances.query,
            ref_points=track_instances.ref_pts,
            img_metas=img_metas,
        )

        output_classes = det_output["all_cls_scores"]
        output_coords = det_output["all_bbox_preds"]
        last_ref_pts = det_output["last_ref_points"]
        query_feats = det_output["query_feats"]

        out = {
            "pred_logits": output_classes,
            "pred_boxes": output_coords,
            "ref_pts": last_ref_pts,
            "bev_embed": bev_embed,
            "query_embeddings": query_feats,
            "all_past_traj_preds": det_output["all_past_traj_preds"],
            "bev_pos": bev_pos,
        }

        """ update track instances with predict results """
        track_scores = output_classes[-1, 0, :].sigmoid().max(dim=-1).values
        track_instances.scores = track_scores
        track_instances.pred_logits = output_classes[-1, 0]
        track_instances.pred_boxes = output_coords[-1, 0]
        track_instances.output_embedding = query_feats[-1][0]
        track_instances.ref_pts = last_ref_pts[0]
        track_instances.obj_idxes[900] = -2
        self.track_base.update(track_instances, None)
        self.track_base.filter_score_thresh = 0.35
        active_index = (track_instances.obj_idxes >= 0) & (
            track_instances.scores >= self.track_base.filter_score_thresh
        )
        out.update(
            self.select_active_track_query(track_instances, active_index, img_metas)
        )
        out.update(
            self.select_sdc_track_query(
                track_instances[track_instances.obj_idxes == -2], img_metas
            )
        )

        """ update with memory_bank """
        if self.memory_bank is not None:
            track_instances = self.memory_bank(track_instances)

        """  Update track instances using matcher """
        tmp = {}
        tmp["init_track_instances"] = self._generate_empty_tracks()
        tmp["track_instances"] = track_instances
        out_track_instances = self.query_interact(tmp)
        out["track_instances_fordet"] = track_instances
        out["track_instances"] = out_track_instances
        out["track_obj_idxes"] = track_instances.obj_idxes
        return out

    def simple_test_track(
        self,
        img=None,
        l2g_t=None,
        l2g_r_mat=None,
        img_metas=None,
        timestamp=None,
    ):
        """only support bs=1 and sequential input"""
        bs = img[0].size(0)

        """ init track instances for first frame """
        if (
            self.test_track_instances is None
            or img_metas[0]["scene_token"] != self.scene_token
        ):
            self.timestamp = timestamp
            self.scene_token = img_metas[0]["scene_token"]
            self.prev_bev = None
        track_instances = self._generate_empty_tracks()
        time_delta, l2g_r1, l2g_t1, l2g_r2, l2g_t2 = None, None, None, None, None

        """ get time_delta and l2g r/t infos """
        """ update frame info for next frame"""
        self.timestamp = timestamp
        self.l2g_t = l2g_t
        self.l2g_r_mat = l2g_r_mat

        """ predict and update """
        prev_bev = self.prev_bev
        frame_res = self._forward_single_frame_inference(
            img,
            img_metas,
            track_instances,
            prev_bev,
            l2g_r1,
            l2g_t1,
            l2g_r2,
            l2g_t2,
            time_delta,
        )

        self.prev_bev = frame_res["bev_embed"]
        track_instances = frame_res["track_instances"]
        track_instances_fordet = frame_res["track_instances_fordet"]
        self.test_track_instances = track_instances
        results = [dict()]
        get_keys = [
            "bev_embed",
            "bev_pos",
            "track_query_embeddings",
            "track_bbox_results",
            "boxes_3d",
            "scores_3d",
            "labels_3d",
            "track_scores",
            "track_ids",
        ]
        if self.with_motion_head:
            get_keys += [
                "sdc_boxes_3d",
                "sdc_scores_3d",
                "sdc_track_scores",
                "sdc_track_bbox_results",
                "sdc_embedding",
            ]
        results[0].update({k: frame_res[k] for k in get_keys})
        results = self._det_instances2results(
            track_instances_fordet, results, img_metas
        )
        return results

    def _track_instances2results(self, track_instances, img_metas, with_mask=True):
        bbox_dict = dict(
            cls_scores=track_instances.pred_logits,
            bbox_preds=track_instances.pred_boxes,
            track_scores=track_instances.scores,
            obj_idxes=track_instances.obj_idxes,
        )
        bboxes_dict = self.bbox_coder.decode(
            bbox_dict, with_mask=with_mask, img_metas=img_metas
        )[0]
        bboxes = bboxes_dict["bboxes"]
        bboxes = img_metas[0]["box_type_3d"](bboxes, 9)
        labels = bboxes_dict["labels"]
        scores = bboxes_dict["scores"]
        bbox_index = bboxes_dict["bbox_index"]

        track_scores = bboxes_dict["track_scores"]
        obj_idxes = bboxes_dict["obj_idxes"]
        result_dict = dict(
            boxes_3d=bboxes,
            scores_3d=scores,
            labels_3d=labels,
            track_scores=track_scores,
            bbox_index=bbox_index,
            track_ids=obj_idxes,
            mask=bboxes_dict["mask"],
            track_bbox_results=[
                [
                    bboxes,
                    scores,
                    labels,
                    bbox_index,
                    bboxes_dict["mask"],
                ]
            ],
        )
        return result_dict

    def _det_instances2results(self, instances, results, img_metas):
        """
        Outs:
        active_instances. keys:
        - 'pred_logits':
        - 'pred_boxes': normalized bboxes
        - 'scores'
        - 'obj_idxes'
        out_dict. keys:
            - boxes_3d (torch.Tensor): 3D boxes.
            - scores (torch.Tensor): Prediction scores.
            - labels_3d (torch.Tensor): Box labels.
            - attrs_3d (torch.Tensor, optional): Box attributes.
            - track_ids
            - tracking_score
        """
        if instances.pred_logits.numel() == 0:
            return [None]
        bbox_dict = dict(
            cls_scores=instances.pred_logits,
            bbox_preds=instances.pred_boxes,
            track_scores=instances.scores,
            obj_idxes=instances.obj_idxes,
        )
        bboxes_dict = self.bbox_coder.decode(bbox_dict, img_metas=img_metas)[0]
        bboxes = bboxes_dict["bboxes"]
        bboxes = img_metas[0]["box_type_3d"](bboxes, 9)
        labels = bboxes_dict["labels"]
        scores = bboxes_dict["scores"]

        track_scores = bboxes_dict["track_scores"]
        obj_idxes = bboxes_dict["obj_idxes"]
        result_dict = results[0]
        result_dict_det = dict(
            boxes_3d_det=bboxes,
            scores_3d_det=scores,
            labels_3d_det=labels,
        )
        result_dict.update(result_dict_det)
        return [result_dict]
