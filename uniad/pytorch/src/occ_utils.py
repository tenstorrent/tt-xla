# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Optional
import torch.utils.checkpoint as checkpoint
import torch
from torchmetrics import Metric
from torchmetrics.functional.classification import multiclass_stat_scores
from torchmetrics.utilities.data import dim_zero_mean
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from collections import OrderedDict
from third_party.tt_forge_models.uniad.pytorch.src.utils import *
from third_party.tt_forge_models.uniad.pytorch.src.transformer import *


class DetrTransformerDecoderLayer(BaseTransformerLayer):
    """Decoder layer in DETR transformer."""

    def __init__(
        self,
        attn_cfgs={
            "type": "MultiheadAttention",
            "embed_dims": 256,
            "num_heads": 8,
            "attn_drop": 0.0,
            "proj_drop": 0.0,
            "dropout_layer": None,
            "batch_first": False,
        },
        ffn_cfgs=dict(
            type="FFN",
            embed_dims=256,
            feedforward_channels=2048,
            num_fcs=2,
            ffn_drop=0.0,
            act_cfg=dict(type="ReLU", inplace=True),
        ),
        operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
        norm_cfg=dict(type="LN"),
        **kwargs
    ):

        super().__init__(
            attn_cfgs=attn_cfgs,
            ffn_cfgs=ffn_cfgs,
            operation_order=operation_order,
            norm_cfg=norm_cfg,
            **kwargs
        )

        assert len(operation_order) == 6
        assert set(operation_order) == set(["self_attn", "norm", "cross_attn", "ffn"])
        self.num_attn = sum([op.endswith("attn") for op in operation_order])


class DetrTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR transformer.

    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        post_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(
        self,
        *args,
        post_norm_cfg=dict(type="LN"),
        return_intermediate=False,
        num_layers=5,
        **kwargs
    ):

        super().__init__(
            transformerlayers=DetrTransformerDecoderLayer, num_layers=num_layers
        )
        self.return_intermediate = return_intermediate
        self.post_norm = torch.nn.LayerNorm((256,), eps=1e-05, elementwise_affine=True)

    def forward(self, query, *args, **kwargs):
        """Forward function for `TransformerDecoder`.

        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.

        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        if not self.return_intermediate:
            x = super().forward(query, *args, **kwargs)
            if self.post_norm:
                x = self.post_norm(x)[None]
            return x

        intermediate = []
        for layer in self.layers:
            query = layer(query, *args, **kwargs)

        return torch.stack(intermediate)


def calculate_birds_eye_view_parameters(x_bounds, y_bounds, z_bounds):
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
    bev_resolution = torch.tensor([row[2] for row in [x_bounds, y_bounds, z_bounds]])
    bev_start_position = torch.tensor(
        [row[0] + row[2] / 2.0 for row in [x_bounds, y_bounds, z_bounds]]
    )
    bev_dimension = torch.tensor(
        [(row[1] - row[0]) / row[2] for row in [x_bounds, y_bounds, z_bounds]],
        dtype=torch.long,
    )

    return bev_resolution, bev_start_position, bev_dimension


class IntersectionOverUnion(Metric):
    """Computes intersection-over-union."""

    def __init__(
        self,
        n_classes: int,
        ignore_index: Optional[int] = None,
        absent_score: float = 0.0,
        reduction: str = "none",
        compute_on_step: bool = False,
    ):
        super().__init__(compute_on_step=compute_on_step)

        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.absent_score = absent_score
        self.reduction = reduction

        self.add_state(
            "true_positive", default=torch.zeros(n_classes), dist_reduce_fx="sum"
        )
        self.add_state(
            "false_positive", default=torch.zeros(n_classes), dist_reduce_fx="sum"
        )
        self.add_state(
            "false_negative", default=torch.zeros(n_classes), dist_reduce_fx="sum"
        )
        self.add_state("support", default=torch.zeros(n_classes), dist_reduce_fx="sum")

    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        tps, fps, _, fns, sups = multiclass_stat_scores(
            prediction, target, self.n_classes
        )

        self.true_positive += tps
        self.false_positive += fps
        self.false_negative += fns
        self.support += sups

    def compute(self):
        scores = torch.zeros(self.n_classes, dtype=torch.float32)

        for class_idx in range(self.n_classes):
            if class_idx == self.ignore_index:
                continue

            tp = self.true_positive[class_idx]
            fp = self.false_positive[class_idx]
            fn = self.false_negative[class_idx]
            sup = self.support[class_idx]

            if sup + tp + fp == 0:
                scores[class_idx] = self.absent_score
                continue

            denominator = tp + fp + fn
            score = tp.to(torch.float) / denominator
            scores[class_idx] = score

        if (self.ignore_index is not None) and (
            0 <= self.ignore_index < self.n_classes
        ):
            scores = torch.cat(
                [scores[: self.ignore_index], scores[self.ignore_index + 1 :]]
            )

        return dim_zero_mean(scores)


class PanopticMetric(Metric):
    def __init__(
        self,
        n_classes: int,
        temporally_consistent: bool = True,
        vehicles_id: int = 1,
        compute_on_step: bool = False,
    ):
        super().__init__(compute_on_step=compute_on_step)

        self.n_classes = n_classes
        self.temporally_consistent = temporally_consistent
        self.vehicles_id = vehicles_id
        self.keys = ["iou", "true_positive", "false_positive", "false_negative"]

        self.add_state("iou", default=torch.zeros(n_classes), dist_reduce_fx="sum")
        self.add_state(
            "true_positive", default=torch.zeros(n_classes), dist_reduce_fx="sum"
        )
        self.add_state(
            "false_positive", default=torch.zeros(n_classes), dist_reduce_fx="sum"
        )
        self.add_state(
            "false_negative", default=torch.zeros(n_classes), dist_reduce_fx="sum"
        )

    def update(self, pred_instance, gt_instance):
        """
        Update state with predictions and targets.

        Parameters
        ----------
            pred_instance: (b, s, h, w)
                Temporally consistent instance segmentation prediction.
            gt_instance: (b, s, h, w)
                Ground truth instance segmentation.
        """
        batch_size, sequence_length = gt_instance.shape[:2]
        assert gt_instance.min() == 0, "ID 0 of gt_instance must be background"
        pred_segmentation = (pred_instance > 0).long()
        gt_segmentation = (gt_instance > 0).long()

        for b in range(batch_size):
            unique_id_mapping = {}
            for t in range(sequence_length):
                result = self.panoptic_metrics(
                    pred_segmentation[b, t].detach(),
                    pred_instance[b, t].detach(),
                    gt_segmentation[b, t],
                    gt_instance[b, t],
                    unique_id_mapping,
                )

                self.iou += result["iou"]
                self.true_positive += result["true_positive"]
                self.false_positive += result["false_positive"]
                self.false_negative += result["false_negative"]

    def compute(self):
        denominator = torch.maximum(
            (self.true_positive + self.false_positive / 2 + self.false_negative / 2),
            torch.ones_like(self.true_positive),
        )
        pq = self.iou / denominator
        sq = self.iou / torch.maximum(
            self.true_positive, torch.ones_like(self.true_positive)
        )
        rq = self.true_positive / denominator

        return {
            "pq": pq,
            "sq": sq,
            "rq": rq,
            "denominator": (
                self.true_positive + self.false_positive / 2 + self.false_negative / 2
            ),
        }

    def panoptic_metrics(
        self,
        pred_segmentation,
        pred_instance,
        gt_segmentation,
        gt_instance,
        unique_id_mapping,
    ):
        """
        Computes panoptic quality metric components.

        Parameters
        ----------
            pred_segmentation: [H, W] range {0, ..., n_classes-1} (>= n_classes is void)
            pred_instance: [H, W] range {0, ..., n_instances} (zero means background)
            gt_segmentation: [H, W] range {0, ..., n_classes-1} (>= n_classes is void)
            gt_instance: [H, W] range {0, ..., n_instances} (zero means background)
            unique_id_mapping: instance id mapping to check consistency
        """
        n_classes = self.n_classes

        result = {key: torch.zeros(n_classes, dtype=torch.float32) for key in self.keys}

        assert pred_segmentation.dim() == 2
        assert (
            pred_segmentation.shape
            == pred_instance.shape
            == gt_segmentation.shape
            == gt_instance.shape
        )

        n_instances = int(torch.cat([pred_instance, gt_instance]).max().item())
        n_all_things = n_instances + n_classes
        n_things_and_void = n_all_things + 1

        prediction, pred_to_cls = self.combine_mask(
            pred_segmentation, pred_instance, n_classes, n_all_things
        )
        target, target_to_cls = self.combine_mask(
            gt_segmentation, gt_instance, n_classes, n_all_things
        )
        x = prediction + n_things_and_void * target
        bincount_2d = torch.bincount(x.long(), minlength=n_things_and_void**2)
        if bincount_2d.shape[0] != n_things_and_void**2:
            raise ValueError("Incorrect bincount size.")
        conf = bincount_2d.reshape((n_things_and_void, n_things_and_void))
        conf = conf[1:, 1:]

        union = conf.sum(0).unsqueeze(0) + conf.sum(1).unsqueeze(1) - conf
        iou = torch.where(
            union > 0,
            (conf.float() + 1e-9) / (union.float() + 1e-9),
            torch.zeros_like(union).float(),
        )

        mapping = (iou > 0.5).nonzero(as_tuple=False)

        is_matching = pred_to_cls[mapping[:, 1]] == target_to_cls[mapping[:, 0]]
        mapping = mapping[is_matching]
        tp_mask = torch.zeros_like(conf, dtype=torch.bool)
        tp_mask[mapping[:, 0], mapping[:, 1]] = True

        for target_id, pred_id in mapping:
            cls_id = pred_to_cls[pred_id]

            if self.temporally_consistent and cls_id == self.vehicles_id:
                if (
                    target_id.item() in unique_id_mapping
                    and unique_id_mapping[target_id.item()] != pred_id.item()
                ):
                    result["false_negative"][target_to_cls[target_id]] += 1
                    result["false_positive"][pred_to_cls[pred_id]] += 1
                    unique_id_mapping[target_id.item()] = pred_id.item()
                    continue

            result["true_positive"][cls_id] += 1
            result["iou"][cls_id] += iou[target_id][pred_id]
            unique_id_mapping[target_id.item()] = pred_id.item()

        for target_id in range(n_classes, n_all_things):
            if tp_mask[target_id, n_classes:].any():
                continue
            if target_to_cls[target_id] != -1:
                result["false_negative"][target_to_cls[target_id]] += 1

        for pred_id in range(n_classes, n_all_things):
            if tp_mask[n_classes:, pred_id].any():
                continue
            if pred_to_cls[pred_id] != -1 and (conf[:, pred_id] > 0).any():
                result["false_positive"][pred_to_cls[pred_id]] += 1

        return result

    def combine_mask(
        self,
        segmentation: torch.Tensor,
        instance: torch.Tensor,
        n_classes: int,
        n_all_things: int,
    ):
        """Shifts all things ids by num_classes and combines things and stuff into a single mask

        Returns a combined mask + a mapping from id to segmentation class.
        """
        instance = instance.view(-1)
        instance_mask = instance > 0
        instance = instance - 1 + n_classes

        segmentation = segmentation.clone().view(-1)
        segmentation_mask = segmentation < n_classes

        instance_id_to_class_tuples = torch.cat(
            (
                instance[instance_mask & segmentation_mask].unsqueeze(1),
                segmentation[instance_mask & segmentation_mask].unsqueeze(1),
            ),
            dim=1,
        )
        instance_id_to_class = -instance_id_to_class_tuples.new_ones((n_all_things,))
        instance_id_to_class[
            instance_id_to_class_tuples[:, 0]
        ] = instance_id_to_class_tuples[:, 1]
        instance_id_to_class[torch.arange(n_classes)] = torch.arange(n_classes)

        segmentation[instance_mask] = instance[instance_mask]
        segmentation += 1
        segmentation[~segmentation_mask] = 0

        return segmentation, instance_id_to_class


class BevFeatureSlicer(nn.Module):
    def __init__(self, grid_conf, map_grid_conf):
        super().__init__()
        self.identity_mapping = False
        (
            bev_resolution,
            bev_start_position,
            bev_dimension,
        ) = calculate_birds_eye_view_parameters(
            grid_conf["xbound"], grid_conf["ybound"], grid_conf["zbound"]
        )

        (
            map_bev_resolution,
            map_bev_start_position,
            map_bev_dimension,
        ) = calculate_birds_eye_view_parameters(
            map_grid_conf["xbound"],
            map_grid_conf["ybound"],
            map_grid_conf["zbound"],
        )

        self.map_x = torch.arange(
            map_bev_start_position[0],
            map_grid_conf["xbound"][1],
            map_bev_resolution[0],
        )

        self.map_y = torch.arange(
            map_bev_start_position[1],
            map_grid_conf["ybound"][1],
            map_bev_resolution[1],
        )

        self.norm_map_x = self.map_x / (-bev_start_position[0])
        self.norm_map_y = self.map_y / (-bev_start_position[1])

        tmp_m, tmp_n = torch.meshgrid(self.norm_map_x, self.norm_map_y)
        tmp_m, tmp_n = tmp_m.T, tmp_n.T

        self.map_grid = torch.stack([tmp_m, tmp_n], dim=2)

    def forward(self, x):
        grid = self.map_grid.unsqueeze(0).type_as(x).repeat(x.shape[0], 1, 1, 1)
        return F.grid_sample(x, grid=grid, mode="bilinear", align_corners=True)


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class ConvModule(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activate = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activate(x)
        return x


class SimpleConv2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, conv_channels=64, num_conv=1, bias=False
    ):
        super().__init__()
        self.out_channels = out_channels
        if num_conv == 1:
            conv_channels = in_channels

        conv_layers = []
        c_in = in_channels
        for i in range(num_conv - 1):
            conv_layers.append(
                ConvModule(
                    in_channels=c_in,
                    out_channels=conv_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=bias,
                )
            )
            c_in = conv_channels

        conv_layers.append(
            nn.Conv2d(
                in_channels=conv_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
        )

        self.conv_layers = nn.Sequential(*conv_layers)

    def forward(self, x):
        b, c_in, h_in, w_in = x.size()
        out = self.conv_layers(x)
        assert out.size() == (b, self.out_channels, h_in, w_in)
        return out


class CVT_DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        skip_dim,
        residual,
        factor,
        upsample,
        with_relu=True,
    ):
        super().__init__()

        dim = out_channels // factor

        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, out_channels, 1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.up = nn.Conv2d(skip_dim, out_channels, 1)

        self.with_relu = with_relu
        if self.with_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = self.conv(x)

        if self.up is not None:
            up = self.up(skip)
            up = F.interpolate(up, x.shape[-2:])

            x = x + up
        if self.with_relu:
            return self.relu(x)
        return x


class CVT_Decoder(nn.Module):
    def __init__(
        self, dim, blocks, residual=True, factor=2, upsample=True, use_checkpoint=False
    ):
        super().__init__()

        layers = []
        channels = dim

        for i, out_channels in enumerate(blocks):
            with_relu = i < len(blocks) - 1
            layer = CVT_DecoderBlock(
                channels,
                out_channels,
                dim,
                residual,
                factor,
                upsample,
                with_relu=with_relu,
            )
            layers.append(layer)

            channels = out_channels

        self.layers = nn.Sequential(*layers)
        self.out_channels = channels
        self.use_checkpoint = use_checkpoint

    def forward(self, x):
        b, t = x.size(0), x.size(1)
        x = rearrange(x, "b t c h w -> (b t) c h w")
        y = x
        for layer in self.layers:
            if self.use_checkpoint:
                y = checkpoint(layer, y, x)
            else:
                y = layer(y, x)
        y = rearrange(y, "(b t) c h w -> b t c h w", b=b, t=t)
        return y


class UpsamplingAdd(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.upsample_layer = nn.Sequential(
            nn.Upsample(
                scale_factor=scale_factor, mode="bilinear", align_corners=False
            ),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x, x_skip):
        x = self.upsample_layer(x)
        return x + x_skip


class Bottleneck(nn.Module):
    """
    Defines a bottleneck module with a residual connection
    """

    def __init__(
        self,
        in_channels,
        out_channels=None,
        kernel_size=3,
        dilation=1,
        groups=1,
        upsample=False,
        downsample=False,
        dropout=0.0,
    ):
        super().__init__()
        self._downsample = downsample
        bottleneck_channels = int(in_channels / 2)
        out_channels = out_channels or in_channels
        padding_size = ((kernel_size - 1) * dilation + 1) // 2

        assert dilation == 1
        bottleneck_conv = nn.Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=kernel_size,
            bias=False,
            dilation=dilation,
            stride=2,
            padding=padding_size,
            groups=groups,
        )

        self.layers = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv_down_project",
                        nn.Conv2d(
                            in_channels, bottleneck_channels, kernel_size=1, bias=False
                        ),
                    ),
                    (
                        "abn_down_project",
                        nn.Sequential(
                            nn.BatchNorm2d(bottleneck_channels), nn.ReLU(inplace=True)
                        ),
                    ),
                    ("conv", bottleneck_conv),
                    (
                        "abn",
                        nn.Sequential(
                            nn.BatchNorm2d(bottleneck_channels), nn.ReLU(inplace=True)
                        ),
                    ),
                    (
                        "conv_up_project",
                        nn.Conv2d(
                            bottleneck_channels, out_channels, kernel_size=1, bias=False
                        ),
                    ),
                    (
                        "abn_up_project",
                        nn.Sequential(
                            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
                        ),
                    ),
                    ("dropout", nn.Dropout2d(p=dropout)),
                ]
            )
        )

        projection = OrderedDict()
        projection.update({"upsample_skip_proj": nn.MaxPool2d(kernel_size=2, stride=2)})
        projection.update(
            {
                "conv_skip_proj": nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, bias=False
                ),
                "bn_skip_proj": nn.BatchNorm2d(out_channels),
            }
        )
        self.projection = nn.Sequential(projection)

    def forward(self, *args):
        (x,) = args
        x_residual = self.layers(x)
        if self.projection is not None:
            if self._downsample:
                x = nn.functional.pad(
                    x, (0, x.shape[-1] % 2, 0, x.shape[-2] % 2), value=0
                )
            return x_residual + self.projection(x)
        return x_residual + x


def update_instance_ids(instance_seg, old_ids, new_ids):
    """
    Parameters
    ----------
        instance_seg: torch.Tensor arbitrary shape
        old_ids: 1D tensor containing the list of old ids, must be all present in instance_seg.
        new_ids: 1D tensor with the new ids, aligned with old_ids

    Returns
        new_instance_seg: torch.Tensor same shape as instance_seg with new ids
    """
    indices = torch.arange(old_ids.max() + 1)
    for old_id, new_id in zip(old_ids, new_ids):
        indices[old_id] = new_id

    return indices[instance_seg].long()


def make_instance_seg_consecutive(instance_seg):
    unique_ids = torch.unique(instance_seg)
    new_ids = torch.arange(len(unique_ids))
    instance_seg = update_instance_ids(instance_seg, unique_ids, new_ids)
    return instance_seg


def predict_instance_segmentation_and_trajectories(
    foreground_masks,
    ins_sigmoid,
    vehicles_id=1,
):
    if foreground_masks.dim() == 5 and foreground_masks.shape[2] == 1:
        foreground_masks = foreground_masks.squeeze(2)
    foreground_masks = foreground_masks == vehicles_id

    argmax_ins = ins_sigmoid.argmax(dim=1)
    argmax_ins = argmax_ins + 1
    instance_seg = (argmax_ins * foreground_masks.float()).long()
    instance_seg = make_instance_seg_consecutive(instance_seg).long()
    return instance_seg
