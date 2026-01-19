# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from ....tools.utils import get_file

import torch.nn as nn
import subprocess
from pathlib import Path
import requests
import math
import numpy as np
import torch
import cv2
import sys
import os
import importlib
import types
from torch.hub import download_url_to_file
from urllib.parse import urljoin
import torchvision
import yaml


def make_divisible(x, divisor):
    """Returns the smallest integer greater than or equal to x that is divisible by divisor."""
    return math.ceil(x / divisor) * divisor


def check_img_size(img_size, s=32):
    """Ensures the image size is a multiple of the given stride value."""
    new_size = make_divisible(img_size, int(s))
    if new_size != img_size:
        print(
            "WARNING: --img-size %g must be multiple of max stride %g, updating to %g"
            % (img_size, s, new_size)
        )
    return new_size


def letterbox(
    img,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=True,
    scaleFill=False,
    scaleup=True,
    stride=32,
):
    """
    Resizes and pads an image to fit a target shape while maintaining aspect ratio.

    Parameters
    ----------
    img : np.ndarray
        Input image in BGR or RGB format (H x W x C).
    new_shape : tuple or int, optional
        Desired output shape (height, width). If an integer is provided, both dimensions
        are set to that value. Default is (640, 640).
    color : tuple, optional
        Border color for padding, in (B, G, R) format. Default is (114, 114, 114).
    auto : bool, optional
        If True, adjusts padding to ensure that the final shape is a multiple of `stride`.
        Default is True.
    scaleFill : bool, optional
        If True, stretches the image to fill the new shape (may distort aspect ratio).
        Default is False.
    scaleup : bool, optional
        If False, only scales down the image, preventing enlargement (useful for validation
        to preserve test mAP). Default is True.
    stride : int, optional
        Stride value to which final image shape will be aligned when `auto=True`.
        Default is 32.

    Returns
    -------
    img : np.ndarray
        The resized and padded image.
    ratio : tuple(float, float)
        The scaling ratios applied to the original width and height.
    (dw, dh) : tuple(float, float)
        The padding added to width and height (half on each side).
    """
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return img, ratio, (dw, dh)


class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output


def attempt_load(weights, map_location=None):
    """
    Loads one or more YOLO model weights into an Ensemble or a single model instance.

    Parameters
    ----------
    weights : str or list of str
        Path(s) to one or more model weight files (.pt checkpoints).
        If a list is provided, multiple models will be loaded and combined into
        an `Ensemble` object; otherwise, a single model is returned.

    map_location : str, torch.device, optional
        Device mapping for loading the weights (e.g., `'cpu'`, `'cuda'`, or a torch.device).
        Defaults to None, letting PyTorch decide.

    Returns
    -------
    model : torch.nn.Module or Ensemble
        - If a single weight file is provided, returns a single fused and evaluated model.
        - If multiple weight files are provided, returns an `Ensemble` object that
          aggregates all loaded models (sharing attributes such as `names` and `stride`).
    """

    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        src_dir = str(Path(__file__).resolve().parent)
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        # # Alias expected pickled module paths (e.g., 'models.common') to the
        # # consolidated single-file implementation 'yolo.py' so torch.load can
        # # resolve classes like models.common.Conv without the original package.
        unified = importlib.import_module("yolo")
        # Ensure parent packages exist in sys.modules
        models_pkg = sys.modules.get("models")
        if models_pkg is None:
            models_pkg = types.ModuleType("models")
            sys.modules["models"] = models_pkg
        utils_pkg = sys.modules.get("utils")
        if utils_pkg is None:
            utils_pkg = types.ModuleType("utils")
            sys.modules["utils"] = utils_pkg

        # Map expected submodules to the unified implementation
        for modname in ("models.common", "models.yolo", "models.experimental"):
            sys.modules[modname] = unified
        for modname in (
            "utils.torch_utils",
            "utils.activations",
            "utils.autoanchor",
            "utils.general",
            "utils.loss",
        ):
            sys.modules[modname] = unified

        # Also expose as attributes on parent packages to satisfy attribute lookups
        setattr(models_pkg, "common", unified)
        setattr(models_pkg, "yolo", unified)
        setattr(models_pkg, "experimental", unified)
        setattr(utils_pkg, "torch_utils", unified)
        setattr(utils_pkg, "activations", unified)
        setattr(utils_pkg, "autoanchor", unified)
        setattr(utils_pkg, "general", unified)
        setattr(utils_pkg, "loss", unified)

        ckpt = torch.load(w, map_location=map_location, weights_only=False)  # load
        model.append(ckpt["ema" if ckpt.get("ema") else "model"].float().fuse().eval())

    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True
        elif type(m) is nn.Upsample:
            m.recompute_scale_factor = None

    if len(model) == 1:
        return model[-1]  # return model
    else:
        print("Ensemble created with %s\n" % weights)
        for k in ["names", "stride"]:
            setattr(model, k, getattr(model[-1], k))
        return model


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (
        (
            torch.min(box1[:, None, 2:], box2[:, 2:])
            - torch.max(box1[:, None, :2], box2[:, :2])
        )
        .clamp(0)
        .prod(2)
    )
    return inter / (
        area1[:, None] + area2 - inter
    )  # iou = inter / (area1 + area2 - inter)


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(
            img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]
        )  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
            img1_shape[0] - img0_shape[0] * gain
        ) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        if nc == 1:
            x[:, 5:] = x[
                :, 4:5
            ]  # for models with one class, cls_loss is 0 and cls_conf is always 0.5,
            # so there is no need to multiplicate.
        else:
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(
                1, keepdim=True
            )  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]

    return output


def yolov7_postprocess(pred, org_img, img):

    # Apply NMS
    pred = non_max_suppression(pred)

    # Get class names
    coco_yaml_path = get_file("test_files/pytorch/yolo/coco.yaml")
    with open(coco_yaml_path, "r") as f:
        coco_yaml = yaml.safe_load(f)
    names = coco_yaml["names"]

    # Process detections
    for i, det in enumerate(pred):  # detections per image

        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], org_img.shape).round()

            # Print detections
            for *xyxy, conf, cls in reversed(det):
                class_num = int(cls.item()) if hasattr(cls, "item") else int(cls)
                conf_value = (
                    float(conf.item()) if hasattr(conf, "item") else float(conf)
                )
                coordinates = [
                    int(x.item()) if hasattr(x, "item") else int(x) for x in xyxy
                ]
                label = names[class_num]

                print(
                    f"Coordinates: {coordinates}, Class: {label}, Confidence: {conf_value:.2f}"
                )

        print("\n")
