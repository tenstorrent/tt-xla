# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
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
