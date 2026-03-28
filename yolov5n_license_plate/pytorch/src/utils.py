# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


import torch
from pathlib import Path
import yolov5
import cv2
import numpy as np
from PIL import Image
from yolov5.models.common import Detections
from yolov5.utils.dataloaders import exif_transpose, letterbox
from yolov5.utils.general import Profile, non_max_suppression, scale_boxes


def data_preprocessing(ims: Image.Image, size: tuple) -> tuple:
    """Data preprocessing function for YOLOv5 object detection.

    Parameters
    ----------
    ims : Image.Image
        Input image
    size : tuple
        Desired image size

    Returns
    -------
    tuple
        List of images, number of samples, filenames, image size, inference size, preprocessed images
    """

    if not isinstance(ims, (list, tuple)):
        ims = [ims]
    num_images = len(ims)
    shape_orig, shape_infer, filenames = [], [], []

    for idx, img in enumerate(ims):
        filename = getattr(img, "filename", f"image{idx}")
        img = np.asarray(exif_transpose(img))
        filename = Path(filename).with_suffix(".jpg").name
        filenames.append(filename)

        if img.shape[0] < 5:
            img = img.transpose((1, 2, 0))

        if img.ndim == 3:
            img = img[..., :3]
        else:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        shape_orig.append(img.shape[:2])
        scale = max(size) / max(img.shape[:2])
        shape_infer.append([int(dim * scale) for dim in img.shape[:2]])
        ims[idx] = img if img.flags["C_CONTIGUOUS"] else np.ascontiguousarray(img)

    shape_infer = [size[0] for _ in np.array(shape_infer).max(0)]
    imgs_padded = [letterbox(img, shape_infer, auto=False)[0] for img in ims]
    imgs_padded = np.ascontiguousarray(np.array(imgs_padded).transpose((0, 3, 1, 2)))
    tensor_imgs = torch.from_numpy(imgs_padded) / 255

    return ims, num_images, filenames, shape_orig, shape_infer, tensor_imgs


def data_postprocessing(
    ims: list,
    x_shape: torch.Size,
    pred: list,
    model: yolov5.models.common.AutoShape,
    n: int,
    shape0: list,
    shape1: list,
    files: list,
) -> Detections:
    """Data postprocessing function for YOLOv5 object detection.

    Parameters
    ----------
    ims : list
        List of input images
    x_shape : torch.Size
        Shape of each image
    pred : list
        List of model predictions
    model : yolov5.models.common.AutoShape
        Model
    n : int
        Number of input samples
    shape0 : list
        Image shape
    shape1 : list
        Inference shape
    files : list
        Filenames

    Returns
    -------
    Detections
        Detection object
    """

    # Create dummy dt tuple (not used but required for Detections)
    dt = (Profile(), Profile(), Profile())

    # Perform NMS
    y = non_max_suppression(
        prediction=pred,
        conf_thres=model.conf,
        iou_thres=model.iou,
        classes=None,
        agnostic=model.agnostic,
        multi_label=model.multi_label,
        labels=(),
        max_det=model.max_det,
    )

    # Scale bounding boxes
    for i in range(n):
        scale_boxes(shape1, y[i][:, :4], shape0[i])

    # Return Detections object
    return Detections(ims, y, files, times=dt, names=model.names, shape=x_shape)
