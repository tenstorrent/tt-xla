# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import skimage
import skimage.io
import skimage.transform
import os
import requests


# Image processing utilities for vision models
def load_image(image_path):
    """Load image using skimage.

    Args:
        image_path: Path to the image file

    Returns:
        numpy.ndarray: Loaded image as float array
    """
    img = skimage.img_as_float(skimage.io.imread(image_path))
    if len(img.shape) == 2:
        img = np.array([img, img, img]).swapaxes(0, 2)
    return img


def rescale_image(img, input_height, input_width):
    """Rescale image maintaining aspect ratio.

    Args:
        img: Input image array
        input_height: Target height
        input_width: Target width

    Returns:
        numpy.ndarray: Rescaled image
    """
    aspect = img.shape[1] / float(img.shape[0])
    if aspect > 1:
        # landscape orientation - wide image
        res = int(aspect * input_height)
        imgScaled = skimage.transform.resize(img, (input_width, res))
    elif aspect < 1:
        # portrait orientation - tall image
        res = int(input_width / aspect)
        imgScaled = skimage.transform.resize(img, (res, input_height))
    else:
        imgScaled = skimage.transform.resize(img, (input_width, input_height))
    return imgScaled


def crop_center(img, cropx, cropy):
    """Crop image from center.

    Args:
        img: Input image array
        cropx: Width of crop
        cropy: Height of crop

    Returns:
        numpy.ndarray: Center-cropped image
    """
    y, x, c = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty : starty + cropy, startx : startx + cropx]


def normalize_image(img, mean=128, std=128):
    """Normalize image.

    Args:
        img: Input image array
        mean: Mean value for normalization
        std: Standard deviation for normalization

    Returns:
        numpy.ndarray: Normalized image
    """
    img = (img * 256 - mean) / std
    return img


def prepare_ssd_input(img_uri):
    """Prepare input image for SSD models.

    Args:
        img_uri: Path or URI to the image

    Returns:
        numpy.ndarray: Preprocessed image ready for SSD model
    """
    img = load_image(img_uri)
    img = rescale_image(img, 300, 300)
    img = crop_center(img, 300, 300)
    img = normalize_image(img)
    return img


def download_checkpoint(checkpoint_url, checkpoint_path):
    """Download a checkpoint file if it doesn't exist.

    Args:
        checkpoint_url: URL to download the checkpoint from
        checkpoint_path: Local path where to save the checkpoint

    Returns:
        str: Path to the downloaded checkpoint file
    """
    if not os.path.exists(checkpoint_path):
        response = requests.get(checkpoint_url)
        response.raise_for_status()  # Raise exception for HTTP errors
        with open(checkpoint_path, "wb") as f:
            f.write(response.content)
    return checkpoint_path
