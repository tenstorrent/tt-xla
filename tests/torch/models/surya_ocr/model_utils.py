# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import math
from typing import List, Optional, Union

import cv2  # type: ignore[reportMissingImports]
import numpy as np  # type: ignore[reportMissingImports]
import torch
from PIL import Image  # type: ignore[reportMissingImports]
from surya.common.surya.processor import (
    SuryaOCRProcessor,  # type: ignore[reportMissingImports]
)
from surya.detection.processor import (
    SegformerImageProcessor,  # type: ignore[reportMissingImports]
)
from surya.foundation.cache.static_ops import (
    StaticOpsCache,  # type: ignore[reportMissingImports]
)
from surya.settings import settings  # type: ignore[reportMissingImports]
from transformers import PretrainedConfig  # type: ignore[reportMissingImports]
from transformers.image_utils import (  # type: ignore[reportMissingImports]
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    infer_channel_dimension_format,
)


def _prepare_image(self, img):
    new_size = (self.processor.size["width"], self.processor.size["height"])

    img.thumbnail(new_size, Image.Resampling.LANCZOS)
    img = img.resize(new_size, Image.Resampling.LANCZOS)
    # Original line
    # img = np.asarray(img, dtype=np.uint8)
    # New line (patch): replace original with explicit copy
    img = np.array(img, dtype=np.uint8, copy=True)
    img = self.processor(img)["pixel_values"][0]
    img = torch.from_numpy(img)
    return img


# Monkeypatch SegformerImageProcessor._preprocess to use torch-friendly ops
def _segformer_preprocess(
    self: SegformerImageProcessor,
    image: ImageInput,
    do_resize: bool,
    do_rescale: bool,
    do_normalize: bool,
    size: Optional[dict[str, int]] = None,
    resample: PILImageResampling = None,
    rescale_factor: Optional[float] = None,
    image_mean: Optional[Union[float, List[float]]] = None,
    image_std: Optional[Union[float, List[float]]] = None,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
):
    if isinstance(image, Image.Image):
        image = np.array(image)

    if isinstance(image, np.ndarray):
        tensor_image = torch.from_numpy(image)
    elif isinstance(image, torch.Tensor):
        tensor_image = image
    else:
        tensor_image = torch.as_tensor(image)

    if not tensor_image.is_floating_point():
        tensor_image = tensor_image.to(torch.float32)

    if do_rescale:
        scale = float(rescale_factor if rescale_factor is not None else 1.0)
        tensor_image = tensor_image * scale

    if do_normalize:
        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(
                image
                if isinstance(image, np.ndarray)
                else tensor_image.detach().cpu().numpy()
            )
        try:
            channel_dim = ChannelDimension(input_data_format)
        except Exception:
            channel_dim = input_data_format

        if channel_dim == ChannelDimension.LAST:
            channel_axis = -1
        elif channel_dim == ChannelDimension.FIRST:
            channel_axis = 0
        else:
            channel_axis = None

        if channel_axis is None or tensor_image.ndim < 3:
            mean_values = (
                image_mean if isinstance(image_mean, (list, tuple)) else [image_mean]
            )
            std_values = (
                image_std if isinstance(image_std, (list, tuple)) else [image_std]
            )
            mean_scalar = float(mean_values[0])
            std_scalar = float(std_values[0])
            tensor_image = (tensor_image - mean_scalar) / std_scalar
        else:
            num_channels = tensor_image.shape[channel_axis]
            if isinstance(image_mean, (list, tuple)):
                mean_list = list(image_mean)
            else:
                mean_list = [image_mean] * num_channels
            if isinstance(image_std, (list, tuple)):
                std_list = list(image_std)
            else:
                std_list = [image_std] * num_channels

            mean_tensor = torch.tensor(
                mean_list, dtype=tensor_image.dtype, device=tensor_image.device
            )
            std_tensor = torch.tensor(
                std_list, dtype=tensor_image.dtype, device=tensor_image.device
            )

            expand_shape = [1] * tensor_image.ndim
            expand_shape[channel_axis] = num_channels
            mean_tensor = mean_tensor.view(
                *(
                    [num_channels]
                    if channel_axis in (0, -1) and tensor_image.ndim == 1
                    else expand_shape
                )
            )
            std_tensor = std_tensor.view(
                *(
                    [num_channels]
                    if channel_axis in (0, -1) and tensor_image.ndim == 1
                    else expand_shape
                )
            )

            if channel_axis == -1:
                mean_tensor = mean_tensor.view(*expand_shape)
                std_tensor = std_tensor.view(*expand_shape)
            elif channel_axis == 0:
                mean_tensor = mean_tensor.view(*expand_shape)
                std_tensor = std_tensor.view(*expand_shape)

            tensor_image = (tensor_image - mean_tensor) / std_tensor

    result = (
        tensor_image.detach().cpu().numpy().astype(np.float32)
        if tensor_image.is_floating_point()
        else tensor_image.detach().cpu().numpy()
    )
    return result


# Torch-friendly replacement for get_dynamic_thresholds in surya.detection.heatmap
def _get_dynamic_thresholds_torch(
    linemap,
    text_threshold: float,
    low_text: float,
    typical_top10_avg: float = 0.7,
):
    tensor_map = torch.as_tensor(linemap, dtype=torch.float32)
    flat = tensor_map.reshape(-1)
    numel = int(flat.numel())
    k = max(1, int(numel * 0.10))  # top 10%
    if k >= numel:
        top_values = flat
    else:
        top_values, _ = torch.topk(flat, k, largest=True)
    avg_intensity = top_values.mean()
    scaling_factor = torch.clamp(
        avg_intensity / float(typical_top10_avg), 0.0, 1.0
    ).pow(0.5)

    low_text_new = torch.clamp(
        torch.tensor(float(low_text)) * scaling_factor, 0.1, 0.6
    ).item()
    text_threshold_new = torch.clamp(
        torch.tensor(float(text_threshold)) * scaling_factor, 0.15, 0.8
    ).item()
    return float(text_threshold_new), float(low_text_new)


# Torch-friendly replacement for detect_boxes in surya.detection.heatmap
def _detect_boxes_torch(linemap, text_threshold, low_text):
    # Ensure torch tensor for elementwise ops to avoid Dynamo numpy interception
    tensor_map = torch.as_tensor(linemap, dtype=torch.float32)
    img_h, img_w = int(tensor_map.shape[0]), int(tensor_map.shape[1])

    # Use the torch-based dynamic thresholds
    text_threshold, low_text = _get_dynamic_thresholds_torch(
        tensor_map, text_threshold, low_text
    )

    # Threshold using torch, then convert to numpy for OpenCV
    text_score_comb = (tensor_map > float(low_text)).to(torch.uint8).cpu().numpy()
    label_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        text_score_comb, connectivity=4
    )

    det: List[np.ndarray] = []
    confidences: List[float] = []
    max_confidence = 0.0

    linemap_np = tensor_map.cpu().numpy()

    for k in range(1, int(label_count)):
        # size filtering
        size = int(stats[k, cv2.CC_STAT_AREA])
        if size < 10:
            continue

        # make segmentation map
        x, y, w, h = stats[
            k,
            [cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT],
        ]

        try:
            niter = int(np.sqrt(min(int(w), int(h))))
        except ValueError:
            niter = 0

        buffer = 1
        sx, sy = max(0, int(x) - niter - buffer), max(0, int(y) - niter - buffer)
        ex, ey = min(img_w, int(x) + int(w) + niter + buffer), min(
            img_h, int(y) + int(h) + niter + buffer
        )

        mask = labels[sy:ey, sx:ex] == k
        line_max = (
            float(np.max(linemap_np[sy:ey, sx:ex][mask])) if np.any(mask) else 0.0
        )

        # thresholding
        if line_max < float(text_threshold):
            continue

        segmap = mask.astype(np.uint8)

        ksize = buffer + niter
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        selected_segmap = cv2.dilate(segmap, kernel)

        # make box
        y_inds, x_inds = np.nonzero(selected_segmap)
        x_inds = x_inds + sx
        y_inds = y_inds + sy
        np_contours = np.column_stack((x_inds, y_inds))
        if np_contours.shape[0] < 3:
            # Need at least a few points for minAreaRect; skip small artifacts
            continue
        rectangle = cv2.minAreaRect(np_contours.astype(np.float32))
        box = cv2.boxPoints(rectangle)

        # align diamond-shape
        w_len = np.linalg.norm(box[0] - box[1])
        h_len = np.linalg.norm(box[1] - box[2])
        box_ratio = max(w_len, h_len) / (min(w_len, h_len) + 1e-5)
        if abs(1 - box_ratio) <= 0.1:
            l, r = np_contours[:, 0].min(), np_contours[:, 0].max()
            t, b = np_contours[:, 1].min(), np_contours[:, 1].max()
            box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

        # make clock-wise order
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - startidx, 0)

        max_confidence = max(max_confidence, line_max)

        confidences.append(line_max)
        det.append(box.astype(np.float32))

    if max_confidence > 0:
        confidences = [float(c / max_confidence) for c in confidences]
    return det, confidences


def _patched_static_ops_cache_init(
    self,
    config: PretrainedConfig,
    batch_size: int,
    max_cache_len: int,
    text_sliding_window: int,
    device: int,
    dtype: int,
):
    self.text_sliding_window = text_sliding_window
    self.num_layers = config.num_hidden_layers
    self.max_batch_size = batch_size
    self.max_cache_len = max_cache_len
    self.head_dim = (
        getattr(config, "head_dim", None)
        or config.hidden_size // config.num_attention_heads
    )
    self._dtype = dtype
    self.num_key_value_heads = (
        config.num_attention_heads
        if getattr(config, "num_key_value_heads", None) is None
        else config.num_key_value_heads
    )

    # Cache init is taken from huggingface StaticCache - https://github.com/huggingface/transformers/blob/67ddc82fbc7e52c6f42a395b4a6d278c55b77a39/src/transformers/cache_utils.py#L1125
    self.key_cache: list[torch.Tensor] = []
    self.value_cache: list[torch.Tensor] = []
    cache_shape = (
        self.max_batch_size,
        self.num_key_value_heads,
        self.max_cache_len,
        self.head_dim,
    )
    device = torch.device(device) if device is not None else None
    for _ in range(config.num_hidden_layers):
        new_layer_key_cache = torch.zeros(cache_shape, dtype=self._dtype, device=device)
        new_layer_value_cache = torch.zeros(
            cache_shape, dtype=self._dtype, device=device
        )
        if not torch._dynamo.is_compiling():
            torch._dynamo.mark_static_address(new_layer_key_cache)
            torch._dynamo.mark_static_address(new_layer_value_cache)
        self.key_cache.append(new_layer_key_cache)
        self.value_cache.append(new_layer_value_cache)

    self.attention_mask = torch.zeros(
        (self.max_batch_size, self.max_cache_len), device=device, dtype=torch.long
    )
    self.text_token_counts = [
        torch.zeros(self.max_batch_size, dtype=torch.long, device=device)
        for _ in range(self.num_layers)
    ]

    self.dtype = dtype
    self.device = device


def _patched_dynamic_ops_cache_init(
    self,
    config: PretrainedConfig,
    batch_size: int,
    max_cache_len: int,
    text_sliding_window: int,
    device: int,
    dtype: int,
):
    self.text_sliding_window = text_sliding_window
    self.num_layers = config.num_hidden_layers
    self.max_batch_size = batch_size
    self.max_cache_len = max_cache_len
    self.head_dim = (
        getattr(config, "head_dim", None)
        or config.hidden_size // config.num_attention_heads
    )
    self._dtype = dtype
    self.num_key_value_heads = (
        config.num_attention_heads
        if getattr(config, "num_key_value_heads", None) is None
        else config.num_key_value_heads
    )

    # Initialize KV caches but avoid Dynamo-forbidden mark_static_address while tracing
    self.key_cache: list[torch.Tensor] = []
    self.value_cache: list[torch.Tensor] = []
    cache_shape = (
        self.max_batch_size,
        self.num_key_value_heads,
        self.max_cache_len,
        self.head_dim,
    )
    device = torch.device(device) if device is not None else None
    for _ in range(config.num_hidden_layers):
        new_layer_key_cache = torch.zeros(cache_shape, dtype=self._dtype, device=device)
        new_layer_value_cache = torch.zeros(
            cache_shape, dtype=self._dtype, device=device
        )
        if not torch._dynamo.is_compiling():
            torch._dynamo.mark_static_address(new_layer_key_cache)
            torch._dynamo.mark_static_address(new_layer_value_cache)
        self.key_cache.append(new_layer_key_cache)
        self.value_cache.append(new_layer_value_cache)

    # Mirror attention mask and per-layer text token counters expected by DynamicOpsCache
    self.attention_mask = torch.zeros(
        (self.max_batch_size, self.max_cache_len), device=device, dtype=torch.long
    )
    self.text_token_counts = [
        torch.zeros(self.max_batch_size, dtype=torch.long, device=device)
        for _ in range(self.num_layers)
    ]

    # Keep parity with static cache helper fields
    self.dtype = dtype
    self.device = device


def _patched_image_processor(self: SuryaOCRProcessor, image: np.ndarray) -> np.ndarray:
    tensor_img = torch.as_tensor(image, dtype=torch.float32)
    scale = float(getattr(self, "rescale_factor", 1.0))
    tensor_img = tensor_img * scale

    # self.image_mean/std may be numpy arrays of shape (3,)
    mean = torch.as_tensor(
        self.image_mean, dtype=tensor_img.dtype, device=tensor_img.device
    )
    std = torch.as_tensor(
        self.image_std, dtype=tensor_img.dtype, device=tensor_img.device
    )

    if tensor_img.ndim == 3 and tensor_img.shape[-1] == int(mean.numel()):
        # HWC layout
        tensor_img = (tensor_img - mean) / std
    else:
        # Fallback broadcast
        tensor_img = (tensor_img - mean.view(1, 1, -1)) / std.view(1, 1, -1)

    return tensor_img.detach().cpu().numpy()


def _patched_process_and_tile_no_xla(self: SuryaOCRProcessor, image: np.ndarray):
    """
    Monkey-patch version of SuryaOCRProcessor._process_and_tile that forces
    FOUNDATION_XLA-like padding off by using extra_multipler = 1.
    """
    # Equivalent to original, but always behaves as if settings.FOUNDATION_XLA is False
    extra_multipler = 1
    factor = self.patch_size * self.merge_size * extra_multipler

    height, width = image.shape[:2]
    h_bar = math.ceil(height / factor) * factor
    w_bar = math.ceil(width / factor) * factor
    if h_bar != height or w_bar != width:
        if height == 0 or width == 0:
            image = np.zeros((h_bar, w_bar, 3), dtype=np.uint8)
        else:
            image = cv2.resize(image, (w_bar, h_bar), interpolation=cv2.INTER_CUBIC)

    # Handle scaling and normalization
    image = self._image_processor(image)
    height, width = image.shape[:2]

    # Numpy array to torch tensor
    img_tensor = torch.from_numpy(image.transpose(2, 0, 1))
    patches = img_tensor.unsqueeze(0)

    channel = patches.shape[1]
    grid_t = patches.shape[0]
    grid_h, grid_w = height // self.patch_size, width // self.patch_size

    patches = patches.reshape(
        grid_t,
        1,
        channel,
        grid_h // self.merge_size,
        self.merge_size,
        self.patch_size,
        grid_w // self.merge_size,
        self.merge_size,
        self.patch_size,
    )
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flatten_patches = patches.reshape(
        grid_t * grid_h * grid_w, channel * 1 * self.patch_size * self.patch_size
    )

    return flatten_patches, (grid_t, grid_h, grid_w)
