# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Utility functions for Ultra-Fast-Lane-Detection model
"""

import torch
import cv2
import numpy as np
import scipy.special
import torchvision.transforms as transforms
from PIL import Image


def load_lane_detection_model(backbone, griding_num, cls_num_per_lane, model_path):
    """Load the lane detection model from a checkpoint.

    Args:
        backbone: ResNet backbone variant (e.g., '18', '34', '50')
        griding_num: Number of grid cells
        cls_num_per_lane: Number of classification points per lane
        model_path: Path to the pretrained model weights (.pth file)

    Returns:
        torch.nn.Module: Loaded model
    """
    # Import the model architecture (now local to this module)
    from .model import parsingNet

    # Create model instance
    net = parsingNet(
        pretrained=False,
        backbone=backbone,
        cls_dim=(griding_num + 1, cls_num_per_lane, 4),
        use_aux=False,  # We don't need auxiliary segmentation in inference
    )

    # Load checkpoint
    state_dict = torch.load(model_path, map_location="cpu")["model"]

    # Handle models trained with DataParallel
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if "module." in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    # Load weights
    net.load_state_dict(compatible_state_dict, strict=False)

    return net


def preprocess_image(image):
    """Preprocess an image for lane detection.

    Args:
        image: Input image (numpy array in BGR format from cv2)

    Returns:
        torch.Tensor: Preprocessed image tensor (1, 3, 288, 800)
    """
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert to PIL Image
    img_pil = Image.fromarray(img_rgb)

    # Define transforms
    img_transforms = transforms.Compose(
        [
            transforms.Resize((288, 800)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    # Apply transforms and add batch dimension
    img_tensor = img_transforms(img_pil).unsqueeze(0)

    return img_tensor


def postprocess_detections(
    model_output, griding_num, cls_num_per_lane, img_w, img_h, row_anchor
):
    """Post-process model output to extract lane coordinates.

    Args:
        model_output: Raw model output tensor
        griding_num: Number of grid cells
        cls_num_per_lane: Number of classification points per lane
        img_w: Output image width
        img_h: Output image height
        row_anchor: List of row anchor points

    Returns:
        tuple: (lanes, num_lanes)
            - lanes: List of detected lanes, each lane is a list of (x, y) tuples
            - num_lanes: Number of detected lanes
    """
    # Process output
    col_sample = np.linspace(0, 800 - 1, griding_num)
    col_sample_w = col_sample[1] - col_sample[0]

    out_j = model_output[0].data
    # Convert to numpy if it's a torch tensor
    if hasattr(out_j, "cpu"):
        # Convert bfloat16 to float32 if needed (numpy doesn't support bfloat16)
        if out_j.dtype == torch.bfloat16:
            out_j = out_j.cpu().float().numpy()
        else:
            out_j = out_j.cpu().numpy()
    elif hasattr(out_j, "numpy"):
        out_j = out_j.numpy()
    out_j = out_j[:, ::-1, :]
    prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
    idx = np.arange(griding_num) + 1
    idx = idx.reshape(-1, 1, 1)
    loc = np.sum(prob * idx, axis=0)
    out_j = np.argmax(out_j, axis=0)
    loc[out_j == griding_num] = 0
    out_j = loc

    # Extract lanes
    lanes = []
    num_lanes = 0

    for i in range(out_j.shape[1]):
        lane_points = []
        num_points = np.sum(out_j[:, i] != 0)

        # Only consider lanes with at least 2 points
        if num_points >= 2:
            num_lanes += 1

            for k in range(out_j.shape[0]):
                if out_j[k, i] > 0:
                    x = int(out_j[k, i] * col_sample_w * img_w / 800) - 1
                    y = int(img_h * (row_anchor[cls_num_per_lane - 1 - k] / 288)) - 1
                    lane_points.append((x, y))

            lanes.append(lane_points)

    return lanes, num_lanes


def visualize_lanes(image, lanes, img_w, img_h):
    """Visualize detected lanes on the image.

    Args:
        image: Original input image (numpy array)
        lanes: List of detected lanes from postprocess_detections
        img_w: Target image width
        img_h: Target image height

    Returns:
        numpy.ndarray: Image with visualized lanes
    """
    # Resize image to target size if needed
    if image.shape[1] != img_w or image.shape[0] != img_h:
        vis = cv2.resize(image, (img_w, img_h))
    else:
        vis = image.copy()

    # Different colors for each lane
    colors = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
    ]

    # Draw each lane
    for i, lane_points in enumerate(lanes):
        if len(lane_points) > 0:
            color = colors[i % len(colors)]

            # Draw circles at each point
            for point in lane_points:
                cv2.circle(vis, point, 8, color, -1)

            # Draw lines connecting points
            for j in range(len(lane_points) - 1):
                cv2.line(vis, lane_points[j], lane_points[j + 1], color, 3)

    return vis


def run_inference_and_visualize(
    model,
    image_path,
    griding_num,
    cls_num_per_lane,
    img_w,
    img_h,
    row_anchor,
    output_path=None,
):
    """Run complete inference pipeline on an image.

    Args:
        model: Loaded lane detection model
        image_path: Path to input image
        griding_num: Number of grid cells
        cls_num_per_lane: Number of classification points per lane
        img_w: Output image width
        img_h: Output image height
        row_anchor: List of row anchor points
        output_path: Optional path to save visualization

    Returns:
        dict: Results containing lanes, visualization, etc.
    """
    # Load image
    img_orig = cv2.imread(image_path)
    if img_orig is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Preprocess
    img_tensor = preprocess_image(img_orig)

    with torch.no_grad():
        output = model(img_tensor)

    # Post-process
    lanes, num_lanes = postprocess_detections(
        model_output=output,
        griding_num=griding_num,
        cls_num_per_lane=cls_num_per_lane,
        img_w=img_w,
        img_h=img_h,
        row_anchor=row_anchor,
    )

    # Visualize
    vis_img = visualize_lanes(img_orig, lanes, img_w, img_h)

    # Save if requested
    if output_path is not None:
        cv2.imwrite(output_path, vis_img)
        print(f"Visualization saved to: {output_path}")

    return {
        "lanes": lanes,
        "num_lanes": num_lanes,
        "visualization": vis_img,
        "original_image": img_orig,
    }


tusimple_row_anchor = [
    64,
    68,
    72,
    76,
    80,
    84,
    88,
    92,
    96,
    100,
    104,
    108,
    112,
    116,
    120,
    124,
    128,
    132,
    136,
    140,
    144,
    148,
    152,
    156,
    160,
    164,
    168,
    172,
    176,
    180,
    184,
    188,
    192,
    196,
    200,
    204,
    208,
    212,
    216,
    220,
    224,
    228,
    232,
    236,
    240,
    244,
    248,
    252,
    256,
    260,
    264,
    268,
    272,
    276,
    280,
    284,
]
culane_row_anchor = [
    121,
    131,
    141,
    150,
    160,
    170,
    180,
    189,
    199,
    209,
    219,
    228,
    238,
    248,
    258,
    267,
    277,
    287,
]
