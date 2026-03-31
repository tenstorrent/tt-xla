#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Extract ground truth tensors from the real CenterPoint (PointPillars) model.

Run this script from the CenterPoint repo directory using its own venv:
    cd /proj_sw/user_dev/ctr-lelanchelian/CenterPoint
    /proj_sw/user_dev/ctr-lelanchelian/venv_centerpoint/bin/python \
        /proj_sw/user_dev/ctr-lelanchelian/tt-xla/scripts/extract_centerpoint_ground_truth.py

Saves to tt-xla/tests/torch/graphs/:
  - centerpoint_bev_image_bf16.pt      : (1, 64, 512, 512) BEV pseudo-image
  - centerpoint_raw_preds_bf16.pt      : list of 6 task dicts (raw CenterHead output)

These are the ground truth for CPU vs TT verification of the real RPN+CenterHead.
"""

import sys, os, logging
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
CENTERPOINT_DIR = "/proj_sw/user_dev/ctr-lelanchelian/CenterPoint"
TTXLA_DIR = "/proj_sw/user_dev/ctr-lelanchelian/tt-xla"
OUTPUT_DIR = os.path.join(TTXLA_DIR, "tests/torch/graphs")

sys.path.insert(0, CENTERPOINT_DIR)
torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
os.environ.setdefault("LD_LIBRARY_PATH", "")
if torch_lib not in os.environ["LD_LIBRARY_PATH"]:
    os.environ["LD_LIBRARY_PATH"] = torch_lib + ":" + os.environ["LD_LIBRARY_PATH"]

import itertools
from easydict import EasyDict

# ── Model config (matches run_cpu_demo.py) ────────────────────────────────────
TASKS = [
    dict(num_class=1, class_names=["car"]),
    dict(num_class=2, class_names=["truck", "construction_vehicle"]),
    dict(num_class=2, class_names=["bus", "trailer"]),
    dict(num_class=1, class_names=["barrier"]),
    dict(num_class=2, class_names=["motorcycle", "bicycle"]),
    dict(num_class=2, class_names=["pedestrian", "traffic_cone"]),
]

VOXEL_SIZE = [0.2, 0.2, 8.0]
PC_RANGE   = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
MAX_PTS_VOX  = 20
MAX_VOXELS   = 30000

TEST_CFG = EasyDict(
    post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
    max_per_img=500,
    nms=EasyDict(
        nms_pre_max_size=1000,
        nms_post_max_size=83,
        nms_iou_threshold=0.2,
    ),
    score_threshold=0.1,
    pc_range=[-51.2, -51.2],
    out_size_factor=8,
    voxel_size=[0.2, 0.2],
)

model_cfg = dict(
    type="PointPillars",
    pretrained=None,
    reader=dict(
        type="PillarFeatureNet",
        num_filters=[64, 64],
        num_input_features=5,
        with_distance=False,
        voxel_size=tuple(VOXEL_SIZE),
        pc_range=tuple(PC_RANGE),
    ),
    backbone=dict(type="PointPillarsScatter", ds_factor=1),
    neck=dict(
        type="RPN",
        layer_nums=[3, 5, 5],
        ds_layer_strides=[2, 2, 2],
        ds_num_filters=[64, 128, 256],
        us_layer_strides=[0.5, 1, 2],
        us_num_filters=[128, 128, 128],
        num_input_features=64,
        logger=logging.getLogger("RPN"),
    ),
    bbox_head=dict(
        type="CenterHead",
        in_channels=384,
        tasks=TASKS,
        dataset="nuscenes",
        weight=0.25,
        code_weights=[1.0]*8 + [0.2, 0.2],
        common_heads={
            "reg":    (2, 2),
            "height": (1, 2),
            "dim":    (3, 2),
            "rot":    (2, 2),
            "vel":    (2, 2),
        },
    ),
    train_cfg=None,
    test_cfg=TEST_CFG,
)

# ── Build model ───────────────────────────────────────────────────────────────
from det3d.models import build_detector

print("Building PointPillars model ...")
model = build_detector(model_cfg, train_cfg=None, test_cfg=TEST_CFG)
model = model.cpu().eval()
print(f"  params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

# ── Synthetic point cloud ─────────────────────────────────────────────────────
np.random.seed(42)
N_POINTS = 35_000
xyz    = np.random.uniform([PC_RANGE[0], PC_RANGE[1], PC_RANGE[2]],
                            [PC_RANGE[3], PC_RANGE[4], PC_RANGE[5]],
                            size=(N_POINTS, 3)).astype(np.float32)
intens = np.random.uniform(0, 1, (N_POINTS, 1)).astype(np.float32)
ts     = np.random.uniform(0, 0.5, (N_POINTS, 1)).astype(np.float32)
points = np.concatenate([xyz, intens, ts], axis=1)

# ── Voxelization ──────────────────────────────────────────────────────────────
from det3d.core.input.voxel_generator import VoxelGenerator

voxel_gen = VoxelGenerator(
    voxel_size=VOXEL_SIZE,
    point_cloud_range=PC_RANGE,
    max_num_points=MAX_PTS_VOX,
    max_voxels=MAX_VOXELS,
)
voxels, coords, num_points = voxel_gen.generate(points, max_voxels=MAX_VOXELS)
print(f"  voxels: {voxels.shape}, coords: {coords.shape}, grid: {voxel_gen.grid_size}")

batch_idx = np.zeros((coords.shape[0], 1), dtype=np.int32)
coords_with_batch = np.concatenate([batch_idx, coords], axis=1)

example = dict(
    voxels      = torch.from_numpy(voxels).float(),
    coordinates = torch.from_numpy(coords_with_batch).int(),
    num_points  = torch.from_numpy(num_points).int(),
    num_voxels  = torch.tensor([voxels.shape[0]]),
    shape       = torch.tensor([[voxel_gen.grid_size[0], voxel_gen.grid_size[1]]]),
    metadata    = [{"token": "synthetic_demo"}],
)

# ── Hook to capture BEV image (output of PointPillarsScatter) ─────────────────
bev_image_capture = {}

def capture_bev(module, input, output):
    bev_image_capture["bev"] = output.detach().cpu()

hook = model.backbone.register_forward_hook(capture_bev)

# ── Forward pass ──────────────────────────────────────────────────────────────
print("Running forward pass ...")
with torch.no_grad():
    data = dict(
        features=example["voxels"],
        num_voxels=example["num_points"],
        coors=example["coordinates"],
        batch_size=1,
        input_shape=example["shape"][0],
    )
    # Run PFN + Scatter + RPN
    pfn_out = model.reader(data["features"], data["num_voxels"], data["coors"])
    bev = model.backbone(pfn_out, data["coors"], data["batch_size"], data["input_shape"])
    rpn_out = model.neck(bev)
    # Raw CenterHead output (before NMS/decode)
    raw_preds, _ = model.bbox_head(rpn_out)

hook.remove()

print(f"  BEV image shape: {bev.shape}")
print(f"  RPN output shape: {rpn_out.shape}")
print(f"  CenterHead tasks: {len(raw_preds)}")
for i, d in enumerate(raw_preds):
    shapes = {k: list(v.shape) for k, v in d.items()}
    print(f"    task[{i}]: {shapes}")

# ── Cast to bfloat16 and save ─────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

# BEV image — this is the input to the TT model
bev_bf16 = bev.to(torch.bfloat16)
bev_path = os.path.join(OUTPUT_DIR, "centerpoint_bev_image_bf16.pt")
torch.save(bev_bf16, bev_path)
print(f"\nSaved BEV image: {bev_path}  shape={list(bev_bf16.shape)}")

# Raw preds — ground truth output of TT model (cast to bf16)
raw_preds_bf16 = []
for task_dict in raw_preds:
    raw_preds_bf16.append({k: v.to(torch.bfloat16) for k, v in task_dict.items()})
preds_path = os.path.join(OUTPUT_DIR, "centerpoint_raw_preds_bf16.pt")
torch.save(raw_preds_bf16, preds_path)
print(f"Saved raw preds:  {preds_path}  ({len(raw_preds_bf16)} tasks)")

# Weights — neck (RPN) + head (CenterHead) state dicts from the SAME model run
# Remap det3d bbox_head key prefix: "tasks." → "task_heads." to match standalone CenterHead
neck_sd = model.neck.state_dict()
head_sd_raw = model.bbox_head.state_dict()
head_sd = {k.replace("tasks.", "task_heads.", 1): v for k, v in head_sd_raw.items()}
weights_path = os.path.join(OUTPUT_DIR, "centerpoint_rpn_head_weights.pt")
torch.save({"rpn": neck_sd, "head": head_sd}, weights_path)
print(f"Saved weights:    {weights_path}  (rpn={len(neck_sd)} keys, head={len(head_sd)} keys)")

print("\nDone. Ground truth saved.")
print("Next: run scripts/verify_centerpoint_vs_cpu_gt.py (in tt-xla venv).")
