# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import numpy as np


def generate_random_lidar2img(n=6):
    lidar2img = []
    for _ in range(n):
        proj = np.random.uniform(-800, 800, size=(3, 3))
        trans = np.random.uniform(-400, 400, size=(3, 1))
        top = np.hstack((proj, trans))
        bottom = np.array([[0, 0, 0, 1]])
        lidar2img_matrix = np.vstack((top, bottom))
        lidar2img.append(lidar2img_matrix)

    return lidar2img


def generate_random_lidar2cam(n=6):
    lidar2cam = []
    for _ in range(n):
        rot = np.random.uniform(-1.0, 1.0, size=(3, 3))
        rot, _ = np.linalg.qr(rot)
        trans = np.random.uniform(-1.0, 1.0, size=(3, 1))
        top = np.hstack((rot, trans))
        bottom = np.array([[0, 0, 0, 1]], dtype=np.float32)
        lidar2cam_matrix = np.vstack((top, bottom)).astype(np.float32)
        lidar2cam.append(lidar2cam_matrix)

    return lidar2cam


def generate_random_can_bus():
    can_bus = np.array(
        [
            np.random.uniform(500, 700),
            np.random.uniform(1500, 1700),
            0.0,
            *np.random.uniform(-1.0, 0.0, 4),
            np.random.uniform(-1.0, 0.0),
            np.random.uniform(-0.1, 0.1),
            np.random.uniform(8.0, 11.0),
            *np.random.uniform(-0.05, 0.05, 3),
            np.random.uniform(8.0, 9.0),
            0.0,
            0.0,
            np.random.uniform(5.0, 6.0),
            np.random.uniform(300, 350),
        ]
    )
    return can_bus


def build_inputs():

    img_shapes = [
        (480, 800, 3),
        (480, 800, 3),
        (480, 800, 3),
        (480, 800, 3),
        (480, 800, 3),
        (480, 800, 3),
    ]
    lidar2img = generate_random_lidar2img()
    can_bus = generate_random_can_bus()
    lidar2cam = generate_random_lidar2cam()
    img = torch.randn(1, 6, 3, 480, 800)
    img_shapes = [
        (480, 800, 3),
        (480, 800, 3),
        (480, 800, 3),
        (480, 800, 3),
        (480, 800, 3),
        (480, 800, 3),
    ]
    pad_shape = [
        (640, 1600, 3),
        (640, 1600, 3),
        (640, 1600, 3),
        (640, 1600, 3),
        (640, 1600, 3),
        (640, 1600, 3),
    ]
    img_norm_cfg = {
        "mean": np.array([103.53, 116.28, 123.675], dtype=np.float32),
        "std": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        "to_rgb": False,
    }
    ego2global_translation = [
        [
            torch.tensor([600.1202], dtype=torch.float64),
            torch.tensor([1647.4908], dtype=torch.float64),
            torch.tensor([0.0], dtype=torch.float64),
        ]
    ]
    ego2global_rotation = [
        [
            torch.tensor([-0.9687], dtype=torch.float64),
            torch.tensor([-0.0040], dtype=torch.float64),
            torch.tensor([-0.0077], dtype=torch.float64),
            torch.tensor([0.2482], dtype=torch.float64),
        ]
    ]
    lidar2ego_translation = [
        [
            torch.tensor([0.9858], dtype=torch.float64),
            torch.tensor([0.0], dtype=torch.float64),
            torch.tensor([1.8402], dtype=torch.float64),
        ]
    ]
    lidar2ego_rotation = [
        [
            torch.tensor([0.7067], dtype=torch.float64),
            torch.tensor([-0.0153], dtype=torch.float64),
            torch.tensor([0.0174], dtype=torch.float64),
            torch.tensor([-0.7071], dtype=torch.float64),
        ]
    ]
    timestamp = [torch.tensor([1.5332e09], dtype=torch.float64)]

    return {
        "can_bus": can_bus,
        "lidar2img": lidar2img,
        "lidar2cam": lidar2cam,
        "img_shapes": img_shapes,
        "img": img,
        "pad_shape": pad_shape,
        "img_norm_cfg": img_norm_cfg,
        "ego2global_translation": ego2global_translation,
        "ego2global_rotation": ego2global_rotation,
        "lidar2ego_translation": lidar2ego_translation,
        "lidar2ego_rotation": lidar2ego_rotation,
        "timestamp": timestamp,
    }
