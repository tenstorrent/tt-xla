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


def build_random_inputs():

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
    img = torch.randn(1, 6, 3, 480, 800)

    return {
        "can_bus": can_bus,
        "lidar2img": lidar2img,
        "img_shapes": img_shapes,
        "img": img,
    }
