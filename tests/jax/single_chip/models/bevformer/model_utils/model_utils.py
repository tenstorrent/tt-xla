# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import numpy as np


def generate_random_lidar2img(n=6):
    lidar2img = []
    for _ in range(n):
        # Random 3x3 values simulating projection matrix (can be large)
        proj = np.random.uniform(-800, 800, size=(3, 3))

        # Simulate translation vector
        trans = np.random.uniform(-400, 400, size=(3, 1))

        # Combine into 3x4 matrix
        top = np.hstack((proj, trans))

        # Add the homogeneous row [0, 0, 0, 1]
        bottom = np.array([[0, 0, 0, 1]])

        # Full 4x4 projection matrix
        lidar2img_matrix = np.vstack((top, bottom))

        lidar2img.append(lidar2img_matrix)

    return lidar2img


def build_from_input_image(input_image):
    img_metas_data_container = input_image["img_metas"][0]
    img_metas = img_metas_data_container.data
    img_data_container = input_image["img"][0]

    filename = img_metas[0][0]["filename"]
    ori_shapes = img_metas[0][0]["ori_shape"]
    img_shapes = img_metas[0][0]["img_shape"]
    # lidar2img = img_metas[0][0]["lidar2img"]
    lidar2img = generate_random_lidar2img()
    lidar2cam = img_metas[0][0]["lidar2cam"]
    pad_shape = img_metas[0][0]["pad_shape"]
    box_mode_3d = img_metas[0][0]["box_mode_3d"]
    box_type_3d = img_metas[0][0]["box_type_3d"]
    img_norm_cfg = img_metas[0][0]["img_norm_cfg"]
    pts_filename = img_metas[0][0]["pts_filename"]
    # can_bus = img_metas[0][0]["can_bus"]
    can_bus = np.array(
        [
            np.random.uniform(500, 700),  # similar to 600
            np.random.uniform(1500, 1700),  # similar to 1647
            0.0,
            *np.random.uniform(-1, 0, 4),  # 4 negative values around -1
            np.random.uniform(-1, 1),  # similar to -0.6
            np.random.uniform(-0.2, 0.2),  # small float like -0.07
            np.random.uniform(8, 11),  # similar to 9.8
            *np.random.uniform(-0.05, 0.05, 3),  # three small values like -0.02
            np.random.uniform(8, 9),  # similar to 8.56
            0.0,
            0.0,
            np.random.uniform(5, 6),  # similar to 5.78
            np.random.uniform(300, 350),  # similar to 331
        ]
    )
    img = img_data_container.data

    # ori_shapes_tensor = torch.tensor(ori_shapes, dtype=torch.float32).unsqueeze(0)
    # img_shapes_tensor = torch.tensor(img_shapes, dtype=torch.float32).unsqueeze(0)
    # pad_shapes_tensor = torch.tensor(pad_shape, dtype=torch.float32).unsqueeze(0)

    # lidar2img_tensor = [torch.tensor(arr, dtype=torch.float32) for arr in lidar2img]
    # lidar2img_stacked_tensor = torch.stack(lidar2img_tensor, dim=0).unsqueeze(0)

    # lidar2cam_tensor = [torch.tensor(arr, dtype=torch.float32) for arr in lidar2cam]
    # lidar2cam_stacked_tensor = torch.stack(lidar2cam_tensor, dim=0).unsqueeze(0)

    img_pybuda = torch.randn(1, 6, 3, 480, 800)
    return {
        # "filename": filename,
        # "box_mode_3d": box_mode_3d,
        # # "box_type_3d": box_type_3d,
        # "img_norm_cfg": img_norm_cfg,
        # "pts_filename": pts_filename,
        "can_bus": can_bus,
        "lidar2img": [lidar2img],
        # "ori_shapes": ori_shapes,
        # "lidar2cam": lidar2cam,
        "img_shapes": img_shapes,
        # "pad_shape": pad_shape,
        # "ori_shapes_tensor": ori_shapes_tensor,
        # "img_shapes_tensor": img_shapes_tensor,
        # "pad_shapes_tensor": pad_shapes_tensor,
        # "lidar2img_stacked_tensor": lidar2img_stacked_tensor,
        # "lidar2cam_stacked_tensor": lidar2cam_stacked_tensor,
        "img_pybuda": img_pybuda,
    }


class BEV_wrapper(torch.nn.Module):
    def __init__(
        self,
        model,
        # filename,
        # box_mode_3d,
        # box_type_3d,
        # img_norm_cfg,
        # pts_filename,
        can_bus,
        lidar2img_orig,
        # ori_shape_orig,
        # lidar2cam_orig,
        img_shape_orig,
        # pad_shape_orig,
    ):
        super().__init__()
        self.model = model
        # self.filename = filename
        # self.box_mode_3d = box_mode_3d
        # self.box_type_3d = box_type_3d
        # self.img_norm_cfg = img_norm_cfg
        # self.pts_filename = pts_filename
        self.can_bus = can_bus
        self.lidar2img = lidar2img_orig
        # self.ori_shape = ori_shape_orig
        # self.lidar2cam = lidar2cam_orig
        self.img_shape = img_shape_orig
        # self.pad_shape = pad_shape_orig

    def forward(
        self,
        # ori_shapes_tensor,
        # img_shapes_tensor,
        # lidar2img_stacked_tensor,
        # lidar2cam_stacked_tensor,
        # pad_shapes_tensor,
        img_pybuda,
    ):
        # lidar2img_stacked_tensor = lidar2img_stacked_tensor.squeeze(0)
        # lidar2cam_stacked_tensor = lidar2cam_stacked_tensor.squeeze(0)
        # ori_shapes_tensor = ori_shapes_tensor.squeeze(0)
        # img_shapes_tensor = img_shapes_tensor.squeeze(0)
        # pad_shapes_tensor = pad_shapes_tensor.squeeze(0)
        img_pybuda = img_pybuda.squeeze(0)
        # lidar2img_array = [tensor.numpy() for tensor in lidar2img_stacked_tensor]
        # lidar2cam_array = [tensor.numpy() for tensor in lidar2cam_stacked_tensor]
        # ori_shapes_list = [tuple(row.tolist()) for row in ori_shapes_tensor]
        # img_shapes_list = [tuple(row.tolist()) for row in img_shapes_tensor]
        # pad_shapes_list = [tuple(row.tolist()) for row in pad_shapes_tensor]
        img_metas = {
            # "filename": self.filename,
            # # "ori_shape": ori_shapes_list,
            # "ori_shape": self.ori_shape,
            # # "img_shape": img_shapes_list,
            # "img_shape": self.img_shape,
            # # "lidar2img": lidar2img_array,
            "lidar2img": self.lidar2img,
            # # "lidar2cam": lidar2cam_array,
            # "lidar2cam": self.lidar2cam,
            # "pad_shape": pad_shapes_list,
            # "pad_shape": self.pad_shape,
            "scale_factor": 1.0,
            "flip": False,
            "pcd_horizontal_flip": False,
            "pcd_vertical_flip": False,
            # "box_mode_3d": self.box_mode_3d,
            # "box_type_3d": self.box_type_3d,
            # "img_norm_cfg": self.img_norm_cfg,
            "sample_idx": "3e8750f331d7499e9b5123e9eb70f2e2",
            "prev_idx": "",
            "next_idx": "3950bd41f74548429c0f7700ff3d8269",
            "pcd_scale_factor": 1.0,
            # "pts_filename": self.pts_filename,
            "scene_token": "fcbccedd61424f1b85dcbf8f897f9754",
            "can_bus": self.can_bus,
        }
        input_pybuda_dict = {
            "rescale": True,
            "img_metas": [[img_metas]],
            "img": [img_pybuda],
        }
        # print("input_pybuda_dict = ",input_pybuda_dict)
        output = self.model(return_loss=False, **input_pybuda_dict)
        # breakpoint()
        # boxes_3d_tensor = output[0]['pts_bbox']['boxes_3d'].tensor
        # output[0]['pts_bbox']['boxes_3d'] = boxes_3d_tensor

        # boxes_3d_tensor = output[0]['pts_bbox']['boxes_3d']
        # scores_3d_tensor = output[0]['pts_bbox']['scores_3d']
        # labels_3d_tensor = output[0]['pts_bbox']['labels_3d']
        # output = (boxes_3d_tensor.detach(), scores_3d_tensor.detach(), labels_3d_tensor)
        # logger.info(f"outputs = {output['all_bbox_preds']}")
        # return output['all_bbox_preds']
        return output
