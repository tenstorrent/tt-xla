# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from third_party.tt_forge_models.bevformer.pytorch.loader import ModelLoader
from tests.jax.single_chip.models.bevformer.model_utils.model_utils import (
    BEV_wrapper,
    build_from_input_image,
)


def test_bevformer():

    # Load model and inputs
    loader = ModelLoader()
    framework_model = loader.load_model()
    inputs_dict = loader.load_inputs()
    print("model = ", framework_model)

    built = build_from_input_image(inputs_dict)

    wrapper_bev_model = BEV_wrapper(
        framework_model,
        # built["filename"],
        # built["box_mode_3d"],
        # built["box_type_3d"],
        # built["img_norm_cfg"],
        # built["pts_filename"],
        built["can_bus"],
        built["lidar2img"],
        # built["ori_shapes"],
        # built["lidar2cam"],
        built["img_shapes"],
        # built["pad_shape"],
    )
    wrapper_bev_model.eval()
    result_wrapper = wrapper_bev_model(
        # built["ori_shapes_tensor"],
        # built["img_shapes_tensor"],
        # built["lidar2img_stacked_tensor"],
        # built["lidar2cam_stacked_tensor"],
        # built["pad_shapes_tensor"],
        built["img_pybuda"],
    )
    print("result_wrapper = ", result_wrapper)


if __name__ == "__main__":
    test_bevformer()
