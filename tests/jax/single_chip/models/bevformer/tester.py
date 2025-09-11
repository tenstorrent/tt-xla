# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
from typing import Any, Dict, Sequence
from infra import ComparisonConfig, Model, RunMode, TorchModelTester
from third_party.tt_forge_models.bevformer.pytorch.loader import (
    ModelLoader,
    ModelVariant,
)
from tests.jax.single_chip.models.bevformer.model_utils.model_utils import (
    BEV_wrapper,
    build_from_input_image,
)


class BEVFormerTester(TorchModelTester):
    """Tester for BEVFormer model."""

    def __init__(
        self,
        variant_name: ModelVariant,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
    ) -> None:
        self._model_loader = ModelLoader(variant_name)
        super().__init__(comparison_config, run_mode)

    # @override
    def _get_model(self) -> Model:
        self._built = build_from_input_image(self._model_loader.load_inputs())
        wrapped_model = BEV_wrapper(
            self._model_loader.load_model(),
            self._built["filename"],
            self._built["box_mode_3d"],
            self._built["box_type_3d"],
            self._built["img_norm_cfg"],
            self._built["pts_filename"],
            self._built["can_bus"],
            self._built["lidar2img"],
            self._built["ori_shapes"],
            self._built["lidar2cam"],
            self._built["img_shapes"],
            self._built["pad_shape"],
        )
        return wrapped_model

    # @override
    def _get_input_activations(self) -> Dict | Sequence[Any]:
        built = getattr(self, "_built", None)
        if built is None:
            built = build_from_input_image(self._model_loader.load_inputs())
            self._built = built
        return [
            built["ori_shapes_tensor"],
            built["img_shapes_tensor"],
            built["lidar2img_stacked_tensor"],
            built["lidar2cam_stacked_tensor"],
            built["pad_shapes_tensor"],
            built["img_pybuda"],
        ]
