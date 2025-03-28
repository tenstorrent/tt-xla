# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from infra import ComparisonConfig, ModelTester, RunMode
import torch
from torch import random
import numpy as np

from typing import Dict, Sequence
from transformers import (
    ViTHybridModel,
    ViTHybridForImageClassification,
    ViTHybridConfig,
    ViTHybridImageProcessor,
)


class ViTHybridTester(ModelTester):
    """Tester for ViT family of models."""

    def __init__(
        self,
        model_name: str,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
    ) -> None:
        self._model_name = model_name
        super().__init__(comparison_config, run_mode)

    # @override
    def _get_model(self) -> ViTHybridModel:
        return ViTHybridForImageClassification.from_pretrained(self._model_name)

    # @override
    def _get_input_activations(self) -> torch.Tensor:
        model_config = ViTHybridConfig.from_pretrained(self._model_name)
        image_size = model_config.image_size
        torch.manual_seed(0)
        random_image = torch.randint(
            low=0,
            high=256,
            size=(image_size, image_size, 3),
            dtype=torch.uint8,
        ).numpy()
        processor = ViTHybridImageProcessor.from_pretrained(self._model_name)
        inputs = processor(images=random_image, return_tensors="pt")
        return inputs["pixel_values"]

    # @override
    def _get_forward_method_kwargs(self) -> Dict[str, torch.Tensor]:
        assert hasattr(self._model, "params")
        return {
            "params": self._model.params,
            "pixel_values": self._get_input_activations(),
        }

    # @override
    def _get_static_argnames(self) -> Sequence[str]:
        return ["train"]
