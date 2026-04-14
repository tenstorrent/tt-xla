# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Optional, Sequence

import torch
from infra import ComparisonConfig, Model, RunMode, TorchModelTester

from third_party.tt_forge_models.whisper.pytorch import ModelLoader

from .model_utils import WhisperWrapper


class WhisperTester(TorchModelTester):
    """Tester for Whisper model."""

    def __init__(
        self,
        variant_name,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ) -> None:
        self._variant_name = variant_name
        self._dtype_override = dtype_override
        self._model_loader = ModelLoader(variant_name)
        super().__init__(comparison_config, run_mode, **kwargs)

    # @override
    def _get_model(self) -> Model:
        model = self._model_loader.load_model(dtype_override=self._dtype_override)
        model = WhisperWrapper(model, variant=self._variant_name)
        return model

    # @override
    def _get_input_activations(self) -> Dict | Sequence[Any]:
        return self._model_loader.load_inputs()
