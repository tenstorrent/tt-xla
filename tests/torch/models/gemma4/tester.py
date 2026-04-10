# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Dict, Optional, Sequence

import torch
import torch_xla.runtime as xr
from infra import ComparisonConfig, Model, RunMode, TorchModelTester
from torch_xla.distributed.spmd import Mesh

from tests.infra.testers.compiler_config import CompilerConfig
from tests.infra.utilities.torch_multichip_utils import get_mesh
from third_party.tt_forge_models.gemma4.causal_lm.pytorch import ModelLoader


class Gemma4Tester(TorchModelTester):
    """Tester for Gemma4 causal LM model."""

    def __init__(
        self,
        variant_name,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
        compiler_config: CompilerConfig = None,
        dtype_override=None,
        num_layers=None,
    ) -> None:
        self._variant_name = variant_name
        self._model_loader = ModelLoader(variant_name, num_layers=num_layers)
        super().__init__(
            comparison_config,
            run_mode,
            compiler_config,
            dtype_override=dtype_override,
        )
        # Disable perf warmup runs — the 31B model segfaults on repeated
        # device execution, blocking the PCC comparison that follows.
        self._disable_perf_measurement = True

    # @override
    def _get_model(self) -> Model:
        return self._model_loader.load_model(dtype_override=torch.bfloat16)

    # @override
    def _get_input_activations(self) -> Dict | Sequence[Any]:
        return self._model_loader.load_inputs()

    # @override
    def _get_mesh(self) -> Optional[Mesh]:
        num_devices = xr.global_runtime_device_count()
        mesh_shape, mesh_names = self._model_loader.get_mesh_config(num_devices)
        if mesh_shape and mesh_names:
            return get_mesh(mesh_shape, mesh_names)
        return None

    # @override
    def _get_shard_specs_function(self) -> Optional[Callable]:
        num_devices = xr.global_runtime_device_count()
        mesh_shape, _ = self._model_loader.get_mesh_config(num_devices)
        if mesh_shape is None:
            return None
        return self._model_loader.load_shard_spec
