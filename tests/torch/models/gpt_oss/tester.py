# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Sequence

import torch_xla.runtime as xr
from infra import ComparisonConfig, Model, RunMode, TorchModelTester
from infra.utilities.torch_multichip_utils import get_mesh
from tt_torch.sparse_mlp import enable_sparse_mlp, get_moe_shard_specs

from tests.infra.testers.compiler_config import CompilerConfig
from third_party.tt_forge_models.config import Parallelism
from third_party.tt_forge_models.gpt_oss.pytorch import ModelLoader


class GptOssTester(TorchModelTester):
    """Tester for GPT-OSS model with MoE injection support."""

    def __init__(
        self,
        variant_name: str,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
        compiler_config: CompilerConfig = None,
        dtype_override=None,
        inject_custom_moe: bool = False,
    ) -> None:
        self._model_loader = ModelLoader(variant_name)
        self._inject_moe = inject_custom_moe
        super().__init__(
            comparison_config,
            run_mode,
            compiler_config,
            parallelism=Parallelism.TENSOR_PARALLEL,
            dtype_override=dtype_override,
        )

        if self._inject_moe:
            self._apply_custom_moe()

    def _apply_custom_moe(self):
        """Replace MoE layers with A2aSparseMLP and update shard specs."""
        mesh_info = self._workload.mesh.shape()
        mesh_shape = tuple(mesh_info.values())
        mesh_names = tuple(mesh_info.keys())
        enable_sparse_mlp(self._model, mesh=mesh_shape, cluster_axis=1)
        shard_spec_fn = self._workload.shard_spec_fn
        if shard_spec_fn:

            def combined_shard_spec_fn(model, _fn=shard_spec_fn, _names=mesh_names):
                return get_moe_shard_specs(model, _fn, _names)

            self._workload.shard_spec_fn = combined_shard_spec_fn

    # @override
    def _get_model(self) -> Model:
        return self._model_loader.load_model()

    # @override
    def _get_input_activations(self) -> Dict | Sequence[Any]:
        return self._model_loader.load_inputs()

    # @override
    def _get_shard_specs_function(self):
        return self._model_loader.load_shard_spec

    # @override
    def _get_mesh(self):
        num_devices = xr.global_runtime_device_count()
        mesh_shape, mesh_names = self._model_loader.get_mesh_config(num_devices)
        return get_mesh(mesh_shape, mesh_names)
