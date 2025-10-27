# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Sequence

from infra.utilities import Framework, Mesh, Model
from infra.utilities.torch_multichip_utils import enable_spmd

from .workload import Workload


class TorchWorkload(Workload):
    """Class encapsulating workload (executable/model with its inputs).

    Workload needs both model and executable fields depending on the tests,
    for example model tests use both and op tests use only executable.
    Please also pay attention that run_on_cpu decorator used in _match_data_types()
    creates a workload with executable.
    """

    def __init__(
        self,
        executable: Optional[Callable] = None,
        compiled_executable: Optional[Callable] = None,
        model: Optional[Model] = None,
        args: Optional[Sequence[Any]] = None,
        kwargs: Optional[Mapping[str, Any]] = None,
        static_argnames: Optional[Sequence[str]] = None,
        mesh: Optional[Mesh] = None,
        shard_spec_fn: Optional[Callable] = None,
    ) -> None:

        super().__init__(
            framework=Framework.TORCH,
            executable=executable,
            compiled_executable=compiled_executable,
            model=model,
            args=args,
            kwargs=kwargs,
            static_argnames=static_argnames,
        )
        self.mesh = mesh
        self.shard_spec_fn = shard_spec_fn
        self._enable_xla_spmd_if_needed()

    # If model has shard specs and running on multichip mesh, then convert StableHLO
    # to Shardy dialect and initialize XLA SPMD runtime.
    def _enable_xla_spmd_if_needed(self) -> None:
        is_multichip = bool(self.mesh and len(self.mesh.device_ids) > 1)
        has_shard_specs = False
        if callable(self.shard_spec_fn) and self.model is not None:
            has_shard_specs = bool(self.shard_spec_fn(self.model))

        if has_shard_specs and is_multichip:
            enable_spmd()
