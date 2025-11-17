# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

from infra.utilities import Framework, ShardingMode
from jax.sharding import Mesh, PartitionSpec

from .workload import Workload


@dataclass
class JaxMultichipWorkload(Workload):
    """
    An extension of the Workload dataclass that includes a mesh and partition specs,
    necessary for multichip sharding for JAX framework.
    """

    def __init__(
        self,
        executable: Callable,
        compiled_executable: Optional[Callable] = None,
        args: Optional[Sequence[Any]] = None,
        kwargs: Optional[Mapping[str, Any]] = None,
        static_argnames: Optional[Sequence[str]] = None,
        device_mesh: Optional[Mesh] = None,
        in_specs: Optional[Sequence[PartitionSpec]] = None,
        out_spec: Optional[PartitionSpec] = None,
        sharding_mode: Optional[ShardingMode] = None,
    ) -> None:

        super().__init__(
            framework=Framework.JAX,
            executable=executable,
            compiled_executable=compiled_executable,
            args=args,
            kwargs=kwargs,
            static_argnames=static_argnames,
        )

        self._device_mesh = device_mesh
        self._in_specs = in_specs
        self._out_spec = out_spec
        self._sharding_mode = sharding_mode

    # --- Convenience getters that remove optionality out of attributes ---

    @property
    def device_mesh(self) -> Mesh:
        assert self._device_mesh is not None
        return self._device_mesh

    @property
    def in_specs(self) -> Sequence[PartitionSpec]:
        assert self._in_specs is not None
        return self._in_specs

    @property
    def out_spec(self) -> PartitionSpec:
        assert self._out_spec is not None
        return self._out_spec

    @property
    def sharding_mode(self) -> ShardingMode:
        assert self._sharding_mode is not None
        return self._sharding_mode

    def execute(self) -> Any:
        with self.device_mesh:
            return super().execute()
