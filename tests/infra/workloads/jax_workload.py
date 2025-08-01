# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

from infra.utilities import ShardingMode
from jax.sharding import Mesh, PartitionSpec

from .workload import Workload


class JaxWorkload(Workload):
    """Workload used with JAX."""

    def __init__(
        self,
        executable: Callable,
        args: Optional[Sequence[Any]] = None,
        kwargs: Optional[Mapping[str, Any]] = None,
        static_argnames: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(args, kwargs)

        self.executable = executable
        self.static_argnames = static_argnames or []

    def as_mlir_module_str(self) -> str:
        """
        Returns self as mlir module string.

        Note `self.executable` must be the result of jax.jit, otherwise exception will
        be raised.
        """
        try:
            return self.executable.lower(*self.args, **self.kwargs).as_text()
        except Exception as e:
            raise RuntimeError("Couldn't produce MLIR module str from workload.") from e

    # @override
    def _execute(self) -> Any:
        """Calls callable passing stored args and kwargs directly."""
        return self.executable(*self.args, **self.kwargs)


@dataclass
class JaxMultichipWorkload(JaxWorkload):
    """
    An extension of the JaxWorkload dataclass that includes a mesh and partition specs,
    necessary for multichip sharding.
    """

    def __init__(
        self,
        executable: Callable,
        args: Optional[Sequence[Any]] = None,
        kwargs: Optional[Mapping[str, Any]] = None,
        static_argnames: Optional[Sequence[str]] = None,
        device_mesh: Optional[Mesh] = None,
        in_specs: Optional[Sequence[PartitionSpec]] = None,
        out_spec: Optional[PartitionSpec] = None,
        sharding_mode: Optional[ShardingMode] = None,
    ) -> None:
        super().__init__(executable, args, kwargs, static_argnames)

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
