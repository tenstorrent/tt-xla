# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

from jax.sharding import Mesh, PartitionSpec
from jaxtyping import PyTree


@dataclass
class Workload:
    """
    Convenience dataclass storing a callable and its positional and keyword arguments.
    """

    executable: Callable
    args: Sequence[Any]
    kwargs: Optional[Mapping[str, Any]] = None
    static_argnames: Optional[Sequence[str]] = None

    def __post_init__(self):
        # If kwargs is None, initialize it to an empty dictionary.
        if self.kwargs is None:
            self.kwargs = {}
        if self.static_argnames is None:
            self.static_argnames = []

    def execute(self) -> Any:
        """Calls callable passing stored args and kwargs directly."""
        return self.executable(*self.args, **self.kwargs)


@dataclass
class MultichipWorkload(Workload):
    """
    An extension of the Workload dataclass that includes a mesh and partition specs, necessary for multichip sharding.
    """

    device_mesh: Mesh = None
    in_specs: Sequence[PartitionSpec | PyTree] = None
