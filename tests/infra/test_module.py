# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Sequence

import jax
from jax import export


class TestModule:
    """
    Wrapper around a callable and its arguments.

    Single-op or multi-op graphs defined as a python function are wrapped in a
    TestModule for convenience. TestModule is then used throughout test infra.
    """

    def __init__(
        self,
        f: Callable,
        args: Sequence[Any],
        kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self._f = f
        self._args = args
        self._kwargs = kwargs if kwargs is not None else {}

        self._inputs = tuple(self._args) + tuple(self._kwargs.values())

    def get_inputs(self) -> Sequence[Any]:
        return self._inputs

    def get_jit_graph(self):
        return jax.jit(self._f)

    def __call__(self):
        """Calls underlying callable with passed underlying args."""
        return self._f(*self._inputs)

    def __repr__(self) -> str:
        return f"TestModule: {self._f.__qualname__}"

    def as_mlir_module(self) -> str:
        """
        Returns jitted graph as a mlir module string.

        Note that this only works if test module can be successfully run in jitted form.
        """
        s = export.export(self.get_jit_graph())(*self.get_inputs()).mlir_module()
        # Remove all #loc lines for cleaner output.
        return "\n".join(line for line in s.splitlines() if not line.startswith("#loc"))
