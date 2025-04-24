# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Sequence

from ..types import Framework, Model
from .jax_workload import JaxWorkload
from .torch_workload import TorchWorkload
from .workload import Workload


class WorkloadFactory:
    """Factory creating Workloads based on provided framework."""

    # -------------------- Public methods --------------------

    def __init__(self, framework: Framework) -> None:
        self._framework = framework

    def create_workload(
        self,
        *,
        executable: Optional[Callable] = None,
        model: Optional[Model] = None,
        args: Optional[Sequence[Any]] = None,
        kwargs: Optional[Mapping[str, Any]] = None,
        static_argnames: Optional[Sequence[str]] = None,
    ) -> Workload:
        if self._framework == Framework.JAX:
            assert (
                executable is not None
            ), f"`executable` must be provided for JaxWorkload."

            return JaxWorkload(executable, args, kwargs, static_argnames)
        elif self._framework == Framework.TORCH:
            assert (
                executable is not None or model is not None
            ), f"Either `executable` or `model` must be provided for TorchWorkload."

            return TorchWorkload.create(executable, model, args, kwargs)
        else:
            raise ValueError(f"Unsupported framework {self._framework}")
