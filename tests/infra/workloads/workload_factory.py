# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Sequence

from infra.utilities import Framework, Model

from .jax_workload import JaxWorkload
from .torch_workload import TorchWorkload
from .workload import Workload


class WorkloadFactory:
    """Factory creating Workloads based on provided framework."""

    @staticmethod
    def create_workload(
        framework: Framework,
        *,
        executable: Optional[Callable] = None,
        model: Optional[Model] = None,
        args: Optional[Sequence[Any]] = None,
        kwargs: Optional[Mapping[str, Any]] = None,
        static_argnames: Optional[Sequence[str]] = None,
    ) -> Workload:
        """
        Creates appropriate workload based on `framework`.

        `JaxWorkload` must have an `executable` provided.
        `TorchWorkload` can either have `executable` or `model` provided.
        See docs of these classes for better understanding.
        """
        if framework == Framework.JAX:
            assert (
                executable is not None
            ), f"`executable` must be provided for JaxWorkload."

            return JaxWorkload(executable, args, kwargs, static_argnames)
        elif framework == Framework.TORCH:
            assert (
                executable is not None or model is not None
            ), f"Either `executable` or `model` must be provided for TorchWorkload."

            return TorchWorkload.create(executable, model, args, kwargs)
        else:
            raise ValueError(f"Unsupported framework {framework}")
