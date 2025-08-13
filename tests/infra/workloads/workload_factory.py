# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Sequence

from infra.utilities import Framework, Model

from .workload import Workload


class WorkloadFactory:
    """Factory creating Workloads."""

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
        
        if not isinstance(framework, Framework):
            raise ValueError(f"Unsupported framework {framework}")
        
        return Workload(framework, executable, model, args, kwargs, static_argnames)
