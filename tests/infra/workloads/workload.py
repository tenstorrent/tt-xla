# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Optional, Sequence


class Workload(ABC):
    """Abstract base class encapsulating workload (executable/model with its inputs)."""

    def __init__(
        self,
        args: Optional[Sequence[Any]] = None,
        kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        assert (
            args is not None or kwargs is not None
        ), f"Workload must either have args or kwargs provided"

        self.args = args or []
        self.kwargs = kwargs or {}

    def execute(self) -> Any:
        return self._execute()

    @abstractmethod
    def _execute(self) -> Any:
        raise NotImplementedError("Subclasses must implement this method")
