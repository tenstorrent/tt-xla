# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Sequence

from .workload import Workload


class JaxWorkload(Workload):
    """Workload used with JAX."""

    # -------------------- Private methods --------------------

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

    # --- Overrides ---

    # @override
    def _execute(self) -> Any:
        """Calls callable passing stored args and kwargs directly."""
        return self.executable(*self.args, **self.kwargs)
