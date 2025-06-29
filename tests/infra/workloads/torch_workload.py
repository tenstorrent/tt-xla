# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Sequence

from infra.utilities import Model

from .workload import Workload


class TorchWorkload(Workload):
    """Workload used with Torch."""

    # -------------------- Public methods --------------------

    @staticmethod
    def create(
        executable: Optional[Callable] = None,
        model: Optional[Model] = None,
        args: Optional[Sequence[Any]] = None,
        kwargs: Optional[Mapping[str, Any]] = None,
    ) -> TorchWorkload:
        """
        Factory method implemented to hide complexity of creation.

        If `model` is provided, we don't need an `executable`. We set it to None
        explicitly. This is done to distinguish torch workloads that are created to
        carry models and those that carry regular functions. Workloads that carry
        models call different Torch `compile` API and `model` needs to explicitly be
        put on device using `.to()`, whereas functions don't.
        """
        if model is not None:
            # Carry only model.
            return TorchWorkload(None, model, args, kwargs)
        else:
            # Carry only executable.
            return TorchWorkload(executable, None, args, kwargs)

    # -------------------- Private methods --------------------

    def __init__(
        self,
        executable: Optional[Callable] = None,
        model: Optional[Model] = None,
        args: Optional[Sequence[Any]] = None,
        kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Private constructor. Use provided factory method instead."""
        super().__init__(args, kwargs)

        self.executable = executable
        self.model = model

    # --- Overrides ---

    # @override
    def _execute(self) -> Any:
        """Calls callable passing stored args and kwargs directly."""
        return (
            self.model(*self.args, **self.kwargs)
            if self.model is not None
            else self.executable(*self.args, **self.kwargs)
        )
