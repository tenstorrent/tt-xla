# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Sequence

from infra.utilities import Framework, Model


class Workload:
    """Class encapsulating workload (executable/model with its inputs).

    Workload needs both model and executable fields depending on the tests,
    for example model tests use both and op tests use only executable.
    Please also pay attention that run_on_cpu decorator used in _match_data_types()
    creates a workload with executable.
    """

    def __init__(
        self,
        framework: Framework,
        executable: Optional[Callable] = None,
        compiled_executable: Optional[Callable] = None,
        model: Optional[Model] = None,
        args: Optional[Sequence[Any]] = None,
        kwargs: Optional[Mapping[str, Any]] = None,
        static_argnames: Optional[Sequence[str]] = None,
    ) -> None:

        self.framework = framework

        assert (
            executable is not None or model is not None
        ), f"Workload must either have executable or model provided"

        self.executable = executable
        self.compiled_executable = compiled_executable
        self.model = model

        assert (
            args is not None or kwargs is not None
        ), f"Workload must either have args or kwargs provided"

        self.args = args or []
        self.kwargs = kwargs or {}
        # TODO: Move static_argnames out of Workload.
        # This field is JAX-specific and only used in compile functions.
        # Currently needed because _safely_put_workload_on_device relies on it to avoid putting those args on device.
        # Consider reworking _safely_put_workload_on_device to eliminate the need for static_argnames in Workload.
        self.static_argnames = static_argnames or []

        self.shard_spec_fn = None
        self.mesh = None

    @property
    def is_jax(self) -> bool:
        return self.framework == Framework.JAX

    @property
    def is_torch(self) -> bool:
        return self.framework == Framework.TORCH

    def execute(self) -> Any:
        """Calls callable passing stored args and kwargs directly."""
        if self.model is not None:
            return self.model(*self.args, **self.kwargs)
        elif self.compiled_executable is not None:
            return self.compiled_executable(*self.args, **self.kwargs)
        elif self.executable is not None:
            return self.executable(*self.args, **self.kwargs)
        else:
            raise ValueError(
                "No model, compiled_executable, or executable provided in Workload."
            )
