# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Sequence

import torch
from infra.utilities import Framework, Model
from torch.utils._pytree import tree_map


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

    @property
    def is_jax(self) -> bool:
        return self.framework == Framework.JAX

    @property
    def is_torch(self) -> bool:
        return self.framework == Framework.TORCH

    def execute(self) -> Any:
        """Calls callable passing stored args and kwargs directly."""
        result = (
            self.model(*self.args, **self.kwargs)
            if self.model is not None
            else self.executable(*self.args, **self.kwargs)
        )

        # PyTorch models will leave their outputs on the device.
        # Since the model output will be an arbitrary collection of tensors,
        # use tree_map to move all tensors to CPU.
        if result is not None and self.framework == Framework.TORCH:
            # NOTE: Each call to .to('cpu') will halt until that tensor's data
            # has been populated. That is to say, the program has finished compiling and executing.
            # However, once the program has finished executing, all outputs will be ready, so in reality
            # the halt will only be for the first tensor, and the rest will be ready immediately.
            result = tree_map(
                lambda x: x.to("cpu") if isinstance(x, torch.Tensor) else x, result
            )

        return result
