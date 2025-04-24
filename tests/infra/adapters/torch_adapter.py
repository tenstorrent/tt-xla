# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
from utilities.types import Model
from utilities.workloads.torch_workload import TorchWorkload, Workload

from .framework_adapter import FrameworkAdapter


class TorchAdapter(FrameworkAdapter):
    """Adapter for Torch."""

    # -------------------- Private methods --------------------

    # --- Overrides ---

    # @override
    def _configure_model_for_inference(self, model: Model) -> None:
        if isinstance(model, torch.nn.Module):
            model.eval()
        else:
            raise TypeError(f"Unknown model type: {type(model)}")

    # @override
    def _configure_model_for_training(self, model: Model) -> None:
        if isinstance(model, torch.nn.Module):
            model.train()
        else:
            raise TypeError(f"Unknown model type: {type(model)}")

    # @override
    def _compile(self, workload: Workload) -> Workload:
        """JIT-compiles model or executable into optimized kernels."""
        assert isinstance(workload, TorchWorkload)

        if workload.model is not None:
            workload.model.compile(backend="openxla")
        else:
            workload.executable = torch.compile(workload.executable, backend="openxla")

        return workload
