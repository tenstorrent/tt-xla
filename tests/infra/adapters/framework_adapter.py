# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod

from utilities.types import Model
from utilities.workloads import Workload


class FrameworkAdapter(ABC):
    """
    Absctract base adapter class implementing framework-specific functionality needed in
    testers while providing a common interface.

    Meant to be injected in a tester and used whenever framework-specific behaviour
    is needed, concerning configuring NN models, compiling them, etc.
    """

    # -------------------- Public methods --------------------

    def configure_model_for_inference(self, model: Model) -> None:
        """Configures model for inference."""
        self._configure_model_for_inference(model)

    def configure_model_for_training(self, model: Model) -> None:
        """Configures model for training."""
        self._configure_model_for_training(model)

    def compile(self, workload: Workload) -> Workload:
        """Compiles workload into optimized kernels."""
        return self._compile(workload)

    # -------------------- Protected methods --------------------

    # --- For subclasses to override ---

    @abstractmethod
    def _configure_model_for_inference(self, model: Model) -> None:
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def _configure_model_for_training(self, model: Model) -> None:
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def _compile(self, workload: Workload) -> Workload:
        raise NotImplementedError("Subclasses must implement this method")
