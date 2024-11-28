# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Optional, Sequence

import jax

from .test_module import TestModule


class TestModel:
    """
    Interface class for testing models.

    Provides methods which models to be tested must override. Provides a way to export
    self to `TestModule` which is then used throughout test infra.
    """

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> jax.Array:
        raise NotImplementedError("Subclasses should implement this method")

    @staticmethod
    @abstractmethod
    def get_model() -> TestModel:
        raise NotImplementedError("Subclasses should implement this method")

    @staticmethod
    @abstractmethod
    def get_model_inputs() -> Sequence[jax.Array]:
        raise NotImplementedError("Subclasses should implement this method")

    def as_test_module(
        self, inputs: Optional[Sequence[jax.Array]] = None
    ) -> TestModule:
        ins = inputs if inputs is not None else self.get_model_inputs()
        return TestModule(self.__call__, ins)

    def __repr__(self) -> str:
        return f"TestModel: {self.__class__.__qualname__}"
