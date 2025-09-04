# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod

from infra.utilities import PyTree, Tensor

from .comparison_config import AllcloseConfig, AtolConfig, ComparisonConfig, PccConfig


class Comparator(ABC):
    """
    Utility class providing comparison functionality.

    Provides an abstract interface for framework specific subclasses to implement.
    """

    def __init__(self, comparison_config: ComparisonConfig) -> None:
        """Initialize the comparator with comparison configuration."""
        self._comparison_config = comparison_config

    def compare(self, device_out: Tensor, golden_out: Tensor) -> None:
        """
        Compares device output with golden output based on ComparisonConfig provided
        during creation.
        """
        # Pack args in an iterable to simulate a pytree.
        device_output, golden_output = self._match_data_types((device_out, golden_out))
        if self._comparison_config.equal.enabled:
            self._compare_equal(device_output, golden_output)
        if self._comparison_config.atol.enabled:
            self._compare_atol(
                device_output, golden_output, self._comparison_config.atol
            )
        if self._comparison_config.pcc.enabled:
            self._compare_pcc(device_output, golden_output, self._comparison_config.pcc)
        if self._comparison_config.allclose.enabled:
            self._compare_allclose(
                device_output, golden_output, self._comparison_config.allclose
            )

    @staticmethod
    @abstractmethod
    def _match_data_types(tensors: PyTree) -> PyTree:
        """Casts tensors to float32."""
        raise NotImplementedError("Subclasses must implement this method")

    @staticmethod
    @abstractmethod
    def _compare_equal(device_output: PyTree, golden_output: PyTree) -> None:
        """Compares if device and golden output are equal. Asserts if not."""
        raise NotImplementedError("Subclasses must implement this method")

    @staticmethod
    @abstractmethod
    def _compare_atol(
        device_output: PyTree, golden_output: PyTree, atol_config: AtolConfig
    ) -> None:
        """
        Compares if maximum absolute difference between device and golden outputs is
        within required tolerance. Asserts if not.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @staticmethod
    @abstractmethod
    def _compare_pcc(
        device_output: PyTree, golden_output: PyTree, pcc_config: PccConfig
    ) -> None:
        """
        Compares if PCC metric between device and golden output is within required PCC.
        Asserts if not.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @staticmethod
    @abstractmethod
    def _compare_allclose(
        self,
        device_output: PyTree,
        golden_output: PyTree,
        allclose_config: AllcloseConfig,
    ) -> None:
        """
        Compares if device and golden output are element-wise equal within a tolerance.
        Asserts if not.
        """
        raise NotImplementedError("Subclasses must implement this method")
