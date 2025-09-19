# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from infra.utilities import PyTree, Tensor

from .comparison_config import AllcloseConfig, AtolConfig, ComparisonConfig, PccConfig


@dataclass
class ComparisonResult:
    """Holds the results of comparison metrics."""

    pcc: Optional[float] = None
    atol: Optional[float] = None
    passed: bool = True
    error_message: Optional[str] = None


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
        result = self.compare_with_metrics(device_out, golden_out)
        if not result.passed:
            raise AssertionError(result.error_message)

    def compare_with_metrics(
        self, device_out: Tensor, golden_out: Tensor
    ) -> ComparisonResult:
        """
        Compares device output with golden output and returns detailed metrics.
        """
        # Pack args in an iterable to simulate a pytree.
        device_output, golden_output = self._match_data_types((device_out, golden_out))

        result = ComparisonResult()

        try:
            if self._comparison_config.equal.enabled:
                self._compare_equal(device_output, golden_output)
            if True or self._comparison_config.atol.enabled:
                result.atol = self._compare_atol_with_metrics(
                    device_output, golden_output, self._comparison_config.atol
                )
            if True or self._comparison_config.pcc.enabled:
                result.pcc = self._compare_pcc_with_metrics(
                    device_output, golden_output, self._comparison_config.pcc
                )
            if self._comparison_config.allclose.enabled:
                self._compare_allclose(
                    device_output, golden_output, self._comparison_config.allclose
                )
        except AssertionError as e:
            result.passed = False
            result.error_message = str(e)

        return result

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
    def _compare_atol_with_metrics(
        device_output: PyTree, golden_output: PyTree, atol_config: AtolConfig
    ) -> float:
        """
        Compares ATOL and returns the calculated ATOL value.
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
    def _compare_pcc_with_metrics(
        device_output: PyTree, golden_output: PyTree, pcc_config: PccConfig
    ) -> float:
        """
        Compares PCC and returns the calculated PCC value.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @staticmethod
    @abstractmethod
    def _compare_allclose(
        device_output: PyTree,
        golden_output: PyTree,
        allclose_config: AllcloseConfig,
    ) -> None:
        """
        Compares if device and golden output are element-wise equal within a tolerance.
        Asserts if not.
        """
        raise NotImplementedError("Subclasses must implement this method")
