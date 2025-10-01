# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod

from infra.utilities import PyTree, Tensor

from .comparison_config import (
    AllcloseConfig,
    AtolConfig,
    ComparisonConfig,
    EqualConfig,
    PccConfig,
)

from dataclasses import dataclass


@dataclass
class ComparisonResult:
    passed: bool | None
    pcc: float | None
    atol: float | None
    allclose: bool | None
    equal: bool | None
    error_message: str | None = None


class Comparator(ABC):
    """
    Utility class providing comparison functionality.

    Provides an abstract interface for framework specific subclasses to implement.
    """

    def __init__(self, comparison_config: ComparisonConfig) -> None:
        """Initialize the comparator with comparison configuration."""
        self._comparison_config = comparison_config

    def compare(self, device_out: Tensor, golden_out: Tensor) -> ComparisonResult:
        """
        Compares device output with golden output based on ComparisonConfig provided
        during creation.

        Returns ComparisonResult with computed metrics.
        If config.assert_on_failure=True (default), also asserts on failure.
        """
        # Pack args in an iterable to simulate a pytree.
        device_output, golden_output = self._match_data_types((device_out, golden_out))
        _comparison_result = ComparisonResult(
            passed=None,
            pcc=None,
            atol=None,
            allclose=None,
            equal=None,
            error_message=None,
        )

        if self._comparison_config.equal.enabled:
            _comparison_result.equal = self._compare_equal(device_output, golden_output)
        if self._comparison_config.atol.enabled:
            _comparison_result.atol = self._compare_atol(
                device_output, golden_output, self._comparison_config.atol
            )
        if self._comparison_config.pcc.enabled:
            _comparison_result.pcc = self._compare_pcc(
                device_output, golden_output, self._comparison_config.pcc
            )
        if self._comparison_config.allclose.enabled:
            _comparison_result.allclose = self._compare_allclose(
                device_output, golden_output, self._comparison_config.allclose
            )

        # Evaluate the overall pass/fail status and capture any error message
        _comparison_result.passed, _comparison_result.error_message = (
            self._evaluate_results(_comparison_result)
        )

        # Check if any comparison failed and optionally assert
        if self._comparison_config.assert_on_failure:
            Comparator._assert_on_results(_comparison_result)

        return _comparison_result

    def _evaluate_results(
        self, comparison_result: ComparisonResult
    ) -> tuple[bool, str | None]:
        """
        Evaluate comparison results and return whether all enabled checks passed along with error message.

        Args:
            comparison_result: The ComparisonResult to evaluate

        Returns (passed, error_message) where:
        - passed: True if all enabled comparisons passed their thresholds, False otherwise
        - error_message: None if passed, combined error message for all failures if any failed
        """
        passed = True
        error_messages = []

        # Check each enabled comparison type and collect all failures
        if self._comparison_config.equal.enabled and comparison_result.equal is False:
            passed = False
            error_messages.append("Equal comparison failed.")

        if self._comparison_config.atol.enabled and comparison_result.atol is not None:
            required_atol = self._comparison_config.atol.required_atol
            if comparison_result.atol > required_atol:
                passed = False
                error_messages.append(
                    f"Atol comparison failed. "
                    f"Calculated: atol={comparison_result.atol}. Required: atol={required_atol}."
                )

        if self._comparison_config.pcc.enabled and comparison_result.pcc is not None:
            required_pcc = self._comparison_config.pcc.required_pcc
            if comparison_result.pcc < required_pcc:
                passed = False
                error_messages.append(
                    f"PCC comparison failed. "
                    f"Calculated: pcc={comparison_result.pcc}. Required: pcc={required_pcc}."
                )

        if (
            self._comparison_config.allclose.enabled
            and comparison_result.allclose is False
        ):
            passed = False
            allclose_config = self._comparison_config.allclose
            error_messages.append(
                f"Allclose comparison failed. "
                f"Required: atol={allclose_config.atol}, rtol={allclose_config.rtol}."
            )

        # Combine all error messages if any failures occurred
        combined_error_message = None
        if error_messages:
            combined_error_message = "\n".join(error_messages)

        return passed, combined_error_message

    @staticmethod
    def _assert_on_results(comparison_result: ComparisonResult) -> None:
        """
        Assert based on comparison results if any checks failed.

        Args:
            comparison_result: The ComparisonResult to assert on
        """
        # Simply assert if we have a failure with the stored error message
        if not comparison_result.passed:
            assert False, comparison_result.error_message

    @staticmethod
    @abstractmethod
    def _match_data_types(tensors: PyTree) -> PyTree:
        """Casts tensors to float32."""
        raise NotImplementedError("Subclasses must implement this method")

    @staticmethod
    @abstractmethod
    def _compare_equal(device_output: PyTree, golden_output: PyTree) -> bool:
        """Compares if device and golden output are equal. Returns True if equal, False otherwise."""
        raise NotImplementedError("Subclasses must implement this method")

    @staticmethod
    @abstractmethod
    def _compare_atol(
        device_output: PyTree, golden_output: PyTree, atol_config: AtolConfig
    ) -> float:
        """
        Compares maximum absolute difference between device and golden outputs.
        Returns the calculated absolute tolerance value.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @staticmethod
    @abstractmethod
    def _compare_pcc(
        device_output: PyTree, golden_output: PyTree, pcc_config: PccConfig
    ) -> float:
        """
        Compares PCC metric between device and golden output.
        Returns the calculated PCC value.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @staticmethod
    @abstractmethod
    def _compare_allclose(
        device_output: PyTree,
        golden_output: PyTree,
        allclose_config: AllcloseConfig,
    ) -> bool:
        """
        Compares if device and golden output are element-wise equal within a tolerance.
        Returns True if all elements are close, False otherwise.
        """
        raise NotImplementedError("Subclasses must implement this method")
