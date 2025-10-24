# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for PCC/ATOL result assertion behavior in the test infrastructure."""

import jax.numpy as jnp
import pytest
import torch
from infra.comparators import AtolConfig, ComparisonConfig, PccConfig

from tests.infra.comparators.comparator import Comparator
from tests.infra.comparators.jax_comparator import JaxComparator
from tests.infra.comparators.torch_comparator import TorchComparator


# Fixture to provide framework-specific test data
@pytest.fixture(params=["torch", "jax"])
def framework_setup(request):
    """Fixture that provides comparator and tensor creation functions for each framework."""
    framework = request.param

    if framework == "torch":

        def create_tensor(data):
            return torch.tensor(data, dtype=torch.float32)

        comparator_class = TorchComparator
    else:  # jax

        def create_tensor(data):
            return jnp.array(data, dtype=jnp.float32)

        comparator_class = JaxComparator

    return {
        "framework": framework,
        "create_tensor": create_tensor,
        "comparator_class": comparator_class,
    }


@pytest.mark.push
def test_fail_atol_triggers_assertion_by_default(framework_setup):
    """Test that failing ATOL check triggers an assertion by default."""
    create_tensor = framework_setup["create_tensor"]
    comparator_class = framework_setup["comparator_class"]

    # Create config with strict ATOL requirement
    config = ComparisonConfig(
        atol=AtolConfig(enabled=True, required_atol=1e-6),
        pcc=PccConfig(enabled=False),
    )
    # assert_on_failure is True by default
    assert config.assert_on_failure is True

    comparator = comparator_class(config)

    # Create tensors that will fail ATOL check
    device_output = create_tensor([1.0, 2.0, 3.0])
    golden_output = create_tensor([1.0, 2.0, 3.5])  # 0.5 difference exceeds 1e-6

    # Should raise AssertionError because assert_on_failure=True by default
    with pytest.raises(AssertionError, match="Atol comparison failed"):
        comparator.compare(device_output, golden_output)


@pytest.mark.push
def test_fail_pcc_triggers_assertion_by_default(framework_setup):
    """Test that failing PCC check triggers an assertion by default."""
    create_tensor = framework_setup["create_tensor"]
    comparator_class = framework_setup["comparator_class"]

    # Create config with strict PCC requirement
    config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.999),
        atol=AtolConfig(enabled=False),
    )
    # assert_on_failure is True by default
    assert config.assert_on_failure is True

    comparator = comparator_class(config)

    # Create tensors that will fail PCC check
    # Scramble the values to break correlation (not just offset)
    device_output = create_tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    golden_output = create_tensor(
        [5.0, 1.0, 4.0, 2.0, 3.0]
    )  # Scrambled order breaks correlation

    # Should raise AssertionError because assert_on_failure=True by default
    with pytest.raises(AssertionError, match="PCC comparison failed"):
        comparator.compare(device_output, golden_output)


def test_pass_atol_no_assertion(framework_setup):
    """Test that passing ATOL check does not trigger an assertion."""
    create_tensor = framework_setup["create_tensor"]
    comparator_class = framework_setup["comparator_class"]

    # Create config with reasonable ATOL requirement
    config = ComparisonConfig(
        atol=AtolConfig(enabled=True, required_atol=1.0),
        pcc=PccConfig(enabled=False),
    )

    comparator = comparator_class(config)

    # Create tensors that will pass ATOL check
    device_output = create_tensor([1.0, 2.0, 3.0])
    golden_output = create_tensor([1.0, 2.0, 3.5])  # 0.5 difference is within 1.0

    # Should not raise - comparison passes
    result = comparator.compare(device_output, golden_output)
    assert result.passed is True
    assert result.atol is not None
    assert result.atol <= 1.0


@pytest.mark.push
def test_pass_pcc_no_assertion(framework_setup):
    """Test that passing PCC check does not trigger an assertion."""
    create_tensor = framework_setup["create_tensor"]
    comparator_class = framework_setup["comparator_class"]

    # Create config with reasonable PCC requirement
    config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.99),
        atol=AtolConfig(enabled=False),
    )

    comparator = comparator_class(config)

    # Create tensors that will pass PCC check (identical tensors)
    device_output = create_tensor([1.0, 2.0, 3.0, 4.0])
    golden_output = create_tensor([1.0, 2.0, 3.0, 4.0])

    # Should not raise - comparison passes
    result = comparator.compare(device_output, golden_output)
    assert result.passed is True
    assert result.pcc is not None
    assert result.pcc >= 0.99


@pytest.mark.push
def test_fail_atol_with_assert_on_failure_false_no_assertion(framework_setup):
    """Test that failing ATOL check with assert_on_failure=False does not trigger an assertion."""
    create_tensor = framework_setup["create_tensor"]
    comparator_class = framework_setup["comparator_class"]

    # Create config with strict ATOL but disable assertion
    config = ComparisonConfig(
        atol=AtolConfig(enabled=True, required_atol=1e-6),
        pcc=PccConfig(enabled=False),
        assert_on_failure=False,
    )

    comparator = comparator_class(config)

    # Create tensors that will fail ATOL check
    device_output = create_tensor([1.0, 2.0, 3.0])
    golden_output = create_tensor([1.0, 2.0, 3.5])  # 0.5 difference exceeds 1e-6

    # Should NOT raise AssertionError because assert_on_failure=False
    result = comparator.compare(device_output, golden_output)

    # But the result should still indicate failure
    assert result.passed is False
    assert result.atol is not None
    assert result.atol > 1e-6
    assert result.error_message is not None
    assert "Atol comparison failed" in result.error_message


@pytest.mark.push
def test_fail_pcc_with_assert_on_failure_false_no_assertion(framework_setup):
    """Test that failing PCC check with assert_on_failure=False does not trigger an assertion."""
    create_tensor = framework_setup["create_tensor"]
    comparator_class = framework_setup["comparator_class"]

    # Create config with strict PCC but disable assertion
    config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.999),
        atol=AtolConfig(enabled=False),
        assert_on_failure=False,
    )

    comparator = comparator_class(config)

    # Create tensors that will fail PCC check
    # Scramble the values to break correlation (not just offset)
    device_output = create_tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    golden_output = create_tensor(
        [5.0, 1.0, 4.0, 2.0, 3.0]
    )  # Scrambled order breaks correlation

    # Should NOT raise AssertionError because assert_on_failure=False
    result = comparator.compare(device_output, golden_output)

    # But the result should still indicate failure
    assert result.passed is False
    assert result.pcc is not None
    assert result.pcc < 0.999
    assert result.error_message is not None
    assert "PCC comparison failed" in result.error_message


@pytest.mark.push
def test_manual_assertion_after_assert_on_failure_false_still_fails(framework_setup):
    """Test that manually calling assert on failed results still raises an assertion."""
    create_tensor = framework_setup["create_tensor"]
    comparator_class = framework_setup["comparator_class"]

    # Create config with strict checks but disable automatic assertion
    config = ComparisonConfig(
        atol=AtolConfig(enabled=True, required_atol=1e-6),
        pcc=PccConfig(enabled=True, required_pcc=0.999),
        assert_on_failure=False,
    )

    comparator = comparator_class(config)

    # Create tensors that will fail both checks
    # Use scrambled values to break PCC correlation
    device_output = create_tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    golden_output = create_tensor(
        [5.0, 1.0, 4.0, 2.0, 3.0]
    )  # Scrambled order breaks correlation and has large ATOL

    # Should NOT raise during compare because assert_on_failure=False
    result = comparator.compare(device_output, golden_output)

    # Verify the result indicates failure
    assert result.passed is False

    # But manually calling _assert_on_results should still raise
    with pytest.raises(AssertionError, match="Comparison result 0 failed"):
        Comparator._assert_on_results(result)


@pytest.mark.push
def test_manual_assertion_on_tuple_of_results(framework_setup):
    """Test that manually calling assert on a tuple of results works correctly."""
    create_tensor = framework_setup["create_tensor"]
    comparator_class = framework_setup["comparator_class"]

    # Create config with checks disabled for automatic assertion
    config = ComparisonConfig(
        atol=AtolConfig(enabled=True, required_atol=1e-6),
        pcc=PccConfig(enabled=False),
        assert_on_failure=False,
    )

    comparator = comparator_class(config)

    # Create first comparison that passes
    device_output_1 = create_tensor([1.0, 2.0, 3.0])
    golden_output_1 = create_tensor([1.0, 2.0, 3.0])
    result_1 = comparator.compare(device_output_1, golden_output_1)
    assert result_1.passed is True

    # Create second comparison that fails
    device_output_2 = create_tensor([1.0, 2.0, 3.0])
    golden_output_2 = create_tensor([1.0, 2.0, 3.5])
    result_2 = comparator.compare(device_output_2, golden_output_2)
    assert result_2.passed is False

    # Manually calling assert on tuple should raise because result_2 failed
    with pytest.raises(AssertionError, match="Comparison result 1 failed"):
        Comparator._assert_on_results((result_1, result_2))


@pytest.mark.push
def test_multiple_failures_combined_error_message(framework_setup):
    """Test that multiple failures produce a combined error message."""
    create_tensor = framework_setup["create_tensor"]
    comparator_class = framework_setup["comparator_class"]

    # Create config with both ATOL and PCC enabled, both strict
    config = ComparisonConfig(
        atol=AtolConfig(enabled=True, required_atol=1e-6),
        pcc=PccConfig(enabled=True, required_pcc=0.999),
        assert_on_failure=False,  # Don't auto-assert so we can inspect the message
    )

    comparator = comparator_class(config)

    # Create tensors that will fail both checks
    # Use scrambled values to break PCC correlation and exceed ATOL threshold
    device_output = create_tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    golden_output = create_tensor(
        [5.0, 1.0, 4.0, 2.0, 3.0]
    )  # Scrambled order breaks correlation

    result = comparator.compare(device_output, golden_output)

    # Verify both failures are in the error message
    assert result.passed is False
    assert result.error_message is not None
    assert "Atol comparison failed" in result.error_message
    assert "PCC comparison failed" in result.error_message
    # Both errors should be on separate lines
    assert "\n" in result.error_message


@pytest.mark.push
def test_all_checks_pass_no_error_message(framework_setup):
    """Test that when all checks pass, there is no error message."""
    create_tensor = framework_setup["create_tensor"]
    comparator_class = framework_setup["comparator_class"]

    # Create config with reasonable thresholds
    config = ComparisonConfig(
        atol=AtolConfig(enabled=True, required_atol=1.0),
        pcc=PccConfig(enabled=True, required_pcc=0.99),
    )

    comparator = comparator_class(config)

    # Create identical tensors
    device_output = create_tensor([1.0, 2.0, 3.0, 4.0])
    golden_output = create_tensor([1.0, 2.0, 3.0, 4.0])

    result = comparator.compare(device_output, golden_output)

    # Verify success with no error message
    assert result.passed is True
    assert result.error_message is None


@pytest.mark.push
@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf"), float("-inf")])
def test_invalid_atol_triggers_assertion(framework_setup, invalid_value):
    """Test that invalid ATOL values (NaN, Inf, -Inf) trigger an assertion."""
    create_tensor = framework_setup["create_tensor"]
    comparator_class = framework_setup["comparator_class"]

    # Create config with ATOL check enabled
    config = ComparisonConfig(
        atol=AtolConfig(enabled=True, required_atol=0.01),
        pcc=PccConfig(enabled=False),
    )
    assert config.assert_on_failure is True

    comparator = comparator_class(config)

    # Create tensors with invalid values - will result in invalid ATOL
    device_output = create_tensor([1.0, 2.0, invalid_value])
    golden_output = create_tensor([1.0, 2.0, 3.0])

    # Should raise AssertionError because ATOL will be invalid
    with pytest.raises(AssertionError, match="Atol comparison failed"):
        comparator.compare(device_output, golden_output)


@pytest.mark.push
@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf"), float("-inf")])
def test_invalid_pcc_triggers_assertion(framework_setup, invalid_value):
    """Test that invalid PCC values (NaN, Inf, -Inf) trigger an assertion."""
    create_tensor = framework_setup["create_tensor"]
    comparator_class = framework_setup["comparator_class"]

    # Create config with PCC check enabled
    config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.99),
        atol=AtolConfig(enabled=False),
    )
    assert config.assert_on_failure is True

    comparator = comparator_class(config)

    # Create tensors with invalid values - will result in invalid PCC
    device_output = create_tensor([1.0, 2.0, invalid_value])
    golden_output = create_tensor([1.0, 2.0, 3.0])

    # Should raise AssertionError because PCC will be invalid
    with pytest.raises(AssertionError, match="PCC comparison failed"):
        comparator.compare(device_output, golden_output)
