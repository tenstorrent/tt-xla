# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from typing import Callable

import pytest
from infra.ttmlir import BaseModel, Test, model_to_dict


@pytest.fixture(scope="function", autouse=True)
def record_test_start_timestamp(record_property: Callable):
    """Autouse fixture used to capture start execution time of a test."""
    record_property("test_start_ts", datetime.now())

    # Run the test.
    yield


@pytest.fixture(scope="function", autouse=True)
def record_tt_xla_property(record_property: Callable):
    """
    Autouse fixture that automatically record some test properties for each test
    function.

    It also yields back callable which can be explicitly used in tests to record
    additional properties.

    Parameters:
    ----------
    record_property: Callable
        A pytest built-in function used to record test metadata, such as custom
        properties or additional information about the test execution.

    Yields:
    -------
    Callable
        The `record_property` callable, allowing tests to add additional properties if
        needed.

    Example:
    -------
    ```
    def test_model(fixture1, fixture2, ..., record_tt_xla_property):
        record_tt_xla_property("key", value)
        # Test logic...
    ```
    """
    # Automatically record some properties of the test.
    record_property("owner", "tt-xla")

    # Run the test.
    yield record_property


def collect_phase_outcome(
    item: pytest.Item,
    call: pytest.CallInfo,
    report: pytest.TestReport,
) -> None:
    if not hasattr(item, "_outcome_per_phase"):
        item._outcome_per_phase = {}

    item._outcome_per_phase[report.when] = {
        "outcome": report.outcome,  # or ("xfailed" if hasattr(report, "wasxfail") else report.outcome)
        "message": str(call.excinfo.value) if call.excinfo else "",
    }


def aggregate_outcomes(item: pytest.Item) -> tuple:
    success, skipped, message = True, False, ""

    for _, outcome_and_msg in item._outcome_per_phase.items():
        # Only if all phases "passed" aggregate outcome is success.
        success = success and outcome_and_msg["outcome"] == "passed"
        # If outcome of any phase is "skipped", aggregate outcome is skipped.
        skipped = skipped or outcome_and_msg["outcome"] == "skipped"
        # Any message that is not empty is capturing skip/xfail or fail reason.
        message = message or outcome_and_msg["message"]

    # We can now get rid of this field that we added in `collect_phase_outcome`
    # first time it executed.
    del item._outcome_per_phase

    return success, skipped, message


def get_recorded_test_properties(item: pytest.Item) -> dict:
    def validate_keys(keys: dict, is_model_test: bool):
        valid_keys = [
            "category",
            "jax_op_name",
            "shlo_op_name",
            "model_name",
            "group",
            "run_mode",
            "bringup_status",
            "pcc",
            "atol",
        ]

        # Check that only valid keys are used.
        if not all(key in valid_keys for key in keys):
            raise KeyError(
                f"Invalid keys found in 'record_test_properties' marker: {', '.join(keys)}. "
                f"Allowed keys are: {', '.join(valid_keys)}"
            )

        # If model test, check all necessary properties are provided.
        if is_model_test:
            mandatory_model_properties = [
                "model_name",
                "group",
                "run_mode",
                "bringup_status",
            ]

            if not all(
                model_property in keys for model_property in mandatory_model_properties
            ):
                raise KeyError(
                    f"Model tests must have following properties: "
                    f"{mandatory_model_properties}."
                )

    # Look for the custom marker.
    record_test_properties_marker = item.get_closest_marker(
        name="record_test_properties"
    )

    if not record_test_properties_marker:
        return {}

    # Extract the key-value pairs passed to the marker.
    recorded_properties: dict = record_test_properties_marker.kwargs

    # Check if the test is marked using the "model_test" marker.
    is_model_test = item.get_closest_marker(name="model_test") is not None
    # Validate that only allowed keys are used.
    validate_keys(recorded_properties.keys(), is_model_test)

    # Cast all keys to strings and return a new dict.
    return {k: str(v) for k, v in recorded_properties.items()}


def get_all_recorded_properties(item: pytest.Item) -> dict:
    # Fetch properties recorded with `record_test_properties` marker.
    marker_properties = get_recorded_test_properties(item)
    # Convert all properties recorded outside of `record_test_properties` marker
    # to dict since it is easier to fetch data from it.
    user_properties = dict(item.user_properties)

    # From this point on, it doesn't matter how we recorded properties, everything
    # recorded will be returned.
    return {**marker_properties, **user_properties}


def create_pydantic_model_from_recorded_properties(
    recorded_properties: dict,
    item: pytest.Item,
) -> BaseModel:
    # Extract everything in variables and set default values.
    # Everything we recorded that wasn't popped here will remain in `tags`
    # dict in the model.
    test_start_ts = recorded_properties.pop("test_start_ts", datetime.now())
    test_end_ts = datetime.now()
    full_test_name = item.originalname
    test_case_name = item.name
    filepath = str(item.path)
    category = recorded_properties.pop("category", "")
    group = recorded_properties.pop("group", None)
    owner = recorded_properties.pop("owner", None)

    # Aggregate outcomes for each phase into one final test outcome.
    success, skipped, message = aggregate_outcomes(item)

    # Now we have everything we need to create a pydantic model.
    pydantic_model = Test(
        test_start_ts=test_start_ts,
        test_end_ts=test_end_ts,
        full_test_name=full_test_name,
        success=success,
        skipped=skipped,
        error_message=message,
        tags=recorded_properties,
        config=None,  # Unused, not sure for other FEs
        test_case_name=test_case_name,
        filepath=filepath,
        category=category,
        group=group,
        owner=owner,
    )

    return pydantic_model


def produce_test_result(item: pytest.Item):
    # Collect all recorded properties.
    recorded_properties = get_all_recorded_properties(item)

    # Create appropriate pydantic model which serves as a final form of the test report.
    pydantic_model = create_pydantic_model_from_recorded_properties(
        recorded_properties, item
    )

    # Replace `user_properties` with dumped model to include it in the report.
    item.user_properties.clear()
    item.user_properties.append(("test_result", model_to_dict(pydantic_model)))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    """
    This hook is executed once for each test phase: setup, call and teardown.

    TODO see https://github.com/pytest-dev/pluggy/issues/569#issuecomment-2726487888
    """
    # In teardown phase test report is created. Make sure to prepare everything we want
    # to include in the report.
    if call.when == "teardown":
        produce_test_result(item)

    # This 'yield' runs the normal behavior of the hook. After it is done, it will
    # return control flow here together with hook result from which test report can
    # be fetched.
    outcome_report = yield

    # Collect `setup` and `call` phase outcomes and store it in `item`.
    # They will be used when producing final result during teardown and combined into
    # one final test outcome.
    if call.when != "teardown":
        collect_phase_outcome(item, call, outcome_report.get_result())
