# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from enum import Enum
from typing import Callable

import pytest


class RecordProperties(Enum):
    """Properties we can record."""

    # Timestamp of test start.
    START_TIMESTAMP = "start_timestamp"
    # Timestamp of test end.
    END_TIMESTAMP = "end_timestamp"
    # Frontend or framework used to run the test.
    FRONTEND = "frontend"
    # Kind of operation. e.g. eltwise.
    OP_KIND = "op_kind"
    # Name of the operation in the framework. e.g. torch.conv2d.
    FRAMEWORK_OP_NAME = "framework_op_name"
    # Name of the operation. e.g. ttir.conv2d.
    OP_NAME = "op_name"
    # Name of the model in which this op appears.
    MODEL_NAME = "model_name"


@pytest.fixture(scope="function", autouse=True)
def record_test_timestamp(record_property: Callable):
    """
    Autouse fixture used to capture execution time of a test.

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
    --------
    ```
    def test_model(fixture1, fixture2, ..., record_tt_xla_property):
        record_tt_xla_property("key", value)

        # Test logic...
    ```
    """
    start_timestamp = datetime.strftime(datetime.now(), "%Y-%m-%dT%H:%M:%S%z")
    record_property(RecordProperties.START_TIMESTAMP.value, start_timestamp)

    # Run the test.
    yield

    end_timestamp = datetime.strftime(datetime.now(), "%Y-%m-%dT%H:%M:%S%z")
    record_property(RecordProperties.END_TIMESTAMP.value, end_timestamp)


@pytest.fixture(scope="function", autouse=True)
def record_tt_xla_property(record_property: Callable):
    """
    Autouse fixture that automatically records some test properties for each test
    function.

    It also yields back callable which can be explicitly used in tests to record
    additional properties.

    Example:

    ```
    def test_model(fixture1, fixture2, ..., record_tt_xla_property):
        record_tt_xla_property("key", value)

        # Test logic...
    ```

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
    """
    # Record default properties for tt-xla.
    record_property(RecordProperties.FRONTEND.value, "tt-xla")

    # Run the test.
    yield record_property
