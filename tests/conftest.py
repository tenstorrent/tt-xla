# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import psutil
import time
import gc
import ctypes
from loguru import logger


def pytest_configure(config: pytest.Config):
    """
    Registers custom pytest marker `record_test_properties(key1=val1, key2=val2, ...)`.

    Allowed keys are:
        - Every test:
            - `category`: utils.Category

        - Op tests:
            - `jax_op_name`: name of the operation in jax, e.g. `jax.numpy.exp`
            - `shlo_op_name`: name of the matching stablehlo operation

        - Model tests:
            - `model_name`: name of the model under test
            - 'model_group': utils.ModelGroup
            - `run_mode`: infra.RunMode
            - `bringup_status`: utils.BringupStatus
            - `pcc`: float
            - `atol`: float

    These are used to tag the function under test with properties which will be dumped
    to the final XML test report. These reports get picked up by other CI workflows and
    are used to display state of tests on a dashboard.
    """
    config.addinivalue_line(
        "markers",
        "record_test_properties(key_value_pairs): Record custom properties for the test",
    )


def pytest_collection_modifyitems(items):
    """
    Pytest hook to process the custom marker and attach recorder properties to the test.
    """

    def validate_keys(keys: dict, is_model_test: bool):
        valid_keys = [
            "category",
            "jax_op_name",
            "shlo_op_name",
            "model_name",
            "model_group",
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
                "model_group",
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

    for item in items:
        # Add some test metadata in a 'tags' dictionary.
        tags = {"test_name": item.originalname, "specific_test_case": item.name}

        # Look for the custom marker.
        properties_marker = item.get_closest_marker(name="record_test_properties")

        # Utils flags helping handling model tests properly.
        is_model_test = False
        model_group = None

        if properties_marker:
            # Extract the key-value pairs passed to the marker.
            properties: dict = properties_marker.kwargs

            # Check if the test is marked using the "model_test" marker.
            is_model_test = item.get_closest_marker(name="model_test") is not None

            # Validate that only allowed keys are used.
            validate_keys(properties.keys(), is_model_test)

            # Turn all properties to strings.
            for k, v in properties.items():
                properties[k] = str(v)

            if is_model_test:
                model_group = properties.get("model_group")

            # Tag them.
            for key, value in properties.items():
                # Skip model_group, we don't need it in tags, we will insert it separately.
                if key != "model_group":
                    tags[key] = value

        # Attach metadata and tags dictionary as a single property.
        item.user_properties.extend([("tags", tags), ("owner", "tt-xla")])
        if is_model_test:
            # Add model group independently of tags dict.
            item.user_properties.append(("group", model_group))

@pytest.fixture(autouse=True)
def memory_usage_tracker(request):
    """
    A pytest fixture that tracks memory usage during the execution of a test.
    This fixture automatically tracks the memory usage of the process running the tests.
    It starts tracking before the test runs, continues tracking in a background thread during the test,
    and stops tracking after the test completes. It logs the memory usage statistics including the
    minimum, maximum, average, and total memory usage by the test.
    The memory usage is measured in megabytes (MB).
    Note:
        - This fixture is automatically used for all tests due to the `autouse=True` parameter.
        - The interval for memory readings can be adjusted by changing the sleep duration in the `track_memory` function.
        - Min, max, and avg memory usage are calculated based on the recorded memory readings from system memory.
    """
    process = psutil.Process()
    # Initialize memory tracking variables
    start_mem = process.memory_info().rss / (1024 * 1024)  # MB
    min_mem = start_mem
    max_mem = start_mem
    total_mem = start_mem
    count = 1
    # Start a background thread or loop to collect memory usage over time
    tracking = True
    def track_memory():
        nonlocal min_mem, max_mem, total_mem, count
        while tracking:
            current_mem = process.memory_info().rss / (1024 * 1024)
            min_mem = min(min_mem, current_mem)
            max_mem = max(max_mem, current_mem)
            total_mem += current_mem
            count += 1
            time.sleep(0.1)  # Adjust the interval as needed
    # Start tracking in a background thread
    import threading
    tracker_thread = threading.Thread(target=track_memory)
    tracker_thread.start()
    # Run the test
    yield
    # Stop tracking and wait for the thread to finish
    tracking = False
    tracker_thread.join()
    # Calculate end memory and memory usage stats
    end_mem = process.memory_info().rss / (1024 * 1024)  # MB
    min_mem = min(min_mem, end_mem)
    max_mem = max(max_mem, end_mem)
    total_mem += end_mem
    count += 1
    avg_mem = total_mem / count
    by_test = max_mem - start_mem
    # Log memory usage statistics
    logger.info(f"Test memory usage:")
    logger.info(f"    By test: {by_test:.2f} MB")
    logger.info(f"    Minimum: {min_mem:.2f} MB")
    logger.info(f"    Maximum: {max_mem:.2f} MB")
    logger.info(f"    Average: {avg_mem:.2f} MB")
    before_gc = process.memory_info().rss / (1024 * 1024)
    logger.info(f"Memory usage before garbage collection: {before_gc:.2f} MB")
    gc.collect()  # Force garbage collection
    libc = ctypes.CDLL("libc.so.6")
    libc.malloc_trim(0)
    after_gc = process.memory_info().rss / (1024 * 1024)
    logger.info(f"Memory usage after garbage collection: {after_gc:.2f} MB")
    should_log = True
    if not should_log:
        return
    # Get the current test name
    test_name = request.node.name
    # Store memory usage stats into a CSV file
    file_name = "pytest-memory-usage.csv"
    with open(file_name, "a") as f:
        if f.tell() == 0:
            # Write header if file is empty
            f.write("test_name,start_mem,end_mem,min_memory,max_memory,by_test (approx), after_gc\n")
        # NOTE: escape test_name in double quotes because some tests have commas in their parameter list...
        f.write(
            f'"{test_name}",{start_mem:.2f},{end_mem:.2f},{min_mem:.2f},{max_mem:.2f},{by_test:2f},{after_gc:2f}\n'
        )