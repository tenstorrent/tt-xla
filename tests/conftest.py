# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ctypes
from contextlib import contextmanager
import gc
import os
import shutil
import sys
import threading
import time

import torch
import psutil
import pytest
from infra import DeviceConnectorFactory, Framework
from loguru import logger
from pathlib import Path
from third_party.tt_forge_models.config import ModelInfo
from typing import Any


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
            - `model_info`: third_party.tt_forge_models.config.ModelInfo
            - `run_mode`: infra.RunMode
            - `parallelism`: third_party.tt_forge_models.config.Parallelism
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

    """
    Register a marker to disable auto user_properties injection at collection time, when they
    would otherwise be populated at runtime.
    """
    config.addinivalue_line(
        "markers",
        "no_auto_properties: disable auto user_properties injection at collection",
    )


def pytest_collection_modifyitems(items):
    """
    Pytest hook to process the custom marker and attach recorder properties to the test.
    Also filters tests based on .pytest_tests_to_run file if it exists.
    """

    def validate_keys(keys: dict, tagged_as_model_test: bool):
        valid_keys = [
            "category",
            "jax_op_name",
            "shlo_op_name",
            "model_name",
            "model_group",
            "model_info",
            "run_mode",
            "parallelism",
            "bringup_status",
            "execution_pass",
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
        if tagged_as_model_test:
            # Check if using new property set
            new_mandatory_properties = [
                "model_info",
                "run_mode",
                "bringup_status",
            ]

            # Check if using old property set
            old_mandatory_properties = [
                "model_name",
                "model_group",
                "run_mode",
                "bringup_status",
            ]

            has_new_properties = all(prop in keys for prop in new_mandatory_properties)
            has_old_properties = all(prop in keys for prop in old_mandatory_properties)

            # Ensure exactly one property set is used (XOR condition)
            if has_new_properties == has_old_properties:
                raise KeyError(
                    f"Model tests must have either new properties: {new_mandatory_properties} "
                    f"or old properties: {old_mandatory_properties}."
                )

    # Filter tests based on .pytest_tests_to_run file if it exists
    tests_to_run_file = Path(".pytest_tests_to_run")
    if tests_to_run_file.exists():
        with open(tests_to_run_file, "r") as f:
            allowed_tests = set(line.strip() for line in f if line.strip())

        # Remove tests not in the allowed list
        items[:] = [item for item in items if item.nodeid in allowed_tests]

    for item in items:

        # Skip collection-time user_properies for this test, populate at runtime.
        if item.get_closest_marker("no_auto_properties"):
            continue

        # Add some test metadata in a 'tags' dictionary.
        tags = {"test_name": item.originalname, "specific_test_case": item.name}

        # Look for the custom marker.
        properties_marker = item.get_closest_marker(name="record_test_properties")

        # Utils flags helping handling model tests properly.
        tagged_as_model_test = False
        model_group = None

        if properties_marker:
            # Extract the key-value pairs passed to the marker.
            properties: dict = properties_marker.kwargs

            # Check if the test is marked using the "model_test" marker.
            tagged_as_model_test = (
                item.get_closest_marker(name="model_test") is not None
            )

            # Validate that only allowed keys are used.
            validate_keys(properties.keys(), tagged_as_model_test)

            # Put all properties in tags.
            for key, value in properties.items():
                if key == "model_info":
                    model_info: ModelInfo = value
                    tags["model_name"] = model_info.name
                    tags["model_info"] = model_info.to_report_dict()
                    model_group = str(model_info.group)
                elif key == "model_group":
                    model_group = str(value)
                else:
                    tags[key] = str(value)

        # Attach tags dictionary as a single property. Also set owner.
        item.user_properties.extend([("tags", tags), ("owner", "tt-xla")])
        if tagged_as_model_test:
            # Add model group independently of tags dict.
            item.user_properties.append(("group", model_group))


def pytest_addoption(parser):
    """
    Custom CLI pytest option to enable memory usage tracking in tests.

    Use it when calling pytest like `pytest --log-memory ...`.
    """
    parser.addoption(
        "--log-memory",
        action="store_true",
        default=False,
        help="Enable memory usage tracking for tests",
    )


# DOCKER_CACHE_ROOT is only meaningful on CIv1 and its presence indicates CIv1 usage.
# TODO: Consider using a more explicit way to differentiate CIv2-specific environment
# Issue: https://github.com/tenstorrent/github-ci-infra/issues/772
# Users of IRD may not have DOCKER_CACHE_ROOT set locally, but do have IRD_ARCH_NAME
# set so also consider that variable and expect it not to be set.
def _is_on_CIv2() -> bool:
    """
    Check if we are on CIv2.
    """
    is_on_civ1 = bool(os.environ.get("DOCKER_CACHE_ROOT"))
    is_user_ird = bool(os.environ.get("IRD_ARCH_NAME"))
    is_on_civ2 = not is_on_civ1 and not is_user_ird
    return is_on_civ2


@contextmanager
def newline_logger():
    """
    Context manager to temporarily set the logger to use a newline at the start of each log message.
    Reverts to the default format after the context exits.
    """
    default_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS Z}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    logger.remove()
    logger.add(
        sys.stdout,
        format=f"\n{default_format}",
        level="INFO",
    )
    try:
        yield
    finally:
        logger.remove()
        logger.add(
            sys.stdout,
            format=default_format,
            level="INFO",
        )


@pytest.fixture(autouse=True)
def memory_usage_tracker(request):
    """
    A pytest fixture that tracks memory usage during the execution of a test.
    Only runs if --log-memory is passed to pytest.
    """
    if request.config.getoption("--log-memory"):
        process = psutil.Process()
        # Initialize memory tracking variables
        vm = psutil.virtual_memory()
        start_mem = (vm.total - vm.available) / (1024 * 1024)  # MB
        min_mem = start_mem
        max_mem = start_mem
        total_mem = start_mem
        count = 1
        tracking = True

        def track_memory():
            nonlocal min_mem, max_mem, total_mem, count
            while tracking:
                vm = psutil.virtual_memory()
                used = (vm.total - vm.available) / (1024 * 1024)
                min_mem = min(min_mem, used)
                max_mem = max(max_mem, used)
                total_mem += used
                count += 1
                time.sleep(0.1)

        tracker_thread = threading.Thread(target=track_memory)
        tracker_thread.start()
        yield
        tracking = False
        tracker_thread.join()

        vm = psutil.virtual_memory()
        end_mem = (vm.total - vm.available) / (1024 * 1024)  # MB
        min_mem = min(min_mem, end_mem)
        max_mem = max(max_mem, end_mem)
        total_mem += end_mem
        count += 1
        avg_mem = total_mem / count
        by_test = max_mem - start_mem

        with newline_logger():
            logger.info(f"Test memory usage:")
        logger.info(f"    By test: {by_test:.2f} MB")
        logger.info(f"    Minimum: {min_mem:.2f} MB")
        logger.info(f"    Maximum: {max_mem:.2f} MB")
        logger.info(f"    Average: {avg_mem:.2f} MB")
    else:
        yield

    # Clean up memory.
    gc.collect()
    libc = ctypes.CDLL("libc.so.6")
    libc.malloc_trim(0)

    if request.config.getoption("--log-memory"):
        vm = psutil.virtual_memory()
        after_gc = (vm.total - vm.available) / (1024 * 1024)  # MB
        logger.info(f"Memory usage after garbage collection: {after_gc:.2f} MB")


@pytest.fixture(scope="session", autouse=True)
def initialize_device_connectors():
    """
    Autouse fixture that establishes connection to devices by creating connector
    instances.

    Done to make sure it is executed before any other jax command during tests.
    """
    DeviceConnectorFactory.create_connector(Framework.JAX)
    DeviceConnectorFactory.create_connector(Framework.TORCH)


CACHE_DIRECTORIES = [
    Path.home() / ".cache" / "lfcache",
    Path.home() / ".cache" / "url_cache",
    Path("/mnt/dockercache/huggingface"),
    Path.home() / ".cache" / "huggingface",
    Path("/tmp") / "huggingface",
    Path.home() / ".cache" / "jax",
    Path.home() / ".cache" / "jaxlib",
    Path("/tmp") / f"torchinductor_{os.environ.get('USER', '')}",
]


def cleanup_cache():
    """
    Cleans up cache directories if we are running on CIv2.
    """
    if not _is_on_CIv2():
        return

    for cache_dir in CACHE_DIRECTORIES:
        if not cache_dir.exists():
            continue

        try:
            shutil.rmtree(cache_dir)
            logger.debug(f"Cleaned up cache directory: {cache_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up cache directory {cache_dir}: {e}")


@pytest.fixture(autouse=True)
def cleanup_cache_fixture():
    """
    Pytest fixture that cleans up cache directories before and after each test.
    Only runs if we are running on CIv2.
    """
    # Cleanup before test
    cleanup_cache()

    yield

    # Cleanup after test
    cleanup_cache()


# TODO(@LPanosTT): We do not need to reset the seed and dynamo state for jax test. Yet this will
# do so blindly around all tests: https://github.com/tenstorrent/tt-xla/issues/1265.
@pytest.fixture(autouse=True)
def run_around_tests():
    torch.manual_seed(0)
    yield
    torch._dynamo.reset()
