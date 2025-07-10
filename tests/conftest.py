# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ctypes
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
import gc
import sys
import threading
import time
from typing import Any

import jax
import psutil
import pytest
import transformers
import transformers.modeling_flax_utils
from infra import DeviceConnectorFactory, Framework
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
        start_mem = process.memory_info().rss / (1024 * 1024)  # MB
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

        end_mem = process.memory_info().rss / (1024 * 1024)  # MB
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
        after_gc = process.memory_info().rss / (1024 * 1024)
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


@dataclass
class MonkeyPatchConfig:
    """Configuration class for managing monkey patching operations.

    This class provides a structured way to temporarily replace functions or methods
    in modules with custom implementations. We primarily use this to wrap JAX operations
    in StableHLO CompositeOps, for easier matching in the compiler.

    Attributes:
        target_module (Any): The module object containing the function to be patched.
        target_function (str): The name of the function/method to be replaced.
        replacement_factory (Callable): A factory function that creates the replacement
            function. Should accept this config instance as a parameter.
        post_patch (Callable): Optional callback function executed after the patch
            is applied. Defaults to a no-op lambda function.
        backup (Any): Storage for the original function before patching. Used to
            restore the original implementation later. Initially None.
    """

    target_module: Any
    target_function: str
    replacement_factory: Callable
    post_patch: Callable = lambda: None
    backup: Any = None

    def patch(self):
        """Apply the monkey patch if not already applied."""
        if self.backup is None:
            self.backup = getattr(self.target_module, self.target_function)

            replacement = self.replacement_factory(self)
            setattr(self.target_module, self.target_function, replacement)

            self.post_patch()


monkeypatches = [
    MonkeyPatchConfig(
        target_module=jax.nn,
        target_function="gelu",
        replacement_factory=lambda config: lambda x, approximate=True: jax.lax.composite(
            lambda x: config.backup(x, approximate=approximate),
            "tenstorrent.gelu_tanh" if approximate else "tenstorrent.gelu",
        )(
            x
        ),
        post_patch=lambda: transformers.modeling_flax_utils.ACT2FN.update(
            {"gelu": partial(jax.nn.gelu, approximate=False)}
        ),
    )
]

# Monkeypatch libraries to use our versions of functions, which will wrap operations in a StableHLO CompositeOp
@pytest.fixture(autouse=True)
def monkeypatch_import(request):
    for patch_config in monkeypatches:
        patch_config.patch()

    yield
