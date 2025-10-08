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
def _is_on_CIv2() -> bool:
    """
    Check if we are on CIv2.
    """
    return not bool(os.environ.get("DOCKER_CACHE_ROOT"))


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

# Additional test artifact directories that should be cleaned
TEST_ARTIFACT_DIRECTORIES = [
    "results",
    "output",
    "outputs",
    "artifacts",
    "orbax",
    "model_outputs",
    "checkpoints",
    "tmp",
    "temp",
    "test_outputs",
    "test_results",
]


def cleanup_cache():
    """
    Cleans up cache directories if we are running on CIv2.
    """
    if not _is_on_CIv2():
        return

    import subprocess

    # Clean standard cache directories
    for cache_dir in CACHE_DIRECTORIES:
        if not cache_dir.exists():
            continue

        try:
            shutil.rmtree(cache_dir)
            logger.debug(f"Cleaned up cache directory: {cache_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up cache directory {cache_dir}: {e}")

    # Aggressive cleanup - remove everything in cache/tmp locations
    aggressive_cleanup_paths = [
        "/mnt/dockercache",
        str(Path.home() / ".cache"),
        "/tmp",
        "/var/tmp",
        "/__w/tt-xla/tt-xla/.pytest_cache",
        "/__w/tt-xla/tt-xla/build",
    ]

    for path in aggressive_cleanup_paths:
        if Path(path).exists():
            try:
                subprocess.run(f"rm -rf {path}/* 2>/dev/null", shell=True)
                logger.debug(f"Aggressively cleaned: {path}")
            except Exception as e:
                logger.warning(f"Failed to clean {path}: {e}")

    # Clean up workspace artifacts
    try:
        subprocess.run("find /__w -name '*.onnx' -o -name '*.safetensors' -o -name '*.bin' -o -name 'core*' | xargs rm -f 2>/dev/null", shell=True)
        subprocess.run("find /__w -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null", shell=True)
        subprocess.run("find /__w -name '*.pyc' -o -name '*.pyo' -delete 2>/dev/null", shell=True)
    except Exception as e:
        logger.warning(f"Failed to clean workspace artifacts: {e}")


def cleanup_test_artifacts():
    """
    Cleans up test artifact directories and files if we are running on CIv2.
    """
    if not _is_on_CIv2():
        return

    import glob
    import subprocess

    # Get the test directory path
    test_dir = Path(__file__).parent
    project_root = test_dir.parent

    # Clean up test artifact directories in both test dir and project root
    for base_dir in [test_dir, project_root]:
        for artifact_dir in TEST_ARTIFACT_DIRECTORIES:
            dir_path = base_dir / artifact_dir
            if dir_path.exists() and dir_path.is_dir():
                try:
                    shutil.rmtree(dir_path)
                    logger.debug(f"Cleaned up test artifact directory: {dir_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up test artifact directory {dir_path}: {e}")

    # Clean up Python cache directories
    for base_dir in [test_dir, project_root]:
        # Find and remove __pycache__ directories
        try:
            result = subprocess.run(
                f'find "{base_dir}" -type d -name "__pycache__" -exec rm -rf {{}} + 2>/dev/null',
                shell=True,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                logger.debug(f"Cleaned up __pycache__ directories in {base_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up __pycache__ directories: {e}")

        # Remove .pyc and .pyo files
        try:
            subprocess.run(
                f'find "{base_dir}" -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete 2>/dev/null',
                shell=True
            )
        except Exception as e:
            logger.warning(f"Failed to clean up Python compiled files: {e}")

    # Clean up core dump files (can be very large)
    # Core files can appear in various locations
    core_patterns = [
        str(project_root / "core"),
        str(project_root / "core.*"),
        str(test_dir / "core"),
        str(test_dir / "core.*"),
        "/tmp/core",
        "/tmp/core.*",
        str(Path.cwd() / "core"),
        str(Path.cwd() / "core.*"),
    ]

    for pattern in core_patterns:
        for core_file in glob.glob(pattern):
            try:
                # Get file size before deleting for logging
                file_size_mb = os.path.getsize(core_file) / (1024*1024)
                os.remove(core_file)
                logger.info(f"Cleaned up core dump file: {core_file} (size: {file_size_mb:.2f} MB)")
            except FileNotFoundError:
                pass  # File was already deleted
            except Exception as e:
                logger.warning(f"Failed to clean up core file {core_file}: {e}")

    # Clean up temporary directories and files in /tmp
    tmp_patterns = [
        "/tmp/pytest-*",
        "/tmp/tt_xla_*",
        "/tmp/test_*",
        "/tmp/model_*",
        "/tmp/*.onnx",
        "/tmp/*.pt",
        "/tmp/*.pth",
        "/tmp/*.safetensors",
        "/tmp/tmp*",
    ]

    for pattern in tmp_patterns:
        for path in glob.glob(pattern):
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
                logger.debug(f"Cleaned up temp path: {path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp path {path}: {e}")

    # Clean up any .log files in the project root
    for log_file in project_root.glob("*.log"):
        try:
            os.remove(log_file)
            logger.debug(f"Cleaned up log file: {log_file}")
        except Exception as e:
            logger.warning(f"Failed to clean up log file {log_file}: {e}")


@pytest.fixture(autouse=True)
def cleanup_cache_fixture(request):
    """
    Pytest fixture that cleans up cache directories before and after each test.
    Only runs if we are running on CIv2.
    """
    # Cleanup before test
    cleanup_cache()
    cleanup_test_artifacts()

    yield

    # Log folder sizes after each test if running in CI
    if _is_on_CIv2():
        try:
            import subprocess
            test_name = request.node.nodeid
            # Find the script path - look for it relative to the project root
            script_path = None
            potential_paths = [
                Path(__file__).parent.parent / ".github" / "scripts" / "log_folder_sizes.py",
                Path("/__w/tt-xla/tt-xla/.github/scripts/log_folder_sizes.py"),
                Path(".github/scripts/log_folder_sizes.py"),
            ]
            for path in potential_paths:
                if path.exists():
                    script_path = str(path)
                    break

            if script_path:
                # Run the folder size logging script
                subprocess.run(
                    [
                        "python",
                        script_path,
                        "--test-name", test_name,
                        "--output", "folder_sizes_per_test.json",
                        "--append",
                        "--find-large-files",
                        "--min-file-size", "50"
                    ],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
            else:
                logger.debug("Folder size logging script not found")
        except Exception as e:
            logger.warning(f"Failed to log folder sizes: {e}")

    # Cleanup after test
    cleanup_cache()
    cleanup_test_artifacts()


# TODO(@LPanosTT): We do not need to reset the seed and dynamo state for jax test. Yet this will
# do so blindly around all tests: https://github.com/tenstorrent/tt-xla/issues/1265.
@pytest.fixture(autouse=True)
def run_around_tests():
    torch.manual_seed(0)
    yield
    torch._dynamo.reset()
