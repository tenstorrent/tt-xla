# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import contextlib
import ctypes
import gc
import io
import os
import select
import shutil
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import psutil
import pytest
import torch
import torch_xla.runtime as xr
from infra import DeviceConnectorFactory, Framework
from loguru import logger

from third_party.tt_forge_models.config import ModelInfo


def pytest_configure(config: pytest.Config):
    """
    Registers custom pytest marker `record_test_properties(key1=val1, key2=val2, ...)`.

    Allowed keys are:
        - Every test:
            - `category`: utils.Category

        - Op tests:
            - `jax_op_name`: name of the operation in jax, e.g. `jax.numpy.exp`
            - `torch_op_name`: name of the operation in torch, e.g. `torch.add`
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
            "torch_op_name",
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
    Custom CLI pytest options for tests.

    Use it when calling pytest like `pytest --log-memory ...` or `pytest --serialize ...`.
    """
    parser.addoption(
        "--log-memory",
        action="store_true",
        default=False,
        help="Enable memory usage tracking for tests",
    )
    parser.addoption(
        "--serialize",
        action="store_true",
        default=False,
        help="Enable serialization of compilation artifacts during tests",
    )
    parser.addoption(
        "--log-pid",
        action="store_true",
        default=False,
        help="Append process PID to log file names specified in TTXLA_LOGGER_FILE, TT_LOGGER_FILE, TTMLIR_RUNTIME_LOGGER_FILE environment variables if set, to facilitate multiprocess debug logging.",
    )
    parser.addoption(
        "--disable-perf-measurement",
        action="store_true",
        default=False,
        help="Disable performance benchmark measurement in tester",
    )
    parser.addoption(
        "--perf-report-dir",
        action="store",
        default=None,
        help="Output directory for perf benchmark reports. If not given, no perf benchmark files will be generated.",
    )
    parser.addoption(
        "--perf-id",
        action="store",
        default=None,
        help="Perf ID for perf benchmark reports.",
    )
    parser.addoption(
        "--dump-irs",
        action="store_true",
        default=False,
        help="Enable IR dumping during model tests",
    )


@pytest.fixture(autouse=True)
def disable_perf_measurement(request):
    """
    A pytest fixture that disables performance benchmark measurement if --disable-perf-measurement is passed to pytest.
    """
    if request.config.getoption("--disable-perf-measurement"):
        os.environ["DISABLE_PERF_MEASUREMENT"] = "1"


# DOCKER_CACHE_ROOT is only meaningful on CIv1 and its presence indicates CIv1 usage.
# TODO: Consider using a more explicit way to differentiate CIv2-specific environment
# Issue: https://github.com/tenstorrent/github-ci-infra/issues/772
# Users of IRD may not have DOCKER_CACHE_ROOT set locally, but do have IRD_ARCH_NAME
# set so also consider that variable and expect it not to be set.
def _is_on_CIv2() -> bool:
    """
    Check if we are on CIv2.
    """
    if bool(os.environ.get("TT_XLA_CI")):
        is_on_civ1 = bool(os.environ.get("DOCKER_CACHE_ROOT"))
        return not is_on_civ1

    # We are not running in CI environment.
    return False


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
def setup_pid_logging(request):
    """
    A pytest fixture that monkeypatches TTXLA_LOGGER_FILE, TTMLIR_RUNTIME_LOGGER_FILE, and TT_LOGGER_FILE environment
    variables to include the process PID before the file extension when --log-pid
    is passed to pytest.

    TT_LOGGER_FILE controls tt-metal's tt-logger log output filepath, see
    https://github.com/tenstorrent/tt-logger?tab=readme-ov-file#environment-variables
    for more information about tt-logger.

    TTMLIR_RUNTIME_LOGGER_FILE controls tt-mlir-runtime's logging output filepath.
    """
    if not request.config.getoption("--log-pid"):
        yield
        return

    # Store original values for restoration
    original_TTXLA_LOGGER_FILE = os.environ.get("TTXLA_LOGGER_FILE")
    original_tt_logger_file = os.environ.get("TT_LOGGER_FILE")
    original_ttmlir_runtime_logger_file = os.environ.get("TTMLIR_RUNTIME_LOGGER_FILE")

    def add_pid_to_filename(filepath):
        """Add PID before file extension"""
        if not filepath:
            return filepath

        path = Path(filepath)
        pid = os.getpid()

        if path.suffix:
            # File has extension, insert PID before it
            new_name = f"{path.stem}.{pid}{path.suffix}"
        else:
            # No extension, just append PID
            new_name = f"{path.name}.{pid}"

        return str(path.parent / new_name)

    # Modify environment variables if they exist
    if original_TTXLA_LOGGER_FILE:
        os.environ["TTXLA_LOGGER_FILE"] = add_pid_to_filename(
            original_TTXLA_LOGGER_FILE
        )

    if original_tt_logger_file:
        os.environ["TT_LOGGER_FILE"] = add_pid_to_filename(original_tt_logger_file)

    if original_ttmlir_runtime_logger_file:
        os.environ["TTMLIR_RUNTIME_LOGGER_FILE"] = add_pid_to_filename(
            original_ttmlir_runtime_logger_file
        )

    try:
        yield
    finally:
        # Restore original values
        if original_TTXLA_LOGGER_FILE:
            os.environ["TTXLA_LOGGER_FILE"] = original_TTXLA_LOGGER_FILE

        if original_tt_logger_file:
            os.environ["TT_LOGGER_FILE"] = original_tt_logger_file

        if original_ttmlir_runtime_logger_file:
            os.environ["TTMLIR_RUNTIME_LOGGER_FILE"] = (
                original_ttmlir_runtime_logger_file
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
    if _is_on_CIv2():
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


@pytest.fixture()
def clear_torchxla_computation_cache():
    """
    Pytest fixture that clears the TorchXLA computation cache before each test.
    This helps avoid consteval-associated DRAM leaks as described in https://github.com/tenstorrent/tt-xla/issues/1940
    """
    yield
    xr.clear_computation_cache()


class TeeCaptureResult:
    """Result object mimicking pytest's CaptureResult."""

    def __init__(self, out: str, err: str):
        self.out = out
        self.err = err


class TeeCapture:
    """
    Captures stderr/stdout at fd level while still writing to terminal in real-time.
    This allows capturing C++ output (like MLIR errors) without suppressing it.

    NOTE: This does NOT work with pytest-forked due to interpreter shutdown issues.
    Use capfd instead when running with --forked.
    """

    def __init__(self):
        self._stderr_buffer = io.StringIO()
        self._stdout_buffer = io.StringIO()
        self._fds = {}
        self._threads = []
        self._stop_event = threading.Event()
        self._started = False

    def _safe_close(self, fd):
        if fd is not None:
            with contextlib.suppress(OSError):
                os.close(fd)

    def _tee_reader(self, read_fd, saved_fd, buffer):
        while not self._stop_event.is_set():
            try:
                ready, _, _ = select.select([read_fd], [], [], 0.1)
                if not ready:
                    continue
                data = os.read(read_fd, 4096)
                if not data:
                    return
                os.write(saved_fd, data)
                with contextlib.suppress(Exception):
                    buffer.write(data.decode("utf-8", errors="replace"))
            except OSError:
                return

    def start(self):
        try:
            # Save original fds and create pipes
            for name, stream, buffer in [
                ("stderr", sys.stderr, self._stderr_buffer),
                ("stdout", sys.stdout, self._stdout_buffer),
            ]:
                original_fd = stream.fileno()
                saved_fd = os.dup(original_fd)
                read_fd, write_fd = os.pipe()
                os.dup2(write_fd, original_fd)

                self._fds[name] = {
                    "original": original_fd,
                    "saved": saved_fd,
                    "read": read_fd,
                    "write": write_fd,
                }

                thread = threading.Thread(
                    target=self._tee_reader,
                    args=(read_fd, saved_fd, buffer),
                    daemon=True,
                )
                thread.start()
                self._threads.append(thread)

            self._started = True
        except OSError:
            self._started = False

    def stop(self):
        if not self._started:
            return

        with contextlib.suppress(OSError):
            sys.stderr.flush()
        with contextlib.suppress(OSError):
            sys.stdout.flush()

        self._stop_event.set()

        # Restore original fds and close pipe write ends
        for fds in self._fds.values():
            with contextlib.suppress(OSError):
                os.dup2(fds["saved"], fds["original"])
            self._safe_close(fds["write"])

        for thread in self._threads:
            thread.join(timeout=1.0)

        # Drain remaining data from pipes
        for name, fds in self._fds.items():
            buffer = self._stderr_buffer if name == "stderr" else self._stdout_buffer
            self._drain_pipe(fds["read"], fds["saved"], buffer)

        # Close all remaining fds
        for fds in self._fds.values():
            self._safe_close(fds["read"])
            self._safe_close(fds["saved"])

    def _drain_pipe(self, read_fd, saved_fd, buffer):
        while True:
            try:
                ready, _, _ = select.select([read_fd], [], [], 0)
                if not ready:
                    return
                data = os.read(read_fd, 4096)
                if not data:
                    return
                with contextlib.suppress(OSError):
                    os.write(saved_fd, data)
                with contextlib.suppress(Exception):
                    buffer.write(data.decode("utf-8", errors="replace"))
            except (OSError, ValueError):
                return

    def readouterr(self):
        return TeeCaptureResult(
            self._stdout_buffer.getvalue(), self._stderr_buffer.getvalue()
        )


def _should_use_capfd(request) -> bool:
    """
    Determine if capfd should be used instead of TeeCapture.

    Returns True when:
    - Running with --forked (TeeCapture doesn't work with pytest-forked)
    - Running in distributed mode (TeeCapture doesn't work in subprocesses)
    - Pytest capture is enabled (TeeCapture conflicts with pytest's capture pipes)
    """
    # Check if --forked option was passed to pytest
    try:
        is_forked = request.config.getoption("--forked", default=False)
    except ValueError:
        is_forked = False

    # Check if running in distributed/multi_host subprocess
    is_distributed = os.environ.get("TT_RUNTIME_ENABLE_DISTRIBUTED") == "1"

    # Check if pytest capture is enabled (not disabled via -s)
    capture_mode = request.config.getoption("capture")
    is_capturing = capture_mode != "no"

    return is_forked or is_distributed or is_capturing


@pytest.fixture()
def captured_output_fixture(request):
    """
    Pytest fixture that captures stdout/stderr at fd level.

    When running normally: Uses TeeCapture to show output in real-time while capturing.
    When running with --forked: Uses pytest's capfd which handles forked processes correctly.
    When running in distributed mode: Uses capfd (TeeCapture doesn't work in subprocesses).
    """
    if _should_use_capfd(request):
        # Use capfd - handles forked/subprocess environments correctly
        capfd = request.getfixturevalue("capfd")
        yield capfd
    else:
        # Use TeeCapture for real-time output
        tee = TeeCapture()
        tee.start()
        yield tee
        tee.stop()
