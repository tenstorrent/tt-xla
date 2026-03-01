# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import contextlib
import ctypes
import gc
import io
import os
import shutil
import sys
import threading
import time
import types
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
        start_free = vm.available / (1024 * 1024)  # MB
        start_proc_rss = process.memory_info().rss / (1024 * 1024)  # MB

        with newline_logger():
            logger.info(f"Memory at test start:")
        logger.info(f"    System used:      {start_mem:.2f} MB")
        logger.info(f"    System free:      {start_free:.2f} MB")
        logger.info(f"    Process RSS:      {start_proc_rss:.2f} MB")

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

        end_proc_rss = process.memory_info().rss / (1024 * 1024)  # MB

        with newline_logger():
            logger.info(f"Test memory usage:")
        logger.info(f"    By test: {by_test:.2f} MB")
        logger.info(f"    Minimum: {min_mem:.2f} MB")
        logger.info(f"    Maximum: {max_mem:.2f} MB")
        logger.info(f"    Average: {avg_mem:.2f} MB")
        logger.info(f"    Process RSS (end): {end_proc_rss:.2f} MB")
    else:
        yield

    # Clean up memory.
    gc.collect()
    libc = ctypes.CDLL("libc.so.6")
    libc.malloc_trim(0)

    if request.config.getoption("--log-memory"):
        vm = psutil.virtual_memory()
        after_gc = (vm.total - vm.available) / (1024 * 1024)  # MB
        after_gc_proc_rss = process.memory_info().rss / (1024 * 1024)  # MB
        logger.info(f"Memory after garbage collection:")
        logger.info(f"    System used:  {after_gc:.2f} MB")
        logger.info(f"    Process RSS:  {after_gc_proc_rss:.2f} MB")

        # Scan for large CPU tensors still alive - helps diagnose memory leaks.
        # Write to a file to bypass pytest's stdout/stderr capture.
        _diag_file = f"/tmp/mem_diag_{os.getpid()}.log"

        def _diag(msg):
            with open(_diag_file, "a") as _f:
                _f.write(msg + "\n")

        large_tensors = []
        total_live_cpu_mb = 0.0
        for obj in gc.get_objects():
            if isinstance(obj, torch.Tensor) and obj.device.type == "cpu":
                size_mb = obj.element_size() * obj.nelement() / (1024 * 1024)
                total_live_cpu_mb += size_mb
                if size_mb > 100:
                    large_tensors.append((size_mb, tuple(obj.shape), obj.dtype))
        _diag(
            f"[MEM_DIAG] Live CPU tensors: total={total_live_cpu_mb:.1f} MB, "
            f"large(>100MB) count={len(large_tensors)}"
        )
        if large_tensors:
            large_tensors.sort(reverse=True)
            for size, shape, dtype in large_tensors[:20]:
                _diag(f"[MEM_DIAG]   {size:.1f} MB: shape={shape}, dtype={dtype}")

        # Scan for live non-CPU tensors (XLA tensors hold PJRT buffers with
        # owned host copies in Distributed/TP mode).
        large_xla_tensor_objs = []
        large_xla_tensors = []
        total_live_xla_mb = 0.0
        for obj in gc.get_objects():
            if isinstance(obj, torch.Tensor) and obj.device.type not in ("cpu", "meta"):
                try:
                    size_mb = obj.element_size() * obj.nelement() / (1024 * 1024)
                    total_live_xla_mb += size_mb
                    if size_mb > 10:
                        large_xla_tensor_objs.append(obj)
                        large_xla_tensors.append(
                            (size_mb, obj.device.type, tuple(obj.shape), obj.dtype)
                        )
                except Exception:
                    pass
        _diag(
            f"[MEM_DIAG] Live non-CPU tensors (xla/etc): total={total_live_xla_mb:.1f} MB, "
            f"large(>10MB) count={len(large_xla_tensors)}"
        )
        if large_xla_tensors:
            large_xla_tensors.sort(reverse=True)
            for size, dev, shape, dtype in large_xla_tensors[:20]:
                _diag(
                    f"[MEM_DIAG]   {size:.1f} MB: device={dev} shape={shape}, dtype={dtype}"
                )

        # Trace immediate referrers for the top-3 largest live XLA tensors.
        if large_xla_tensor_objs:
            large_xla_tensor_objs.sort(
                key=lambda t: t.element_size() * t.nelement(), reverse=True
            )
            _diag_vars = {id(large_xla_tensor_objs), id(large_xla_tensors)}
            for sample in large_xla_tensor_objs[:3]:
                _diag(
                    f"[MEM_DIAG] Referrers of XLA tensor shape={tuple(sample.shape)}, "
                    f"dtype={sample.dtype}:"
                )
                refs1 = [
                    r
                    for r in gc.get_referrers(sample)
                    if id(r) not in _diag_vars and not isinstance(r, type)
                ]
                for ref in refs1[:5]:
                    _diag(f"[MEM_DIAG]   ← {type(ref).__name__}: {repr(ref)[:200]}")
                    # One more level: referrers of this referrer
                    refs2 = [
                        r
                        for r in gc.get_referrers(ref)
                        if id(r) not in _diag_vars
                        and r is not refs1
                        and not isinstance(r, type)
                    ]
                    for ref2 in refs2[:3]:
                        _diag(
                            f"[MEM_DIAG]     ← {type(ref2).__name__}: {repr(ref2)[:150]}"
                        )

        # Scan for live torch.fx.GraphModule objects — these hold FX nodes whose
        # meta['val'] can pin XLA tensors.  Find what holds them.
        try:
            import torch.fx as _torch_fx

            live_gms = [
                obj
                for obj in gc.get_objects()
                if isinstance(obj, _torch_fx.GraphModule)
            ]
            _diag(f"[MEM_DIAG] Live torch.fx.GraphModule objects: {len(live_gms)}")
            for gm in live_gms[:3]:
                _diag(f"[MEM_DIAG]   GraphModule type={type(gm).__name__}")

            # Find functions/closures that capture any of the live GraphModules.
            # A function's __closure__ is a tuple of cell objects; each cell's
            # cell_contents is the captured variable value.
            if live_gms:
                live_gm_ids = {id(gm) for gm in live_gms}
                holding_functions = []
                for obj in gc.get_objects():
                    if isinstance(obj, types.FunctionType) and obj.__closure__:
                        for cell in obj.__closure__:
                            try:
                                val = cell.cell_contents
                                if id(val) in live_gm_ids:
                                    holding_functions.append((obj, cell, val))
                                    break
                            except ValueError:
                                pass  # empty cell
                _diag(
                    f"[MEM_DIAG] Functions with closures holding live GraphModules: "
                    f"{len(holding_functions)}"
                )
                for fn, cell, gm in holding_functions[:5]:
                    _diag(
                        f"[MEM_DIAG]   function={fn.__qualname__!r} "
                        f"defined at {fn.__code__.co_filename}:{fn.__code__.co_firstlineno}"
                    )
                    # Find who holds the function itself
                    fn_refs = [
                        r
                        for r in gc.get_referrers(fn)
                        if not isinstance(r, type)
                        and r is not holding_functions
                        and r is not live_gms
                    ]
                    for ref in fn_refs[:4]:
                        _diag(
                            f"[MEM_DIAG]     ← {type(ref).__name__}: {repr(ref)[:200]}"
                        )
        except Exception as e:
            _diag(f"[MEM_DIAG] GraphModule scan failed: {e}")

        # Scan for live XLAExecutor objects (compiled "tt" backend instances).
        # Each holds params_and_consts (XLA tensors) and compiled_graph closure.
        try:
            from tt_torch.backend.backend import XLAExecutor

            live_executors = [
                obj for obj in gc.get_objects() if isinstance(obj, XLAExecutor)
            ]
            _diag(f"[MEM_DIAG] Live XLAExecutor objects: {len(live_executors)}")
            for ex in live_executors[:3]:
                pc = len(ex.params_and_consts) if ex.params_and_consts else 0
                _diag(
                    f"[MEM_DIAG]   XLAExecutor: params_and_consts count={pc}, "
                    f"compiled_graph={'set' if ex.compiled_graph is not None else 'None'}"
                )
                referrers = gc.get_referrers(ex)
                _diag(f"[MEM_DIAG]   XLAExecutor referrers ({len(referrers)} total):")
                for ref in referrers[:5]:
                    _diag(f"[MEM_DIAG]     {type(ref).__name__}: {repr(ref)[:120]}")
        except Exception as e:
            _diag(f"[MEM_DIAG] XLAExecutor scan failed: {e}")

        # Scan for live torch.nn.Module objects (possible unreleased model refs).
        live_modules = [
            obj for obj in gc.get_objects() if isinstance(obj, torch.nn.Module)
        ]
        _diag(f"[MEM_DIAG] Live torch.nn.Module objects: {len(live_modules)}")
        for mod in live_modules[:5]:
            num_params = sum(p.numel() for p in mod.parameters())
            _diag(f"[MEM_DIAG]   {type(mod).__name__}: {num_params:,} params")

        # Use smaps_rollup for a high-level breakdown of memory types.
        try:
            rollup_stats = {}
            with open(f"/proc/{os.getpid()}/smaps_rollup", "r") as f:
                for line in f:
                    if ":" in line:
                        key, val = line.split(":", 1)
                        rollup_stats[key.strip()] = val.strip()
            _diag(f"[MEM_DIAG] smaps_rollup (process RSS={after_gc_proc_rss:.0f} MB):")
            for key in (
                "Rss",
                "Private_Dirty",
                "Private_Clean",
                "Shared_Dirty",
                "Shared_Clean",
                "Anonymous",
                "Swap",
            ):
                if key in rollup_stats:
                    kb = int(rollup_stats[key].split()[0])
                    _diag(f"[MEM_DIAG]   {key:20s}: {kb / 1024:.1f} MB")
        except Exception as e:
            _diag(f"[MEM_DIAG] smaps_rollup parse failed: {e}")

        # Parse /proc/self/smaps for per-region virtual size + RSS.
        # Shows which anonymous regions are actually resident (not just virtual).
        try:
            regions = []
            cur_size_mb = 0.0
            cur_rss_mb = 0.0
            cur_name = "(anon)"
            cur_perms = ""
            with open(f"/proc/{os.getpid()}/smaps", "r") as smaps_f:
                for line in smaps_f:
                    # Header line: starts with hex address range
                    if line and line[0] in "0123456789abcdef":
                        if cur_size_mb > 0:
                            regions.append(
                                (cur_rss_mb, cur_size_mb, cur_name, cur_perms)
                            )
                        parts = line.split()
                        start, end = parts[0].split("-")
                        cur_size_mb = (int(end, 16) - int(start, 16)) / (1024 * 1024)
                        cur_rss_mb = 0.0
                        cur_perms = parts[1] if len(parts) >= 2 else ""
                        cur_name = parts[5] if len(parts) >= 6 else "(anon)"
                    elif line.startswith("Rss:"):
                        kb = int(line.split()[1])
                        cur_rss_mb = kb / 1024
            if cur_size_mb > 0:
                regions.append((cur_rss_mb, cur_size_mb, cur_name, cur_perms))
            # Sort by RSS descending
            regions.sort(reverse=True)
            _diag(f"[MEM_DIAG] Top 15 regions by RSS (rss / virt):")
            for rss_mb, virt_mb, name, perms in regions[:15]:
                _diag(
                    f"[MEM_DIAG]   rss={rss_mb:7.0f} MB  virt={virt_mb:7.0f} MB"
                    f"  {perms}  {name}"
                )
        except Exception as e:
            _diag(f"[MEM_DIAG] smaps parse failed: {e}")

        # Use malloc_info to see glibc's internal heap state (free vs in-use).
        try:
            import ctypes as _ctypes
            import ctypes.util as _ctypes_util

            libc = _ctypes.CDLL(_ctypes_util.find_library("c"))
            libc.fopen.restype = _ctypes.c_void_p
            libc.fopen.argtypes = [_ctypes.c_char_p, _ctypes.c_char_p]
            libc.fclose.argtypes = [_ctypes.c_void_p]
            libc.malloc_info.argtypes = [_ctypes.c_int, _ctypes.c_void_p]

            _mi_path = f"/tmp/malloc_info_{os.getpid()}.xml"
            fp = libc.fopen(_mi_path.encode(), b"w")
            if fp:
                libc.malloc_info(0, _ctypes.c_void_p(fp))
                libc.fclose(_ctypes.c_void_p(fp))
                with open(_mi_path) as _mif:
                    _mi_content = _mif.read()
                # Extract totals from the XML: system=total_system, in_use=allocated
                import re as _re

                system_m = _re.search(r'<total type="system" size="(\d+)"', _mi_content)
                inuse_m = _re.search(r'<total type="in_use" size="(\d+)"', _mi_content)
                if system_m and inuse_m:
                    system_bytes = int(system_m.group(1))
                    inuse_bytes = int(inuse_m.group(1))
                    free_bytes = system_bytes - inuse_bytes
                    _diag(
                        f"[MEM_DIAG] glibc malloc_info totals:"
                        f"  system={system_bytes/(1<<20):.0f} MB"
                        f"  in_use={inuse_bytes/(1<<20):.0f} MB"
                        f"  free={free_bytes/(1<<20):.0f} MB"
                    )
                else:
                    _diag(f"[MEM_DIAG] malloc_info: (could not parse totals)")
        except Exception as e:
            _diag(f"[MEM_DIAG] malloc_info failed: {e}")


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


def _release_dynamo_bridge_tensors():
    """Release XLA tensor references retained by torch_xla dynamo bridge closures.

    Workaround for torch_xla leak: extract_internal() in dynamo_bridge.py creates
    an optimized_mod closure capturing sym_constants_to_graph_vars, a dict holding
    GraphInputMatcher with ALL model-weight XLA tensors (~26 GB for 8B TP).
    torch._dynamo.reset() frees XLAExecutor but this closure survives.
    """
    try:
        all_objects = gc.get_objects()
    except Exception:
        return
    cleared = False
    for obj in all_objects:
        if not (
            isinstance(obj, types.FunctionType)
            and obj.__qualname__ == "extract_internal.<locals>.optimized_mod"
            and obj.__closure__
        ):
            continue
        for varname, cell in zip(obj.__code__.co_freevars, obj.__closure__):
            try:
                val = cell.cell_contents
            except ValueError:
                continue
            if varname == "sym_constants_to_graph_vars" and isinstance(val, dict):
                # Clear each GraphInputMatcher's tensor list individually to avoid
                # PyTorch "Deallocating UntypedStorage" warnings from bulk free.
                for entry in val.values():
                    if isinstance(entry, tuple):
                        for item in entry:
                            if hasattr(item, "graph_input_xla_values"):
                                item.graph_input_xla_values.clear()
                val.clear()
                cleared = True
            elif varname == "xla_model" and hasattr(val, "xla_args"):
                val.xla_args = None
    if cleared:
        gc.collect()


# TODO(@LPanosTT): We do not need to reset the seed and dynamo state for jax test. Yet this will
# do so blindly around all tests: https://github.com/tenstorrent/tt-xla/issues/1265.
@pytest.fixture(autouse=True)
def run_around_tests():
    torch.manual_seed(0)
    yield
    torch._dynamo.reset()
    _release_dynamo_bridge_tensors()


@pytest.fixture()
def clear_torchxla_computation_cache():
    """
    Pytest fixture that clears the TorchXLA computation cache before each test.
    This helps avoid consteval-associated DRAM leaks as described in https://github.com/tenstorrent/tt-xla/issues/1940
    """
    yield
    try:
        xr.clear_computation_cache()
    except Exception as e:
        logger.warning(f"Failed to clear TorchXLA computation cache: {e}")
        logger.warning(
            "This is expected if the test throws an exception, https://github.com/tenstorrent/tt-xla/issues/2814"
        )


class TeeCaptureResult:
    """Result object mimicking pytest's CaptureResult."""

    def __init__(self, out: str, err: str):
        self.out = out
        self.err = err


class _StreamTee:
    """
    Tee capture for a single stream (stdout or stderr).

    Redirects a file descriptor to a memory-backed file (memfd). A background
    thread reads from the memfd and forwards to both a capture buffer and the
    original terminal.

    Architecture::

        Process -> stdout -> memfd -> Reader Thread -> Buffer + Terminal

    Attributes:
        _CHUNK_SIZE: Maximum bytes to read per iteration (class constant).
        _original_fd: File descriptor of the stream being captured.
        _saved_fd: Duplicated fd pointing to the original terminal for forwarding.
        _memfd: Memory-backed file descriptor that receives redirected writes.
        _read_fd: Separate fd for reading memfd with independent file position.
        _buffer: StringIO buffer accumulating captured output.
        _thread: Background thread running the reader loop.
        _read_pos: Current byte position in the memfd for reading.
        _final_size: Termination signal for reader thread. None means keep looping,
            a numeric value N means exit after reading N bytes.
    """

    _CHUNK_SIZE = 65536

    def __init__(self, stream):
        self._original_fd = stream.fileno()
        self._saved_fd = None
        self._memfd = None
        self._read_fd = None
        self._buffer = io.StringIO()
        self._thread = None
        self._read_pos = 0
        self._final_size = None

    def start(self):
        """Redirect stream to memfd and start reader thread."""
        self._saved_fd = os.dup(self._original_fd)
        self._memfd = os.memfd_create(f"tee_capture_{self._original_fd}")
        self._read_fd = os.open(f"/proc/self/fd/{self._memfd}", os.O_RDONLY)
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        os.dup2(self._memfd, self._original_fd)

    def _reader_loop(self):
        """Read from memfd, write to buffer and terminal."""
        while True:
            try:
                file_size = os.fstat(self._memfd).st_size
                if file_size > self._read_pos:
                    data = os.pread(
                        self._read_fd, file_size - self._read_pos, self._read_pos
                    )
                    if data:
                        self._read_pos += len(data)
                        self._buffer.write(data.decode("utf-8", errors="replace"))
                        with contextlib.suppress(BlockingIOError, OSError):
                            os.write(self._saved_fd, data)

                if self._final_size is not None and self._read_pos >= self._final_size:
                    return
            except OSError:
                return

    def stop(self):
        """Restore stream and wait for reader to finish."""
        if self._saved_fd is not None:
            with contextlib.suppress(OSError):
                os.dup2(self._saved_fd, self._original_fd)

        if self._memfd is not None:
            self._final_size = os.fstat(self._memfd).st_size

        if self._thread:
            self._thread.join(timeout=5.0)

        if self._memfd is not None and self._read_fd is not None:
            try:
                file_size = os.fstat(self._memfd).st_size
                if file_size > self._read_pos:
                    data = os.pread(
                        self._read_fd, file_size - self._read_pos, self._read_pos
                    )
                    if data:
                        self._buffer.write(data.decode("utf-8", errors="replace"))
                        with contextlib.suppress(OSError):
                            os.write(self._saved_fd, data)
            except OSError:
                pass

        # Cleanup
        for fd in (self._memfd, self._read_fd, self._saved_fd):
            if fd is not None:
                with contextlib.suppress(OSError):
                    os.close(fd)
        self._memfd = self._read_fd = self._saved_fd = None

    def getvalue(self):
        """Return captured output."""
        return self._buffer.getvalue()


class TeeCapture:
    """
    Captures stderr/stdout at fd level while still writing to terminal in real-time.
    This allows capturing C++ output (like MLIR errors) without suppressing it.

    NOTE: This does NOT work with pytest-forked due to interpreter shutdown issues.
    Use capfd instead when running with --forked.
    """

    def __init__(self):
        self._stdout_tee = _StreamTee(sys.stdout)
        self._stderr_tee = _StreamTee(sys.stderr)
        self._started = False

    def start(self):
        try:
            self._stdout_tee.start()
            self._stderr_tee.start()
            self._started = True
        except OSError:
            self._started = False

    def stop(self):
        if not self._started:
            return
        with contextlib.suppress(OSError):
            sys.stdout.flush()
        with contextlib.suppress(OSError):
            sys.stderr.flush()
        self._stdout_tee.stop()
        self._stderr_tee.stop()

    def readouterr(self):
        return TeeCaptureResult(
            self._stdout_tee.getvalue(), self._stderr_tee.getvalue()
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
