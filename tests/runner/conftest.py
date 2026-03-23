# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import socket

import pytest

from infra import RunMode

from tests.runner.test_config.constants import ALLOWED_ARCHES
from tests.runner.test_config.jax import test_config as jax_test_config
from tests.runner.test_config.torch import test_config as torch_test_config
from tests.runner.test_config.torch_llm import test_config as torch_llm_test_config
from tests.infra.utilities.types import Framework
from tests.runner.test_utils import (
    ModelTestConfig,
    ModelTestStatus,
    RunPhase,
    _to_marshal_safe,
    build_model_tags,
    create_benchmark_result,
    get_input_shape_info,
    get_xla_device_arch,
    record_model_test_properties,
    update_test_metadata_for_exception,
)
from tests.utils import BringupStatus

_BRINGUP_STAGE_FILE = "._bringup_stage.txt"

# Maps nodeid -> item, populated during collection for crash-report fallback.
_item_by_nodeid: dict = {}


def _get_model_group_from_item(item):
    """Extract ModelGroup enum from a parametrized test item's test_entry parameter.

    Args:
        item: pytest test item

    Returns:
        ModelGroup if available, None otherwise
    """
    # Get the test_entry from parametrized test's callspec
    if not hasattr(item, "callspec"):
        return None

    test_entry = item.callspec.params.get("test_entry")
    if test_entry is None:
        # Try test_entry_and_phase (for test_llms_torch which uses tuple parametrization)
        test_entry_and_phase = item.callspec.params.get("test_entry_and_phase")
        if test_entry_and_phase is not None:
            test_entry, _ = test_entry_and_phase  # Unpack tuple

    if test_entry is None:
        return None

    # test_entry.variant_info is (variant_enum, ModelLoader)
    variant, ModelLoader = test_entry.variant_info
    model_info = ModelLoader.get_model_info(variant=variant)
    return model_info.group


def pytest_addoption(parser):
    """Register CLI options for selecting target arch."""
    parser.addoption(
        "--arch",
        action="store",
        default=None,
        choices=sorted(ALLOWED_ARCHES),
        help="Target architecture (e.g., n150, p150) for which to match via arch_overrides in test_config files",
    )


@pytest.fixture(autouse=True)
def bringup_stage_file():
    """Create and cleanup bringup stage file for each test when ENABLE_BRINGUP_STAGE_LOGGING=1."""
    if os.environ.get("ENABLE_BRINGUP_STAGE_LOGGING") == "1":
        # Setup: Create/truncate file before test
        with open(_BRINGUP_STAGE_FILE, "w") as f:
            f.write("FE_COMPILATION_START")

    yield

    # Teardown: Remove file after test
    if os.environ.get("ENABLE_BRINGUP_STAGE_LOGGING") == "1" and os.path.exists(
        _BRINGUP_STAGE_FILE
    ):
        os.remove(_BRINGUP_STAGE_FILE)


@pytest.fixture
def test_metadata(request) -> ModelTestConfig:
    """Expose per-test ModelTestConfig attached during collection."""
    meta = getattr(request.node, "_test_meta", None)
    assert meta is not None, f"No ModelTestConfig attached for {request.node.nodeid}"
    return meta


def pytest_collection_modifyitems(config, items):
    """During collection, attach ModelTestConfig, apply markers.

    Also deselect tests explicitly marked with EXCLUDE_MODEL so they do not run.
    """
    arch = config.getoption("--arch")

    # Merge torch and jax test configs once outside the loop
    combined_test_config = torch_test_config | jax_test_config | torch_llm_test_config

    deselected = []

    for item in items:

        nodeid = item.nodeid

        if "[" in nodeid:
            nodeid = nodeid[nodeid.index("[") + 1 : -1]

        meta = ModelTestConfig(combined_test_config.get(nodeid), arch)
        item._test_meta = meta  # attach for fixture access

        # Uncomment this to print info for each test collected.
        # print(f"DEBUG nodeid: {nodeid} meta.status: {meta.status}")

        # Skip auto-marking if test already has the placeholder marker. This simplifies the running
        # on -m unspecified tests in experimental nightly, don't need to exclude placeholder
        if item.get_closest_marker("placeholder") is not None:
            continue

        # Ability to mark models we don't want to run via test_models.py.
        if meta.status == ModelTestStatus.EXCLUDE_MODEL:
            deselected.append(item)
            continue

        if meta.status == ModelTestStatus.EXPECTED_PASSING:
            item.add_marker(pytest.mark.expected_passing)
        elif meta.status == ModelTestStatus.KNOWN_FAILURE_XFAIL:
            item.add_marker(pytest.mark.known_failure_xfail)
            item.add_marker(
                pytest.mark.xfail(
                    reason=getattr(meta, "reason", "known failure"),
                    strict=False,
                )
            )
        elif meta.status == ModelTestStatus.NOT_SUPPORTED_SKIP:
            item.add_marker(pytest.mark.not_supported_skip)
            item.add_marker(
                pytest.mark.skip(
                    reason=getattr(meta, "reason", "not supported"),
                )
            )
        elif meta.status == ModelTestStatus.UNSPECIFIED:
            item.add_marker(pytest.mark.unspecified)

        # Apply any custom/extra markers from config (e.g., "push", "nightly", "weekly")
        config_markers = getattr(meta, "markers", []) or []
        for marker_name in config_markers:
            item.add_marker(getattr(pytest.mark, marker_name))

        # Apply default nightly/weekly marker based on ModelGroup if not already specified in config.
        # RED and PRIORITY models run nightly, GENERALITY models run weekly.
        # Config markers take precedence and can override this default behavior.
        model_group = _get_model_group_from_item(item)
        if model_group is not None:
            # Get enum members from the same class to avoid import path identity issues
            ModelGroup = type(model_group)

            # Add "red" marker for RED models to enable filtering like: -m red
            if model_group == ModelGroup.RED:
                item.add_marker(pytest.mark.red)

            # Apply schedule markers if not already specified in config
            has_schedule_marker = any(
                m in config_markers for m in ("nightly", "weekly")
            )
            if not has_schedule_marker:
                if model_group in (ModelGroup.RED, ModelGroup.PRIORITY):
                    # RED and PRIORITY models run nightly
                    item.add_marker(pytest.mark.nightly)
                else:
                    # GENERALITY models run weekly
                    item.add_marker(pytest.mark.weekly)

        # Apply marker based on bringup_status to enable filtering like: -m incorrect_result
        bringup_status = getattr(meta, "bringup_status", None)
        if bringup_status:
            # Normalize enum or string to a pytest-safe, lowercase marker name
            status_str = str(bringup_status)
            # In case string includes enum class name, keep the last segment
            status_str = status_str.split(".")[-1]
            normalized_marker = status_str.lower()
            item.add_marker(getattr(pytest.mark, normalized_marker))

        # Define default set of supported archs, which can be optionally overridden in test_config files
        # by a model (ie. n300, n300-llmbox), and are applied as markers for filtering tests on CI.
        default_archs = ["n150", "p150"]
        archs_to_mark = getattr(meta, "supported_archs", None) or default_archs
        for arch_marker in archs_to_mark:
            # Prefer the exact string; if it contains a hyphen and pytest disallows it, also add underscore variant
            item.add_marker(getattr(pytest.mark, arch_marker))
            if "-" in arch_marker:
                item.add_marker(getattr(pytest.mark, arch_marker.replace("-", "_")))

    # Exclude deselected tests from the collected items and properly report them as deselected.
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = [i for i in items if i not in deselected]

    # Build nodeid→item mapping for crash-report fallback (see pytest_runtest_logreport).
    for item in items:
        _item_by_nodeid[item.nodeid] = item


@pytest.hookimpl(hookwrapper=True, trylast=True)
def pytest_runtest_logreport(report):
    """Inject static model properties into crash reports missing tags (forked process died).

    Uses hookwrapper=True so our pre-yield code runs before non-wrapper hooks
    (including pytest-junitxml), ensuring property injection and report.when
    normalisation are visible when junitxml reads the report.
    trylast=True makes this the outermost wrapper so it runs first in pre-yield.
    """
    # Only process failed crash reports (when="???") (this is produced by pytest-forked)
    # normal failures are handled by record_model_test_properties.
    if report.when != "???" or report.passed:
        yield
        return

    if any(key == "tags" for key, _ in report.user_properties):
        yield
        return

    item = _item_by_nodeid.get(report.nodeid)
    if item is None or not hasattr(item, "callspec") or not hasattr(item, "_test_meta"):
        yield
        return

    meta = item._test_meta

    test_entry, run_phase = _resolve_test_entry(item)
    if test_entry is None:
        yield
        return

    variant, ModelLoader = test_entry.variant_info
    model_info = ModelLoader.get_model_info(variant=variant)

    params = item.callspec.params
    run_mode = params.get("run_mode")
    parallelism = params.get("parallelism")
    weights_dtype = (
        "bfp8" if getattr(meta, "enable_weight_bfp8_conversion", False) else "bfloat16"
    )

    tags = build_model_tags(
        item,
        meta,
        model_info,
        run_mode=run_mode,
        run_phase=run_phase,
        parallelism=parallelism,
        weights_dtype=weights_dtype,
        bringup_status=BringupStatus.UNKNOWN,
    )

    report.user_properties.append(("tags", _to_marshal_safe(tags)))
    report.user_properties.append(("owner", "tt-xla"))
    if hasattr(model_info, "group") and model_info.group is not None:
        report.user_properties.append(("group", str(model_info.group)))

    # pytest-junitxml writes user_properties only when finalize() is
    # called, which happens on teardown reports. Crash reports (when="???")
    # never produce a teardown, so finalize() is never triggered.
    # Re-classifying as "teardown" causes junitxml to call finalize(report)
    # and write our properties. The crash message in longrepr is preserved.
    report.when = "teardown"

    yield


def _resolve_test_entry(item):
    """Resolve test_entry and run_phase from item's callspec params."""
    if not hasattr(item, "callspec"):
        return None, RunPhase.DEFAULT
    params = item.callspec.params
    test_entry = params.get("test_entry")
    run_phase = RunPhase.DEFAULT
    if test_entry is None:
        test_entry_and_phase = params.get("test_entry_and_phase")
        if test_entry_and_phase is not None:
            test_entry, run_phase = test_entry_and_phase
    return test_entry, run_phase


def _record_properties_from_hook(report, item, meta):
    """Record model test properties from the makereport hook."""

    def record_property(key, value):
        report.user_properties.append((key, value))

    model_info = getattr(item, "_model_info", None)
    if model_info is None:
        return

    record_model_test_properties(
        record_property,
        item,
        model_info=model_info,
        test_metadata=meta,
        run_mode=getattr(item, "_run_mode", None),
        run_phase=getattr(item, "_run_phase", RunPhase.DEFAULT),
        parallelism=getattr(item, "_parallelism", None),
        test_passed=getattr(item, "_test_passed", False),
        comparison_results=getattr(item, "_comparison_result", []),
        comparison_config=getattr(item, "_comparison_config", None),
        model_size=getattr(item, "_model_size", None),
        weights_dtype=getattr(item, "_weights_dtype", None),
    )


def _maybe_record_perf_benchmarks(item):
    """Record perf benchmark results for torch inference tests."""
    framework = getattr(item, "_framework", None)
    run_mode = getattr(item, "_run_mode", None)
    if framework != Framework.TORCH or run_mode != RunMode.INFERENCE:
        return

    tester = getattr(item, "_tester", None)
    loader = getattr(item, "_loader", None)
    model_info = getattr(item, "_model_info", None)
    parallelism = getattr(item, "_parallelism", None)

    if model_info is None:
        return

    measurements = getattr(tester, "_perf_measurements", None)
    model_config = loader.load_config() if loader else None
    batch_size, input_sequence_length, input_size = (
        get_input_shape_info(getattr(tester, "_input_activations", None))
        if tester
        else (1, -1, (-1,))
    )
    create_benchmark_result(
        full_model_name=model_info.name,
        output_dir=item.config.getoption("--perf-report-dir"),
        perf_id=item.config.getoption("--perf-id"),
        measurements=measurements,
        model_type=str(model_info.task),
        training=False,
        model_info=model_info.name,
        model_rawname=f"{model_info.model}_{model_info.variant}",
        model_group=str(model_info.group),
        parallelism=str(parallelism),
        device_arch=get_xla_device_arch(),
        run_mode=str(run_mode),
        device_name=socket.gethostname(),
        batch_size=batch_size,
        input_size=input_size,
        num_layers=getattr(model_config, "num_hidden_layers", 0) if model_config else 0,
        total_time=(
            measurements[0].get("total_time", -1)
            if measurements and len(measurements) > 0
            else -1
        ),
        total_samples=(
            measurements[0].get("perf_iters_count", -1)
            if measurements and len(measurements) > 0
            else -1
        ),
        input_sequence_length=input_sequence_length,
        data_format="bfloat16",
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Record model test properties and classify failures.

    Replaces the try/except/finally block that was in test_models.py.
    Handles:
    1. Failure classification using captured stdout/stderr
    2. Recording all model test properties (tags, owner, group)
    3. Perf benchmark recording (torch inference only)

    Skipped tests (NOT_SUPPORTED_SKIP) are handled for the 'setup' phase
    since they are skipped at collection time and never reach 'call'.
    """
    outcome = yield
    report = outcome.get_result()

    # Only process tests with model test metadata
    if not hasattr(item, "_test_meta"):
        return
    # Only process tests marked no_auto_properties (model tests)
    if not item.get_closest_marker("no_auto_properties"):
        return

    meta = item._test_meta

    # Handle skipped tests (skipped at collection time via pytest.mark.skip)
    if report.when == "setup" and report.skipped:
        # For NOT_SUPPORTED_SKIP tests, record properties on the setup report
        if meta.status == ModelTestStatus.NOT_SUPPORTED_SKIP:
            _record_skip_properties(report, item, meta)
        return

    if report.when != "call":
        return

    # 1. On failure: classify exception using captured stdout/stderr
    if report.failed and call.excinfo is not None:
        stdout = report.capstdout or None
        stderr = report.capstderr or None
        update_test_metadata_for_exception(
            meta, call.excinfo.value, stdout=stdout, stderr=stderr
        )

    # 2. Record all model test properties
    _record_properties_from_hook(report, item, meta)

    # 3. Perf benchmarks (torch inference only, on success)
    if report.passed:
        _maybe_record_perf_benchmarks(item)


def _record_skip_properties(report, item, meta):
    """Record properties for tests skipped at collection time."""

    def record_property(key, value):
        report.user_properties.append((key, value))

    # Resolve model_info from test_entry
    test_entry, run_phase = _resolve_test_entry(item)
    if test_entry is None:
        return

    variant, ModelLoader = test_entry.variant_info
    model_info = ModelLoader.get_model_info(variant=variant)

    params = item.callspec.params if hasattr(item, "callspec") else {}
    run_mode = params.get("run_mode")
    parallelism = params.get("parallelism")
    weights_dtype = (
        "bfp8" if getattr(meta, "enable_weight_bfp8_conversion", False) else "bfloat16"
    )

    record_model_test_properties(
        record_property,
        item,
        model_info=model_info,
        test_metadata=meta,
        run_mode=run_mode,
        run_phase=run_phase,
        parallelism=parallelism,
        weights_dtype=weights_dtype,
    )
