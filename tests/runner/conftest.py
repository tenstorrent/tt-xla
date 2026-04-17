# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess

import pytest

from tests.runner.requirements import RequirementsManager
from tests.runner.test_config.constants import ALLOWED_ARCHES
from tests.runner.test_config.jax import test_config as jax_test_config
from tests.runner.test_config.torch import test_config as torch_test_config
from tests.runner.test_config.torch_llm import test_config as torch_llm_test_config
from tests.runner.test_utils import (
    ModelTestConfig,
    ModelTestStatus,
    RunPhase,
    to_marshal_safe,
)
from tests.utils import BringupStatus, Category

_BRINGUP_STAGE_FILE = "._bringup_stage.txt"

# Maps nodeid -> item, populated during collection for crash-report fallback.
_item_by_nodeid: dict = {}


def pytest_sessionstart(session):
    """Capture the clean pip environment at session start for crash recovery.

    This must be a hook (not a session-scoped fixture) because pytest-forked
    runs ``runtestprotocol`` — including fixture setup — inside each fork()ed
    child.  A session fixture would lazily re-initialise in every child,
    capturing the (possibly dirty) on-disk state instead of the original
    clean state.  ``pytest_sessionstart`` runs once in the parent process
    before any forking occurs, so the snapshot is taken exactly once and
    inherited by all children via fork().
    """
    RequirementsManager.capture_golden_state()


@pytest.fixture(autouse=True)
def restore_pip_env_if_dirty():
    """Restore pip environment before each test if a previous fork was killed."""
    RequirementsManager.check_and_restore_environment()


@pytest.fixture(autouse=True)
def reset_card_between_tests():
    """Reset TT hardware card before each test when TT_RESET_CARD_BETWEEN_TESTS=1.

    Enabled via qb2-model-sweep.json preset to ensure clean hardware state
    between model tests. Runs inside each forked child process before the test.
    """
    if os.environ.get("TT_RESET_CARD_BETWEEN_TESTS") == "1":
        print("Resetting card before test (tt-smi -r)...", flush=True)
        result = subprocess.run(
            ["tt-smi", "-r"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("Card reset successful.", flush=True)
        else:
            print(
                f"WARNING: tt-smi -r failed (exit {result.returncode}): {result.stderr}",
                flush=True,
            )
    yield


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
        elif meta.status == ModelTestStatus.NOT_SUPPORTED_SKIP:
            item.add_marker(pytest.mark.not_supported_skip)
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
            elif model_group == ModelGroup.GENERALITY:
                item.add_marker(pytest.mark.generality)
            elif model_group == ModelGroup.VULCAN:
                item.add_marker(pytest.mark.vulcan)

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

    # Build nodeid->item mapping for crash-report fallback (see pytest_runtest_logreport).
    for item in items:
        _item_by_nodeid[item.nodeid] = item


@pytest.hookimpl(hookwrapper=True, trylast=True)
def pytest_runtest_logreport(report):
    """Inject static model properties into crash reports missing tags (forked process died)."""

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
    params = item.callspec.params

    # Resolve test_entry (regular: "test_entry"; LLM: "test_entry_and_phase")
    test_entry = params.get("test_entry")
    run_phase = RunPhase.DEFAULT
    if test_entry is None:
        test_entry_and_phase = params.get("test_entry_and_phase")
        if test_entry_and_phase is not None:
            test_entry, run_phase = test_entry_and_phase

    if test_entry is None:
        yield
        return

    variant, ModelLoader = test_entry.variant_info
    model_info = ModelLoader.get_model_info(variant=variant)

    run_mode = params.get("run_mode")
    parallelism = params.get("parallelism")
    weights_dtype = (
        "bfp8" if getattr(meta, "enable_weight_bfp8_conversion", False) else "bfloat16"
    )

    tags = {
        "test_name": str(item.originalname),
        "specific_test_case": str(item.name),
        "category": str(Category.MODEL_TEST),
        "model_name": str(model_info.name),
        "model_info": model_info.to_report_dict(),
        "run_mode": str(run_mode) if run_mode is not None else None,
        "run_phase": str(run_phase),
        "parallelism": str(parallelism) if parallelism is not None else None,
        "bringup_status": str(BringupStatus.UNKNOWN),
        "model_test_status": str(meta.status),
        "arch": getattr(meta, "arch", None),
        "seq_len": getattr(meta, "seq_len", None),
        "batch_size": getattr(meta, "batch_size", None),
        "weights_dtype": weights_dtype,
        "failing_reason": {
            "name": None,
            "description": None,
            "component": None,
            "summary": None,
        },
        "guidance": [],
    }

    report.user_properties.append(("tags", to_marshal_safe(tags)))
    report.user_properties.append(("owner", "tt-xla"))
    if hasattr(model_info, "group") and model_info.group is not None:
        report.user_properties.append(("group", str(model_info.group)))

    # pytest-junitxml writes user_properties only when finalize() is
    # called, which happens on teardown reports. Crash reports (when="???")
    # never produce a teardown, so finalize() is never triggered.
    # Re-classifying as "teardown" causes junitxml to call finalize(report)
    # and write our properties. The crash message in longrepr is preserved.
    # Verified against pytest 9.0.2 and pytest-forked 1.6.0.
    report.when = "teardown"

    yield
