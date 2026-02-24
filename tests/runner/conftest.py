# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

from tests.runner.test_config.constants import ALLOWED_ARCHES
from tests.runner.test_config.jax import test_config as jax_test_config
from tests.runner.test_config.torch import test_config as torch_test_config
from tests.runner.test_config.torch_llm import test_config as torch_llm_test_config
from tests.runner.test_utils import ModelTestConfig, ModelTestStatus

_BRINGUP_STAGE_FILE = "._bringup_stage.txt"


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


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """
    Print detected failing reason to console when available.
    """
    outcome = yield
    report = outcome.get_result()

    # Only print once per test call phase
    if report.when != "call":
        return

    meta = getattr(item, "_test_meta", None)
    failing_reason = getattr(meta, "failing_reason", None) if meta else None
    if failing_reason:
        desc = failing_reason.value.description
        print(f"Failing reason - {desc}")
