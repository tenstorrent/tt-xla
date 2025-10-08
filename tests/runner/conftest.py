# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from tests.runner.test_config import test_config
from tests.runner.test_utils import ModelTestConfig, ModelTestStatus
import difflib

# Global set to track collected test node IDs
_collected_nodeids = set()

# Allowed architecture identifiers for arch_overrides and --arch option
ALLOWED_ARCHES = {"n150", "p150", "n300", "n300-llmbox"}


def pytest_addoption(parser):
    """Register CLI options for selecting target arch and enabling config validation."""
    parser.addoption(
        "--arch",
        action="store",
        default=None,
        choices=sorted(ALLOWED_ARCHES),
        help="Target architecture (e.g., n150, p150) for which to match via arch_overrides in test_config.py",
    )
    parser.addoption(
        "--validate-test-config",
        action="store_true",
        default=False,
        help="Fail if test_config.py and collected test IDs are out of sync",
    )


@pytest.fixture
def test_metadata(request) -> ModelTestConfig:
    """Expose per-test ModelTestConfig attached during collection."""
    meta = getattr(request.node, "_test_meta", None)
    assert meta is not None, f"No ModelTestConfig attached for {request.node.nodeid}"
    return meta


def pytest_collection_modifyitems(config, items):
    """During collection, attach ModelTestConfig, apply markers, and optionally clear tests when validating config.

    Also deselect tests explicitly marked with EXCLUDE_MODEL so they do not run.
    """
    arch = config.getoption("--arch")
    validate_config = config.getoption("--validate-test-config")

    deselected = []

    for item in items:
        nodeid = item.nodeid
        if "[" in nodeid:
            nodeid = nodeid[nodeid.index("[") + 1 : -1]

        _collected_nodeids.add(nodeid)  # Track for final validation

        meta = ModelTestConfig(test_config.get(nodeid), arch)
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

        # Apply any custom/extra markers from config (e.g., "push", "nightly")
        for marker_name in getattr(meta, "markers", []) or []:
            item.add_marker(getattr(pytest.mark, marker_name))

        # Define default set of supported archs, which can be optionally overridden in test_config.py
        # by a model (ie. n300, n300-llmbox), and are applied as markers for filtering tests on CI.
        default_archs = ["n150", "p150"]
        archs_to_mark = getattr(meta, "supported_archs", None) or default_archs
        for arch_marker in archs_to_mark:
            # Prefer the exact string; if it contains a hyphen and pytest disallows it, also add underscore variant
            item.add_marker(getattr(pytest.mark, arch_marker))
            if "-" in arch_marker:
                item.add_marker(getattr(pytest.mark, arch_marker.replace("-", "_")))

    # Exclude deselected tests from the collected items.
    if deselected:
        items[:] = [i for i in items if i not in deselected]

    # If validating config, clear all items so no tests run
    if validate_config:
        items.clear()


def pytest_sessionfinish(session, exitstatus):
    """At session end, validate test_config entries and arch_overrides against collected tests."""
    if not session.config.getoption("--validate-test-config"):
        return  # Skip check unless explicitly requested

    print("\n" + "=" * 60)
    print("VALIDATING TEST CONFIGURATIONS")
    print("=" * 60 + "\n")

    # Basic validation: ensure all arch_overrides keys use allowed arches
    invalid_arch_entries = []
    for test_name, cfg in test_config.items():
        if not isinstance(cfg, dict):
            continue
        overrides = cfg.get("arch_overrides")
        if overrides is None:
            continue
        if not isinstance(overrides, dict):
            invalid_arch_entries.append((test_name, "arch_overrides is not a dict"))
            continue
        for arch_key in overrides.keys():
            if arch_key not in ALLOWED_ARCHES:
                invalid_arch_entries.append((test_name, f"unknown arch '{arch_key}'"))

    if invalid_arch_entries:
        print("ERROR: Found invalid arch_overrides entries (unknown arches):")
        for test_name, reason in sorted(invalid_arch_entries):
            print(f"  - {test_name}: {reason}")
        print("\nAllowed arches:", ", ".join(sorted(ALLOWED_ARCHES)))
        print("\n" + "=" * 60)
        raise pytest.UsageError(
            "test_config.py contains arch_overrides with unknown arches"
        )
    else:
        print("All arch_overrides entries are valid")

    # Validate that entries in test_config.py are found in the collected tests. They can diverge if
    # model variants are renamed, removed, have import errors, etc.
    declared_nodeids = set(test_config.keys())
    unknown = declared_nodeids - _collected_nodeids
    unlisted = _collected_nodeids - declared_nodeids
    print(
        f"Found {len(unknown)} unknown tests and {len(unlisted)} unlisted tests",
        flush=True,
    )

    # Unlisted tests are just warnings, for informational purposes.
    if unlisted:
        print("\nWARNING: The following tests are missing from test_config.py:")
        for test_name in sorted(unlisted):
            print(f"  - {test_name}")
    else:
        print("\nAll collected tests are properly defined in test_config.py")

    # Unknown tests are tests listed that no longer exist, treat as error.
    if unknown:
        msg = "test_config.py contains entries not found in collected tests."
        print(f"\nERROR: {msg}")
        for test_name in sorted(unknown):
            print(f"  - {test_name}")
            suggestion = difflib.get_close_matches(test_name, _collected_nodeids, n=1)
            if suggestion:
                print(f"    Did you mean: {suggestion[0]}?")
        print("\n" + "=" * 60)
        raise pytest.UsageError(msg)
    else:
        session.exitstatus = 0
