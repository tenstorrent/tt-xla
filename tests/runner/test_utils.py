# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import collections
import importlib.util
import inspect
import numbers
import os
import sys
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pytest
import torch
import torch_xla.runtime as xr
from infra import ComparisonConfig, RunMode, TorchModelTester
from infra.utilities.torch_multichip_utils import get_mesh
from torch_xla.distributed.spmd import Mesh

from tests.infra.comparators import comparison_config
from tests.utils import BringupStatus, Category
from third_party.tt_forge_models.config import Parallelism

BRINGUP_STAGE_FILE = "._bringup_stage.txt"


def fix_venv_isolation():
    """
    Fix venv isolation issue: ensure venv packages take precedence over system packages.

    This function adjusts the Python path to prioritize virtual environment packages
    over system packages, preventing package conflicts and ensuring proper isolation
    during test execution.
    """
    venv_site = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "..",
        "venv",
        "lib",
        "python3.11",
        "site-packages",
    )
    if os.path.exists(venv_site) and venv_site not in sys.path:
        sys.path.insert(0, os.path.abspath(venv_site))

    # Remove system packages from path to ensure proper isolation
    sys.path = [
        p
        for p in sys.path
        if "/usr/local/lib/python3.11/dist-packages" not in p
        or p == "/usr/local/lib/python3.11/dist-packages"
    ]
    # Re-add at the end as fallback
    if "/usr/local/lib/python3.11/dist-packages" not in sys.path:
        sys.path.append("/usr/local/lib/python3.11/dist-packages")


class ModelTestStatus(Enum):
    # Passing tests
    EXPECTED_PASSING = "expected_passing"
    # Known failures that should be xfailed
    KNOWN_FAILURE_XFAIL = "known_failure_xfail"
    # Not supported on this architecture or low priority
    NOT_SUPPORTED_SKIP = "not_supported_skip"
    # New model, awaiting triage
    UNSPECIFIED = "unspecified"
    # Avoid import and auto discovery. Can be used if model test is hand written.
    EXCLUDE_MODEL = "exclude_model"


class ModelTestConfig:
    def __init__(self, data, arch=None):
        self.data = data or {}
        self.arch = arch

        # For marking tests as expected passing, known failures, etc
        self.status = self._resolve("status", default=ModelTestStatus.UNSPECIFIED)

        # Arguments to ModelTester
        self.required_pcc = self._resolve("required_pcc", default=None)
        self.assert_pcc = self._resolve("assert_pcc", default=None)
        # Enable/override absolute-tolerance comparator
        self.assert_atol = self._resolve("assert_atol", default=False)
        self.required_atol = self._resolve("required_atol", default=None)
        # Allclose comparator controls
        self.assert_allclose = self._resolve("assert_allclose", default=False)
        self.allclose_rtol = self._resolve("allclose_rtol", default=None)
        self.allclose_atol = self._resolve("allclose_atol", default=None)

        # Misc arguments used in test
        self.batch_size = self._resolve("batch_size", default=None)

        # Arguments to skip_full_eval_test() for skipping tests
        self.reason = self._resolve("reason", default=None)
        self.bringup_status = self._resolve("bringup_status", default=None)

        # Optional list of pytest markers to apply (e.g. ["push", "nightly"]) - normalized to list[str]
        self.markers = self._normalize_markers(self._resolve("markers", default=[]))

        # Optional list of supported architectures (e.g. ["p150", "n300", "n300-llmbox"]) - normalized to list[str]
        self.supported_archs = self._normalize_markers(
            self._resolve("supported_archs", default=[])
        )

    def _resolve(self, key, default=None):
        overrides = self.data.get("arch_overrides", {})
        if self.arch in overrides and key in overrides[self.arch]:
            return overrides[self.arch][key]
        return self.data.get(key, default)

    def to_comparison_config(self) -> ComparisonConfig:
        """Build a ComparisonConfig directly from this test metadata."""
        config = ComparisonConfig()
        # PCC comparator
        if self.assert_pcc is False:
            config.pcc.disable()
        else:
            config.pcc.enable()
        if self.required_pcc is not None:
            config.pcc.required_pcc = self.required_pcc

        # ATOL comparator (absolute tolerance only, separate from allclose)
        if self.assert_atol or (self.required_atol is not None):
            config.atol.enable()
            if self.required_atol is not None:
                config.atol.required_atol = self.required_atol

        # ALLCLOSE comparator (rtol/atol)
        enable_allclose = bool(
            self.assert_allclose
            or self.allclose_rtol is not None
            or self.allclose_atol is not None
        )
        if enable_allclose:
            config.allclose.enable()

            # Apply provided thresholds
            if self.allclose_rtol is not None:
                config.allclose.rtol = self.allclose_rtol
            if self.allclose_atol is not None:
                config.allclose.atol = self.allclose_atol

            # Keep PCC fallback allclose thresholds in sync if user provided overrides
            if self.allclose_rtol is not None:
                config.pcc.allclose.rtol = self.allclose_rtol
            if self.allclose_atol is not None:
                config.pcc.allclose.atol = self.allclose_atol

        config.assert_on_failure = False
        return config

    def _normalize_markers(self, markers_value):
        if markers_value is None:
            return []
        if isinstance(markers_value, str):
            return [markers_value]
        try:
            return [str(m) for m in markers_value if m]
        except TypeError:
            return []


def parse_last_bringup_stage() -> BringupStatus | None:
    """
    Read the current stage from file and map to BringupStatus.

    This function reads the structured logging marker written to ._bringup_stage.txt (at repo root)
    by the C++ compilation/execution pipeline when ENABLE_BRINGUP_STAGE_LOGGING=1 is set.

    Returns:
        BringupStatus: The bringup status based on the last stage reached before failure.
        None: If the file doesn't exist or cannot be read.
    """
    try:
        with open(BRINGUP_STAGE_FILE, "r") as f:
            stage_name = f.read().strip()
    except (FileNotFoundError, IOError):
        return None

    if not stage_name:
        return None

    # Map stage to BringupStatus
    stage_to_status = {
        "FE_COMPILATION_START": BringupStatus.FAILED_FE_COMPILATION,
        "TTMLIR_COMPILATION_START": BringupStatus.FAILED_TTMLIR_COMPILATION,
        "RUNTIME_EXECUTION_START": BringupStatus.FAILED_RUNTIME,
    }

    return stage_to_status.get(stage_name)


# This is needed for combination of pytest-forked and using ruamel.yaml
# ruamel returns ScalarFloat/ScalarString types (subclasses of float/str).
# pytest-forked uses Python's marshal, which rejects non-builtin subclasses inside the
# test report's user_properties, causing ValueError: unmarshallable object.
def _to_marshal_safe(value):
    """Recursively convert values to marshal-safe builtin types for pytest-forked."""
    # None stays None
    if value is None:
        return None

    # Enums -> string representation
    if isinstance(value, Enum):
        return str(value)

    # Numpy scalar types -> corresponding python scalar
    if isinstance(value, np.generic):
        return value.item()

    # Primitive scalars, ensure builtin types
    # Note: bool must be checked before Integral (since bool is a subclass of int)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value)
    if isinstance(value, (str, bytes)):
        return value.decode() if isinstance(value, bytes) else value

    # Collections
    if isinstance(value, dict):
        return {str(k): _to_marshal_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_marshal_safe(v) for v in value]

    # Fallback to string to avoid unmarshallable objects
    return str(value)


def record_model_test_properties(
    record_property,
    request,
    *,
    model_info,
    test_metadata,
    run_mode: RunMode,
    parallelism: Parallelism,
    test_passed: bool = False,
    comparison_result=None,
    comparison_config=None,
):
    """
    Record standard runtime properties for model tests and optionally control flow.

    - Always records tags (including test_name, specific_test_case, category, model_name, run_mode, bringup_status),
      plus owner and group properties.
    - Passing tests (test_passed=True) set bringup_status based on PCC comparison.
    - Failing tests classify bringup info based on the last stage reached before failure.
    - If test_metadata.status is NOT_SUPPORTED_SKIP, set bringup_status and reason from config and call pytest.skip(reason).
    - If test_metadata.status is KNOWN_FAILURE_XFAIL, call pytest.xfail(reason) at the end.
    """

    reason = None
    arch = getattr(test_metadata, "arch", None)

    if test_metadata.status == ModelTestStatus.NOT_SUPPORTED_SKIP:
        bringup_status = getattr(test_metadata, "bringup_status", None)
        reason = getattr(test_metadata, "reason", None)

    elif comparison_result is not None:
        pcc = comparison_result.pcc
        required_pcc = comparison_config.pcc.required_pcc
        if pcc < required_pcc:
            bringup_status = BringupStatus.INCORRECT_RESULT
            required_pcc = comparison_config.pcc.required_pcc
            pcc_check_str = "enabled" if comparison_config.pcc.enabled else "disabled"
            reason = f"Test marked w/ INCORRECT_RESULT. PCC check {pcc_check_str}. Calculated: pcc={pcc}. Required: pcc={required_pcc}."

    elif test_passed:
        bringup_status = BringupStatus.PASSED

    else:
        # If test fails, use the bringup status from the last stage reached before failure.
        # TODO: add better way to set the reason dynamically.
        static_reason = getattr(test_metadata, "reason", None)
        runtime_reason = getattr(test_metadata, "runtime_reason", None)

        if comparison_result is None:
            bringup_status = parse_last_bringup_stage()
            if bringup_status is None:
                bringup_status = BringupStatus.UNKNOWN
        else:
            bringup_status = BringupStatus.INCORRECT_RESULT

        reason = static_reason or runtime_reason or "Not specified"

    tags = {
        "test_name": str(request.node.originalname),
        "specific_test_case": str(request.node.name),
        "category": str(Category.MODEL_TEST),
        "model_name": str(model_info.name),
        "model_info": model_info.to_report_dict(),
        "run_mode": str(run_mode),
        "bringup_status": str(bringup_status),
        "parallelism": str(parallelism),
        "arch": arch,
    }

    # Add comparison result metrics if available
    if comparison_result is not None:
        tags.update(
            {
                "pcc": comparison_result.pcc,
                "atol": comparison_result.atol,
                "comparison_passed": comparison_result.passed,
                "comparison_error_message": comparison_result.error_message,
            }
        )
    if comparison_config is not None:
        tags.update(
            {
                "pcc_threshold": comparison_config.pcc.required_pcc,
                "atol_threshold": comparison_config.atol.required_atol,
                "pcc_assertion_enabled": comparison_config.pcc.enabled,
                "atol_assertion_enabled": comparison_config.atol.enabled,
            }
        )

    # If we have an explanatory reason, include it as a top-level property too for DB visibility
    # which is especially useful for passing tests (used to just from xkip/xfail reason)
    if reason:
        record_property("error_message", _to_marshal_safe(reason))

    # Write properties
    record_property("tags", _to_marshal_safe(tags))
    record_property("owner", "tt-xla")
    if hasattr(model_info, "group") and model_info.group is not None:
        record_property("group", str(model_info.group))

    # Control flow for skipped and xfailed tests is handled by pytest.
    if test_metadata.status == ModelTestStatus.NOT_SUPPORTED_SKIP:
        pytest.skip(reason)
    elif test_metadata.status == ModelTestStatus.KNOWN_FAILURE_XFAIL:
        pytest.xfail(reason)
