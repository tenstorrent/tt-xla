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
import torch
import torch_xla.runtime as xr
from infra import ComparisonConfig, RunMode, TorchModelTester
from infra.utilities.torch_multichip_utils import get_mesh
from torch_xla.distributed.spmd import Mesh

from tests.infra.comparators import comparison_config
from tests.utils import BringupStatus, Category
from third_party.tt_forge_models.config import Parallelism


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


# This attempts to classify various exception types but is not robust at all.
# Soon https://github.com/tenstorrent/tt-xla/issues/1052 will improve bringup_status reporting
# and this would be updated to use actual bringup_status achieved by a model.
def update_test_metadata_for_exception(
    test_metadata, exc: Exception, stderr: str
) -> None:
    """
    Inspect exception message, stderr and set `runtime_bringup_status` and `runtime_reason` on `test_metadata`.
    """
    try:
        message = str(exc)
    except Exception:
        message = repr(exc)

    msg = message.lower() if message else ""
    err = (stderr or "").lower()
    # print(f"Found exception: {repr(exc)} message: {msg} stderr: {err}")

    if isinstance(exc, AssertionError) and "comparison failed" in msg:
        status = BringupStatus.INCORRECT_RESULT
    elif isinstance(exc, RuntimeError):
        if (
            "failed to legalize" in err
            or "stablehlo" in err
            or "mhlo" in err
            or "mlir" in err
        ):
            status = BringupStatus.FAILED_TTMLIR_COMPILATION
        elif "bad statusor access" in msg or "internal: error code: 13" in msg:
            status = BringupStatus.FAILED_RUNTIME
        else:
            status = BringupStatus.FAILED_RUNTIME
    else:
        status = BringupStatus.UNKNOWN

    setattr(test_metadata, "runtime_bringup_status", status)
    setattr(test_metadata, "runtime_reason", message)


# DynamicTorchModelTester class for data parallel support
class DynamicTorchModelTester(TorchModelTester):
    def __init__(
        self,
        run_mode: RunMode,
        *,
        loader,
        comparison_config: ComparisonConfig | None = None,
        parallelism: Parallelism = Parallelism.SINGLE_DEVICE,
    ) -> None:
        self.loader = loader
        # Optional: store requested parallelism for reporting/consumers
        self.parallelism = parallelism

        super().__init__(
            comparison_config=comparison_config or ComparisonConfig(),
            run_mode=run_mode,
            parallelism=self.parallelism,
        )

    # --- TorchModelTester interface implementations ---

    def _get_model(self):
        sig = inspect.signature(self.loader.load_model)
        if "dtype_override" in sig.parameters:
            return self.loader.load_model(dtype_override=torch.bfloat16)
        return self.loader.load_model()

    def _get_input_activations(self):
        sig = inspect.signature(self.loader.load_inputs)
        inputs = None
        if "dtype_override" in sig.parameters:
            inputs = self.loader.load_inputs(dtype_override=torch.bfloat16)
        else:
            inputs = self.loader.load_inputs()

        if self.parallelism == Parallelism.DATA_PARALLEL:

            def batch_tensor(tensor, num_devices):
                if isinstance(tensor, torch.Tensor):
                    if tensor.dim() == 0:
                        return tensor.repeat(num_devices)
                    else:
                        if tensor.dim() == 1:
                            tensor = tensor.unsqueeze(0)
                        return tensor.repeat_interleave(num_devices, dim=0)
                return tensor

            num_devices = xr.global_runtime_device_count()
            if isinstance(inputs, collections.abc.Mapping):
                inputs = {k: batch_tensor(v, num_devices) for k, v in inputs.items()}
            elif isinstance(inputs, collections.abc.Sequence):
                inputs = [batch_tensor(inp, num_devices) for inp in inputs]
            else:
                inputs = batch_tensor(inputs, num_devices)
        return inputs

    def _get_shard_specs_function(self):
        if self.parallelism == Parallelism.DATA_PARALLEL:

            def load_shard_spec(args, kwargs):
                shard_specs = {}
                for arg in args:
                    if isinstance(arg, torch.Tensor) and arg.dim() > 0:
                        shard_spec = [None] * len(arg.shape)
                        shard_spec[0] = "data"
                        shard_specs[arg] = tuple(shard_spec)
                for kwarg_value in kwargs.values():
                    if isinstance(kwarg_value, torch.Tensor) and kwarg_value.dim() > 0:
                        shard_spec = [None] * len(kwarg_value.shape)
                        shard_spec[0] = "data"
                        shard_specs[kwarg_value] = tuple(shard_spec)
                return shard_specs

            return load_shard_spec
        else:
            return self.loader.load_shard_spec

    def _get_mesh(self):
        num_devices = xr.global_runtime_device_count()
        if self.parallelism == Parallelism.DATA_PARALLEL:
            mesh_shape, mesh_names = (1, num_devices), ("model", "data")
        else:
            mesh_shape, mesh_names = self.loader.get_mesh_config(num_devices)

        return get_mesh(mesh_shape, mesh_names)


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
    - Passing tests (test_passed=True) always record bringup_status=PASSED, ignoring configured/static values.
    - Failing tests classify bringup info in this order:
      1) Static: use test_metadata.bringup_status/reason from config when both are present
      2) Runtime: else use test_metadata.runtime_bringup_status/runtime_reason when both are present
      3) Default: else use UNKNOWN/"Not specified"
    - If test_metadata.status is NOT_SUPPORTED_SKIP, set bringup_status from static/default logic and call pytest.skip(reason).
    - If test_metadata.status is KNOWN_FAILURE_XFAIL, leave execution to xfail via marker; properties still reflect runtime/static/default classification.
    """

    # Determine bringup status and reason based on runtime/test outcome
    reason = None
    static_bringup_status = getattr(test_metadata, "bringup_status", None)
    static_reason = getattr(test_metadata, "reason", None)
    arch = getattr(test_metadata, "arch", None)

    if test_passed:
        # If custom bringup_status and reason are provided, use them.
        reason = static_reason or None
        bringup_status = static_bringup_status or BringupStatus.PASSED

        # Handle common case where test passes but is statically marked as INCORRECT_RESULT and doesn't contain a reason.
        # In this case, report PCC check enablement and results for superset dashboard visibility on latest results.
        if (
            static_reason is None
            and static_bringup_status == BringupStatus.INCORRECT_RESULT
        ):
            pcc = comparison_result.pcc
            required_pcc = comparison_config.pcc.required_pcc
            pcc_check_str = "enabled" if comparison_config.pcc.enabled else "disabled"
            reason = f"Test marked w/ INCORRECT_RESULT. PCC check {pcc_check_str}. Calculated: pcc={pcc}. Required: pcc={required_pcc}."

    else:
        runtime_bringup_status = getattr(test_metadata, "runtime_bringup_status", None)
        runtime_reason = getattr(test_metadata, "runtime_reason", None)

        if static_bringup_status and static_reason:
            bringup_status = static_bringup_status
            reason = static_reason
        elif runtime_bringup_status and runtime_reason:
            bringup_status = runtime_bringup_status
            reason = runtime_reason or "Runtime failure"
        else:
            bringup_status = BringupStatus.UNKNOWN
            reason = "Not specified"

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
        import pytest

        pytest.skip(reason)
    elif test_metadata.status == ModelTestStatus.KNOWN_FAILURE_XFAIL:
        import pytest

        pytest.xfail(reason)
