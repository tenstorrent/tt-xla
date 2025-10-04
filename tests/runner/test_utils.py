# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import importlib.util
import torch
import inspect
from enum import Enum
import collections
from dataclasses import dataclass
from infra import ComparisonConfig, RunMode, TorchModelTester
from tests.utils import BringupStatus, Category
from third_party.tt_forge_models.config import Parallelism
from infra.utilities.xla_multichip_utils import get_mesh
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh
import numpy as np


@dataclass
class ModelTestEntry:
    path: str
    variant_info: tuple


class ModelTestStatus(Enum):
    # Passing tests
    EXPECTED_PASSING = "expected_passing"
    # Known failures that should be xfailed
    KNOWN_FAILURE_XFAIL = "known_failure_xfail"
    # Not supported on this architecture or low priority
    NOT_SUPPORTED_SKIP = "not_supported_skip"
    # New model, awaiting triage
    UNSPECIFIED = "unspecified"


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


def get_models_root(project_root):
    """Return the filesystem path to the given module, supporting both installed and source-tree use cases."""
    module_name = "third_party.tt_forge_models"
    spec = importlib.util.find_spec(module_name)
    if spec:
        if spec.submodule_search_locations:
            return spec.submodule_search_locations[0]
        elif spec.origin:
            return os.path.dirname(os.path.abspath(spec.origin))

    # Derive filesystem path from module name
    rel_path = os.path.join(*module_name.split("."))
    fallback = os.path.join(project_root, rel_path)
    print(f"No installed {module_name}; falling back to {fallback}")
    return fallback


def import_model_loader(loader_path, models_root):
    """Dynamically import and return ModelLoader class from a loader.py path, ensuring relative imports work."""
    # Import the base module first to ensure it's available
    models_parent = os.path.dirname(models_root)
    if models_parent not in sys.path:
        sys.path.insert(0, models_parent)

    # Get the relative path from models_root to construct proper module name
    rel_path = os.path.relpath(loader_path, models_root)
    rel_path_without_ext = rel_path.replace(".py", "")

    # Use different/dummy module name to avoid conflicts with real package name
    module_path = "tt-forge-models." + rel_path_without_ext.replace(os.sep, ".")

    spec = importlib.util.spec_from_file_location(module_path, location=loader_path)
    mod = importlib.util.module_from_spec(spec)

    # Set the module's __package__ for relative imports to work
    loader_dir = os.path.dirname(loader_path)
    package_name = "tt_forge_models." + os.path.relpath(
        loader_dir, models_root
    ).replace(os.sep, ".")
    mod.__package__ = package_name
    mod.__name__ = module_path

    # Add the module to sys.modules to support relative imports
    sys.modules[module_path] = mod
    spec.loader.exec_module(mod)

    return mod.ModelLoader


def get_model_variants(loader_path, loader_paths, models_root):
    """Fill loader_paths[loader_path] with (variant_name, ModelLoader) tuples by querying the loader; on failure, log and continue."""
    try:
        # Import the ModelLoader class from the module
        ModelLoader = import_model_loader(loader_path, models_root)
        variants = ModelLoader.query_available_variants()

        # Store variant_name, ModelLoader together for usage, or empty one if no variants found.
        if variants:
            for variant_name in variants.keys():
                loader_paths[loader_path].append((variant_name, ModelLoader))
        else:
            loader_paths[loader_path].append((None, ModelLoader))

    except Exception as e:
        print(f"Cannot import path: {loader_path}: {e}")


def generate_test_id(test_entry, models_root):
    """Generate test ID from test entry with optional variant."""
    model_path = os.path.relpath(os.path.dirname(test_entry.path), models_root)
    variant_info = test_entry.variant_info
    variant_name = variant_info[0] if variant_info else None
    return f"{model_path}-{variant_name}" if variant_name else model_path


class DynamicTorchModelTester(TorchModelTester):
    def __init__(
        self,
        run_mode: RunMode,
        *,
        loader,
        comparison_config: ComparisonConfig | None = None,
    ) -> None:
        self.loader = loader

        super().__init__(
            comparison_config=comparison_config or ComparisonConfig(),
            run_mode=run_mode,
        )

    # --- TorchModelTester interface implementations ---

    def _get_model(self):
        sig = inspect.signature(self.loader.load_model)
        if "dtype_override" in sig.parameters:
            return self.loader.load_model(dtype_override=torch.bfloat16)
        return self.loader.load_model()

    def _get_input_activations(self):
        sig = inspect.signature(self.loader.load_inputs)
        if "dtype_override" in sig.parameters:
            return self.loader.load_inputs(dtype_override=torch.bfloat16)
        return self.loader.load_inputs()

    def _get_shard_specs_function(self):
        return self.loader.load_shard_spec

    def _get_mesh(self):
        num_devices = xr.global_runtime_device_count()
        mesh_shape, mesh_names = self.loader.get_mesh_config(num_devices)
        return get_mesh(mesh_shape, mesh_names)


def setup_models_path(project_root):
    """Setup models root path and add to sys.path for imports."""
    models_root = get_models_root(project_root)

    # Add the models root to sys.path so relative imports work
    if models_root not in sys.path:
        sys.path.insert(0, models_root)

    return models_root


def discover_loader_paths(models_root):
    """Discover all pytorch (for now) loader.py files in the models directory, with exclusions."""
    loader_paths = {}

    # TODO(kmabee) - Temporary workaround to exclude models with fatal issues.
    # Surya OCR imports and initializes torch_xla runtime which causes issues
    # https://github.com/tenstorrent/tt-xla/issues/1166
    excluded_model_dirs = {"suryaocr"}

    for root, dirs, files in os.walk(models_root):

        model_dir_name = os.path.basename(os.path.dirname(root))
        if model_dir_name in excluded_model_dirs:
            print(
                f"Workaround to exclude model: {model_dir_name} from discovery. Issue #1166",
                flush=True,
            )
            continue

        if os.path.basename(root) == "pytorch" and "loader.py" in files:
            loader_paths[os.path.join(root, "loader.py")] = []

    # Populate variants for each loader path
    for path in loader_paths.keys():
        get_model_variants(path, loader_paths, models_root)

    return loader_paths


def create_test_entries(loader_paths):
    """Create test entries combining loader paths and variants."""
    test_entries = []

    # Development / Debug workaround to collect only red models.
    red_only = os.environ.get("TT_XLA_RED_ONLY", "0") == "1"

    # Store variant tuple along with the ModelLoader
    for loader_path, variant_tuples in loader_paths.items():
        for variant_info in variant_tuples:

            if red_only:
                model_info = variant_info[1].get_model_info(variant_info[0])
                if model_info.group.value != "red":
                    continue

            test_entries.append(
                ModelTestEntry(path=loader_path, variant_info=variant_info)
            )

    return test_entries


def setup_test_discovery(project_root):
    """Complete test discovery setup - combines all the setup steps."""
    models_root = setup_models_path(project_root)
    loader_paths = discover_loader_paths(models_root)
    test_entries = create_test_entries(loader_paths)
    return models_root, test_entries


def create_test_id_generator(models_root):
    """Create a function for generating test IDs."""

    def _generate_test_id(test_entry):
        """Generate test ID from test entry using the utility function."""
        return generate_test_id(test_entry, models_root)

    return _generate_test_id


def record_model_test_properties(
    record_property,
    request,
    *,
    model_info,
    test_metadata,
    run_mode: RunMode = RunMode.INFERENCE,
    test_passed: bool = False,
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

    if test_passed:
        # If custom bringup_status and reason are provided, use them.
        reason = static_reason or None
        bringup_status = static_bringup_status or BringupStatus.PASSED

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

    # TODO (kmabee) - This is temporary workaround to populate parallelism tag for superset
    # database correctly in very short term for newly added TP models on n300-llmbox.
    # This will be replaced by something more robust in near future.
    arch = request.config.getoption("--arch")
    if arch is not None and "llmbox" in arch:
        parallelism = Parallelism.TENSOR_PARALLEL
    else:
        parallelism = Parallelism.SINGLE_DEVICE

    tags = {
        "test_name": str(request.node.originalname),
        "specific_test_case": str(request.node.name),
        "category": str(Category.MODEL_TEST),
        "model_name": str(model_info.name),
        "model_info": model_info.to_report_dict(),
        "run_mode": str(run_mode),
        "bringup_status": str(bringup_status),
        "parallelism": str(parallelism),
    }

    # If we have an explanatory reason, include it as a top-level property too for DB visibility
    # which is especially useful for passing tests (used to just from xkip/xfail reason)
    if reason:
        record_property("error_message", reason)

    # Write properties
    record_property("tags", tags)
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
