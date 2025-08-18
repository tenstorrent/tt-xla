# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import importlib.util
import torch
import inspect
from enum import Enum
from tests.utils import ModelTester
from tt_torch.tools.utils import OpByOpBackend


class ModelStatus(Enum):
    # Passing tests
    EXPECTED_PASSING = "expected_passing"
    # Known failures that should be xfailed
    KNOWN_FAILURE_XFAIL = "known_failure_xfail"
    # Not supported on this architecture or low priority
    NOT_SUPPORTED_SKIP = "not_supported_skip"
    # New model, awaiting triage
    UNSPECIFIED = "unspecified"


class ModelTestConfig:
    def __init__(self, data, arch):
        self.data = data or {}
        self.arch = arch

        # For marking tests as expected passing, known failures, etc
        self.status = self._resolve("status", default=ModelStatus.UNSPECIFIED)

        # Arguments to ModelTester
        self.required_pcc = self._resolve("required_pcc", default=None)
        self.assert_pcc = self._resolve("assert_pcc", default=None)
        # FIXME - Consider enabling atol checking.
        self.assert_atol = self._resolve("assert_atol", default=False)
        self.relative_atol = self._resolve("relative_atol", default=None)

        # Misc arguments used in test
        self.batch_size = self._resolve("batch_size", default=None)

        # Arguments to skip_full_eval_test() for skipping tests
        self.skip_reason = self._resolve("skip_reason", default="Unknown Reason")
        self.skip_bringup_status = self._resolve(
            "skip_bringup_status", default="FAILED_RUNTIME"
        )
        self.xfail_reason = self._resolve("xfail_reason", default="Unknown Reason")

    def _resolve(self, key, default=None):
        overrides = self.data.get("arch_overrides", {})
        if self.arch in overrides and key in overrides[self.arch]:
            return overrides[self.arch][key]
        return self.data.get(key, default)

    def to_tester_args(self):
        args = {}
        if self.assert_pcc is not None:
            args["assert_pcc"] = self.assert_pcc
        if self.assert_atol is not None:
            args["assert_atol"] = self.assert_atol
        if self.required_pcc is not None:
            args["required_pcc"] = self.required_pcc
        if self.relative_atol is not None:
            args["relative_atol"] = self.relative_atol
        return args


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


def import_model_loader_and_variant(loader_path, models_root):
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

    # Find ModelVariant class in the module
    ModelVariant = None
    for name, obj in mod.__dict__.items():
        if name == "ModelVariant":
            ModelVariant = obj
            break

    return mod.ModelLoader, ModelVariant


def get_model_variants(loader_path, loader_paths, models_root):
    try:
        # Import both the ModelLoader and ModelVariant class from the same module
        ModelLoader, ModelVariant = import_model_loader_and_variant(
            loader_path, models_root
        )
        variants = ModelLoader.query_available_variants()
        for variant in variants.keys():
            # Store the variant, ModelLoader class, and ModelVariant class together
            loader_paths[loader_path].append((variant, ModelLoader, ModelVariant))

    except Exception as e:
        print(f"Cannot import path: {loader_path}: {e}")


def generate_test_id(test_entry, models_root):
    """Generate test ID from test entry."""
    model_path = os.path.relpath(os.path.dirname(test_entry["path"]), models_root)
    variant_info = test_entry["variant_info"]

    if variant_info:
        variant, _, _ = variant_info  # Unpack the tuple to get just the variant
        return f"{model_path}-{variant}"
    else:
        return model_path


class DynamicTester(ModelTester):
    def _load_model(self):
        # Check if load_model method supports dtype_override parameter
        sig = inspect.signature(self.loader.load_model)
        if "dtype_override" in sig.parameters:
            return self.loader.load_model(dtype_override=torch.bfloat16)
        else:
            return self.loader.load_model()

    def _load_inputs(self):
        # Check if load_inputs method supports dtype_override parameter
        sig = inspect.signature(self.loader.load_inputs)
        if "dtype_override" in sig.parameters:
            return self.loader.load_inputs(dtype_override=torch.bfloat16)
        else:
            return self.loader.load_inputs()


def setup_models_path(project_root):
    """Setup models root path and add to sys.path for imports."""
    models_root = get_models_root(project_root)

    # Add the models root to sys.path so relative imports work
    if models_root not in sys.path:
        sys.path.insert(0, models_root)

    return models_root


def discover_loader_paths(models_root):
    """Discover all loader.py files in the models directory."""
    loader_paths = {}
    for root, dirs, files in os.walk(models_root):
        if os.path.basename(root) == "pytorch" and "loader.py" in files:
            loader_paths[os.path.join(root, "loader.py")] = []

    # Populate variants for each loader path
    for path in loader_paths.keys():
        get_model_variants(path, loader_paths, models_root)

    return loader_paths


def create_test_entries(loader_paths):
    """Create test entries combining loader paths and variants."""
    test_entries = []

    # Store variant info along with the ModelLoader and ModelVariant classes
    for loader_path, variant_tuples in loader_paths.items():
        if variant_tuples:  # Model has variants
            for variant_tuple in variant_tuples:
                # Each tuple contains (variant, ModelLoader, ModelVariant)
                test_entries.append(
                    {"path": loader_path, "variant_info": variant_tuple}
                )
        else:  # Model has no variants
            test_entries.append({"path": loader_path, "variant_info": None})

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
