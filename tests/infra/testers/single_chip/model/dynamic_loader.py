# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Dynamic loader helper class for model testers."""

import importlib.util
import os
import sys
import inspect
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


@dataclass
class ModelTestEntry:
    """Entry for a model test with path and variant information."""

    path: str
    variant_info: tuple


class DynamicLoader:
    """Helper class that encapsulates dynamic loader functionality.

    This class provides a unified interface for loading models and inputs
    from loader objects, with support for mesh configuration for parallelism.
    """

    def __init__(self, loader):
        """Initialize DynamicLoader with a loader object.

        Args:
            loader: Loader object that implements load_model and load_inputs methods
        """
        self.loader = loader

    def load_model(self) -> Any:
        """Load model from the loader.

        Returns:
            Model instance loaded from the loader
        """
        return self.loader.load_model()

    def load_inputs(self) -> Any:
        """Load input activations from the loader.

        Returns:
            Input tensors/arrays that can be fed to the model
        """
        sig = inspect.signature(self.loader.load_inputs)
        if "dtype_override" in sig.parameters:
            return self.loader.load_inputs(dtype_override=torch.bfloat16)
        else:
            return self.loader.load_inputs()

    def get_shard_spec_function(self):
        """Get shard spec function from loader if available.

        Returns:
            Shard spec function if loader implements load_shard_spec, None otherwise
        """
        if hasattr(self.loader, "load_shard_spec"):
            return self.loader.load_shard_spec
        return None

    def get_mesh_config(
        self, num_devices: int
    ) -> Tuple[Optional[Tuple], Optional[Tuple]]:
        """Get mesh configuration from loader if available.

        Args:
            num_devices: Number of devices to configure mesh for

        Returns:
            Tuple of (mesh_shape, mesh_names) if loader implements get_mesh_config,
            (None, None) otherwise
        """
        if hasattr(self.loader, "get_mesh_config"):
            return self.loader.get_mesh_config(num_devices)
        return None, None

    def has_mesh_support(self) -> bool:
        """Check if the loader supports mesh configuration.

        Returns:
            True if loader has get_mesh_config method, False otherwise
        """
        return hasattr(self.loader, "get_mesh_config")

    def has_shard_spec_support(self) -> bool:
        """Check if the loader supports shard specifications.

        Returns:
            True if loader has load_shard_spec method, False otherwise
        """
        return hasattr(self.loader, "load_shard_spec")

    # ========== Test Discovery Methods (Class Methods) ==========

    @classmethod
    def setup_test_discovery(
        cls, project_root: str
    ) -> Tuple[str, List[ModelTestEntry]]:
        """Complete test discovery setup - combines all the setup steps.

        Args:
            project_root: Root directory of the project

        Returns:
            Tuple of (models_root, test_entries)
        """
        models_root = cls.setup_models_path(project_root)
        loader_paths = cls.discover_loader_paths(models_root)
        test_entries = cls.create_test_entries(loader_paths)
        return models_root, test_entries

    @classmethod
    def get_models_root(cls, project_root: str) -> str:
        """Return the filesystem path to the given module, supporting both installed and source-tree use cases.

        Args:
            project_root: Root directory of the project

        Returns:
            Path to the models root directory
        """
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
        logger.warning(f"No installed {module_name}; falling back to {fallback}")
        return fallback

    @classmethod
    def setup_models_path(cls, project_root: str) -> str:
        """Setup models root path and add to sys.path for imports.

        Args:
            project_root: Root directory of the project

        Returns:
            Path to the models root directory
        """
        models_root = cls.get_models_root(project_root)

        # Add the models root to sys.path so relative imports work
        if models_root not in sys.path:
            sys.path.insert(0, models_root)

        return models_root

    @classmethod
    def import_model_loader(cls, loader_path: str, models_root: str):
        """Dynamically import and return ModelLoader class from a loader.py path, ensuring relative imports work.

        Args:
            loader_path: Path to the loader.py file
            models_root: Root directory of the models

        Returns:
            ModelLoader class from the imported module
        """
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

    @classmethod
    def get_model_variants(cls, loader_path: str, loader_paths: Dict, models_root: str):
        """Fill loader_paths[loader_path] with (variant_name, ModelLoader) tuples by querying the loader; on failure, log and continue.

        Args:
            loader_path: Path to the loader.py file
            loader_paths: Dictionary to fill with variant information
            models_root: Root directory of the models
        """
        try:
            # Import the ModelLoader class from the module
            ModelLoader = cls.import_model_loader(loader_path, models_root)
            variants = ModelLoader.query_available_variants()

            # Store variant_name, ModelLoader together for usage, or empty one if no variants found.
            if variants:
                for variant_name in variants.keys():
                    loader_paths[loader_path].append((variant_name, ModelLoader))
            else:
                loader_paths[loader_path].append((None, ModelLoader))

        except Exception as e:
            logger.warning(f"Cannot import path: {loader_path}: {e}")

    @classmethod
    def discover_loader_paths(cls, models_root: str) -> Dict:
        """Discover all loader.py files in the models directory.

        This method should be implemented by framework-specific subclasses.

        Args:
            models_root: Root directory of the models

        Returns:
            Dictionary mapping loader paths to their variants

        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError(
            "Subclasses must implement discover_loader_paths() for their specific framework"
        )

    @classmethod
    def create_test_entries(cls, loader_paths: Dict) -> List[ModelTestEntry]:
        """Create test entries combining loader paths and variants.

        Args:
            loader_paths: Dictionary mapping loader paths to their variants

        Returns:
            List of ModelTestEntry objects
        """
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

    @classmethod
    def generate_test_id(cls, test_entry: ModelTestEntry, models_root: str) -> str:
        """Generate test ID from test entry with optional variant.

        Args:
            test_entry: ModelTestEntry object
            models_root: Root directory of the models

        Returns:
            Test ID string
        """
        model_path = os.path.relpath(os.path.dirname(test_entry.path), models_root)
        variant_info = test_entry.variant_info
        variant_name = variant_info[0] if variant_info else None
        return f"{model_path}-{variant_name}" if variant_name else model_path

    @classmethod
    def create_test_id_generator(cls, models_root: str):
        """Create a function for generating test IDs.

        Args:
            models_root: Root directory of the models

        Returns:
            Function that generates test IDs from test entries
        """

        def _generate_test_id(test_entry):
            """Generate test ID from test entry using the utility function."""
            return cls.generate_test_id(test_entry, models_root)

        return _generate_test_id


class TorchDynamicLoader(DynamicLoader):
    """PyTorch-specific implementation of DynamicLoader.

    This class provides PyTorch-specific test discovery functionality,
    particularly for discovering PyTorch model loader files.
    """

    @classmethod
    def discover_loader_paths(cls, models_root: str) -> Dict:
        """Discover all PyTorch loader.py files in the models directory, with exclusions.

        Args:
            models_root: Root directory of the models

        Returns:
            Dictionary mapping loader paths to their variants
        """
        loader_paths = {}

        # TODO(kmabee) - Temporary workaround to exclude models with fatal issues.
        # Dictionary mapping model directory names to their exclusion reasons
        excluded_model_dirs = {
            "suryaocr": "Issue #1166"  # Surya OCR imports and initializes torch_xla runtime which causes issues
        }

        for root, dirs, files in os.walk(models_root):
            model_dir_name = os.path.basename(os.path.dirname(root))
            if model_dir_name in excluded_model_dirs:
                warning_msg = excluded_model_dirs[model_dir_name]
                logger.warning(
                    f"Workaround to exclude model: {model_dir_name} from discovery. {warning_msg}"
                )
                continue

            # Look specifically for PyTorch loader files
            if os.path.basename(root) == "pytorch" and "loader.py" in files:
                loader_paths[os.path.join(root, "loader.py")] = []

        # Populate variants for each loader path
        for path in loader_paths.keys():
            cls.get_model_variants(path, loader_paths, models_root)

        return loader_paths
    
    @classmethod
    def batch_tensor(cls, tensor, num_devices):
        if isinstance(tensor, torch.Tensor):
            if tensor.dim() == 0:
                return tensor.repeat(num_devices)
            else:
                if tensor.dim() == 1:
                    tensor = tensor.unsqueeze(0)
                return tensor.repeat_interleave(num_devices, dim=0)
        return tensor
    
    @classmethod
    def load_shard_spec_data_parallel(cls, args, kwargs):
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


class JaxDynamicLoader(DynamicLoader):
    """JAX-specific implementation of DynamicLoader.

    This class provides JAX-specific test discovery functionality,
    particularly for discovering JAX model loader files.
    """

    @classmethod
    def discover_loader_paths(cls, models_root: str) -> Dict:
        """Discover all JAX loader.py files in the models directory, with exclusions.

        Args:
            models_root: Root directory of the models

        Returns:
            Dictionary mapping loader paths to their variants
        """
        loader_paths = {}

        # TODO - Add JAX-specific exclusions if needed
        # Dictionary mapping model directory names to their exclusion reasons
        excluded_model_dirs = {}

        for root, dirs, files in os.walk(models_root):
            model_dir_name = os.path.basename(os.path.dirname(root))
            if model_dir_name in excluded_model_dirs:
                warning_msg = excluded_model_dirs[model_dir_name]
                logger.warning(
                    f"Workaround to exclude model: {model_dir_name} from discovery. {warning_msg}"
                )
                continue

            # Look specifically for JAX loader files
            if os.path.basename(root) == "jax" and "loader.py" in files:
                loader_paths[os.path.join(root, "loader.py")] = []

        # Populate variants for each loader path
        for path in loader_paths.keys():
            cls.get_model_variants(path, loader_paths, models_root)

        return loader_paths
