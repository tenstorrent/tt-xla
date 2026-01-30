# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Shared wrapper utilities for tt-metal modules.

Provides MetaPath import hooks for redirecting package imports to
their _original counterparts (e.g., tracy.X -> tracy._original.X).
"""

import importlib
import importlib.util
import sys
from contextlib import contextmanager
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ExtensionFileLoader, ModuleSpec
from pathlib import Path
from types import ModuleType


def create_wrapper_redirector(
    package_name: str,
    original_path: Path,
    skip_submodules: tuple[str, ...] = (),
    extensions: dict[str, str] | None = None,
):
    """
    Create a MetaPathFinder that redirects pkg.X imports to pkg._original.X.

    Args:
        package_name: The package name (e.g., "tracy", "ttnn")
        original_path: Path to the _original directory
        skip_submodules: Submodules to skip (e.g., ("__main__",))
        extensions: Dict mapping module names to .so filenames (e.g., {"_ttnn": "_ttnn.so"})

    Returns:
        A MetaPathFinder instance
    """
    extensions = extensions or {}

    class _Redirector(MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            prefix = f"{package_name}."
            original_prefix = f"{package_name}._original"

            if not fullname.startswith(prefix):
                return None
            if fullname.startswith(original_prefix):
                return None

            parts = fullname.split(".")
            if len(parts) < 2:
                return None

            submodule = parts[1]
            rest = ".".join(parts[2:]) if len(parts) > 2 else ""

            if submodule in skip_submodules:
                return None

            # Handle C extensions (e.g., _ttnn.so)
            if submodule in extensions and not rest:
                return self._find_extension_spec(fullname, submodule)

            original_name = f"{original_prefix}.{submodule}" + (
                f".{rest}" if rest else ""
            )
            try:
                original_spec = importlib.util.find_spec(original_name)
                if original_spec:
                    return ModuleSpec(
                        fullname,
                        WrapperLoader(original_name),
                        origin=original_spec.origin,
                        is_package=original_spec.submodule_search_locations is not None,
                    )
            except (ImportError, ModuleNotFoundError, ValueError):
                pass
            return None

        def _find_extension_spec(self, fullname, submodule):
            """Find spec for C extension (.so file) from shared lib directory."""
            ext_filename = extensions[submodule]

            # Load from pjrt_plugin_tt/lib/ instead of _original/ directory
            import pjrt_plugin_tt

            shared_lib_path = (
                Path(pjrt_plugin_tt.__file__).parent / "lib" / ext_filename
            )

            if shared_lib_path.exists():
                return ModuleSpec(
                    fullname,
                    ExtensionFileLoader(fullname, str(shared_lib_path)),
                    origin=str(shared_lib_path),
                )

            # Fallback to original location for editable installs
            ext_path = original_path / ext_filename
            if ext_path.exists():
                return ModuleSpec(
                    fullname,
                    ExtensionFileLoader(fullname, str(ext_path)),
                    origin=str(ext_path),
                )
            return None

    return _Redirector()


class WrapperLoader(Loader):
    """Loader that imports pkg._original.X but registers as pkg.X"""

    def __init__(self, original_name: str):
        self.original_name = original_name

    def create_module(self, spec):
        return None  # Use default module creation

    def exec_module(self, module):
        original = importlib.import_module(self.original_name)
        module.__dict__.update(original.__dict__)


class ProxyModule(ModuleType):
    """
    Proxy module that forwards attribute lookups to {package_name}._original.

    Used to handle circular imports during wrapped module initialization.
    """

    def __init__(self, package_name: str):
        super().__init__(package_name)
        self._package_name = package_name

    def __getattr__(self, name):
        original = sys.modules.get(f"{self._package_name}._original")
        if original is None:
            raise AttributeError(f"module '{self.__name__}' has no attribute '{name}'")
        return getattr(original, name)


@contextmanager
def proxy_import(package_name: str):
    """
    Context manager that temporarily replaces a module with a proxy during import.

    Handles circular imports by forwarding attribute lookups to {package_name}._original.

    Usage:
        with proxy_import("ttnn"):
            from ttnn._original import *
    """
    wrapper_module = sys.modules[package_name]
    proxy = ProxyModule(package_name)
    proxy.__path__ = wrapper_module.__path__
    proxy.__file__ = wrapper_module.__file__

    sys.modules[package_name] = proxy
    try:
        yield proxy
    finally:
        sys.modules[package_name] = wrapper_module
