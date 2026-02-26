# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import hashlib
import inspect
import json
import os
import shutil
import subprocess
from dataclasses import dataclass, fields
from pathlib import Path
from sys import stderr, stdout

from setuptools import Extension, find_packages, setup
from setuptools.command.build_py import build_py
from wheel.bdist_wheel import bdist_wheel

THIS_DIR = Path(os.path.realpath(os.path.dirname(__file__)))
REPO_DIR = Path(os.path.join(THIS_DIR, "..")).resolve()


@dataclass
class SetupConfig:
    """
    Helper dataclass storing wheel config for TT-XLA package.

    The wheel structure is as follows:
    ```
    pjrt_plugin_tt/                     # PJRT plugin package
        |-- __init__.py
        |-- pjrt_plugin_tt.so               # PJRT plugin binary
        |-- tt-metal/                       # tt-metal runtime dependencies (kernels, riscv compiler/linker, etc.)
        `-- lib/                            # shared library dependencies (tt-mlir, tt-metal)
    jax_plugin_tt/                      # Thin JAX wrapper
        `-- __init__.py                     # imports and sets up pjrt_plugin_tt for XLA
    torch_plugin_tt                     # Thin PyTorch/XLA wrapper
        `-- __init__.py                     # imports and sets up pjrt_plugin_tt for PyTorch/XLA
    tracy/                              # Wrapped Tracy profiler - this is a thin wrapper arount `tracy` module from tt-metal
        `-- _original/                       # Wrapped tracy code from tt-metal
    ttnn/                               # Wrapped TTNN module - this is a thin wrapper around `ttnn` module from tt-metal
        `-- _original/                       # Wrapped ttnn code from tt-metal
    ```
    """

    # --- Dataclass fields ---
    build_type: str = "release"

    # --- Necessary wheel properties ---

    @property
    def version(self) -> str:
        """Wheel version."""
        short_hash = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode("ascii")
            .strip()
        )
        date = (
            subprocess.check_output(
                ["git", "show", "-s", "--format=%cd", "--date=format:%y%m%d", "HEAD"]
            )
            .decode("ascii")
            .strip()
        )

        # NOTE this is how tt-forge-fe does it.
        return "0.1." + date + "+dev." + short_hash

    @property
    def requirements(self) -> list:
        """
        List of requirements needed for plugins to actually work.
        """
        reqs = []
        requirements_path = THIS_DIR / "requirements.txt"

        with requirements_path.open() as f:
            reqs = [
                line
                for line in f.read().splitlines()
                if not line.strip().startswith("--extra-index-url")
            ]

        return reqs

    @property
    def long_description(self) -> str:
        """Package description."""
        readme = REPO_DIR / "README.md"

        with readme.open() as f:
            return f.read()

    @property
    def description_with_versions(self) -> str:
        """Generate description with version information."""
        import re
        import urllib.request
        from datetime import datetime

        # Extract tt-mlir SHA from third_party/CMakeLists.txt
        cmake_file = REPO_DIR / "third_party" / "CMakeLists.txt"
        with cmake_file.open() as f:
            cmake_content = f.read()

        mlir_match = re.search(r'set\(TT_MLIR_VERSION "([^"]+)"\)', cmake_content)
        if not mlir_match:
            raise RuntimeError(
                "Failed to extract TT_MLIR_VERSION from third_party/CMakeLists.txt"
            )
        mlir_sha = mlir_match.group(1)

        # Fetch tt-metal SHA from tt-mlir repo
        tt_mlir_url = f"https://raw.githubusercontent.com/tenstorrent/tt-mlir/{mlir_sha}/third_party/CMakeLists.txt"
        try:
            with urllib.request.urlopen(tt_mlir_url) as response:
                tt_mlir_content = response.read().decode("utf-8")
        except Exception as e:
            raise RuntimeError(
                f"Failed to fetch tt-mlir CMakeLists.txt from {tt_mlir_url}: {e}"
            )

        metal_match = re.search(r'set\(TT_METAL_VERSION "([^"]+)"\)', tt_mlir_content)
        if not metal_match:
            raise RuntimeError(
                "Failed to extract TT_METAL_VERSION from tt-mlir CMakeLists.txt"
            )
        metal_sha = metal_match.group(1)

        # Get frontend SHA from current repo
        try:
            commit = (
                subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_DIR)
                .decode("ascii")
                .strip()
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to get frontend SHA: {e}")

        # Get build date
        build_date = datetime.now().strftime("%Y-%m-%d")

        # Format the description
        return f"commit={commit}, tt-mlir-commit={mlir_sha}, tt-metal-commit={metal_sha}, built-date={build_date}, build-type={self.build_type}"

    # --- Properties of wheel bundle ---

    @property
    def shared_device_package_target_dir_relpath(self) -> Path:
        """Path to shared pjrt_plugin_tt package relative to this script."""
        return Path("pjrt_plugin_tt")

    def __repr__(self) -> str:
        """Representes self as json string."""
        # Fields too long to display.
        ignore_fields = ["long_description"]

        # Collect results as `attribute_name: attribute_value` mapping.
        result = {}

        # Include fields.
        for f in fields(self):
            if f.name in ignore_fields:
                continue

            value = getattr(self, f.name)
            if isinstance(value, Path):
                value = str(value)

            result[f.name] = value

        # Include properties.
        for name, prop in inspect.getmembers(
            type(self), lambda o: isinstance(o, property)
        ):
            try:
                if name in ignore_fields:
                    continue

                value = getattr(self, name)
                if isinstance(value, Path):
                    value = str(value)

                result[name] = value
            except Exception:
                result[name] = "<error>"

        return json.dumps(result, indent=4)

    enable_explorer: bool = False


# Instantiate config.
config = SetupConfig()


class BdistWheel(bdist_wheel):
    """
    Custom wheel builder for a platform-specific Python package.

    - Marks the wheel as non-pure (`root_is_pure = False`) to ensure proper installation
      of native binaries.
    - Overrides the tag to be Python 3.12-specific (`cp312-cp312`) while preserving
      platform specificity.
    """

    user_options = bdist_wheel.user_options + [
        ("build-type=", None, "Build type: release, codecov, debug, or explorer"),
    ]

    def initialize_options(self):
        super().initialize_options()
        # Default build type is release
        self.build_type = "release"

    def finalize_options(self):
        build_types = ["release", "codecov", "debug", "explorer"]
        if self.build_type not in build_types:
            raise ValueError(
                f"Invalid build type: {self.build_type}. Valid options are: {', '.join(build_types)}"
            )

        config.build_type = self.build_type
        config.enable_explorer = self.build_type == "explorer"

        self.root_is_pure = False
        bdist_wheel.finalize_options(self)

    def run(self):
        # Update the description with version info after options are finalized (e.g. self.build_type)
        from setuptools.dist import Distribution

        dist = self.distribution
        dist.metadata.description = config.description_with_versions

        # Call the parent run method
        bdist_wheel.run(self)

    def get_tag(self):
        python, abi, plat = bdist_wheel.get_tag(self)
        # Force specific Python 3.12 ABI format for the wheel
        python, abi = "cp312", "cp312"
        # Ensure platform-specific tag for x86_64 architecture
        # This prevents 'any' platform and enables auditwheel to properly repair
        import platform

        if plat == "any" or not plat:
            machine = platform.machine().lower()
            if machine in ("x86_64", "amd64"):
                plat = "linux_x86_64"
        return python, abi, plat


class CMakeBuildPy(build_py):
    """
    Custom build_py command that builds the native CMake-based PJRT plugin and prepares
    the package for wheel creation.

    It first ensures project is built, then it copies pre-written __init__.py file
    containing plugin initialization code inside the plugin dir, afterwards copies
    created JAX plugin (product of the build) `pjrt_plugin_tt.so` inside the plugin dir,
    and finally copies entire tt-mlir installation dir inside the plugin dir as well,
    for them all to be packaged together.

    NOTE MANIFEST.in defines command through which additional non-python files (like
    .yaml, .so, .a, etc.) are going to be included in the final package. This cannot be
    done solely using `package_data` parameter of `setup` which expects python modules.
    """

    def in_ci(self) -> bool:
        return os.environ.get("IN_CIBW_ENV") == "ON"

    def run(self):
        if hasattr(self, "editable_mode") and self.editable_mode:
            # No need to built the project in editable mode.
            return

        print(f"Building wheel with following settings:\n{config}")

        # Install project to the shared device package directory.
        print("Building project...")
        self.build_cmake_project()

        # Refresh package list to include any packages that are discoverable after the build.
        # E.g. `tracy` will show up only after the first cmake build is completed, since
        # it is pulled in as an external project.
        discovered_packages = find_packages()
        self.distribution.packages = discovered_packages
        self.packages = discovered_packages

        # Continue with the rest of the Python build.
        super().run()

    def build_cmake_project(self):
        install_dir = (
            THIS_DIR / self.build_lib / config.shared_device_package_target_dir_relpath
        )

        code_coverage = "OFF"
        enable_explorer = "OFF"

        if config.build_type == "codecov":
            code_coverage = "ON"
        if config.enable_explorer:
            enable_explorer = "ON"

        cmake_args = [
            "-G",
            "Ninja",
            "-B",
            "build",
            "-DTTXLA_ENABLE_EWHEEL_INSTALL=OFF",
            "-DTTXLA_ENABLE_TOOLS=OFF" + enable_explorer,
            "-DCODE_COVERAGE=" + code_coverage,
            "-DTTXLA_ENABLE_EXPLORER=" + enable_explorer,
            "-DCMAKE_INSTALL_PREFIX=" + str(install_dir),
        ]
        build_command = ["--build", "build"]
        install_command = ["--install", "build"]

        cmake_cmd = ["cmake"]
        # Run source env/activate if in ci, otherwise onus is on dev
        if self.in_ci():
            cmake_cmd = [
                "source",
                "venv/activate",
                "&&",
                "cmake",
            ]

        print(f"CMake arguments: {[*cmake_cmd, *cmake_args]}")

        # Set environment variables to create a more portable build.
        os.environ["TRACY_NO_ISA_EXTENSIONS"] = "1"
        os.environ["TRACY_NO_INVARIANT_CHECK"] = "1"

        # Execute cmake from top level project dir, where root CMakeLists.txt resides.
        print("Setting up CMake project...")
        stdout.flush()
        stderr.flush()
        subprocess.run(
            " ".join([*cmake_cmd, *cmake_args]),
            check=True,
            shell=True,
            capture_output=False,
            cwd=REPO_DIR,
        )
        subprocess.run(
            " ".join([*cmake_cmd, *build_command]),
            check=True,
            shell=True,
            capture_output=False,
            cwd=REPO_DIR,
        )
        subprocess.run(
            " ".join([*cmake_cmd, *install_command]),
            check=True,
            shell=True,
            capture_output=False,
            cwd=REPO_DIR,
        )

        self._prune_install_tree(install_dir)

    def _prune_install_tree(self, install_dir: Path) -> None:
        if not install_dir.exists():
            return
        # Broken symlinks introduced in tt-umd -> tt-metal -> tt-mlir uplift
        # issue: https://github.com/tenstorrent/tt-umd/issues/1864
        _remove_broken_symlinks(install_dir)
        _remove_static_archives(install_dir)

        # remove cmake and pkgconfig files
        # _remove_bloat_dir(install_dir / "lib" / "cmake")
        # _remove_bloat_dir(install_dir / "lib" / "pkgconfig")
        # _remove_bloat_dir(install_dir / "lib64" / "cmake")
        # _remove_bloat_dir(install_dir / "lib64" / "pkgconfig")
        # _remove_bloat_dir(install_dir / "include")
        # _remove_bloat_dir(install_dir / "tt-metal" / "tests")
        if config.build_type == "release":
            _strip_shared_objects(install_dir)

        _deduplicate_shared_objects(install_dir)

    def copy_plugin_scripts(self):
        scripts_to_copy = ["__init__.py", "monkeypatch.py"]
        for script_file in scripts_to_copy:
            script_src = THIS_DIR / script_file
            script_dst = config.jax_plugin_target_dir / script_file
            if not script_dst.exists():
                print(f"Copying {script_file}...")
                shutil.copy2(script_src, config.jax_plugin_target_dir)
            else:
                print(f"{script_file} already copied.")


def _remove_bloat_dir(dir_path: Path) -> None:
    if dir_path.exists() and dir_path.is_dir():
        print(f"Removing bloat directory: {dir_path}")
        shutil.rmtree(dir_path)


def _remove_static_archives(root: Path) -> None:
    for archive in root.rglob("*.a"):
        if archive.is_symlink() or not archive.is_file():
            continue
        rel = archive.relative_to(root)
        if rel.parts and rel.parts[0] in ("lib", "lib64"):
            print(f"Removing static archive: {rel}")
            archive.unlink()


def _remove_broken_symlinks(root: Path) -> None:
    """Remove broken symlinks that would cause wheel packaging to fail."""
    for path in root.rglob("*"):
        if path.is_symlink() and not path.exists():
            rel = path.relative_to(root)
            print(f"Removing broken symlink: {rel}")
            path.unlink()


def _strip_shared_objects(root: Path) -> None:
    strip_path = shutil.which("strip")
    if strip_path is None:
        print("strip tool not found; skipping debug symbol stripping")
        return

    for so_file in root.rglob("*.so"):
        if so_file.is_symlink() or not so_file.is_file():
            continue
        try:
            subprocess.run([strip_path, "--strip-unneeded", str(so_file)], check=True)
            rel = so_file.relative_to(root)
            print(f"Stripped debug symbols: {rel}")
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Failed to strip {so_file}: {exc}")


def _deduplicate_shared_objects(root: Path) -> None:
    seen: dict[str, Path] = {}
    for so_file in sorted(
        root.rglob("*.so*"), key=lambda p: p.relative_to(root).as_posix()
    ):
        if so_file.is_symlink() or not so_file.is_file():
            continue

        checksum = _sha256_file(so_file)
        if checksum in seen:
            target = seen[checksum]
            link_target = os.path.relpath(target, so_file.parent)
            print(
                f"Deduplicating shared object: {so_file.relative_to(root)} -> {link_target}"
            )
            so_file.unlink()
            os.symlink(link_target, so_file)
        else:
            seen[checksum] = so_file


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


setup(
    author="tt-xla team",
    author_email="tt-xla@tenstorrent.com",
    description=config.description_with_versions,
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Development Status :: 3 - Alpha",
    ],
    cmdclass={
        "bdist_wheel": BdistWheel,
        "build_py": CMakeBuildPy,
    },
    entry_points={
        # We must advertise which Python modules should be treated as loadable
        # plugins. This augments the path based scanning that Jax does, which
        # is not always robust to all packaging circumstances.
        "jax_plugins": ["pjrt_plugin_tt = jax_plugin_tt"],
        # Entry point used by torch xla to register the plugin automatically.
        "torch_xla.plugins": ["tt = torch_plugin_tt:TTPlugin"],
        # Console scripts
        "console_scripts": [
            "tt-forge-install = ttxla_tools.install_sfpi:main",
            "tracy = tracy.__main__:main",
        ],
    },
    include_package_data=True,
    install_requires=config.requirements,
    license="Apache-2.0",
    long_description_content_type="text/markdown",
    long_description=config.long_description,
    name="pjrt-plugin-tt",
    packages=find_packages(),
    python_requires=">=3.12, <3.13",
    url="https://github.com/tenstorrent/tt-xla",
    version=config.version,
    # Needs to reference embedded shared libraries (i.e. .so file), so not zip safe.
    zip_safe=False,
)
