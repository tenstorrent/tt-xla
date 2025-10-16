# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import inspect
import json
import os
import shutil
import subprocess
from dataclasses import dataclass, fields
from pathlib import Path

from setuptools import find_packages, setup
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
    ```
    """

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
            reqs = f.read().splitlines()

        return reqs

    @property
    def long_description(self) -> str:
        """Package description."""
        readme = REPO_DIR / "README.md"

        with readme.open() as f:
            return f.read()

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

    code_coverage: bool = False


# Instantiate config.
config = SetupConfig()


def find_metal_packages() -> list[str]:
    # Find all python packages in ttnn - skip test packages.
    file_dir = os.path.dirname(os.path.abspath(__file__))

    tt_metal_root_dir = (
        Path(file_dir)
        / ".."
        / "third_party"
        / "tt-mlir"
        / "src"
        / "tt-mlir"
        / "third_party"
        / "tt-metal"
        / "src"
        / "tt-metal"
    )
    ttnn_dir = tt_metal_root_dir / "ttnn"

    tools_dir = tt_metal_root_dir / "tools"

    ttnn_packages = find_packages(
        where=ttnn_dir,
        exclude=["ttnn.examples", "ttnn.examples.*", "test"],
        include=["ttnn", "ttnn.*"],
    )
    print(f"Found ttnn packages: {ttnn_packages}")

    packages = {}
    packages["ttnn"] = str((ttnn_dir / "ttnn").relative_to(file_dir))

    tools_packages = find_packages(
        where=tools_dir,
        include=["tracy"],
    )
    print(f"Found metal tools packages: {tools_packages}")
    packages["tracy"] = str((tools_dir / "tracy").relative_to(file_dir))

    return (ttnn_packages + tools_packages, packages)


class BdistWheel(bdist_wheel):
    """
    Custom wheel builder for a platform-specific Python package.

    - Marks the wheel as non-pure (`root_is_pure = False`) to ensure proper installation
      of native binaries.
    - Overrides the tag to be Python 3.11-specific (`cp311-cp311`) while preserving
      platform specificity.
    """

    user_options = bdist_wheel.user_options + [
        ("code-coverage", None, "Enable code coverage for the build")
    ]

    def initialize_options(self):
        super().initialize_options()
        self.code_coverage = False  # Default value for code coverage

    def finalize_options(self):
        if self.code_coverage is None:
            self.code_coverage = False

        config.code_coverage = self.code_coverage

        bdist_wheel.finalize_options(self)
        self.root_is_pure = False

    def get_tag(self):
        python, abi, plat = bdist_wheel.get_tag(self)
        # Force specific Python 3.11 ABI format for the wheel
        python, abi = "cp311", "cp311"
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

    def finalize_options(self):
        super().finalize_options()
        (metal_packages, dirs) = find_metal_packages()
        self.distribution.packages.extend(list(metal_packages))
        self.packages = (self.packages or []).extend(list(metal_packages))
        self.package_dir.update(dirs)
        print("packages:", self.distribution.packages)
        print("package dirs:", self.package_dir)

    def run(self):
        if hasattr(self, "editable_mode") and self.editable_mode:
            # No need to built the project in editable mode.
            # Find all interesting tt-metal packages and install them as well.
            super().run()
            return

        print(f"Building wheel with following settings:\n{config}")

        # Install project to the shared device package directory.
        print("Building project...")
        self.build_cmake_project()

        # Continue with the rest of the Python build.
        super().run()

    def build_cmake_project(self):
        install_dir = (
            THIS_DIR / self.build_lib / config.shared_device_package_target_dir_relpath
        )

        code_coverage = "OFF"

        if config.code_coverage:
            code_coverage = "ON"

        cmake_args = [
            "-G",
            "Ninja",
            "-B",
            "build",
            "-DCODE_COVERAGE=" + code_coverage,
            "-DCMAKE_INSTALL_PREFIX=" + str(install_dir),
        ]
        build_command = ["--build", "build"]
        install_command = ["--install", "build"]

        print(f"CMake arguments: {cmake_args}")

        # Execute cmake from top level project dir, where root CMakeLists.txt resides.
        subprocess.check_call(["cmake", *cmake_args], cwd=REPO_DIR)
        subprocess.check_call(["cmake", *build_command], cwd=REPO_DIR)
        subprocess.check_call(["cmake", *install_command], cwd=REPO_DIR)

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


setup(
    author="tt-xla team",
    author_email="tt-xla@tenstorrent.com",
    description="Tenstorrent PJRT plugin",
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
    },
    include_package_data=True,
    install_requires=config.requirements,
    license="Apache-2.0",
    long_description_content_type="text/markdown",
    long_description=config.long_description,
    name="pjrt-plugin-tt",
    packages=find_packages(),
    package_dir={
        "tracy": "../third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/tools/tracy",
        "ttnn": "../third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/ttnn",
    },
    python_requires=">=3.11, <3.12",
    url="https://github.com/tenstorrent/tt-xla",
    version=config.version,
    # Needs to reference embedded shared libraries (i.e. .so file), so not zip safe.
    zip_safe=False,
)
