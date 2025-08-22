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

from setuptools import setup
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
    pjrt_plugin_tt/                     # Shared device package (single copy)
    |-- __init__.py                     # get_library_path(), setup_environment()
    |-- pjrt_plugin_tt.so               # PJRT plugin binary
    `-- tt-mlir/                        # TT-MLIR installation
        `-- install/
            |-- lib/                    # libTTMLIRCompiler.so, etc.
            |-- include/                # Headers
            `-- tt-metal/               # TT_METAL_HOME points here

    jax_plugin_tt/                      # Thin JAX wrapper (Python only)
    `-- __init__.py                     # imports pjrt_plugin_tt

    torch_plugin_tt                     # Thin PyTorch/XLA wrapper (Python only)
    `-- __init__.py                     # imports pjrt_plugin_tt
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
        List of requirements needed for plugin to actually work.

        requirements.txt is parsed and only JAX requirements are pulled from it.
        """
        reqs = []
        requirements_path = REPO_DIR / "requirements.txt"

        with requirements_path.open() as f:
            # Filter for just pinned versions.
            pin_pairs = [line.strip().split("==") for line in f if "==" in line]
            pin_versions = dict(pin_pairs)

            # Convert pinned versions to >= for install_requires.
            for pin_name in ("jax", "jaxlib"):
                assert (
                    pin_name in pin_versions.keys()
                ), f"Requirement {pin_name} not found in {requirements_path}"

                pin_version = pin_versions[pin_name]
                reqs.append(f"{pin_name}>={pin_version}")

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


class BdistWheel(bdist_wheel):
    """
    Custom wheel builder for a platform-specific, non-pure Python package.

    - Marks the wheel as non-pure (`root_is_pure = False`) to ensure proper installation
      of native binaries.
    - Overrides the tag to be Python-version agnostic (`py3-none`) while preserving
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
        # We don't contain any python extensions so are version agnostic
        # but still want to be platform specific.
        python, abi = "py3", "none"
        return python, abi, plat


class CMakeBuildPy(build_py):
    """
    Custom build_py command that builds the native CMake-based PJRT plugin and prepares
    the package for wheel creation.

    It first ensures project is built, then it copies pre-written __init__.py file
    containing plugin initialization code inside the plugin di, afterwards copies
    created JAX plugin (product of the build) `pjrt_plugin_tt.so` inside the plugin dir,
    and finally copies entire tt-mlir installation dir inside the plugin dir as well,
    for them all to be packaged together.

    NOTE MANIFEST.in defines command through which additional non-python files (like
    .yaml, .so, .a, etc.) are going to be included in the final package. This cannot be
    done solely using `package_data` parameter of `setup` which expects python modules.
    """

    def run(self):
        if hasattr(self, "editable_mode") and self.editable_mode:
            # No need to built the project in editable mode.
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
    packages=[
        "pjrt_plugin_tt",
        "jax_plugin_tt",
        "torch_plugin_tt",
        "ttxla_tools",
    ],
    package_dir={
        "pjrt_plugin_tt": "pjrt_plugin_tt",
        "jax_plugin_tt": "jax_plugin_tt",
        "torch_plugin_tt": "torch_plugin_tt",
        "ttxla_tools": os.path.join("..", "ttxla_tools"),
    },
    python_requires=">=3.10, <3.11",
    url="https://github.com/tenstorrent/tt-xla",
    version=config.version,
    # Needs to reference embedded shared libraries (i.e. .so file), so not zip safe.
    zip_safe=False,
)
