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
    Helper dataclass storing wheel config.

    Jax plugin must be bundled in following form:
    ```
    jax_plugins
    `-- <custom-plugin-name>
        |-- __init__.py         # Contains plugin registration function
        `-- <custom-plugin-name>.so   # Plugin itself
    ```
    to be automatically detected by `jax` lib. Upon installation, it will be unpacked
    from wheel into user's python `env/lib` dir.

    In our case we also want to bundle other necessary things in wheel. So our final
    package tree looks like this:
    ```
    jax_plugins
    `-- pjrt_plugin_tt
        |-- __init__.py
        |-- pjrt_plugin_tt.so   # Plugin itself.
        `-- tt-mlir             # Entire tt-mlir installation folder
            `-- install
                |-- include
                |   `-- ...
                |-- lib
                |   |-- libTTMLIRCompiler.so
                |   |-- libTTMLIRRuntime.so
                |   `-- ...
                `-- tt-metal    # We need to set TT_METAL_HOME to this dir when loading plugin
                    |-- runtime
                    |   `-- ...
                    |-- tt_metal
                    |   `-- ...
                    `-- ttnn
                        `-- ...
    ```
    """

    # --- Necessary wheel properties ---

    package_name: str = "pjrt_plugin_tt"
    description_content_type: str = "text/markdown"
    # Needs to reference embedded shared libraries (i.e. .so file), so not zip safe.
    zip_safe: bool = False

    @property
    def project_name(self) -> str:
        """Name of the lib as pip will display it."""
        return self.package_name.replace("_", "-")

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
                pin_version = pin_versions[pin_name]
                reqs.append(f"{pin_name}>={pin_version}")

        return reqs

    @property
    def long_description(self) -> str:
        """Package description."""
        readme = REPO_DIR / "README.md"

        with readme.open() as f:
            return f.read()

    # --- Properties of project ---

    @property
    def pjrt_plugin(self) -> str:
        """Full file name of custom TT PJRT plugin."""
        return f"{self.package_name}.so"

    @property
    def pjrt_plugin_path(self) -> Path:
        """Full path to custom TT PJRT plugin."""
        return REPO_DIR / f"build/src/tt/{self.pjrt_plugin}"

    @property
    def tt_mlir_install_dir(self) -> Path:
        """Full path to tt-mlir installation dir."""
        return REPO_DIR / "third_party/tt-mlir/install"

    @property
    def project_is_built(self) -> bool:
        """
        Flag indicating project is already built.

        It is considered built if PJRT plugin exists and tt-mlir install dir exists.
        """
        return self.pjrt_plugin_path.exists() and self.tt_mlir_install_dir.exists()

    # --- Properties of wheel bundle ---

    @property
    def jax_plugin_init(self) -> Path:
        """Path to __init__.py which initializes jax plugin."""
        return THIS_DIR / "__init__.py"

    @property
    def jax_plugin_init_copied(self) -> bool:
        """Returns True if __init__.py is already copied to destination."""
        return (self.jax_plugin_target_dir / "__init__.py").exists()

    @property
    def jax_plugin_target_dir_relpath(self) -> Path:
        """Path to our custom jax plugin relative to this script."""
        return Path(f"jax_plugins/{self.package_name}")

    @property
    def jax_plugin_target_dir(self) -> Path:
        """
        Full path to target dir in which .so file and tt-mlir installation tree root
        will be copied.
        """
        return THIS_DIR / self.jax_plugin_target_dir_relpath

    @property
    def pjrt_plugin_copied(self) -> bool:
        """Returns True if .so file is already copied to destination."""
        return (self.jax_plugin_target_dir / self.pjrt_plugin).exists()

    @property
    def tt_mlir_target_dir(self) -> Path:
        """
        Convenience accessor for tt-mlir target dir which is nested in jax plugin
        target dir.
        """
        return self.jax_plugin_target_dir / "tt-mlir/install"

    @property
    def tt_mlir_copied(self) -> bool:
        """Returns True if tt-mlir installation is already copied to destination."""
        return self.tt_mlir_target_dir.exists()

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

    def finalize_options(self):
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

    It first ensures project is built, then it copies the created JAX plugin (product
    of the build) `pjrt_plugin_tt.so` inside the plugin dir, and finally copies entire
    tt-mlir installation dir inside the plugin dir as well, for them all to be packaged
    together.

    NOTE MANIFEST.in defines command through which additional non-python files (like
    .yaml, .so, .a, etc.) are going to be included in the final package. This cannot be
    done solely using `package_data` parameter of `setup` which expects python modules.
    """

    def run(self):
        print(f"Building wheel with following settings:\n{config}")

        # Build the project if not already built.
        if not config.project_is_built:
            print("Building project...")
            self.build_cmake_project()
        else:
            print("Project already built.")

        # Create temp dir.
        config.jax_plugin_target_dir.mkdir(parents=True, exist_ok=True)

        # Copy __init__.py file into the python jax_plugins package directory.
        if not config.jax_plugin_init_copied:
            print("Copying __init__.py...")
            shutil.copy2(config.jax_plugin_init, config.jax_plugin_target_dir)
        else:
            print("__init__.py already copied.")

        # Copy the .so file into the python jax_plugins package directory.
        if not config.pjrt_plugin_copied:
            print("Copying PJRT plugin...")
            shutil.copy2(config.pjrt_plugin_path, config.jax_plugin_target_dir)
        else:
            print("PJRT plugin already copied.")

        # Copy tt-mlir install dir into the python jax_plugins package directory.
        # TODO it might be an overkill that we are copying entire tt-mlir install dir,
        # but various issues pop up if some tt-mlir lib is missing or `TT_METAL_HOME`
        # doesn't point to tt-metal install dir. It might be worthwhile searching for a
        # minimum set of things we need to copy from tt-mlir installation in order for
        # plugin to work.
        # See issue https://github.com/tenstorrent/tt-xla/issues/595.
        if not config.tt_mlir_copied:
            print("Copying tt-mlir installation...")
            shutil.copytree(
                config.tt_mlir_install_dir,
                config.tt_mlir_target_dir,
                dirs_exist_ok=True,
            )
        else:
            print("tt-mlir installation already copied.")

        # Continue with the rest of the Python build.
        super().run()

        print("Wheel built.")

    def build_cmake_project(self):
        cmake_args = [
            "-G",
            "Ninja",
            "-B",
            "build",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_C_COMPILER=clang-17",
            "-DCMAKE_CXX_COMPILER=clang++-17",
            "-DTTMLIR_ENABLE_RUNTIME=ON",
            "-DTTMLIR_ENABLE_STABLEHLO=ON",
            "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
            "-DTT_RUNTIME_ENABLE_PERF_TRACE=ON",
        ]
        build_command = ["--build", "build"]

        # Execute cmake from top level project dir, where root CMakeLists.txt resides.
        subprocess.check_call(["cmake", *cmake_args], cwd=REPO_DIR)
        subprocess.check_call(["cmake", *build_command], cwd=REPO_DIR)


setup(
    name=config.project_name,
    version=config.version,
    install_requires=config.requirements,
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
    ],
    cmdclass={
        "bdist_wheel": BdistWheel,
        "build_py": CMakeBuildPy,
    },
    package_dir={
        f"jax_plugins.{config.package_name}": f"{config.jax_plugin_target_dir_relpath}",
    },
    packages=[
        f"jax_plugins.{config.package_name}",
    ],
    include_package_data=True,
    package_data={
        f"jax_plugins.{config.package_name}": [f"{config.pjrt_plugin}"],
    },
    entry_points={
        # We must advertise which Python modules should be treated as loadable
        # plugins. This augments the path based scanning that Jax does, which
        # is not always robust to all packaging circumstances.
        "jax_plugins": [
            f"{config.package_name} = jax_plugins.{config.package_name}",
        ],
    },
    description="Tenstorrent PJRT plugin",
    long_description=config.long_description,
    long_description_content_type=config.description_content_type,
    zip_safe=config.zip_safe,
)
