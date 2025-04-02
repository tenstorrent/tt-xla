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
from typing import List

from setuptools import setup
from setuptools.command.build_py import build_py
from wheel.bdist_wheel import bdist_wheel

THIS_DIR = Path(os.path.realpath(os.path.dirname(__file__)))
REPO_DIR = Path(os.path.join(THIS_DIR, "..", "..")).resolve()


@dataclass
class SetupConfig:
    """
    Helper dataclass storing wheel config.

    Jax plugin must be bundled in following form:
    ```
    jax_plugins
    `-- <custom-plugin-name>
        `-- __init__.py     # contains plugin registration function
    ```
    to be automatically detected by `jax` lib. Upon installation, it will be unpacked
    from wheel into user's python `env/lib` dir.

    In our case we also want to bundle other necessary things in wheel. So our final
    package tree looks like this:
    ```
    jax_plugins
    `-- pjrt_plugin_tt
        |-- __init__.py
        |-- pjrt_plugin_tt.so
        `-- tt-metal    # Entire tt-metal installation folder
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

        requirements.txt is parsed and only jax or flax related stuff is pulled from it.
        """
        reqs = []
        requirements_path = REPO_DIR / "requirements.txt"

        with requirements_path.open() as f:
            for line in f.read().splitlines():
                if "jax" in line or "flax" in line:
                    reqs.append(line)

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
        return REPO_DIR / f"build/src/tt/{config.pjrt_plugin}"

    @property
    def tt_metal_install_dir(self) -> Path:
        """Full path to tt-metal installation dir."""
        return REPO_DIR / "third_party/tt-mlir/install/tt-metal"

    @property
    def project_is_built(self) -> bool:
        """
        Flag indicating project is already built.

        It is considered built if PJRT plugin exists and tt-metal install dir exists.
        """
        return self.pjrt_plugin_path.exists() and self.tt_metal_install_dir.exists()

    # --- Properties of wheel bundle ---

    @property
    def jax_plugin_target_dir_relpath(self) -> Path:
        """Path to our custom jax plugin relative to this script."""
        return Path(f"jax_plugins/{config.package_name}")

    @property
    def jax_plugin_target_dir(self) -> Path:
        """
        Full path to target dir in which .so file and metal installation tree root will
        be copied.
        """
        return THIS_DIR / self.jax_plugin_target_dir_relpath

    @property
    def pjrt_plugin_copied(self) -> bool:
        """Returns True if .so file is already copied to destination."""
        return (self.jax_plugin_target_dir / self.pjrt_plugin).exists()

    @property
    def tt_metal_target_dir(self) -> Path:
        """
        Convenience accessor for tt-metal target dir which is nested in jax plugin
        target dir.
        """
        return self.jax_plugin_target_dir / "tt-metal"

    @property
    def tt_metal_copied(self) -> bool:
        """Returns True if tt-metal installation is already copied to destination."""
        return self.tt_metal_target_dir.exists()

    @property
    def tt_metal_data(self) -> List:
        """
        List of all tt-metal files to include in the wheel.

        Value of `setup` parameter `data_files` must be set to this list to make sure
        all files in tt-metal installation directory (like .yaml, .a, .ld...) are
        included in wheel. However, since we must first copy tt-metal installation
        before collecting data files, we don't send it as `data_files` parameter to
        `setup` but rather explicitly set it in `CMakeBuildPy.run`. Check out comments
        in that method.

        This format is required by `setup` function, we can't just provide tree root
        unfortunately.
        """

        def collect_data_files(src_root: Path) -> List:
            """
            Recursively collects (target_dir, [file1, file2, ...]) pairs for data_files.
            """
            entries = []
            for path in src_root.rglob("*"):
                if path.is_file():
                    rel_dir = path.parent.relative_to(THIS_DIR)
                    entries.append((str(rel_dir), [str(path.relative_to(THIS_DIR))]))

            return entries

        assert self.tt_metal_copied
        return collect_data_files(self.tt_metal_target_dir)

    def __repr__(self) -> str:
        """Representes self as json string."""
        # Fields too long to display.
        # NOTE `tt_metal_data` cannot even be evaluated before tt-metal installation dir
        # is copied to target, i.e. before `self.tt_metal_copied == True`, which is
        # another reason for ignoring it.
        ignore_fields = ["long_description", "tt_metal_data"]

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
    def run(self):
        print(f"Building wheel with following settings:\n{config}")

        # Build the project if not already built.
        if not config.project_is_built:
            print("Building project...")
            self.build_cmake_project()
        else:
            print("Project already built.")

        # Copy the .so file into the python jax_plugins package directory.
        if not config.pjrt_plugin_copied:
            print("Copying PJRT plugin...")
            shutil.copy2(config.pjrt_plugin_path, config.jax_plugin_target_dir)
        else:
            print("PJRT plugin already copied.")

        # Copy tt-metal install dir into the python jax_plugins package directory.
        # TODO it might be an overkill that we are copying entire tt-metal install dir,
        # but various issues pop up if something is missing from where `TT_METAL_HOME`
        # is pointing. It might be worthwhile searching for a minimum set of things we
        # need to copy from tt-metal installation in order for plugin to work.
        # See issue https://github.com/tenstorrent/tt-xla/issues/595.
        if not config.tt_metal_copied:
            print("Copying tt-metal installation...")
            shutil.copytree(
                config.tt_metal_install_dir,
                config.tt_metal_target_dir,
                dirs_exist_ok=True,
            )
        else:
            print("tt-metal installation already copied.")

        # Explicitly set list of all tt-metal files to include in the wheel.
        # NOTE this setting can also be provided as `data_files` parameter of `setup`,
        # but we cannot determine it before tt-metal installation is copied to target
        # dir, so we explicitly set it here.
        self.distribution.data_files = config.tt_metal_data

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
