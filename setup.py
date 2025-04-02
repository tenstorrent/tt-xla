# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
from dataclasses import dataclass

from setuptools import setup
from setuptools.command.build import build as _build
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.install import install
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel


@dataclass
class SetupMetadata:
    package_name: str = "tt_pjrt_plugin"
    description_content_type: str = "text/markdown"
    zip_safe: bool = False  # Needs to reference embedded shared libraries.

    @property
    def package(self) -> dict:
        return {"tt": ["pjrt_plugin_tt.so"]}

    @property
    def cmdclass(self) -> dict:
        return {"build_py": CMakeBuildPy}

    @property
    def version(self) -> str:
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
        return date + "+main." + short_hash

    @property
    def requirements(self) -> list:
        with open("requirements.txt", "r") as f:
            return f.read().splitlines()

    @property
    def description(self) -> str:
        with open("README.md", "r") as f:
            return f.read()


class bdist_wheel(_bdist_wheel):
    def finalize_options(self):
        _bdist_wheel.finalize_options(self)
        self.root_is_pure = False

    def get_tag(self):
        python, abi, plat = _bdist_wheel.get_tag(self)
        # We don't contain any python extensions so are version agnostic
        # but still want to be platform specific.
        python, abi = "py3", "none"
        return python, abi, plat


# Force installation into platlib.
# Since this is a pure-python library with platform binaries, it is
# mis-detected as "pure", which fails audit. Usually, the presence of an
# extension triggers non-pure install. We force it here.
class platlib_install(install):
    def finalize_options(self):
        install.finalize_options(self)
        self.install_lib = self.install_platlib


def populate_built_package(abs_dir):
    """Makes sure that a directory and __init__.py exist.

    This needs to unfortunately happen before any of the build process
    takes place so that setuptools can plan what needs to be built.
    We do this for any built packages (vs pure source packages).
    """
    os.makedirs(abs_dir, exist_ok=True)
    with open(os.path.join(abs_dir, "__init__.py"), "wt"):
        pass


class PjrtPluginBuild(_build):
    def run(self):
        self.run_command("build_py")


class CMakeBuildPy(_build_py):
    def run(self):
        # Build the CMake project
        self.build_cmake_project()

        super().run()

        # Copy the shared object (.so) file to the package directory
        # source_file = "build/src/tt/pjrt_plugin_tt.so"
        # if os.path.exists(source_file):
        #     destination_dir = os.path.join("build/src/tt/pjrt_plugin_tt.so")
        #     os.makedirs(destination_dir, exist_ok=True)
        #     shutil.copy(source_file, destination_dir)
        # else:
        #     raise FileNotFoundError(f"{source_file} not found after build.")

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

        subprocess.check_call(["cmake", *cmake_args])
        subprocess.check_call(["cmake", *build_command])

        print("Build complete.")


metadata = SetupMetadata()

setup(
    name=metadata.package_name,
    version=metadata.version,
    install_requires=metadata.requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
    ],
    # cmdclass=metadata.cmdclass,
    cmdclass={
        "build": PjrtPluginBuild,
        "build_py": CMakeBuildPy,
        "bdist_wheel": bdist_wheel,
        "install": platlib_install,
    },
    package_dir={
        "jax_plugins.tt": "jax_plugins/tt",
        "tt": "build/src/tt/",
    },
    packages=[
        "jax_plugins.tt",
        "tt",
    ],
    package_data={
        "tt": ["pjrt_plugin_tt.so"],
    },
    entry_points={
        # We must advertise which Python modules should be treated as loadable
        # plugins. This augments the path based scanning that Jax does, which
        # is not always robust to all packaging circumstances.
        "jax_plugins": [
            "tt = jax_plugins.tt",
        ],
    },
    long_description=metadata.description,
    long_description_content_type=metadata.description_content_type,
    zip_safe=metadata.zip_safe,
)
