# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import subprocess
from dataclasses import dataclass
import os
from setuptools import setup
from setuptools.command.build_py import build_py
from wheel.bdist_wheel import bdist_wheel


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
        if not os.path.exists(
            os.path.join(os.getcwd(), "build/src/tt/pjrt_plugin_tt.so")
        ):
            # Build the CMake project
            self.build_cmake_project()

        super().run()

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


@dataclass
class SetupMetadata:
    project_name: str = "pjrt-plugin-tt"
    package_name: str = "pjrt_plugin_tt"
    description_content_type: str = "text/markdown"
    zip_safe: bool = False  # Needs to reference embedded shared libraries.

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
        reqs = []

        with open("requirements.txt", "r") as f:
            for line in f.read().splitlines():
                if "jax" in line or "flax" in line:
                    reqs.append(line)

        return reqs

    @property
    def description(self) -> str:
        with open("README.md", "r") as f:
            return f.read()


metadata = SetupMetadata()

setup(
    name=metadata.project_name,
    version=metadata.version,
    install_requires=metadata.requirements,
    classifiers=[
        "License :: Apache-2.0",
        "Programming Language :: Python :: 3",
    ],
    cmdclass={
        "bdist_wheel": BdistWheel,
        "build_py": CMakeBuildPy,
    },
    package_dir={
        "jax_plugins.pjrt_plugin_tt": "jax_plugins/pjrt_plugin_tt",
        "pjrt_plugin_tt": "build/src/tt/",
    },
    packages=[
        "jax_plugins.pjrt_plugin_tt",
        "pjrt_plugin_tt",
    ],
    package_data={
        "pjrt_plugin_tt": ["pjrt_plugin_tt.so"],
    },
    entry_points={
        # We must advertise which Python modules should be treated as loadable
        # plugins. This augments the path based scanning that Jax does, which
        # is not always robust to all packaging circumstances.
        "jax_plugins": [
            f"pjrt_plugin_tt = jax_plugins.pjrt_plugin_tt",
        ],
    },
    long_description=metadata.description,
    long_description_content_type=metadata.description_content_type,
    zip_safe=metadata.zip_safe,
)
