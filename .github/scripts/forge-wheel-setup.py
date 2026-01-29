# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os

from setuptools import setup

version_tag = os.environ.get("VERSION_TAG")
if not version_tag:
    raise RuntimeError("VERSION_TAG environment variable must be set")
testing_tag = "0.9.0.dev20260129"

setup(
    name="tt-forge",
    version=version_tag,
    homepage="https://github.com/tenstorrent/tt-forge",
    # wheel versions hardcoded for testing since publishing to pypi doesn't work rn
    install_requires=[
        f"pjrt-plugin-tt @https://pypi.eng.aws.tenstorrent.com/pjrt-plugin-tt/pjrt_plugin_tt-{testing_tag}-cp311-cp311-linux_x86_64.whl",
        f"vllm_tt @https://pypi.eng.aws.tenstorrent.com/vllm-tt/vllm_tt-{testing_tag}-cp311-cp311-linux_x86_64.whl",
    ],
    python_requires=">=3.11",
    py_modules=[],
    has_ext_modules=lambda: True,
)
