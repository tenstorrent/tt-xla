# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from setuptools import setup

setup(
    name="vllm_tt",
    version="0.1",
    packages=["vllm_tt"],
    install_requires=["vllm==0.10.1.1"],
    entry_points={"vllm.platform_plugins": ["tt = vllm_tt:register"]},
)
