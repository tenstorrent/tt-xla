# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


# This test file is a proxy to tests/runner/test_models.py.
# It runs models through test_models.py on a subprocess with distributed runtime enabled.

import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "model_variant",
    ["llama/causal_lm/pytorch-llama_3_1_8b-tensor_parallel-full-inference"],
)
def test_multihost_models(model_variant):
    os.environ["TT_RUNTIME_ENABLE_DISTRIBUTED"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-ssv",
            f"tests/runner/test_models.py::test_all_models[{model_variant}]",
        ],
        env=os.environ.copy(),
        cwd=os.getcwd(),
    )

    assert (
        result.returncode == 0
    ), f"Failed to run multihost test for model variant: {model_variant}"
