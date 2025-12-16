# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


# This test file is a proxy to tests/runner/test_models.py.
# It runs models through test_models.py on a subprocess with distributed runtime enabled.

import os
import subprocess
import sys

import pytest


def get_distributed_worker_path():
    worker_path = os.environ.get("TT_DISTRIBUTED_WORKER_PATH")
    if worker_path:
        assert os.path.exists(worker_path), (
            "Distributed worker file does not exist at path: " + worker_path
        )
        return worker_path

    pjrt_plugin_dir = os.environ.get("TT_PJRT_PLUGIN_DIR")
    assert pjrt_plugin_dir, "TT_PJRT_PLUGIN_DIR environment variable is not set"

    worker_path = os.path.join(pjrt_plugin_dir, "bin/ttmlir/runtime/distributed/worker")
    assert os.path.exists(worker_path), (
        "Distributed worker file does not exist at path: " + worker_path
    )

    return worker_path


@pytest.mark.push
@pytest.mark.multi_host_cluster
@pytest.mark.parametrize(
    "model_variant",
    ["llama/causal_lm/pytorch-llama_3_1_8b-tensor_parallel-full-inference"],
)
def test_multihost_models(model_variant):
    distributed_env = os.environ.copy()
    distributed_env["TT_RUNTIME_ENABLE_PROGRAM_CACHE"] = "1"
    distributed_env["TT_DISTRIBUTED_WORKER_PATH"] = get_distributed_worker_path()
    distributed_env["TT_RUNTIME_ENABLE_DISTRIBUTED"] = "1"
    distributed_env["TT_DISTRIBUTED_RANK_BINDING"] = "2x4_multiprocess"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-ssv",
            f"tests/runner/test_models.py::test_all_models_torch[{model_variant}]",
            "--disable-perf-measurement",
        ],
        env=distributed_env,
        cwd=os.getcwd(),
    )

    assert (
        result.returncode == 0
    ), f"Failed to run multihost test for model variant: {model_variant}"
