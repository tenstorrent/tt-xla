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

def configure_multihost_environment():
    distributed_env = os.environ.copy()
    distributed_env["TT_RUNTIME_ENABLE_PROGRAM_CACHE"] = "1"
    distributed_env["TT_DISTRIBUTED_WORKER_PATH"] = get_distributed_worker_path()
    distributed_env["TT_RUNTIME_ENABLE_DISTRIBUTED"] = "1"
    distributed_env["TT_DISTRIBUTED_RANK_BINDING"] = "2x4_multiprocess"
    return distributed_env


#  TT_DISTRIBUTED_WORKER_PATH=$TT_MLIR_HOME/build/runtime/bin/distributed/worker pytest -svv tests/torch/multi_host/llmbox/test_multihost.py
@pytest.mark.parametrize("generic_test", ["tests/torch/multi_chip/n300/test_torch_xla_multichip_basic.py"])
def test_multihost_runtime_stitching(generic_test):
    distributed_env = configure_multihost_environment()
    result = subprocess.run([
        sys.executable,
        "-m",
        "pytest",
        "-svv",
        f"{generic_test}"
    ],
    env=distributed_env,
    cwd=os.getcwd())

@pytest.mark.push
@pytest.mark.llmbox
@pytest.mark.parametrize(
    "model_variant",
    ["falcon/pytorch-tiiuae/Falcon3-7B-Base-tensor_parallel-full-inference",
"falcon/pytorch-tiiuae/Falcon3-10B-Base-tensor_parallel-full-inference",
"falcon/pytorch-tiiuae/Falcon3-Mamba-7B-Base-tensor_parallel-full-inference",
"gemma/pytorch-google/gemma-1.1-7b-it-tensor_parallel-full-inference",
"gemma/pytorch-google/gemma-2-9b-it-tensor_parallel-full-inference",
"falcon/pytorch-tiiuae/falcon-7b-instruct-tensor_parallel-full-inference",
"llama/causal_lm/pytorch-llama_3_1_8b-tensor_parallel-full-inference",
"llama/causal_lm/pytorch-llama_3_1_8b_instruct-tensor_parallel-full-inference",
"llama/causal_lm/pytorch-llama_3_8b_instruct-tensor_parallel-full-inference",
"mistral/pixtral/pytorch-tensor_parallel-full-inference",
"mistral/pytorch-7b_instruct_v03-tensor_parallel-full-inference",
"qwen_3/causal_lm/pytorch-0_6b-tensor_parallel-full-inference",
"qwen_3/causal_lm/pytorch-14b-tensor_parallel-full-inference",
"qwen_3/causal_lm/pytorch-1_7b-tensor_parallel-full-inference",
"qwen_3/causal_lm/pytorch-32b-tensor_parallel-full-inference",
"qwen_3/causal_lm/pytorch-8b-tensor_parallel-full-inference",
"qwen_3/embedding/pytorch-embedding_8b-tensor_parallel-full-inference",
"mistral/pytorch-ministral_8b_instruct-tensor_parallel-full-inference",
"mistral/pytorch-mistral_small_24b_instruct_2501-tensor_parallel-full-inference",
"mistral/pytorch-mistral_nemo_instruct_2407-tensor_parallel-full-inference",
"mistral/pytorch-devstral_small_2505-tensor_parallel-full-inference",
"mistral/pytorch-magistral_small_2506-tensor_parallel-full-inference",
"qwen_2_5/causal_lm/pytorch-14b_instruct-tensor_parallel-full-inference",
"qwen_2_5/causal_lm/pytorch-32b_instruct-tensor_parallel-full-inference",
"qwen_2_5_coder/pytorch-32b_instruct-tensor_parallel-full-inference",]
)
def test_multihost_models(model_variant):
    distributed_env = configure_multihost_environment()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-ssv",
            f"tests/runner/test_models.py::test_all_models_torch[{model_variant}]",
        ],
        env=distributed_env,
        cwd=os.getcwd(),
    )

    assert (
        result.returncode == 0
    ), f"Failed to run multihost test for model variant: {model_variant}"
