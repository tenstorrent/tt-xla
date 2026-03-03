# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.evaluators import ComparisonConfig, PccConfig
from engram_demo import Engram, engram_cfg, backbone_config
from torch_xla.distributed.spmd import Mesh
from transformers import AutoTokenizer

from tests.utils import failed_ttmlir_compilation

def test_engram_single_layer():
    xr.set_device_type("TT")
    engram_cfg.layer_ids = [0]

    text = "Only Alexander the Great could tame the horse Bucephalus."
    tokenizer = AutoTokenizer.from_pretrained(engram_cfg.tokenizer_name_or_path,trust_remote_code=True)
    input_ids = tokenizer(text,return_tensors='pt').input_ids

    print("input_ids.dtype:", input_ids.dtype)
    B,L = input_ids.shape

    hidden_states = torch.randn(B, L, backbone_config.hc_mult, backbone_config.hidden_size).to(torch.bfloat16)

    engram = Engram(layer_id=0)

    num_devices = xr.global_runtime_device_count()
    mesh_shape = (2, 4)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.95),
    )

    run_graph_test(
        engram,
        [hidden_states, input_ids],
        framework=Framework.TORCH,
        mesh=mesh,
        comparison_config=comparison_config,
    )



