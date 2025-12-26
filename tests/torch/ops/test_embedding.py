# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
import torch_xla.runtime as xr
from infra import Framework, run_op_test
from infra.evaluators import TorchComparisonEvaluator
from torch_xla.distributed.spmd import Mesh
from utils import Category

from tests.infra.evaluators import ComparisonConfig
from tests.infra.testers.compiler_config import CompilerConfig
from third_party.tt_forge_models.gemma.pytorch.loader import (
    ModelLoader as GemmaModelLoader,
)
from third_party.tt_forge_models.gemma.pytorch.loader import (
    ModelVariant as GemmaModelVariant,
)


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        super().__init__()
        self.embedding = torch.nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )

    def forward(self, input_ids: torch.LongTensor):
        return self.embedding(input_ids)


@pytest.mark.nightly
@pytest.mark.llmbox
@pytest.mark.parametrize(
    "mesh_shape,shard_spec",
    [
        ((1, 8), ("batch", "model")),
        ((2, 4), (None, "batch")),
    ],
)
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_gemma2_27b_embed_tokens(mesh_shape, shard_spec):
    loader = GemmaModelLoader(variant=GemmaModelVariant.GEMMA_2_27B_IT)
    config = loader.load_config()

    embed_tokens = Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)

    def get_shard_spec(embed, args, kwargs):
        shard_specs = {}
        shard_specs[embed_tokens.embedding.weight] = shard_spec
        return shard_specs

    num_devices = xr.global_runtime_device_count()
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

    batch_size = 1
    seq_len = 1024
    input_ids = torch.randint(
        0, config.vocab_size, (batch_size, seq_len), dtype=torch.long
    )

    comparison_config = ComparisonConfig()

    run_op_test(
        embed_tokens,
        [input_ids],
        comparison_config=comparison_config,
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )
