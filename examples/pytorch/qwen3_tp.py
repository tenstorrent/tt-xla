# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
import torch
import torch.nn.functional as F
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh
from transformers import AutoModel, AutoTokenizer


# --------------------------------
# Test run
# --------------------------------
def qwen3_tp():
    # Set SPMD mode and get number of devices.
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()

    num_devices = xr.global_runtime_device_count()

    # Instantiate model.
    model_name = "Qwen/Qwen3-Embedding-8B"
    model: torch.nn.Module = AutoModel.from_pretrained(
        model_name, use_cache=False, torch_dtype=torch.bfloat16
    )
    model.eval()

    # Instantiate tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype=torch.bfloat16)

    # Prepare inputs.
    input_texts, sample_queries = create_input_texts()
    inputs = tokenizer(
        input_texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    # Create a mesh.
    mesh_shape = (1, num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

    # Move inputs and model to device.
    device = torch_xla.device()

    inputs = inputs.to(device)
    model = model.to(device)

    # Tensor-parallel sharding:
    # Linear weight layout (out_features, in_features).
    # column-parallel: shard rows (out_features) -> each device computes a distinct output slice.
    # row-parallel: shard columns (in_features) -> devices compute partial outputs that require an all-reduce.
    shard_specs = {}
    for layer in model.layers:
        # MLP expansion -> column-parallel
        shard_specs[layer.mlp.up_proj.weight] = ("model", None)
        shard_specs[layer.mlp.gate_proj.weight] = ("model", None)
        # MLP contraction -> row-parallel
        shard_specs[layer.mlp.down_proj.weight] = (None, "model")

        # Attention per-head projections -> column-parallel
        shard_specs[layer.self_attn.q_proj.weight] = ("model", None)
        shard_specs[layer.self_attn.k_proj.weight] = ("model", None)
        shard_specs[layer.self_attn.v_proj.weight] = ("model", None)
        # Attention mix-heads projection -> row-parallel
        shard_specs[layer.self_attn.o_proj.weight] = (None, "model")

    for tensor, shard_spec in shard_specs.items():
        xs.mark_sharding(tensor, mesh, shard_spec)

    # Compile model.
    compiled_model = torch.compile(model, backend="tt")

    # Run model.
    with torch.no_grad():
        output = compiled_model(**inputs)

        last_hidden_states = output.last_hidden_state.cpu()
        attention_mask = inputs["attention_mask"].cpu()

        postprocessing(last_hidden_states, attention_mask, sample_queries)


def create_input_texts():
    sample_queries = [
        "What is the capital of China?",
        "Explain gravity",
    ]
    sample_documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
    ]
    formatted_queries = [
        f"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: {query}"
        for query in sample_queries
    ]
    input_texts = formatted_queries + sample_documents

    return input_texts, sample_queries


def postprocessing(last_hidden_states, attention_mask, sample_queries):
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        embeddings = last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        embeddings = last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
        ]
    embeddings = F.normalize(embeddings, p=2, dim=1)
    num_queries = len(sample_queries)
    scores = embeddings[:num_queries] @ embeddings[num_queries:].T

    print("Similarity scores:")
    print(scores.tolist())


# --------------------------------
# main
# --------------------------------
if __name__ == "__main__":
    # By default torch_xla uses the CPU device so we have to set it to TT device.
    xr.set_device_type("TT")

    qwen3_tp()
