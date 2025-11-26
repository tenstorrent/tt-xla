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

"""
Overview of tensor parallelism (TP) strategy for LLM MLP and attention layers:

For MLP, we want to shard in a way that minimizes CCLs. The most common method is megatron style,
where we shard the up and gate weights on column parallel and the down weights on row parallel. This
will produce a column sharded intermediate tensor after up and gate, and a partial result on each
device after proj that will be all_reduced at the end of MLP.

For attention, we want to do something similar, with the caveat that the sharded intermediate after
QKV projection must be aligned with the attention heads. To state another way, the number of heads
must be divisible by the number of devices. The added benefit of this shard scheme (head parallel)
is that both QKV projection and self-attention (MM+softmax+MM) can happen locally without any CCLs.

- Column-parallel sharding splits the rows across devices. Each device computes its local output fully,
  producing distinct output chunks.

- Row-parallel sharding splits the columns across devices. Each device computes a partial output, and
  then an all-reduce (sum) merges partial results to form the final output.



Devices:  [Device 0]           [Device 1]           ...           [Device N-1]
Inputs X replicated on all devices
================================================================================
1) MLP (Megatron-style)
--------------------------------------------------------------------------------
up_proj, gate_proj → column-parallel
down_proj          → row-parallel + ALL-REDUCE
--------------------------------------------------------------------------------
                        ┌──────────────────────────────────────────┐
                        │                MLP Block                 │
                        └──────────────────────────────────────────┘

                 [Device 0]            [Device 1]                   [Device N-1]
X (replicated)      |                     |                               |
                    v                     v                               v
              ┌──────────┐         ┌──────────┐                     ┌──────────┐
              │ up_proj0 │         │ up_proj1 │          ...        │ up_projN │
              │ (column) │         │ (column) │                     │ (column) │
              └──────────┘         └──────────┘                     └──────────┘
                    |                     |                               |
                    |                     |                               |
              ┌──────────┐         ┌──────────┐                     ┌──────────┐
              │gate_proj0│         │gate_proj1│          ...        │gate_projN│
              │ (column) │         │ (column) │                     │ (column) │
              └──────────┘         └──────────┘                     └──────────┘
                    |                     |                               |
                    v                     v                               v
                H_0 (local)           H_1 (local)           ...       H_{N-1} (local)
        (distinct intermediate slices; no communication)

                    |                     |                               |
                    v                     v                               v
              ┌──────────┐         ┌──────────┐                     ┌──────────┐
              │down_proj0│         │down_proj1│          ...        │down_projN│
              │   (row)  │         │   (row)  │                     │   (row)  │
              └──────────┘         └──────────┘                     └──────────┘
                    |                     |                               |
                    v                     v                               v
             Partial O_0             Partial O_1            ...     Partial O_{N-1}
           (each is a partial output that must be summed)

                    \                     |                               /
                     \____________________|______________________________/
                                          v
                              ┌──────────────────────────┐
                              │    ALL-REDUCE (sum)      │
                              └──────────────────────────┘
                                          |
                                          v
                                      Final O

================================================================================
2) Attention (Head-parallel Q/K/V, Row-parallel o_proj)
--------------------------------------------------------------------------------
Requires: num_heads % num_devices == 0
q_proj / k_proj / v_proj → column-parallel (head-parallel)
Self-attention per device → local, no CCL
o_proj                    → row-parallel + ALL-REDUCE
--------------------------------------------------------------------------------
                        ┌──────────────────────────────────────────┐
                        │               Attention Block            │
                        └──────────────────────────────────────────┘

                 [Device 0]            [Device 1]                   [Device N-1]
X (replicated)      |                     |                               |
                    v                     v                               v
              ┌──────────┐         ┌──────────┐                     ┌──────────┐
              │  q_proj0 │         │  q_proj1 │          ...        │  q_projN │
              │ (column) │         │ (column) │                     │ (column) │
              ├──────────┤         ├──────────┤                     ├──────────┤
              │  k_proj0 │         │  k_proj1 │          ...        │  k_projN │
              │ (column) │         │ (column) │                     │ (column) │
              ├──────────┤         ├──────────┤                     ├──────────┤
              │  v_proj0 │         │  v_proj1 │          ...        │  v_projN │
              │ (column) │         │ (column) │                     │ (column) │
              └──────────┘         └──────────┘                     └──────────┘
                    |                     |                               |
                    v                     v                               v
        Heads H_0 (local)     Heads H_1 (local)           ...    Heads H_{N-1} (local)
     (Each device owns disjoint attention heads; head-parallel)

                    |                     |                               |
                    v                     v                               v
            ┌──────────────────── Self-Attention (local) ─────────────────────┐
            │  Q*K^T, softmax, (·V) executed locally on assigned heads        │
            │  No cross-device communication required (“no CCL”).             │
            └─────────────────────────────────────────────────────────────────┘

                    |                     |                               |
                    v                     v                               v
              ┌──────────┐         ┌──────────┐                     ┌──────────┐
              │  o_proj0 │         │  o_proj1 │          ...        │  o_projN │
              │   (row)  │         │   (row)  │                     │   (row)  │
              └──────────┘         └──────────┘                     └──────────┘
                    |                     |                               |
                    v                     v                               v
             Partial O_0             Partial O_1            ...     Partial O_{N-1}
           (partial outputs; require summation)

                    \                     |                               /
                     \____________________|______________________________/
                                          v
                              ┌──────────────────────────┐
                              │    ALL-REDUCE (sum)      │
                              └──────────────────────────┘
                                          |
                                          v
                                      Final O
"""


# --------------------------------
# Qwen3 TP example
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

    # Validate attention heads divisibility for head-parallel sharding.
    num_attention_heads = model.config.num_attention_heads
    if num_attention_heads % num_devices != 0:
        raise ValueError(
            f"Number of attention heads ({num_attention_heads}) must be divisible by number of devices ({num_devices}) for head-parallel sharding."
        )

    # Tensor-parallel sharding:
    shard_specs = {}
    for layer in model.layers:
        # MLP
        shard_specs[layer.mlp.up_proj.weight] = ("model", None)  # column-parallel
        shard_specs[layer.mlp.gate_proj.weight] = ("model", None)  # column-parallel
        shard_specs[layer.mlp.down_proj.weight] = (None, "model")  # row-parallel

        # Attention
        shard_specs[layer.self_attn.q_proj.weight] = ("model", None)  # column-parallel
        shard_specs[layer.self_attn.k_proj.weight] = ("model", None)  # column-parallel
        shard_specs[layer.self_attn.v_proj.weight] = ("model", None)  # column-parallel
        shard_specs[layer.self_attn.o_proj.weight] = (None, "model")  # row-parallel

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


if __name__ == "__main__":
    # By default torch_xla uses the CPU device so we have to set it to TT device.
    xr.set_device_type("TT")

    qwen3_tp()
