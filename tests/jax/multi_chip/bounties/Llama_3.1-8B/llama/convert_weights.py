# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import json
import numpy as np
import gc
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jaxtyping import PyTree
from config import LLaMAConfig
from pathlib import Path
from typing import Optional, Tuple
from transformers import AutoTokenizer
from dataclasses import dataclass


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = (
        1024  # make SwiGLU hidden layer size multiple of large power of 2
    )
    ffn_dim_multiplier: Optional[float] = 1.3
    norm_eps: float = 1e-5
    rope_theta: float = 500000.0

    max_batch_size: int = 1
    max_seq_len: int = 2048


def config_from_params(params: ModelArgs) -> LLaMAConfig:
    return LLaMAConfig(
        vocab_size=params.vocab_size,
        hidden_size=params.dim,
        intermediate_size=14336,  # Correct value for Llama 3.1-8B
        num_hidden_layers=params.n_layers,
        num_attention_heads=params.n_heads,
        num_key_value_heads=params.n_kv_heads or params.n_heads,
        max_sequence_length=params.max_seq_len,  # THIS WAS MISSING!
        max_position_embeddings=params.max_seq_len,
        rms_norm_eps=params.norm_eps,
        rope_theta=params.rope_theta,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        partial_rotary_factor=1.0,
        tie_word_embeddings=False,
        use_bias=False,
    )


def create_sharded_param(mesh, full_param, shard_axis):
    """Create a sharded array directly from full parameter"""
    if shard_axis == 0:  # Shard along first axis (e.g., vocab dimension)
        sharding_spec = P("mp", None)
    elif shard_axis == 1:  # Shard along second axis (e.g., output features)
        sharding_spec = P(None, "mp")
    else:  # Replicated
        sharding_spec = P(None)

    # Convert to JAX array and shard directly - avoid intermediate copies
    sharded_array = jax.device_put(
        full_param, jax.sharding.NamedSharding(mesh, sharding_spec)
    )

    # Force garbage collection after each parameter
    gc.collect()

    return sharded_array


def convert_llama_weights(
    ckpt_dir: str,
    tokenizer: AutoTokenizer,
    max_seq_len: int = 2048,
    n_layers: int = 32,
    verbose: bool = False,
    mesh=None,  # Pass mesh for direct sharding
) -> Tuple[PyTree[np.ndarray], LLaMAConfig]:
    """
    Convert LLaMA weights with optimal tensor parallel sharding.
    Each device gets only ITS portion of weights (memory efficient).
    """
    ckpt_paths = sorted(Path(ckpt_dir).glob("*.pth"))
    if len(ckpt_paths) == 0:
        raise FileNotFoundError(f"No checkpoint files found in {ckpt_dir}")

    # Use only the first checkpoint file
    ckpt_path = ckpt_paths[0]
    if verbose:
        print(f"Loading checkpoint: {ckpt_path.name}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if verbose:
        print("Loaded.")

    # Immediately convert to FP16 and numpy to save memory
    if verbose:
        print("Converting to FP16 numpy arrays...")

    # Pre-convert all tensors to numpy to free torch memory ASAP
    numpy_weights = {}
    for key, tensor in ckpt.items():
        numpy_weights[key] = tensor.type(torch.float16).numpy()

    # Delete the original torch checkpoint immediately
    del ckpt
    gc.collect()

    if verbose:
        print("Torch tensors converted and freed.")

    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())
    params.pop("use_scaled_rope", None)

    params.update(
        {"vocab_size": len(tokenizer), "max_seq_len": max_seq_len, "n_layers": n_layers}
    )
    llama_config = config_from_params(ModelArgs(**params))

    # Create OPTIMAL SHARDED weights (only what THIS device needs)
    # Process embedding and lm_head first
    jax_weights = {
        "transformer": {
            "wte": {
                # Vocab parallel: each device gets its vocab slice
                "embedding": create_sharded_param(
                    mesh,
                    numpy_weights["tok_embeddings.weight"],
                    shard_axis=0,  # Shard along vocab dimension
                )
                if mesh is not None
                else numpy_weights["tok_embeddings.weight"]
            },
            "ln_f": {
                # Layer norm: replicated (small, no sharding needed)
                "kernel": numpy_weights["norm.weight"]
            },
            "h": {},  # Will populate layer by layer
        },
        "lm_head": {
            # Vocab parallel: each device gets its vocab slice
            "kernel": create_sharded_param(
                mesh,
                numpy_weights["output.weight"].transpose(),
                shard_axis=1,  # Shard along output dimension (second axis after transpose)
            )
            if mesh is not None
            else numpy_weights["output.weight"].transpose()
        },
    }

    # Free the embedding and lm_head weights from numpy_weights immediately
    del numpy_weights["tok_embeddings.weight"]
    del numpy_weights["norm.weight"]
    del numpy_weights["output.weight"]
    gc.collect()

    if verbose:
        print("Processed embedding and lm_head weights, freed from numpy_weights")

    # Process transformer layers one by one to minimize memory usage
    for layer in range(n_layers):
        if verbose and layer % 8 == 0:
            print(f"Processing layer {layer}/{n_layers}...")

        # Create layer weights with sharding
        layer_weights = {
            "attention": {
                "wq": {
                    # Column parallel: split output dimension (heads)
                    "kernel": create_sharded_param(
                        mesh,
                        numpy_weights[
                            f"layers.{layer}.attention.wq.weight"
                        ].transpose(),
                        shard_axis=1,  # Column parallel
                    )
                    if mesh is not None
                    else numpy_weights[
                        f"layers.{layer}.attention.wq.weight"
                    ].transpose()
                },
                "wk": {
                    # Column parallel: split output dimension (kv_heads)
                    "kernel": create_sharded_param(
                        mesh,
                        numpy_weights[
                            f"layers.{layer}.attention.wk.weight"
                        ].transpose(),
                        shard_axis=1,  # Column parallel
                    )
                    if mesh is not None
                    else numpy_weights[
                        f"layers.{layer}.attention.wk.weight"
                    ].transpose()
                },
                "wv": {
                    # Column parallel: split output dimension (kv_heads)
                    "kernel": create_sharded_param(
                        mesh,
                        numpy_weights[
                            f"layers.{layer}.attention.wv.weight"
                        ].transpose(),
                        shard_axis=1,  # Column parallel
                    )
                    if mesh is not None
                    else numpy_weights[
                        f"layers.{layer}.attention.wv.weight"
                    ].transpose()
                },
                "wo": {
                    # Row parallel: split input dimension (heads)
                    "kernel": create_sharded_param(
                        mesh,
                        numpy_weights[
                            f"layers.{layer}.attention.wo.weight"
                        ].transpose(),
                        shard_axis=0,  # Row parallel
                    )
                    if mesh is not None
                    else numpy_weights[
                        f"layers.{layer}.attention.wo.weight"
                    ].transpose()
                },
            },
            "feed_forward": {
                "w1": {
                    # Column parallel: split output dimension
                    "kernel": create_sharded_param(
                        mesh,
                        numpy_weights[
                            f"layers.{layer}.feed_forward.w1.weight"
                        ].transpose(),
                        shard_axis=1,  # Column parallel
                    )
                    if mesh is not None
                    else numpy_weights[
                        f"layers.{layer}.feed_forward.w1.weight"
                    ].transpose()
                },
                "w2": {
                    # Row parallel: split input dimension
                    "kernel": create_sharded_param(
                        mesh,
                        numpy_weights[
                            f"layers.{layer}.feed_forward.w2.weight"
                        ].transpose(),
                        shard_axis=0,  # Row parallel
                    )
                    if mesh is not None
                    else numpy_weights[
                        f"layers.{layer}.feed_forward.w2.weight"
                    ].transpose()
                },
                "w3": {
                    # Column parallel: split output dimension
                    "kernel": create_sharded_param(
                        mesh,
                        numpy_weights[
                            f"layers.{layer}.feed_forward.w3.weight"
                        ].transpose(),
                        shard_axis=1,  # Column parallel
                    )
                    if mesh is not None
                    else numpy_weights[
                        f"layers.{layer}.feed_forward.w3.weight"
                    ].transpose()
                },
            },
            "attention_norm": {
                # Layer norm: replicated (small, no sharding needed)
                "kernel": create_sharded_param(
                    mesh,
                    numpy_weights[f"layers.{layer}.attention_norm.weight"],
                    shard_axis=-1,  # Replicated
                )
                if mesh is not None
                else numpy_weights[f"layers.{layer}.attention_norm.weight"]
            },
            "ffn_norm": {
                # Layer norm: replicated (small, no sharding needed)
                "kernel": create_sharded_param(
                    mesh,
                    numpy_weights[f"layers.{layer}.ffn_norm.weight"],
                    shard_axis=-1,  # Replicated
                )
                if mesh is not None
                else numpy_weights[f"layers.{layer}.ffn_norm.weight"]
            },
        }

        # Add to jax_weights
        jax_weights["transformer"]["h"][str(layer)] = layer_weights

        # Immediately delete this layer's weights from numpy_weights to free memory
        del numpy_weights[f"layers.{layer}.attention.wq.weight"]
        del numpy_weights[f"layers.{layer}.attention.wk.weight"]
        del numpy_weights[f"layers.{layer}.attention.wv.weight"]
        del numpy_weights[f"layers.{layer}.attention.wo.weight"]
        del numpy_weights[f"layers.{layer}.feed_forward.w1.weight"]
        del numpy_weights[f"layers.{layer}.feed_forward.w2.weight"]
        del numpy_weights[f"layers.{layer}.feed_forward.w3.weight"]
        del numpy_weights[f"layers.{layer}.attention_norm.weight"]
        del numpy_weights[f"layers.{layer}.ffn_norm.weight"]

        # Force garbage collection after processing each layer
        gc.collect()

    # Clean up numpy weights after use
    del numpy_weights
    gc.collect()

    if verbose:
        print("Weight conversion completed and memory cleaned up.")

    return jax_weights, llama_config
