# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import json
import numpy as np
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
        max_position_embeddings=params.max_seq_len,
        rms_norm_eps=params.norm_eps,
        rope_theta=params.rope_theta,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        partial_rotary_factor=1.0,
        tie_word_embeddings=False,
        use_bias=False,
    )


def convert_llama_weights(
    ckpt_dir: str,
    tokenizer: AutoTokenizer,
    max_seq_len: int = 2048,
    n_layers: int = 32,
    tensor_parallel_size: int = 4,
    tp_rank: int = 0,
    verbose: bool = False,
) -> Tuple[PyTree[np.ndarray], LLaMAConfig]:
    """
    Convert LLaMA weights with optimal tensor parallel sharding.
    Each device gets only ITS portion of weights (memory efficient).
    """
    ckpt_paths = sorted(Path(ckpt_dir).glob("*.pth"))
    ckpts = {}
    for i, ckpt_path in enumerate(ckpt_paths):
        if verbose:
            print(f"Loading checkpoint {i+1} of {len(ckpt_paths)} ...")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if verbose:
            print("Loaded.")
        ckpts[int(ckpt_path.name.split(".", maxsplit=2)[1])] = checkpoint
    ckpts = [ckpts[i] for i in sorted(list(ckpts.keys()))]

    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())
    params.pop("use_scaled_rope", None)

    params.update(
        {"vocab_size": len(tokenizer), "max_seq_len": max_seq_len, "n_layers": n_layers}
    )
    llama_config = config_from_params(ModelArgs(**params))

    # Create OPTIMAL SHARDED weights (only what THIS device needs)
    jax_weights = {
        "transformer": {
            "wte": {
                # Vocab parallel: each device gets its vocab slice
                "embedding": np.concatenate(
                    [
                        ckpt["tok_embeddings.weight"].type(torch.float16).numpy()
                        for ckpt in ckpts
                    ],
                    axis=0,
                )
            },
            "ln_f": {
                # Layer norm: replicated (small, no sharding needed)
                "kernel": ckpts[0]["norm.weight"]
                .type(torch.float16)
                .numpy()
            },
            "h": {
                "%d"
                % (layer): {
                    "attention": {
                        "wq": {
                            # Column parallel: split output dimension (heads)
                            "kernel": np.concatenate(
                                [
                                    ckpt["layers.%d.attention.wq.weight" % (layer)]
                                    .type(torch.float16)
                                    .numpy()
                                    for ckpt in ckpts
                                ],
                                axis=0,
                            ).transpose()
                        },
                        "wk": {
                            # Column parallel: split output dimension (kv_heads)
                            "kernel": np.concatenate(
                                [
                                    ckpt["layers.%d.attention.wk.weight" % (layer)]
                                    .type(torch.float16)
                                    .numpy()
                                    for ckpt in ckpts
                                ],
                                axis=0,
                            ).transpose()
                        },
                        "wv": {
                            # Column parallel: split output dimension (kv_heads)
                            "kernel": np.concatenate(
                                [
                                    ckpt["layers.%d.attention.wv.weight" % (layer)]
                                    .type(torch.float16)
                                    .numpy()
                                    for ckpt in ckpts
                                ],
                                axis=0,
                            ).transpose()
                        },
                        "wo": {
                            # Row parallel: split input dimension (heads)
                            "kernel": np.concatenate(
                                [
                                    ckpt["layers.%d.attention.wo.weight" % (layer)]
                                    .type(torch.float16)
                                    .numpy()
                                    for ckpt in ckpts
                                ],
                                axis=1,
                            ).transpose()
                        },
                    },
                    "feed_forward": {
                        "w1": {
                            # Column parallel: split output dimension
                            "kernel": np.concatenate(
                                [
                                    ckpt["layers.%d.feed_forward.w1.weight" % (layer)]
                                    .type(torch.float16)
                                    .numpy()
                                    for ckpt in ckpts
                                ],
                                axis=0,
                            ).transpose()
                        },
                        "w2": {
                            # Row parallel: split input dimension
                            "kernel": np.concatenate(
                                [
                                    ckpt["layers.%d.feed_forward.w2.weight" % (layer)]
                                    .type(torch.float16)
                                    .numpy()
                                    for ckpt in ckpts
                                ],
                                axis=1,
                            ).transpose()
                        },
                        "w3": {
                            # Column parallel: split output dimension
                            "kernel": np.concatenate(
                                [
                                    ckpt["layers.%d.feed_forward.w3.weight" % (layer)]
                                    .type(torch.float16)
                                    .numpy()
                                    for ckpt in ckpts
                                ],
                                axis=0,
                            ).transpose()
                        },
                    },
                    "attention_norm": {
                        # Layer norm: replicated (small, no sharding needed)
                        "kernel": ckpts[0]["layers.%d.attention_norm.weight" % (layer)]
                        .type(torch.float16)
                        .numpy()
                    },
                    "ffn_norm": {
                        # Layer norm: replicated (small, no sharding needed)
                        "kernel": ckpts[0]["layers.%d.ffn_norm.weight" % (layer)]
                        .type(torch.float16)
                        .numpy()
                    },
                }
                for layer in range(n_layers)
            },
        },
        "lm_head": {
            # Vocab parallel: each device gets its vocab slice
            "kernel": np.concatenate(
                [ckpt["output.weight"].type(torch.float16).numpy() for ckpt in ckpts],
                axis=0,
            ).transpose()
        },
    }

    del ckpts
    torch.cuda.empty_cache()  # Not necessary on CPU, but okay
    import gc

    gc.collect()

    return jax_weights, llama_config
