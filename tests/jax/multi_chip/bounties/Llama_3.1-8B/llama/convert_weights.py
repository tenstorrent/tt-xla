# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import torch
import json
import numpy as np
from jaxtyping import PyTree
from config import LLaMAConfig
from transformers import AutoTokenizer
from typing import Tuple, Optional
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


def config_from_params(args: ModelArgs) -> LLaMAConfig:
    intermediate_size = int(2 * (args.dim * 4) / 3)
    if args.ffn_dim_multiplier is not None:
        intermediate_size = int(args.ffn_dim_multiplier * intermediate_size)
    intermediate_size = args.multiple_of * (
        (intermediate_size + args.multiple_of - 1) // args.multiple_of
    )
    return LLaMAConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.dim,
        intermediate_size=intermediate_size,
        num_hidden_layers=args.n_layers,
        num_attention_heads=args.n_heads,
        num_key_value_heads=args.n_kv_heads,
        max_sequence_length=args.max_seq_len,
        rms_norm_eps=args.norm_eps,
        rope_theta=args.rope_theta,
    )


def convert_llama_weights(
    ckpt_dir: str,
    tokenizer: AutoTokenizer,
    max_seq_len: int = 2048,
    n_layers: int = 32,
    verbose: bool = False,
) -> Tuple[PyTree[np.ndarray], LLaMAConfig]:
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

    jax_weights = {
        "transformer": {
            "wte": {
                "embedding": np.concatenate(
                    [
                        ckpt["tok_embeddings.weight"].type(torch.float32).numpy()
                        for ckpt in ckpts
                    ],
                    axis=1,
                )
            },
            "ln_f": {"kernel": ckpts[0]["norm.weight"].type(torch.float32).numpy()},
            "h": {
                "%d"
                % (layer): {
                    "attention": {
                        "wq": {
                            "kernel": np.concatenate(
                                [
                                    ckpt["layers.%d.attention.wq.weight" % (layer)]
                                    .type(torch.float32)
                                    .numpy()
                                    for ckpt in ckpts
                                ],
                                axis=0,
                            ).transpose()
                        },
                        "wk": {
                            "kernel": np.concatenate(
                                [
                                    ckpt["layers.%d.attention.wk.weight" % (layer)]
                                    .type(torch.float32)
                                    .numpy()
                                    for ckpt in ckpts
                                ],
                                axis=0,
                            ).transpose()
                        },
                        "wv": {
                            "kernel": np.concatenate(
                                [
                                    ckpt["layers.%d.attention.wv.weight" % (layer)]
                                    .type(torch.float32)
                                    .numpy()
                                    for ckpt in ckpts
                                ],
                                axis=0,
                            ).transpose()
                        },
                        "wo": {
                            "kernel": np.concatenate(
                                [
                                    ckpt["layers.%d.attention.wo.weight" % (layer)]
                                    .type(torch.float32)
                                    .numpy()
                                    for ckpt in ckpts
                                ],
                                axis=1,
                            ).transpose()
                        },
                    },
                    "feed_forward": {
                        "w1": {
                            "kernel": np.concatenate(
                                [
                                    ckpt["layers.%d.feed_forward.w1.weight" % (layer)]
                                    .type(torch.float32)
                                    .numpy()
                                    for ckpt in ckpts
                                ],
                                axis=0,
                            ).transpose()
                        },
                        "w2": {
                            "kernel": np.concatenate(
                                [
                                    ckpt["layers.%d.feed_forward.w2.weight" % (layer)]
                                    .type(torch.float32)
                                    .numpy()
                                    for ckpt in ckpts
                                ],
                                axis=1,
                            ).transpose()
                        },
                        "w3": {
                            "kernel": np.concatenate(
                                [
                                    ckpt["layers.%d.feed_forward.w3.weight" % (layer)]
                                    .type(torch.float32)
                                    .numpy()
                                    for ckpt in ckpts
                                ],
                                axis=0,
                            ).transpose()
                        },
                    },
                    "attention_norm": {
                        "kernel": ckpts[0]["layers.%d.attention_norm.weight" % (layer)]
                        .type(torch.float32)
                        .numpy()
                    },
                    "ffn_norm": {
                        "kernel": ckpts[0]["layers.%d.ffn_norm.weight" % (layer)]
                        .type(torch.float32)
                        .numpy()
                    },
                }
                for layer in range(n_layers)
            },
        },
        "lm_head": {
            "kernel": np.concatenate(
                [ckpt["output.weight"].type(torch.float32).numpy() for ckpt in ckpts],
                axis=0,
            ).transpose()
        },
    }

    params.update(
        {"vocab_size": len(tokenizer), "max_seq_len": max_seq_len, "n_layers": n_layers}
    )
    llama_config = config_from_params(ModelArgs(**params))

    del ckpts
    torch.cuda.empty_cache()  # Not necessary on CPU, but okay
    import gc

    gc.collect()

    return jax_weights, llama_config
