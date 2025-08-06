# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
import jax

# Enable Shardy partitioner for advanced sharding analysis
jax.config.update("jax_use_shardy_partitioner", True)

import jax.numpy as jnp
import numpy as np
import fire
from flax.core.frozen_dict import freeze
from model import FlaxLLaMAForCausalLM
from convert_weights import convert_llama_weights
from transformers import AutoTokenizer
from generation import LLaMA
from jax.sharding import Mesh, PartitionSpec as P
import os
from pathlib import Path

ROOT = Path(__file__).parent


def jax_load(
    model_id: str,
    ckpt_dir: str,
    tokenizer_path: str,
    mesh,
    max_seq_length: int = 2048,
    n_layers: int = 32,
) -> LLaMA:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Get tensor parallel size from mesh
    tensor_parallel_size = mesh.shape["mp"]
    full_weights, jax_config = convert_llama_weights(
        ckpt_dir=ckpt_dir,
        tokenizer=tokenizer,
        max_seq_len=max_seq_length,
        n_layers=n_layers,
        tensor_parallel_size=1,  # tp_size=1 = full model
        tp_rank=0,
        verbose=True,
    )

    # Get devices from mesh
    devices = mesh.devices.flatten()

    def create_sharded_param(param_name, full_param, shard_axis):
        """Create a sharded array directly from full parameter"""
        if shard_axis == 0:  # Shard along first axis (e.g., vocab dimension)
            sharding_spec = P("mp", None)
        elif shard_axis == 1:  # Shard along second axis (e.g., output features)
            sharding_spec = P(None, "mp")
        else:  # Replicated
            sharding_spec = P(None)

        # Convert to JAX array and shard directly
        full_array = jnp.asarray(full_param)
        sharded_array = jax.device_put(
            full_array, jax.sharding.NamedSharding(mesh, sharding_spec)
        )

        return sharded_array

    # Build the sharded parameter tree directly from full weights
    jax_params = {
        "transformer": {
            "wte": {
                "embedding": create_sharded_param(
                    "wte.embedding",
                    full_weights["transformer"]["wte"]["embedding"],
                    shard_axis=0,  # Shard along vocab dimension
                )
            },
            "h": {},
            "ln_f": full_weights["transformer"]["ln_f"],  # Replicated
        },
        "lm_head": {
            "kernel": create_sharded_param(
                "lm_head.kernel",
                full_weights["lm_head"]["kernel"],
                shard_axis=1,  # Shard along output dimension (second axis after transpose)
            )
        },
    }

    # Create sharded transformer layers directly from full weights
    for layer_idx in range(n_layers):
        layer_key = str(layer_idx)
        full_layer = full_weights["transformer"]["h"][layer_key]

        jax_params["transformer"]["h"][layer_key] = {
            "attention": {
                "wq": {
                    "kernel": create_sharded_param(
                        f"h.{layer_key}.wq.kernel",
                        full_layer["attention"]["wq"]["kernel"],
                        shard_axis=1,  # Column parallel
                    )
                },
                "wk": {
                    "kernel": create_sharded_param(
                        f"h.{layer_key}.wk.kernel",
                        full_layer["attention"]["wk"]["kernel"],
                        shard_axis=1,  # Column parallel
                    )
                },
                "wv": {
                    "kernel": create_sharded_param(
                        f"h.{layer_key}.wv.kernel",
                        full_layer["attention"]["wv"]["kernel"],
                        shard_axis=1,  # Column parallel
                    )
                },
                "wo": {
                    "kernel": create_sharded_param(
                        f"h.{layer_key}.wo.kernel",
                        full_layer["attention"]["wo"]["kernel"],
                        shard_axis=0,  # Row parallel
                    )
                },
            },
            "feed_forward": {
                "w1": {
                    "kernel": create_sharded_param(
                        f"h.{layer_key}.w1.kernel",
                        full_layer["feed_forward"]["w1"]["kernel"],
                        shard_axis=1,  # Column parallel
                    )
                },
                "w2": {
                    "kernel": create_sharded_param(
                        f"h.{layer_key}.w2.kernel",
                        full_layer["feed_forward"]["w2"]["kernel"],
                        shard_axis=0,  # Row parallel
                    )
                },
                "w3": {
                    "kernel": create_sharded_param(
                        f"h.{layer_key}.w3.kernel",
                        full_layer["feed_forward"]["w3"]["kernel"],
                        shard_axis=1,  # Column parallel
                    )
                },
            },
            "attention_norm": full_layer["attention_norm"],  # Replicated
            "ffn_norm": full_layer["ffn_norm"],  # Replicated
        }

    jax_params = freeze(jax_params)
    model = FlaxLLaMAForCausalLM(config=jax_config, _do_init=False)
    llama = LLaMA(params=jax_params, model=model, tokenizer=tokenizer, mesh=mesh)

    return llama


def main(
    model_id="meta-llama/Meta-Llama-3.1-8B",
    ckpt_dir=str(ROOT / "llama3.1-8B/8B/original"),
    tokenizer_path=str(ROOT / "llama3.1-8B/original/original/tokenizer.model"),
    prompt=("How much is 10 squared?"),
    max_gen_len: int = 5,
    temperature: float = 0.0,
    top_p: float = 1.0,
    n_layers: int = 32,
    max_seq_length: int = 16,
):
    # Define mesh
    devices = jax.devices()
    mesh = Mesh(devices, axis_names=("mp",))
    print("✅ Mesh initialized:", mesh)

    print("🚀 Loading LLaMA...")
    llama = jax_load(
        model_id,
        ckpt_dir,
        tokenizer_path,
        mesh,
        max_seq_length=max_seq_length,
        n_layers=n_layers,
    )

    print("✍️ Generating...")

    with mesh:
        results = llama.generate_from_str(
            [prompt],
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            do_sample=False,
        )
    print("✅ Generation complete.")
    print("🧠 Output:", llama.tokenizer.decode(results[0]))

    if not os.path.isdir("results"):
        os.mkdir("results")

    np.savetxt("results/multi_chip.txt", results, fmt="%d")


if __name__ == "__main__":
    fire.Fire(main)
