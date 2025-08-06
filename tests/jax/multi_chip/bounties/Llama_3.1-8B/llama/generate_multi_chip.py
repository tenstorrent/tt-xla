# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
import jax
import jax.numpy as jnp
import numpy as np
import fire
from flax.core.frozen_dict import freeze
from model import FlaxLLaMAForCausalLM
from convert_weights import convert_llama_weights
from transformers import AutoTokenizer
from generation import LLaMA
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map
import gc
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
    print("🔧 Loading tokenizer and weights...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Get tensor parallel size from mesh
    tensor_parallel_size = mesh.shape["mp"]
    print(f"🔧 Tensor parallel size: {tensor_parallel_size}")

    # STAGE 1: Load full model ONCE and slice into parts
    print("🔧 Stage 1: Loading full model and creating sharded parts...")

    # Load full model (tp_rank=0 gives us the full model before sharding)
    print("🔧 Loading full model...")
    full_weights, jax_config = convert_llama_weights(
        ckpt_dir=ckpt_dir,
        tokenizer=tokenizer,
        max_seq_len=max_seq_length,
        n_layers=n_layers,
        tensor_parallel_size=1,  # tp_size=1 = full model
        tp_rank=0,
        verbose=True,
    )

    print("🔧 Slicing full model into 4 tensor parallel parts...")
    all_rank_weights = {}

    for tp_rank in range(tensor_parallel_size):
        print(f"🔧 Creating part {tp_rank}/{tensor_parallel_size}...")

        # Calculate slice indices for this rank
        heads_per_rank = jax_config.num_attention_heads // tensor_parallel_size
        kv_heads_per_rank = jax_config.num_key_value_heads // tensor_parallel_size
        head_dim = jax_config.hidden_size // jax_config.num_attention_heads
        inter_per_rank = jax_config.intermediate_size // tensor_parallel_size
        vocab_per_rank = jax_config.vocab_size // tensor_parallel_size

        q_start = tp_rank * heads_per_rank * head_dim
        q_end = (tp_rank + 1) * heads_per_rank * head_dim
        kv_start = tp_rank * kv_heads_per_rank * head_dim
        kv_end = (tp_rank + 1) * kv_heads_per_rank * head_dim
        inter_start = tp_rank * inter_per_rank
        inter_end = (tp_rank + 1) * inter_per_rank
        vocab_start = tp_rank * vocab_per_rank
        vocab_end = (tp_rank + 1) * vocab_per_rank

        # Slice the full weights to create this rank's part
        rank_weights = {
            "transformer": {
                "wte": {
                    "embedding": full_weights["transformer"]["wte"]["embedding"][
                        vocab_start:vocab_end, :
                    ]
                },
                "h": {},
                "ln_f": full_weights["transformer"]["ln_f"],  # Replicated
            },
            "lm_head": {
                "kernel": full_weights["lm_head"]["kernel"][
                    :, vocab_start:vocab_end
                ]  # Note: transposed in convert_weights
            },
        }

        # Slice transformer layers
        for layer_idx in range(n_layers):
            layer_key = str(layer_idx)
            full_layer = full_weights["transformer"]["h"][layer_key]

            rank_weights["transformer"]["h"][layer_key] = {
                "attention": {
                    "wq": {
                        "kernel": full_layer["attention"]["wq"]["kernel"][
                            :, q_start:q_end
                        ]
                    },
                    "wk": {
                        "kernel": full_layer["attention"]["wk"]["kernel"][
                            :, kv_start:kv_end
                        ]
                    },
                    "wv": {
                        "kernel": full_layer["attention"]["wv"]["kernel"][
                            :, kv_start:kv_end
                        ]
                    },
                    "wo": {
                        "kernel": full_layer["attention"]["wo"]["kernel"][
                            q_start:q_end, :
                        ]
                    },
                },
                "feed_forward": {
                    "w1": {
                        "kernel": full_layer["feed_forward"]["w1"]["kernel"][
                            :, inter_start:inter_end
                        ]
                    },
                    "w2": {
                        "kernel": full_layer["feed_forward"]["w2"]["kernel"][
                            inter_start:inter_end, :
                        ]
                    },
                    "w3": {
                        "kernel": full_layer["feed_forward"]["w3"]["kernel"][
                            :, inter_start:inter_end
                        ]
                    },
                },
                "attention_norm": full_layer["attention_norm"],  # Replicated
                "ffn_norm": full_layer["ffn_norm"],  # Replicated
            }

        all_rank_weights[tp_rank] = rank_weights

        # Debug: Print shapes for this rank
        wq_shape = rank_weights["transformer"]["h"]["0"]["attention"]["wq"][
            "kernel"
        ].shape
        lm_head_shape = rank_weights["lm_head"]["kernel"].shape
        wte_shape = rank_weights["transformer"]["wte"]["embedding"].shape
        print(
            f"   🔍 Part {tp_rank}: wq={wq_shape}, lm_head={lm_head_shape}, wte={wte_shape}"
        )
        print(f"   ✅ Part {tp_rank} created")

    # Clean up full model from memory
    del full_weights
    gc.collect()
    print(f"🔧 Created {len(all_rank_weights)} tensor parallel parts")

    # STAGE 2: Create properly sharded parameter tree for JAX
    print("🔧 Stage 2: Creating sharded parameter tree...")

    # Get devices from mesh
    devices = mesh.devices.flatten()
    print(f"🔧 Available devices: {devices}")

    # Create a single parameter tree with properly sharded arrays
    # Each parameter will be an array that spans multiple devices
    print("🔧 Creating sharded arrays...")

    def create_sharded_param(param_name, param_parts, shard_axis):
        """Create a sharded array from parts distributed across devices"""
        # Concatenate the parts along the sharding axis to form the full array
        if shard_axis == 0:  # Shard along first axis (e.g., vocab dimension)
            full_array = jnp.concatenate(
                [jnp.asarray(part) for part in param_parts], axis=0
            )
            sharding_spec = P("mp", None)  # Shard along first axis
        elif shard_axis == 1:  # Shard along second axis (e.g., output features)
            full_array = jnp.concatenate(
                [jnp.asarray(part) for part in param_parts], axis=1
            )
            sharding_spec = P(None, "mp")  # Shard along second axis
        else:  # Replicated
            full_array = jnp.asarray(param_parts[0])  # Just take first copy
            sharding_spec = P(None)

        sharded_array = jax.device_put(
            full_array, jax.sharding.NamedSharding(mesh, sharding_spec)
        )

        print(
            f"   📊 {param_name}: parts={[p.shape for p in param_parts]} -> {sharded_array.shape} (sharded on axis {shard_axis})"
        )
        print(f"       Sharding: {sharding_spec}")
        return sharded_array

    # Build the sharded parameter tree
    jax_params = {
        "transformer": {
            "wte": {
                "embedding": create_sharded_param(
                    "wte.embedding",
                    [
                        all_rank_weights[i]["transformer"]["wte"]["embedding"]
                        for i in range(tensor_parallel_size)
                    ],
                    shard_axis=0,  # Shard along vocab dimension
                )
            },
            "h": {},
            "ln_f": all_rank_weights[0]["transformer"]["ln_f"],  # Replicated
        },
        "lm_head": {
            "kernel": create_sharded_param(
                "lm_head.kernel",
                [
                    all_rank_weights[i]["lm_head"]["kernel"]
                    for i in range(tensor_parallel_size)
                ],
                shard_axis=1,  # Shard along output dimension (second axis after transpose)
            )
        },
    }

    # Create sharded transformer layers
    for layer_idx in range(n_layers):
        layer_key = str(layer_idx)
        jax_params["transformer"]["h"][layer_key] = {
            "attention": {
                "wq": {
                    "kernel": create_sharded_param(
                        f"h.{layer_key}.wq.kernel",
                        [
                            all_rank_weights[i]["transformer"]["h"][layer_key][
                                "attention"
                            ]["wq"]["kernel"]
                            for i in range(tensor_parallel_size)
                        ],
                        shard_axis=1,  # Column parallel
                    )
                },
                "wk": {
                    "kernel": create_sharded_param(
                        f"h.{layer_key}.wk.kernel",
                        [
                            all_rank_weights[i]["transformer"]["h"][layer_key][
                                "attention"
                            ]["wk"]["kernel"]
                            for i in range(tensor_parallel_size)
                        ],
                        shard_axis=1,  # Column parallel
                    )
                },
                "wv": {
                    "kernel": create_sharded_param(
                        f"h.{layer_key}.wv.kernel",
                        [
                            all_rank_weights[i]["transformer"]["h"][layer_key][
                                "attention"
                            ]["wv"]["kernel"]
                            for i in range(tensor_parallel_size)
                        ],
                        shard_axis=1,  # Column parallel
                    )
                },
                "wo": {
                    "kernel": create_sharded_param(
                        f"h.{layer_key}.wo.kernel",
                        [
                            all_rank_weights[i]["transformer"]["h"][layer_key][
                                "attention"
                            ]["wo"]["kernel"]
                            for i in range(tensor_parallel_size)
                        ],
                        shard_axis=0,  # Row parallel
                    )
                },
            },
            "feed_forward": {
                "w1": {
                    "kernel": create_sharded_param(
                        f"h.{layer_key}.w1.kernel",
                        [
                            all_rank_weights[i]["transformer"]["h"][layer_key][
                                "feed_forward"
                            ]["w1"]["kernel"]
                            for i in range(tensor_parallel_size)
                        ],
                        shard_axis=1,  # Column parallel
                    )
                },
                "w2": {
                    "kernel": create_sharded_param(
                        f"h.{layer_key}.w2.kernel",
                        [
                            all_rank_weights[i]["transformer"]["h"][layer_key][
                                "feed_forward"
                            ]["w2"]["kernel"]
                            for i in range(tensor_parallel_size)
                        ],
                        shard_axis=0,  # Row parallel
                    )
                },
                "w3": {
                    "kernel": create_sharded_param(
                        f"h.{layer_key}.w3.kernel",
                        [
                            all_rank_weights[i]["transformer"]["h"][layer_key][
                                "feed_forward"
                            ]["w3"]["kernel"]
                            for i in range(tensor_parallel_size)
                        ],
                        shard_axis=1,  # Column parallel
                    )
                },
            },
            "attention_norm": all_rank_weights[0]["transformer"]["h"][layer_key][
                "attention_norm"
            ],  # Replicated
            "ffn_norm": all_rank_weights[0]["transformer"]["h"][layer_key][
                "ffn_norm"
            ],  # Replicated
        }

    jax_params = freeze(jax_params)

    # Clean up CPU memory
    del all_rank_weights
    gc.collect()

    print("🔧 Creating model...")
    model = FlaxLLaMAForCausalLM(config=jax_config, _do_init=False)
    llama = LLaMA(params=jax_params, model=model, tokenizer=tokenizer, mesh=mesh)

    del jax_params
    gc.collect()

    print("✅ Model loaded with optimal tensor parallel sharding!")
    return llama


def main(
    model_id="meta-llama/Meta-Llama-3.1-8B",
    ckpt_dir=str(ROOT / "llama3.1-8B/8B/original"),
    tokenizer_path=str(ROOT / "llama3.1-8B/original/original/tokenizer.model"),
    prompt=("How much is 10 squared?"),
    max_gen_len: int = 64,
    temperature: float = 0.0,
    top_p: float = 1.0,
    n_layers: int = 32,
    max_seq_length: int = 128,
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
