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

ROOT = Path(__file__).parent.parent


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

    # STAGE 1: Pre-load all ranks' sharded weights into CPU memory
    print("🔧 Stage 1: Loading sharded weights for all ranks...")
    all_rank_weights = {}
    jax_config = None

    for tp_rank in range(tensor_parallel_size):
        print(f"🔧 Loading sharded weights for rank {tp_rank}...")
        weights, config = convert_llama_weights(
            ckpt_dir=ckpt_dir,
            tokenizer=tokenizer,
            max_seq_len=max_seq_length,
            n_layers=n_layers,
            tensor_parallel_size=tensor_parallel_size,
            tp_rank=tp_rank,
            verbose=True,
        )
        all_rank_weights[tp_rank] = weights
        if jax_config is None:
            jax_config = config
        gc.collect()  # Clean up after each rank

    print(f"🔧 Pre-loaded {len(all_rank_weights)} ranks into CPU memory")

    # STAGE 2: Distribute sharded weights to devices using shard_map
    print("🔧 Stage 2: Distributing weights to devices...")

    def select_rank_weights():
        """Each device selects its corresponding rank's pre-sharded weights"""
        my_rank = jax.lax.axis_index("mp")  # Get device's logical index (0,1,2,3)

        # Use jax.lax.switch for tracer-safe selection
        return jax.lax.switch(
            my_rank, [lambda: all_rank_weights[i] for i in range(tensor_parallel_size)]
        )

    # Distribute weights to devices
    params_np = shard_map(
        select_rank_weights,
        mesh=mesh,
        in_specs=(),  # No inputs
        out_specs=P(),  # Output is replicated (each device gets its own weights)
        check_rep=False,
    )()

    # Convert to JAX arrays
    jax_params = freeze(jax.tree.map(jnp.asarray, params_np))

    # Clean up CPU memory
    del all_rank_weights, params_np
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
    prompt=("What is the name of the largest planet in our solar system?"),
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
