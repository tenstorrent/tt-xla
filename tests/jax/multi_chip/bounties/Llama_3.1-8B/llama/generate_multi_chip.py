# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
import gc
from pathlib import Path

import fire
import jax
import jax.numpy as jnp
import numpy as np
from convert_weights import convert_llama_weights
from flax.core.frozen_dict import freeze
from generation import LLaMA
from jax.sharding import Mesh
from model import FlaxLLaMAForCausalLM
from transformers import AutoTokenizer

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

    params_np, jax_config = convert_llama_weights(
        ckpt_dir=ckpt_dir,
        tokenizer=tokenizer,
        max_seq_len=max_seq_length,
        n_layers=n_layers,
    )
    jax_params = freeze(jax.tree.map(jnp.asarray, params_np))

    del params_np
    gc.collect()

    model = FlaxLLaMAForCausalLM(config=jax_config, _do_init=False)
    llama = LLaMA(params=jax_params, model=model, tokenizer=tokenizer, mesh=mesh)
    del jax_params
    gc.collect()
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
