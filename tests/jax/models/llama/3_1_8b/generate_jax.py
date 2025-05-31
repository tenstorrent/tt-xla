import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
import jax
import jax.numpy as jnp
import numpy as np
import fire
from flax.core.frozen_dict import freeze
from jax_llama import FlaxLLaMAForCausalLM, convert_llama_weights
from jax_llama.llama3_tokenizer import Tokenizer as LLaMA3Tokenizer
from jax_llama.generation import LLaMA  # your class is here
from jax.sharding import Mesh
from flax.traverse_util import flatten_dict
import jax.debug
from jax_llama.partition import get_llama_param_partition_spec
from jax.experimental import pjit as old_pjit
import gc

def jax_load(ckpt_dir: str, tokenizer_path: str, mesh, max_seq_length: int = 2048) -> LLaMA:
    print("🔧 Loading tokenizer and weights...")
    tokenizer = LLaMA3Tokenizer(tokenizer_path)

    params_np, jax_config = convert_llama_weights(
        ckpt_dir=ckpt_dir,
        tokenizer=tokenizer,
        max_seq_len=max_seq_length
    )
    jax_params = freeze(jax.tree.map(jnp.asarray, params_np))

    del params_np
    gc.collect()    

    param_spec = get_llama_param_partition_spec(jax_params)
    shard_fn = old_pjit.pjit(
        lambda x: x,
        None,
        param_spec
    )

    with mesh:
        jax_params = shard_fn(jax_params)


    model = FlaxLLaMAForCausalLM(config=jax_config, _do_init=False)
    llama = LLaMA(params=jax_params, model=model, tokenizer=tokenizer, mesh=mesh)

    return llama

def main(
    ckpt_dir: str = "/root/tt/sw/llama3.1-8B/8B",
    tokenizer_path: str = "/root/tt/sw/llama3.1-8B/original/tokenizer.model",
    prompt: str = (
    "Q: Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?\n"
    "A: The cost of the house and repairs came out to 80,000 + 50,000 = $<<80000+50000=130000>>130,000\n"
    "He increased the value of the house by 80,000 * 1.5 = <<80000*1.5=120000>>120,000\n"
    "So the new value of the house is 120,000 + 80,000 = $<<120000+80000=200000>>200,000\n"
    "So he made a profit of 200,000 - 130,000 = $<<200000-130000=70000>>70,000\n"
    "F: #### 70000\n\n"
    "Q: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?\n"
    "A:"
),
    max_gen_len: int = 16,
    temperature: float = 0.1,
    top_p: float = 0.99
):
    # Define mesh
    devices = np.array(jax.devices()).reshape(2, 4)
    mesh = Mesh(devices, axis_names=("dp", "mp"))
    print("✅ Mesh initialized:", mesh)

    print("🚀 Loading LLaMA...")
    llama = jax_load(ckpt_dir, tokenizer_path, mesh=mesh)

    print("\n🔍 Visualizing sharded parameter placements (first few):")
    flat_params = flatten_dict(llama.params)
    for k, v in list(flat_params.items())[:5]:
        print(f"🔹 Param {k} sharding:")
        jax.debug.visualize_array_sharding(v)


    print("✍️ Generating...")
    with mesh:
        results = llama.generate_from_str(
            [prompt],
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
    for i, r in enumerate(results):
        print(f"\n🧾 Prompt {i + 1}: {prompt}")
        print("🧠 Output:", r)

if __name__ == "__main__":
    fire.Fire(main)
