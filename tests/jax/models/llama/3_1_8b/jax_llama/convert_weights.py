from typing import Union
from pathlib import Path
import torch
import json
import numpy as np
from jaxtyping import PyTree
from jax_llama.config import LLaMAConfig
from jax_llama.llama3_tokenizer import Tokenizer as LLaMA3Tokenizer
from typing import Tuple, Optional
from dataclasses import dataclass
import os
import psutil

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000.0

    max_batch_size: int = 1
    max_seq_len: int = 2048

def config_from_params(args: ModelArgs) -> LLaMAConfig:
    intermediate_size = int(2 * (args.dim * 4) / 3)
    if args.ffn_dim_multiplier is not None:
        intermediate_size = int(args.ffn_dim_multiplier * intermediate_size)
    intermediate_size = args.multiple_of * ((intermediate_size + args.multiple_of - 1) // args.multiple_of)
    return LLaMAConfig(
        vocab_size=args.vocab_size, 
        hidden_size=args.dim, 
        intermediate_size=intermediate_size, 
        num_hidden_layers=args.n_layers, 
        num_attention_heads=args.n_heads, 
        num_key_value_heads=args.n_kv_heads,
        max_sequence_length=args.max_seq_len, 
        rms_norm_eps=args.norm_eps, 
        rope_theta=args.rope_theta
    )

def convert_llama_weights(ckpt_dir: str, tokenizer: LLaMA3Tokenizer, max_seq_len: int=2048, verbose: bool=False) -> Tuple[PyTree[np.ndarray], LLaMAConfig]:
    ckpt_paths = sorted(Path(ckpt_dir).glob("*.pth"))
    ckpts = {}
    for i, ckpt_path in enumerate(ckpt_paths):
        if verbose:
            print(f"Loading checkpoint {i+1} of {len(ckpt_paths)} ...")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if verbose:
            print('Loaded.')
        ckpts[int(ckpt_path.name.split('.', maxsplit=2)[1])] = checkpoint
    ckpts = [ckpts[i] for i in sorted(list(ckpts.keys()))]
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())
    params.pop("use_scaled_rope", None) 

    jax_weights = {
        'transformer': {
            'wte': {'embedding': np.concatenate([ckpt['tok_embeddings.weight'].type(torch.float16).numpy() for ckpt in ckpts], axis=1)}, 
            'ln_f': {'kernel': ckpts[0]['norm.weight'].type(torch.float16).numpy()}, 
            'h': {
                '%d' % (layer): {
                    'attention': {
                        'wq': {'kernel': np.concatenate([ckpt['layers.%d.attention.wq.weight' % (layer)].type(torch.float16).numpy() for ckpt in ckpts], axis=0).transpose()}, 
                        'wk': {'kernel': np.concatenate([ckpt['layers.%d.attention.wk.weight' % (layer)].type(torch.float16).numpy() for ckpt in ckpts], axis=0).transpose()}, 
                        'wv': {'kernel': np.concatenate([ckpt['layers.%d.attention.wv.weight' % (layer)].type(torch.float16).numpy() for ckpt in ckpts], axis=0).transpose()}, 
                        'wo': {'kernel': np.concatenate([ckpt['layers.%d.attention.wo.weight' % (layer)].type(torch.float16).numpy() for ckpt in ckpts], axis=1).transpose()}, 
                    }, 
                    'feed_forward': {
                        'w1': {'kernel': np.concatenate([ckpt['layers.%d.feed_forward.w1.weight' % (layer)].type(torch.float16).numpy() for ckpt in ckpts], axis=0).transpose()}, 
                        'w2': {'kernel': np.concatenate([ckpt['layers.%d.feed_forward.w2.weight' % (layer)].type(torch.float16).numpy() for ckpt in ckpts], axis=1).transpose()}, 
                        'w3': {'kernel': np.concatenate([ckpt['layers.%d.feed_forward.w3.weight' % (layer)].type(torch.float16).numpy() for ckpt in ckpts], axis=0).transpose()}, 
                    }, 
                    'attention_norm': {'kernel': ckpts[0]['layers.%d.attention_norm.weight' % (layer)].type(torch.float16).numpy()}, 
                    'ffn_norm': {'kernel': ckpts[0]['layers.%d.ffn_norm.weight' % (layer)].type(torch.float16).numpy()}, 
                }
            for layer in range(params['n_layers'])}, 
        }, 
        'lm_head': {'kernel': np.concatenate([ckpt['output.weight'].type(torch.float16).numpy() for ckpt in ckpts], axis=0).transpose()}, 
    }
    
    params.update({'vocab_size': len(tokenizer), 'max_seq_len': max_seq_len})
    llama_config = config_from_params(ModelArgs(**params))

    del ckpts
    torch.cuda.empty_cache()  # Not necessary on CPU, but okay
    import gc; gc.collect()


    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)

    print(f"\n✅ Peak memory usage before return from converting: {mem_mb:.2f} MB")


    return jax_weights, llama_config


def convert_state_dict_keys(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("tok_embeddings.weight"):
            new_key = "model.embed_tokens.weight"
        elif key.startswith("output.weight"):
            new_key = "lm_head.weight"
        elif key.startswith("norm.weight"):
            new_key = "model.norm.weight"
        elif key.startswith("layers."):
            parts = key.split('.')
            layer_num = parts[1]
            sublayer = parts[2]
            weight_type = parts[3]

            if sublayer == "attention":
                if weight_type == "wq":
                    new_key = f"model.layers.{layer_num}.self_attn.q_proj.weight"
                elif weight_type == "wk":
                    new_key = f"model.layers.{layer_num}.self_attn.k_proj.weight"
                elif weight_type == "wv":
                    new_key = f"model.layers.{layer_num}.self_attn.v_proj.weight"
                elif weight_type == "wo":
                    new_key = f"model.layers.{layer_num}.self_attn.o_proj.weight"
            elif sublayer == "feed_forward":
                if weight_type == "w1":
                    new_key = f"model.layers.{layer_num}.mlp.gate_proj.weight"
                elif weight_type == "w2":
                    new_key = f"model.layers.{layer_num}.mlp.down_proj.weight"
                elif weight_type == "w3":
                    new_key = f"model.layers.{layer_num}.mlp.up_proj.weight"
            elif sublayer == "attention_norm":
                new_key = f"model.layers.{layer_num}.input_layernorm.weight"
            elif sublayer == "ffn_norm":
                new_key = f"model.layers.{layer_num}.post_attention_layernorm.weight"
            else:
                print(f"⚠️ Unknown sublayer: {key}")
                continue
        else:
            print(f"⚠️ Unknown key: {key}")
            continue

        new_state_dict[new_key] = value
    return new_state_dict