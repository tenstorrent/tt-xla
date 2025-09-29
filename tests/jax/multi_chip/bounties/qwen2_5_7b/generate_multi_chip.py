#!/usr/bin/env python3
"""
Multi-device generation script for Qwen2.5-7B TP in JAX.
Imports model from model.py and runs inference on simulated multi-device mesh.
"""

import argparse
import logging
import psutil
import gc
import time
from transformers import AutoTokenizer
import os
import json
"""
Note: JAX and model imports are deferred until after environment flags are applied
to ensure device discovery honors CLI-provided settings.
"""

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("qwen25_generate_multi_chip")

def generate_text(model, params, tokenizer, max_tokens, prompt, show_realtime=True):
    print("Starting text generation...")
    
    # Monitor memory usage
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024**3
    print(f"Initial memory before generation: {initial_memory:.2f} GB used")
    print(f"Free memory: {psutil.virtual_memory().available / 1024**3:.2f} GB")
    
    if show_realtime:
        print("\n=== REAL-TIME GENERATION ===")
        print("(Text will appear as it's generated)")
        print("=" * 50)
    
    # Tokenize input with chat template
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(formatted_text, return_tensors="jax")
    batch, seq = input_ids.shape
    position_ids = jnp.arange(seq, dtype=jnp.int32)[None, :]
    past_key_values = None
    
    generated_tokens = []
    start_time = time.time()
    peak_memory = initial_memory
    
    num_tokens_generated = 0
    print(f"Entering generation loop for {max_tokens} tokens...")
    print("Generating tokens (this may take a while on CPU)...")
    
    for i in range(max_tokens):
        print(f"Generating token {i+1}/{max_tokens}...", end="", flush=True)
        # Create attention mask with proper shape for current sequence
        current_seq_len = input_ids.shape[1]
        key_len = current_seq_len if past_key_values is None or past_key_values[0] is None else past_key_values[0][0].shape[1] + current_seq_len
        attention_mask = jnp.ones((batch, 1, current_seq_len, key_len), dtype=jnp.float32)
        
        # Use model.apply directly since ParallelDense handles tensor parallelism
        outputs = model.apply(params, input_ids=input_ids, attention_mask=attention_mask, 
                             position_ids=position_ids, past_key_values=past_key_values, return_dict=True)
        logits = outputs["logits"]
        past_key_values = outputs["past_key_values"]
        
        next_token = sample_next_token(logits[:, -1, :])
        generated_tokens.append(next_token)
        input_ids = jnp.array([[next_token]])
        position_ids = position_ids[:, -1:] + 1
        num_tokens_generated += 1
        
        # Update peak mem
        current_mem = psutil.virtual_memory().used / (1024**3)
        if current_mem > peak_memory:
            peak_memory = current_mem
        
        # Show the generated token
        token_text = tokenizer.decode(next_token, skip_special_tokens=True)
        print(f" -> '{token_text}'")
        
        if next_token == tokenizer.eos_token_id or "<|im_end|>" in token_text:
            print("Stopping generation: EOS token encountered.")
            break
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_token = total_time / num_tokens_generated if num_tokens_generated > 0 else 0
    
    print(f"Memory after generation: {psutil.virtual_memory().used / (1024**3):.2f} GB used")
    print(f"Peak memory during generation: {peak_memory:.2f} GB used")
    print(f"Free memory: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    print(f"Total tokens generated: {num_tokens_generated}")
    print(f"Average time per token: {avg_time_per_token:.2f} seconds")
    
    full_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print("Generation complete.")
    return full_output, peak_memory, avg_time_per_token

def _apply_env_from_args(args):
    # Optional: set platform to suppress TPU warnings or force backend
    if getattr(args, "platform", None):
        os.environ["JAX_PLATFORMS"] = args.platform
    # Optional: set number of devices via XLA_FLAGS
    if getattr(args, "num_devices", None):
        flags = os.environ.get("XLA_FLAGS", "")
        if f"--xla_force_host_platform_device_count" not in flags:
            flags = (flags + f" --xla_force_host_platform_device_count={args.num_devices}").strip()
        os.environ["XLA_FLAGS"] = flags
    # Optional: use shardy partitioner
    if getattr(args, "use_shardy", False):
        try:
            import jax as _jax
            _jax.config.update("jax_use_shardy_partitioner", True)
        except Exception:
            pass

def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-7B-Instruct JAX Inference with Custom Prompts")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model weights")
    parser.add_argument("--prompt", type=str, default="Question: Sam scores 80 on the first test and 90 on the second. What score does he need on the third test to have an average of 85?", help="Custom prompt to generate text for")
    parser.add_argument("--max_tokens", type=int, default=500, help="Maximum number of tokens to generate")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--no_realtime", action="store_true", help="Disable real-time text display")
    parser.add_argument("--num_devices", type=int, default=None, help="Simulate N devices via XLA_FLAGS")
    parser.add_argument("--platform", type=str, default=None, choices=["cpu", "cuda"], help="Force JAX platform")
    parser.add_argument("--use_shardy", action="store_true", help="Enable Shardy partitioner")
    args = parser.parse_args()
    _apply_env_from_args(args)
    # Defer imports until after env flags are set
    import jax
    import jax.numpy as jnp
    globals()["jnp"] = jnp
    from model import Qwen25ForCausalLM, setup_device_mesh, load_params, sample_next_token, mesh  # noqa: E402
    globals()["sample_next_token"] = sample_next_token

    dtype = jnp.bfloat16 if args.dtype == "bfloat16" else jnp.float32

    # Setup device mesh for tensor parallelism
    global mesh
    mesh = setup_device_mesh()
    
    with open(os.path.join(args.model_path, "config.json")) as f:
        config = json.load(f)
    model = Qwen25ForCausalLM(config=config, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    params = load_params(model, args.model_path, dtype)
    
    print(f"\n{'='*80}")
    print("Custom Prompt Generation:")
    print(f"Prompt: {args.prompt}")
    # Generate with specified max tokens
    show_realtime = not args.no_realtime
    output, peak_mem, avg_time_per_token = generate_text(model, params, tokenizer, args.max_tokens, args.prompt, show_realtime)
    print(f"Output: {output}")
    print(f"Peak memory: {peak_mem:.2f} GB")
    print(f"Avg time per token: {avg_time_per_token:.4f} seconds")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()