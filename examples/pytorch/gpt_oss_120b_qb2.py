# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import time
from typing import List

import numpy as np
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from transformers.cache_utils import StaticCache
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils.quantization_config import Mxfp4Config
from tt_torch.sparse_mlp import A2aSparseMLP, enable_sparse_mlp
from tt_torch.weight_dtype import apply_weight_dtype_overrides

DEFAULT_PROMPT = "Explain quantum mechanics."


def gpt_oss_120b(
    interactive: bool = False,
    sparse_moe: bool = False,
    batch_size: int = 8,
    verbose: bool = False,
    perf_metrics: bool = False,
):

    max_cache_len: int = 256
    model_name: str = "openai/gpt-oss-120b"

    setup_spmd()

    device: torch.device = torch_xla.device()
    mesh, mesh_shape = create_device_mesh()

    if verbose:
        print("Setting up model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(model_name, verbose=verbose)

    if sparse_moe:
        if verbose:
            print("Enabling sparse MoE implementation...")
        enable_sparse_mlp(model, mesh=mesh_shape, cluster_axis=1)

    while True:
        if interactive:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ("quit()", "exit", "quit"):
                break
            if not user_input:
                user_input = DEFAULT_PROMPT
            user_prompt = [user_input] * batch_size
        else:
            user_prompt = [DEFAULT_PROMPT] * batch_size

        if verbose:
            print("Constructing inputs...")
        input_args, formatted_prompts = construct_inputs(
            user_prompt, tokenizer, model.config, batch_size, max_cache_len, sparse_moe
        )

        max_tokens_to_generate: int = max_cache_len - input_args["input_ids"].shape[1]

        if verbose:
            print("Moving model and inputs to device...")
        model, input_args = transfer_to_device(model, input_args, device)

        mark_sharding_on_inputs_and_model(model, input_args, mesh, sparse_moe)
        compiled_model = torch.compile(model, backend="tt")

        if verbose:
            print("Begin generation...")
        run_generate(
            compiled_model,
            input_args,
            tokenizer,
            device,
            mesh,
            max_tokens_to_generate,
            formatted_prompts,
            interactive,
            verbose=verbose,
            perf_metrics=perf_metrics,
        )

        if not interactive:
            break


def setup_spmd():
    """Initializes SPMD mode, compilation options, and persistent cache."""
    print("Setting up XLA environment...")

    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()
    torch_xla.set_custom_compile_options({"experimental_weight_dtype": "bfp_bf8"})

    cache_dir = os.path.expanduser("~/.cache/tt_xla/gpt_oss_120b")
    xr.initialize_cache(cache_dir)
    cache_files = list(os.scandir(cache_dir)) if os.path.isdir(cache_dir) else []
    cache_status = (
        f"WARM ({len(cache_files)} entries)" if cache_files else "COLD (will compile)"
    )
    print(f"Compilation cache: {cache_dir} [{cache_status}]")
    print("XLA environment configured.")


def create_device_mesh() -> tuple[Mesh, tuple]:
    """Create device mesh for tensor parallelism."""
    num_devices = xr.global_runtime_device_count()

    if num_devices == 32:  # Galaxy
        mesh_shape = (4, 8)
    elif num_devices == 8:  # llmbox
        mesh_shape = (2, 4)
    elif num_devices == 4:  # QB2 (2x p300)
        mesh_shape = (1, 4)
    else:
        raise RuntimeError(
            f"Gpt-oss-120b requires 4, 8, or 32 devices (got {num_devices})"
        )

    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))
    print(f"Created device mesh: {mesh_shape} with {num_devices} devices")
    return mesh, mesh_shape


def setup_model_and_tokenizer(
    model_name: str,
    verbose: bool = False,
) -> tuple[torch.nn.Module, PreTrainedTokenizer]:
    """Instantiate model and tokenizer with MXFP4 quantization and mixed precision."""
    quantization_config = Mxfp4Config(dequantize=True)

    model: torch.nn.Module = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model = model.eval()

    applied = apply_weight_dtype_overrides(
        model,
        {
            "default": "bfp_bf8",
            "model.layers.*.mlp.experts.gate_up_proj": "bfp_bf4",
            "model.layers.*.mlp.experts.down_proj": "bfp_bf4",
        },
    )
    if verbose:
        print(f"Applied {len(applied)} weight dtype overrides")

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def construct_inputs(
    input_prompt: List[str],
    tokenizer: PreTrainedTokenizer,
    model_config: PretrainedConfig,
    batch_size: int,
    max_cache_len: int,
    sparse_moe: bool = False,
) -> tuple[dict, List[str]]:
    """Construct tokenized inputs with static KV cache for generation."""
    formatted_prompts = []
    for prompt in input_prompt:
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        formatted_prompts.append(formatted_prompt)

    prompt_lengths = [
        len(tokenizer.encode(p, add_special_tokens=False)) for p in formatted_prompts
    ]
    actual_max = max(prompt_lengths)

    # Fixed pad length ensures all prefills hit the same cached compilation graph.
    PREFILL_PAD_LEN = 96
    if actual_max > PREFILL_PAD_LEN:
        raise ValueError(
            f"Prompt is {actual_max} tokens, exceeds PREFILL_PAD_LEN={PREFILL_PAD_LEN}. "
            f"Increase PREFILL_PAD_LEN in construct_inputs."
        )
    max_length = PREFILL_PAD_LEN

    inputs = tokenizer(
        formatted_prompts,
        return_tensors="pt",
        max_length=max_length,
        padding="max_length",
        padding_side="left",
        return_attention_mask=True,
    )

    cache_config = disable_sliding_window_attention(model_config)
    static_cache: StaticCache = StaticCache(
        config=cache_config,
        max_batch_size=batch_size,
        max_cache_len=max_cache_len,
        device="cpu",
        dtype=torch.bfloat16,
    )
    num_key_value_heads = model_config.num_key_value_heads
    head_dim = model_config.head_dim
    static_cache.early_initialization(
        batch_size=batch_size,
        num_heads=num_key_value_heads,
        head_dim=head_dim,
        dtype=torch.bfloat16,
        device="cpu",
    )

    cache_position: torch.Tensor = torch.arange(0, inputs.input_ids.shape[1])

    prompt_len = inputs.input_ids.shape[1]
    full_attention_mask = torch.ones(
        (batch_size, max_cache_len), dtype=inputs.attention_mask.dtype
    )
    full_attention_mask[:, :prompt_len] = inputs.attention_mask

    input_args = {
        "input_ids": inputs.input_ids,
        "past_key_values": static_cache,
        "cache_position": cache_position,
        "use_cache": True,
        "attention_mask": full_attention_mask,
    }

    print(f"Prompt tokens: {inputs.input_ids.shape[1]} (batch {batch_size})")

    return input_args, formatted_prompts


def disable_sliding_window_attention(
    config: PretrainedConfig,
) -> PretrainedConfig:
    """Force all layers to full_attention to avoid sliding window recompilation."""
    config.layer_types = ["full_attention"] * config.num_hidden_layers
    return config


def transfer_to_device(
    model: torch.nn.Module, input_args: dict, device: torch.device
) -> tuple[torch.nn.Module, dict]:
    """Transfer model and inputs to device."""
    for layer in input_args["past_key_values"].layers:
        layer.keys = layer.keys.to(device)
        layer.values = layer.values.to(device)
    input_args["input_ids"] = input_args["input_ids"].to(device)
    input_args["cache_position"] = input_args["cache_position"].to(device)
    input_args["attention_mask"] = input_args["attention_mask"].to(device)

    model = model.to(device)

    return model, input_args


def mark_sharding_on_inputs_and_model(
    model: torch.nn.Module, input_args: dict, mesh: Mesh, sparse_moe: bool = False
):
    """Mark SPMD sharding on model weights and KV cache for tensor parallelism."""
    for layer in input_args["past_key_values"].layers:
        xs.mark_sharding(layer.keys, mesh, (None, "model", None, None))
        xs.mark_sharding(layer.values, mesh, (None, "model", None, None))

    xs.mark_sharding(model.model.embed_tokens.weight, mesh, (None, None))
    xs.mark_sharding(model.model.norm.weight, mesh, (None,))

    for layer in model.model.layers:
        xs.mark_sharding(layer.self_attn.q_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.self_attn.k_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.self_attn.v_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.self_attn.o_proj.weight, mesh, (None, "model"))
        xs.mark_sharding(layer.self_attn.sinks, mesh, (None,))

        if sparse_moe and isinstance(layer.mlp, A2aSparseMLP):
            expert_e_shard = (("batch", "model"), None, None)
            expert_e_bias_shard = (("batch", "model"), None)
        else:
            expert_e_shard = ("model", None, None)
            expert_e_bias_shard = ("model", None)
        xs.mark_sharding(layer.mlp.experts.gate_up_proj, mesh, expert_e_shard)
        xs.mark_sharding(layer.mlp.experts.gate_up_proj_bias, mesh, expert_e_bias_shard)
        xs.mark_sharding(layer.mlp.experts.down_proj, mesh, expert_e_shard)
        xs.mark_sharding(layer.mlp.experts.down_proj_bias, mesh, expert_e_bias_shard)


def run_generate(
    compiled_model: torch.nn.Module,
    input_args: dict,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    mesh: Mesh = None,
    max_tokens_to_generate: int = 128,
    formatted_prompts: List[str] = [""],
    is_interactive: bool = False,
    verbose: bool = False,
    perf_metrics: bool = False,
):
    """Run the autoregressive generation loop with optional interactive streaming."""
    num_users = input_args["input_ids"].shape[0]
    output_tokens: List[List[str]] = [[] for _ in range(num_users)]

    compile_time_prefill = 0.0
    compile_time_decode = 0.0
    decode_times: List[float] = []

    # GPT-OSS output uses <|channel|>name<|message|>content<|end|> format.
    # Only the "final" channel is printed; "analysis" and "commentary" are skipped.
    _END_TOKENS = {"<|end|>", "<|return|>"}
    stream_state = "seek_channel"
    channel_name_buf = ""

    with torch.no_grad():
        for step in range(max_tokens_to_generate):
            if step == 0 and verbose:
                print("Compiling and running prefill...", flush=True)

            step_start = time.perf_counter()

            output: CausalLMOutputWithPast = compiled_model(**input_args)
            output_logits: torch.Tensor = output.logits.to("cpu")

            step_elapsed = time.perf_counter() - step_start

            if step == 0:
                compile_time_prefill = step_elapsed
                if verbose:
                    print(
                        f"Prefill done ({compile_time_prefill:.1f}s). Compiling decode..."
                    )
            elif step == 1:
                compile_time_decode = step_elapsed
            else:
                decode_times.append(step_elapsed)

            next_token_id = output_logits[:, -1].argmax(dim=-1)
            output_text = [tokenizer.decode(next_token_id[i]) for i in range(num_users)]
            for i, output_tokens_list in enumerate(output_tokens):
                output_tokens_list.append(output_text[i])
            if is_interactive:
                # Stream user 0's response, filtering to only the "final" channel
                tok = output_text[0]
                if stream_state == "seek_channel":
                    if tok == "<|channel|>":
                        stream_state = "seek_message"
                        channel_name_buf = ""
                elif stream_state == "seek_message":
                    if tok == "<|message|>":
                        if channel_name_buf.strip() == "final":
                            stream_state = "printing"
                            print("\nGPT: ", end="", flush=True)
                        else:
                            stream_state = "seek_channel"
                    elif tok in _END_TOKENS or tok == "<|start|>":
                        stream_state = "seek_channel"
                    else:
                        channel_name_buf += tok
                elif stream_state == "printing":
                    if tok in _END_TOKENS:
                        stream_state = "done"
                        print()  # newline after response
                    else:
                        print(tok, end="", flush=True)

            if torch.all(next_token_id == tokenizer.eos_token_id):
                if is_interactive:
                    print()  # newline after streamed response
                break

            input_args["input_ids"] = next_token_id.unsqueeze(-1).to(device)

            host_cache_pos = input_args["cache_position"].to("cpu")
            host_cache_pos = torch.tensor([host_cache_pos[-1:] + 1])
            input_args["cache_position"] = host_cache_pos.to(device)

    if is_interactive and stream_state != "done":
        if stream_state == "printing":
            print()  # newline after incomplete response
        else:
            print(
                "\nGPT: [No response generated — model did not produce a 'final' channel]"
            )
    if perf_metrics:
        print()
        print("=" * 50)
        print("PERFORMANCE SUMMARY")
        print(f"  Prefill (incl. compile):      {compile_time_prefill:.2f}s")
        print(f"  First decode (incl. compile): {compile_time_decode:.2f}s")
        if decode_times:
            avg_decode = sum(decode_times) / len(decode_times)
            print(f"  Steady-state decode steps:    {len(decode_times)}")
            print(f"  Avg decode latency:           {avg_decode * 1000:.1f}ms")
            print(
                f"  Throughput (all users):       {num_users / avg_decode:.1f} tokens/s"
            )
            print(f"  Throughput (per user):        {1 / avg_decode:.1f} tokens/s")
        print("=" * 50)
    if is_interactive and verbose:
        print("\n--- RAW TOKEN STREAM (user 0, unfiltered) ---")
        print("".join(output_tokens[0]))
        print("--- END RAW TOKEN STREAM ---")
    if not is_interactive:
        for i in range(num_users):
            print(f"=" * 80)
            print(f"Result for user {i}:")
            print(f"-" * 80)
            print("PROMPT:")
            print(formatted_prompts[i])
            print(f"-" * 80)
            print("GENERATED:")
            print("".join(output_tokens[i]))
            print(f"=" * 80)
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT-OSS 120B generation example")
    parser.add_argument(
        "--interactive",
        action="store_true",
        default=False,
        help="Enable interactive mode for entering custom prompts",
    )
    parser.add_argument(
        "--sparse-moe",
        action="store_true",
        default=False,
        help="Use sparse MoE implementation (A2aSparseMLP) instead of dense expert computation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help=(
            "Number of prompts to process in one batch (default: 8). "
            "When using --sparse-moe, batch_size must be a multiple of 8 because "
            "sparse MoE requires batch_size * num_devices_along_dispact_axis (4) to be a multiple of 32."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print detailed signpost messages, compilation status, and raw token stream",
    )
    parser.add_argument(
        "--perf-metrics",
        action="store_true",
        default=False,
        help="Print performance summary (prefill/decode latency, throughput) after each generation",
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error(
            f"--batch-size must be a positive integer (got {args.batch_size})."
        )

    if args.sparse_moe and args.batch_size % 8 != 0:
        parser.error(
            f"--batch-size must be a multiple of 8 when using --sparse-moe (got {args.batch_size}). "
            "Sparse MoE requires batch_size * device_on_axis (4) to be a multiple of 32."
        )

    xr.set_device_type("TT")

    gpt_oss_120b(
        interactive=args.interactive,
        sparse_moe=args.sparse_moe,
        batch_size=args.batch_size,
        verbose=args.verbose,
        perf_metrics=args.perf_metrics,
    )
