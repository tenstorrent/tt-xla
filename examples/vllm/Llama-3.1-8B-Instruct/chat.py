# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Interactive single-process chat with Llama-3.1-8B-Instruct using vllm.LLM.

No server required — runs the engine directly in this process.

Usage:
    python examples/vllm/Llama-3.1-8B-Instruct/chat.py
"""

import time

import vllm

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
MAX_MODEL_LEN = 2048
MAX_TOKENS = 256
GPU_MEMORY_UTILIZATION = 0.05


def create_engine():
    additional_config = {
        "enable_const_eval": False,
        "min_context_len": 32,
    }
    print(f"Loading {MODEL} ...")
    llm = vllm.LLM(
        model=MODEL,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_MODEL_LEN,
        max_num_seqs=1,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        disable_log_stats=True,
        additional_config=additional_config,
    )
    print("Engine ready.\n")
    return llm


def warmup(llm):
    print("Warming up ...")
    sampling_params = vllm.SamplingParams(max_tokens=16, temperature=0.0)
    llm.generate(["Hello"], sampling_params)
    print("Warmup complete.\n")


def main():
    llm = create_engine()
    warmup(llm)

    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    tokenizer = llm.get_tokenizer()

    while True:
        user_input = input("Enter a message (or 'q' to quit): ")
        if user_input.strip().lower() == "q":
            break

        messages.append({"role": "user", "content": user_input})
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        sampling_params = vllm.SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0)

        start = time.perf_counter()
        outputs = llm.generate([prompt], sampling_params)
        elapsed = time.perf_counter() - start

        output = outputs[0]
        text = output.outputs[0].text
        num_tokens = len(output.outputs[0].token_ids)
        prompt_tokens = len(output.prompt_token_ids)
        tok_s = num_tokens / elapsed if elapsed > 0 else 0

        print(text)
        print(
            f"[{num_tokens} tokens, {prompt_tokens} prompt tokens, "
            f"{tok_s:.1f} tok/s, {elapsed:.2f}s]"
        )

        messages.append({"role": "assistant", "content": text})


if __name__ == "__main__":
    main()
