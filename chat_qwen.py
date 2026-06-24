# Scratch: interactive chat with the FULL Qwen3-0.6B on TT via vLLM's offline
# API (no FastAPI/prometheus, so it sidesteps the OpenAI-server middleware bug).
#
#   python chat_qwen.py            # interactive: type a message, Ctrl-D to quit
#   echo "hi" | python chat_qwen.py   # one-shot smoke test
#
# The __main__ guard is required: the TT plugin runs the engine under
# multiprocessing "spawn", which re-imports this module in the worker, so engine
# creation must not run at import time.
import sys

import vllm

MODEL = "Qwen/Qwen3-0.6B"


def main():
    llm = vllm.LLM(
        model=MODEL,
        max_model_len=512,
        max_num_batched_tokens=512,
        max_num_seqs=1,
        gpu_memory_utilization=0.4,
        additional_config={"enable_const_eval": False, "min_context_len": 32},
    )
    params = vllm.SamplingParams(temperature=0.7, max_tokens=200)

    messages = []
    print(">>> chat ready (Ctrl-D to quit) <<<", flush=True)
    for line in sys.stdin:
        user = line.strip()
        if not user:
            continue
        messages.append({"role": "user", "content": user})
        # enable_thinking=False keeps replies concise within the small context.
        out = llm.chat(
            messages, params, chat_template_kwargs={"enable_thinking": False}
        )
        reply = out[0].outputs[0].text
        messages.append({"role": "assistant", "content": reply})
        print(f"\nassistant> {reply}\n", flush=True)


if __name__ == "__main__":
    main()
