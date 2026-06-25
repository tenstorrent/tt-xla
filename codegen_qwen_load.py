# Scratch: interactively run the FULL Qwen3-0.6B from previously emitted codegen
# instead of compiling. Point TTXLA_CODEGEN_LOAD_DIR at a dir produced by
# codegen_qwen.py; the TT plugin matches each graph by StableHLO hash against the
# saved dirs and runs the (optionally edited) main.py via PythonModelRunner --
# skipping SHLO->TTIR->TTNN compilation entirely.
#
#   python codegen_qwen_load.py [load_dir]   # default ./qwen_codegen
#   then type a prompt per line; Ctrl-D to quit.
#
# Matching is by a hash of the StableHLO text, which is shape-specialized. The
# emit run used a short prompt (padded to min_context_len=32) + max_tokens=16,
# so keep prompts short and params identical here; a prompt/length that produces
# a graph shape the emit run never compiled will fail with "no saved graph with
# hash" (re-emit to capture it).
#
# The LLM(...) config below MUST match codegen_qwen.py exactly. Same spawn/env
# caveat: read the env var first so the re-imported worker reuses it and never
# touches sys.argv.
import os
import sys
from pathlib import Path

LOAD_DIR = os.environ.get("TTXLA_CODEGEN_LOAD_DIR")
if LOAD_DIR is None:
    LOAD_DIR = str(
        Path(sys.argv[1] if len(sys.argv) > 1 else "qwen_codegen").resolve()
    )
    os.environ["TTXLA_CODEGEN_LOAD_DIR"] = LOAD_DIR

import vllm


def main():
    load_dir = Path(LOAD_DIR)
    if not load_dir.is_dir():
        sys.exit(f"load dir {load_dir} does not exist -- run codegen_qwen.py first")
    print(f"loading codegen from {load_dir}", flush=True)

    llm = vllm.LLM(
        model="Qwen/Qwen3-0.6B",
        max_model_len=512,
        max_num_batched_tokens=512,
        max_num_seqs=1,
        gpu_memory_utilization=0.02,
        additional_config={"enable_const_eval": False, "min_context_len": 32},
    )
    params = vllm.SamplingParams(temperature=0, max_tokens=16)

    print(">>> load-mode ready (type a prompt, Ctrl-D to quit) <<<", flush=True)
    for line in sys.stdin:
        prompt = line.strip()
        if not prompt:
            continue
        out = llm.generate([prompt], params)
        print(f"\n{out[0].outputs[0].text}\n", flush=True)


if __name__ == "__main__":
    main()
