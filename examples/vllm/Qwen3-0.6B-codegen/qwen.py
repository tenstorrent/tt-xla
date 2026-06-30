# Emit OR load TTNN Python codegen for the full Qwen3-0.6B via vLLM.
#
#   python qwen.py --emit [dir]   # compile + emit per-graph Python to dir/
#   python qwen.py --load [dir]   # skip compile, run saved code from dir/;
#                                 # then type a prompt per line, Ctrl-D to quit
#   (dir defaults to ./qwen_codegen)
#
# --emit sets TTXLA_CODEGEN_EXPORT_DIR: the TT plugin emits generated Python per
#   compiled graph into hash-keyed subdirs (warmup precompiles every shape).
# --load sets TTXLA_CODEGEN_LOAD_DIR: the plugin matches each graph by StableHLO
#   hash against those subdirs and runs the (optionally edited) main.py via
#   PythonModelRunner, skipping SHLO->TTIR->TTNN compilation entirely. A graph
#   shape that emit never captured fails with "no saved graph with hash".
#
# The mode/dir must be decided BEFORE importing vllm AND must survive the
# engine's "spawn" worker, which re-imports this module without inheriting
# sys.argv (but DOES inherit env vars). So we parse argv only in the parent, set
# the matching env var, and let the worker recover the mode from that env var.
import argparse
import os
import sys
from pathlib import Path

_EMIT_DIR = os.environ.get("TTXLA_CODEGEN_EXPORT_DIR")
_LOAD_DIR = os.environ.get("TTXLA_CODEGEN_LOAD_DIR")

# Parent process: neither env var is set yet -> parse args and set one. The
# spawn worker re-enters here with the env var already set and skips this block
# (and never touches sys.argv, which spawn does not preserve).
if _EMIT_DIR is None and _LOAD_DIR is None:
    parser = argparse.ArgumentParser(
        description="Emit or load TTNN codegen for Qwen3-0.6B via vLLM."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--emit", action="store_true", help="compile and emit per-graph Python to DIR"
    )
    mode.add_argument(
        "--load", action="store_true", help="run previously emitted code from DIR"
    )
    parser.add_argument(
        "dir", nargs="?", default="qwen_codegen", help="codegen dir (default: ./qwen_codegen)"
    )
    args = parser.parse_args()
    resolved = str(Path(args.dir).resolve())
    if args.emit:
        _EMIT_DIR = os.environ["TTXLA_CODEGEN_EXPORT_DIR"] = resolved
        os.environ["XLA_HLO_DEBUG"] = "1"  # proper location data in the emitted IRs
    else:
        _LOAD_DIR = os.environ["TTXLA_CODEGEN_LOAD_DIR"] = resolved

EMIT = _EMIT_DIR is not None
CODEGEN_DIR = Path(_EMIT_DIR if EMIT else _LOAD_DIR)

import vllm


def build_llm():
    # This config MUST be identical between emit and load so the StableHLO (and
    # thus each graph's hash) matches; unifying both modes here enforces that.
    #
    # Additionally, must match params in service.sh
    return vllm.LLM(
        model="Qwen/Qwen3-0.6B",
        max_model_len=4096,
        max_num_batched_tokens=4096,
        max_num_seqs=1,
        gpu_memory_utilization=0.02,
        additional_config={"enable_const_eval": False, "min_context_len": 256},
    )


def run_emit():
    print(f"emitting codegen to {CODEGEN_DIR}", flush=True)
    build_llm()  # warmup precompiles every bucket -> emits all graphs

    print(f"\ncodegen emitted under {CODEGEN_DIR}:")
    if CODEGEN_DIR.is_dir():
        for d in sorted(CODEGEN_DIR.iterdir()):
            main_py = d / "main.py"
            if main_py.exists():
                print(f"  {d.name}/main.py  ({main_py.stat().st_size} bytes)")
        manifest = CODEGEN_DIR / "manifest.json"
        if manifest.exists():
            n = len([l for l in manifest.read_text().splitlines() if l.strip()])
            print(f"  manifest.json  ({n} graphs)")
    else:
        print("  (nothing emitted — check the log above for errors)")


def run_load():
    if not CODEGEN_DIR.is_dir():
        sys.exit(f"load dir {CODEGEN_DIR} does not exist -- run with --emit first")
    print(f"loading codegen from {CODEGEN_DIR}", flush=True)
    llm = build_llm()
    params = vllm.SamplingParams(temperature=0, max_tokens=512)

    print(">>> load-mode ready (Ctrl-D to quit) <<<", flush=True)
    # vLLM keeps no conversation state, so we maintain the history ourselves:
    # each turn appends the user message, sends the WHOLE list, then appends the
    # reply so the next turn has context. With prefix caching on, the shared
    # prefix is reused so only the new suffix actually prefills. Type "reset" to
    # clear the history (and drop back to the small prefill bucket).
    messages = []
    while True:
        try:
            prompt = input(">>> ").strip()
        except EOFError:
            break
        if not prompt:
            continue
        if prompt == "reset":
            messages = []
            print("(history cleared)", flush=True)
            continue
        messages.append({"role": "user", "content": prompt})
        # chat() applies Qwen3's chat template so the model answers instead of
        # just continuing the text; enable_thinking=False keeps it concise.
        out = llm.chat(
            messages,
            params,
            chat_template_kwargs={"enable_thinking": False},
        )
        reply = out[0].outputs[0].text
        messages.append({"role": "assistant", "content": reply})
        print(f"\n{reply}\n", flush=True)


def main():
    run_emit() if EMIT else run_load()


if __name__ == "__main__":
    main()
