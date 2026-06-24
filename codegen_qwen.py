# Scratch: emit TTNN Python codegen for the FULL Qwen3-0.6B via a vLLM run.
#
# Setting TTXLA_CODEGEN_EXPORT_DIR makes the TT plugin emit generated Python per
# compiled graph into hash-keyed subdirs under the export dir (and run it live
# via PythonModelRunner). Reload the (optionally edited) code later by pointing
# TTXLA_CODEGEN_LOAD_DIR at the same dir.
#
#   python codegen_qwen.py [export_dir]   # default ./qwen_codegen
#
# The env var must be set before importing vllm and must survive the engine's
# "spawn" worker, which re-imports this module. We read it from the environment
# first so the inherited worker reuses the parent's value (and never touches
# sys.argv, which spawn does not preserve).
import os
import sys
from pathlib import Path

EXPORT_DIR = os.environ.get("TTXLA_CODEGEN_EXPORT_DIR")
if EXPORT_DIR is None:
    EXPORT_DIR = str(
        Path(sys.argv[1] if len(sys.argv) > 1 else "qwen_codegen").resolve()
    )
    os.environ["TTXLA_CODEGEN_EXPORT_DIR"] = EXPORT_DIR

import vllm


def main():
    export_dir = Path(EXPORT_DIR)
    print(f"emitting codegen to {export_dir}", flush=True)

    llm = vllm.LLM(
        model="Qwen/Qwen3-0.6B",
        max_model_len=512,
        max_num_batched_tokens=512,
        max_num_seqs=1,
        gpu_memory_utilization=0.4,
        additional_config={"enable_const_eval": False, "min_context_len": 32},
    )
    params = vllm.SamplingParams(temperature=0, max_tokens=16)
    out = llm.generate(["Hello"], params)
    print("generated:", repr(out[0].outputs[0].text), flush=True)

    print(f"\ncodegen emitted under {export_dir}:")
    if export_dir.is_dir():
        for d in sorted(export_dir.iterdir()):
            main_py = d / "main.py"
            if main_py.exists():
                print(f"  {d.name}/main.py  ({main_py.stat().st_size} bytes)")
        manifest = export_dir / "manifest.json"
        if manifest.exists():
            n = len([l for l in manifest.read_text().splitlines() if l.strip()])
            print(f"  manifest.json  ({n} graphs)")
    else:
        print("  (nothing emitted — check the log above for errors)")


if __name__ == "__main__":
    main()
