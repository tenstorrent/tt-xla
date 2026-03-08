#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Standalone repro: record_metadata_for_reloading causes host memory
regression during vLLM engine initialization with torch_xla backend.

Requirements: pip install vllm torch_xla psutil
Works on any XLA device with a vLLM platform plugin.

Usage (run each in a separate invocation and compare peak RSS):

  /usr/bin/time -v python3 repro.py                  2>&1 | grep "Maximum resident"
  # then reset device if needed
  /usr/bin/time -v python3 repro.py --no-metadata     2>&1 | grep "Maximum resident"

The --no-metadata variant monkey-patches record_metadata_for_reloading
inside the vLLM engine subprocess via a site-customize hook.
"""
import os
import sys
import tempfile

PATCH = "--no-metadata" in sys.argv

if PATCH:
    # Write a sitecustomize.py that patches record_metadata_for_reloading
    # in the engine subprocess. This is the only reliable way since vLLM
    # spawns EngineCore as a separate process.
    hook_dir = tempfile.mkdtemp()
    with open(os.path.join(hook_dir, "sitecustomize.py"), "w") as f:
        f.write(
            """
import importlib
try:
    import vllm.model_executor.model_loader.utils as u
    u.record_metadata_for_reloading = lambda model: None
except Exception:
    pass
"""
        )
    os.environ["PYTHONPATH"] = hook_dir + ":" + os.environ.get("PYTHONPATH", "")
    print("record_metadata_for_reloading: PATCHED via sitecustomize")
else:
    print("record_metadata_for_reloading: DEFAULT (enabled)")


def main():
    import vllm

    print(f"vllm=={vllm.__version__}, model=Qwen/Qwen3-0.6B")

    llm = vllm.LLM(
        model="Qwen/Qwen3-0.6B",
        max_model_len=32,
        max_num_batched_tokens=32,
        max_num_seqs=1,
        gpu_memory_utilization=0.002,
        disable_log_stats=True,
    )

    output = llm.generate(["Hello world"], vllm.SamplingParams(max_tokens=5))
    print(f"Output: {output[0].outputs[0].text!r}")
    print("PASSED")


if __name__ == "__main__":
    main()
