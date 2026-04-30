# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SmolLM3-3B vLLM server launch script for Tenstorrent hardware.

Applies necessary workarounds for SmolLM3's NoPE (no positional embedding) layers:
- Forces RoPE on all layers to produce uniform graph shapes for TTNN
- Disables torch.compile to avoid SymInt issues from NoPE pattern

The SmolLM3 NoPE workaround is installed via a .pth file in site-packages
(smollm3_tt_patch.py) so it applies in both parent and EngineCore subprocess.

Prerequisites:
    1. Install the patch: cp smollm3_tt_patch.py <venv>/lib/python3.11/site-packages/
    2. Create .pth:      echo "import smollm3_tt_patch" > <venv>/lib/.../smollm3_tt.pth

Usage:
    cd /path/to/tt-xla && source venv/activate
    TORCH_COMPILE_DISABLE=1 python examples/vllm/SmolLM3-3B/serve_tt.py [--port 8000]

Status:
    - SmolLM3 model loads and runs forward pass on TT hardware (verified)
    - NoPE layer workaround applied (RoPE forced on all layers)
    - Blocked by: tt-mlir 'ttir.paged_fill_cache' op not yet legalized for SmolLM3's
      GQA config (4 KV heads, head_dim 128). Direct forward passes work; vLLM paged
      attention KV cache ops need tt-mlir support.
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="SmolLM3-3B on TT hardware via vLLM")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--max-model-len", type=int, default=1024)
    args = parser.parse_args()

    # Ensure env vars propagate to subprocesses
    os.environ["TORCH_COMPILE_DISABLE"] = "1"

    # Set up vLLM CLI args
    sys.argv = [
        "vllm_server",
        "--model", "HuggingFaceTB/SmolLM3-3B",
        "--max-model-len", str(args.max_model_len),
        "--max-num-batched-tokens", str(args.max_model_len),
        "--max-num-seqs", "1",
        "--no-enable-prefix-caching",
        "--gpu-memory-utilization", "0.1",
        "--dtype", "bfloat16",
        "--host", args.host,
        "--port", str(args.port),
        "--additional-config",
        '{"enable_const_eval": false, "min_context_len": 32, "enable_precompile_all": false}',
    ]

    # Launch the server
    import runpy
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")


if __name__ == "__main__":
    main()
