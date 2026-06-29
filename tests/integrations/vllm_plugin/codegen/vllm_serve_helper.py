# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Standalone vLLM generation driver for test_codegen_emit_load.

Run in its own process (see the test) so torch_xla's in-process graph cache
doesn't leak between the emit and load phases. Codegen emit vs. load is selected
by the TTXLA_CODEGEN_EXPORT_DIR / TTXLA_CODEGEN_LOAD_DIR env vars the test sets;
this script just runs a tiny generation and writes the result.

Usage: python vllm_serve_helper.py <out_file>
"""

import sys

import vllm


def main(out_file: str) -> None:
    llm = vllm.LLM(
        model="meta-llama/Llama-3.2-3B",
        max_num_batched_tokens=16,
        max_num_seqs=1,
        max_model_len=16,
        gpu_memory_utilization=0.002,
        additional_config={
            "enable_const_eval": False,
            "min_context_len": 16,
            "num_hidden_layers": 1,
        },
    )
    params = vllm.SamplingParams(temperature=0, max_tokens=4)
    outputs = llm.generate(["Hello"], params)
    text = outputs[0].outputs[0].text
    print(f"generated: {text!r}")
    with open(out_file, "w") as f:
        f.write(text)


if __name__ == "__main__":
    main(sys.argv[1])
