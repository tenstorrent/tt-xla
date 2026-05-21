# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Quick TP validation using opt-125m (no HuggingFace token needed).

Tests both use_2d_mesh=True and use_2d_mesh=False to exercise the
mark_sharding paths added in the decode_postprocess fuse PR.
"""
import re
import sys

import vllm

MODEL = "tiiuae/Falcon3-7B-Instruct"
PROMPT = "I like taking walks in the"
MAX_TOKENS = 32

_STOPWORDS = frozenset(
    "the a an and or but i you he she it we they is are was were be been "
    "have has had do does did of to in on at for with by from as that this "
    "my your her his their can will would should like go get make me so "
    "not if when what how there here".split()
)
_WORD_RE = re.compile(r"[A-Za-z']+")


def assert_output_coherent(text: str, label: str) -> None:
    words = [w.lower() for w in _WORD_RE.findall(text)]
    assert words, f"[{label}] no word characters in output: {text!r}"
    if len(words) >= 5:
        sr = sum(1 for w in words if w in _STOPWORDS) / len(words)
        assert (
            sr >= 0.10
        ), f"[{label}] token-soup output (stopword ratio {sr:.3f}): {text!r}"
    print(f"[{label}] OK — output: {text!r}")


def run(use_2d_mesh: bool) -> None:
    label = f"use_2d_mesh={use_2d_mesh}"
    print(f"\n{'='*60}")
    print(f"Testing TP with {label}")
    print(f"{'='*60}")

    llm = vllm.LLM(
        model=MODEL,
        max_num_batched_tokens=32,
        max_num_seqs=1,
        max_model_len=32,
        gpu_memory_utilization=0.002,
        additional_config={
            "enable_const_eval": False,
            "min_context_len": 32,
            "enable_tensor_parallel": True,
            "use_2d_mesh": use_2d_mesh,
        },
    )

    sampling_params = vllm.SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)
    outputs = llm.generate([PROMPT], sampling_params)
    text = outputs[0].outputs[0].text
    assert_output_coherent(text, label)

    # Clean up before next test
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception:
        pass
    del llm


if __name__ == "__main__":
    failures = []
    for use_2d_mesh in [True, False]:
        try:
            run(use_2d_mesh)
        except Exception as e:
            print(f"[use_2d_mesh={use_2d_mesh}] FAILED: {e}", file=sys.stderr)
            failures.append((use_2d_mesh, e))

    print("\n" + "=" * 60)
    if failures:
        print(f"FAILED: {len(failures)} case(s)")
        for mesh, err in failures:
            print(f"  use_2d_mesh={mesh}: {err}")
        sys.exit(1)
    else:
        print("ALL PASSED")
