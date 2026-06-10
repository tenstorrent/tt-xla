# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Emit TTNN Python codegen during a vLLM run (TTXLA_CODEGEN_EXPORT_DIR), edit
the emitted code, then reload it instead of compiling (TTXLA_CODEGEN_LOAD_DIR).
Each phase runs in its own process since torch_xla caches graphs in-process."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

VLLM_SCRIPT = textwrap.dedent(
    """
    import sys
    import vllm

    out_file = sys.argv[1]

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
    open(out_file, "w").write(text)
    """
)


def run_vllm(tmp_path, env_extra, name):
    out_file = tmp_path / f"{name}.txt"
    env = {**os.environ, **env_extra}
    result = subprocess.run(
        [sys.executable, "-c", VLLM_SCRIPT, str(out_file)],
        env=env,
        capture_output=True,
        text=True,
        timeout=3000,
    )
    return result, out_file


def insert_sentinel(main_py: Path, sentinel: Path):
    lines = main_py.read_text().splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.startswith("def forward("):
            lines.insert(i + 1, f'    open(r"{sentinel}", "a").write("hit")\n')
            break
    else:
        raise AssertionError(f"no forward() in {main_py}")
    main_py.write_text("".join(lines))


@pytest.mark.nightly
@pytest.mark.single_device
def test_vllm_codegen_emit_then_load(tmp_path):
    export_dir = tmp_path / "emitted"

    result, emit_out = run_vllm(
        tmp_path, {"TTXLA_CODEGEN_EXPORT_DIR": str(export_dir)}, "emit"
    )
    assert result.returncode == 0, result.stderr[-4000:]
    assert emit_out.exists()

    graph_dirs = sorted(d for d in export_dir.iterdir() if (d / "module_key").exists())
    assert len(graph_dirs) >= 3, f"expected several graphs, got {graph_dirs}"
    assert (export_dir / "manifest.json").exists()

    sentinels = []
    for d in graph_dirs:
        sentinel = tmp_path / f"sentinel_{d.name}"
        insert_sentinel(d / "main.py", sentinel)
        sentinels.append(sentinel)

    result, load_out = run_vllm(
        tmp_path, {"TTXLA_CODEGEN_LOAD_DIR": str(export_dir)}, "load"
    )
    assert result.returncode == 0, result.stderr[-4000:]

    executed = [s.name for s in sentinels if s.exists()]
    assert executed, "no edited graph was executed in load mode"
    assert load_out.read_text() == emit_out.read_text()
