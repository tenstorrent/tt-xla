# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Standalone e2e sampling-throughput driver for the sharded-topk work (#4494).

Runs the vLLM benchmark on a TP config with device sampling (temperature>0 ->
composite_topk path). Parameterized by env so the same script works on lb and
galaxy. Must be a real module with an `if __name__ == '__main__'` guard: vLLM v1
spawns the engine core, which re-imports the entry module.

  uv pip install -e integrations/vllm_plugin   # once
  E2E_MODEL=meta-llama/Llama-3.1-70B-Instruct E2E_MESH=4,8 \
      python tests/benchmark/run_sampling_e2e.py

Env knobs:
  E2E_MODEL   HF model id        (default meta-llama/Llama-3.1-8B-Instruct)
  E2E_MESH    "rows,cols"        (default 2,4; galaxy = 4,8)
  E2E_TEMP    sampling temp      (default 1.0; 0.0 = greedy/composite_argmax)
  E2E_BATCH   batch size         (default 32)
  E2E_GMU     gpu_memory_util    (default 0.2)
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "benchmarks"))


def main():
    from benchmarks.vllm_benchmark import benchmark_vllm
    from test_vllm_benchmarks import _tp_config

    model = os.environ.get("E2E_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    mesh = [int(x) for x in os.environ.get("E2E_MESH", "2,4").split(",")]
    temp = float(os.environ.get("E2E_TEMP", "1.0"))
    batch = int(os.environ.get("E2E_BATCH", "32"))
    gmu = float(os.environ.get("E2E_GMU", "0.2"))

    cfg = _tp_config(
        model,
        batch,
        mesh_shape=mesh,
        use_2d_mesh=True,
        enable_const_eval=True,
        experimental_weight_dtype="bfp_bf8",
        gpu_memory_utilization=gmu,
    )
    cfg.temperature = temp
    label = model.split("/")[-1]
    cfg.additional_config["export_path"] = "modules"
    cfg.additional_config["export_model_name"] = label.lower().replace(".", "_")

    print(f"RESULT_BEGIN model={model} mesh={mesh} temp={temp} batch={batch}")
    print(benchmark_vllm(cfg, f"{label}-t{temp}-{'x'.join(map(str, mesh))}"))
    print("RESULT_END")


if __name__ == "__main__":
    main()
