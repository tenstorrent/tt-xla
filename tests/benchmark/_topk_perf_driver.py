# Local-only driver to compare the new sharded-topk sampling path against the
# legacy full-vocab all_gather path on a single build, via the
# TT_LEGACY_REPLICATE_LOGITS env toggle in model_runner.py. Not for CI.
import json
import os
import sys

from benchmarks.vllm_benchmark import VLLMBenchmarkConfig, benchmark_vllm


def _find(obj, key):
    if isinstance(obj, dict):
        if obj.get("measurement_name") == key and "value" in obj:
            return obj["value"]
        for v in obj.values():
            r = _find(v, key)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _find(v, key)
            if r is not None:
                return r
    return None


def main():
    model = os.environ.get("DRIVER_MODEL", "tiiuae/Falcon3-1B-Base")
    batch = int(os.environ.get("TT_BENCHMARK_BATCH_SIZE", "1"))
    legacy = os.environ.get("TT_LEGACY_REPLICATE_LOGITS") == "1"
    out = os.environ.get("DRIVER_OUT", "/localdev/akhan/topk_perf_result.json")

    # Mirror _tp_config() defaults from test_vllm_benchmarks.py for an
    # apples-to-apples TP run (2D mesh, device sampling, trace on).
    cfg = VLLMBenchmarkConfig(
        model=model,
        batch_size=batch,
        max_model_len=128,
        gpu_memory_utilization=0.05,
        temperature=0.0,  # greedy: the k=1 path the regression is about
        additional_config={
            "enable_tensor_parallel": True,
            "use_2d_mesh": True,
            "min_context_len": 32,
            "enable_trace": True,
        },
    )

    name = f"vllm_{model.split('/')[-1].lower()}_tp"
    results = benchmark_vllm(cfg, name)

    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)

    sps = _find(results, "samples_per_sec")
    ttft = _find(results, "ttft")
    line = (
        f"TOPK_PERF_RESULT model={model} batch={batch} "
        f"legacy={int(legacy)} samples_per_sec={sps} ttft_ms={ttft}"
    )
    print("\n" + line, file=sys.stderr)
    print(line)


if __name__ == "__main__":
    main()
