# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""vLLM startup and inference benchmark.

Measures time spent in engine creation, first generate (compilation),
steady-state generate, and teardown.  Useful for tracking regressions
and understanding where wall-clock time goes.

Usage:
    python benchmark_vllm_startup.py                        # default: opt_125m
    python benchmark_vllm_startup.py --preset opt_125m llama_3b
    python benchmark_vllm_startup.py --all
    python benchmark_vllm_startup.py --all --json results.json
"""

import argparse
import gc
import json
import socket
import subprocess
import sys
import time
from datetime import datetime

import vllm

# ---------------------------------------------------------------------------
# Model presets — mirrored from test_sampling_params.py fixtures
# ---------------------------------------------------------------------------

MODEL_PRESETS = {
    "opt_125m": {
        "model": "facebook/opt-125m",
        "max_num_batched_tokens": 128,
        "max_num_seqs": 1,
        "max_model_len": 128,
        "gpu_memory_utilization": 0.001,
        "enable_prefix_caching": False,
        "disable_log_stats": True,
        "enforce_eager": True,
        "additional_config": {
            "enable_const_eval": False,
            "min_context_len": 32,
        },
    },
    "llama_3b": {
        "model": "meta-llama/Llama-3.2-3B",
        "max_num_batched_tokens": 128,
        "max_num_seqs": 1,
        "max_model_len": 128,
        "gpu_memory_utilization": 0.002,
        "additional_config": {
            "enable_const_eval": False,
            "min_context_len": 32,
            "enable_tensor_parallel": True,
        },
    },
    "qwen3_0_6b": {
        "model": "Qwen/Qwen3-0.6B",
        "max_num_batched_tokens": 128,
        "max_num_seqs": 1,
        "max_model_len": 128,
        "gpu_memory_utilization": 0.002,
        "additional_config": {
            "enable_const_eval": False,
            "min_context_len": 32,
            "enable_tensor_parallel": True,
        },
    },
}

PROMPT = "Once upon a time, there was a"


# ---------------------------------------------------------------------------
# Core benchmark
# ---------------------------------------------------------------------------


def benchmark_single_model(preset_name, num_steady_state=3):
    """Run the full timing loop for one model preset. Returns a results dict."""
    cfg = MODEL_PRESETS[preset_name]
    results = {"preset": preset_name, "model": cfg["model"]}

    # --- Engine create ---
    t0 = time.perf_counter()
    llm = vllm.LLM(**cfg)
    results["engine_create_s"] = time.perf_counter() - t0

    sampling_params = vllm.SamplingParams(max_tokens=32, temperature=0)

    # --- First generate (includes lazy compilation if enforce_eager) ---
    t0 = time.perf_counter()
    llm.generate([PROMPT], sampling_params)
    results["first_generate_s"] = time.perf_counter() - t0

    # --- Steady-state generates ---
    steady_times = []
    for _ in range(num_steady_state):
        t0 = time.perf_counter()
        llm.generate([PROMPT], sampling_params)
        steady_times.append(time.perf_counter() - t0)
    results["steady_state_mean_s"] = sum(steady_times) / len(steady_times)
    results["steady_state_times_s"] = steady_times

    # --- Compilation overhead estimate ---
    results["compile_overhead_s"] = max(
        0.0, results["first_generate_s"] - results["steady_state_mean_s"]
    )

    # --- Teardown ---
    t0 = time.perf_counter()
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception:
        pass
    del llm
    gc.collect()
    results["teardown_s"] = time.perf_counter() - t0

    # --- Total ---
    results["total_s"] = (
        results["engine_create_s"]
        + results["first_generate_s"]
        + sum(steady_times)
        + results["teardown_s"]
    )

    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _fmt(seconds):
    return f"{seconds:.1f}s"


def print_results_table(results_list):
    """Print a formatted table of benchmark results."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    host = socket.gethostname()

    sep = "=" * 100
    print(sep)
    print(f"vLLM Startup Benchmark — {now} — {host}")
    print(sep)

    header = (
        f"{'Preset':<16}"
        f"{'Model':<28}"
        f"{'Create':>8}"
        f"{'1st Gen':>9}"
        f"{'Steady':>8}"
        f"{'Compile':>9}"
        f"{'Teardown':>10}"
        f"{'Total':>8}"
    )
    print(header)
    print("-" * 100)

    for r in results_list:
        row = (
            f"{r['preset']:<16}"
            f"{r['model']:<28}"
            f"{_fmt(r['engine_create_s']):>8}"
            f"{_fmt(r['first_generate_s']):>9}"
            f"{_fmt(r['steady_state_mean_s']):>8}"
            f"{_fmt(r['compile_overhead_s']):>9}"
            f"{_fmt(r['teardown_s']):>10}"
            f"{_fmt(r['total_s']):>8}"
        )
        print(row)

    print(sep)


def write_json_report(results_list, path):
    """Write benchmark results to a JSON file."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "hostname": socket.gethostname(),
        "results": results_list,
    }
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"JSON report written to {path}")


# ---------------------------------------------------------------------------
# Multi-model subprocess isolation
# ---------------------------------------------------------------------------


def _run_preset_in_subprocess(preset_name):
    """Re-invoke this script for a single preset, capturing JSON from stdout.

    Stderr (vLLM logs, progress bars) streams through to the terminal in
    real time so the user can see progress during long runs.
    """
    proc = subprocess.Popen(
        [sys.executable, __file__, "--preset", preset_name, "--_json_stdout"],
        stdout=subprocess.PIPE,
        stderr=None,  # inherit — streams to terminal in real time
        text=True,
    )
    stdout, _ = proc.communicate()

    if proc.returncode != 0:
        print(f"  FAILED: {preset_name}", file=sys.stderr)
        return None

    # The child prints JSON as the last line of stdout.  Any vLLM logging
    # goes to stderr (streamed above), but some may leak to stdout — find
    # the JSON line.
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    print(f"  FAILED: {preset_name} — no JSON in stdout", file=sys.stderr)
    if stdout:
        print(stdout, file=sys.stderr)
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM startup and inference timing."
    )
    parser.add_argument(
        "--preset",
        nargs="+",
        choices=list(MODEL_PRESETS.keys()),
        help="One or more model presets to benchmark.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Benchmark all presets.",
    )
    parser.add_argument(
        "--json",
        metavar="PATH",
        help="Write results to a JSON file.",
    )
    parser.add_argument(
        "--steady-state-runs",
        type=int,
        default=3,
        help="Number of steady-state generate calls (default: 3).",
    )
    # Internal flag: emit JSON to stdout and suppress the table.
    parser.add_argument("--_json_stdout", action="store_true", help=argparse.SUPPRESS)

    args = parser.parse_args()

    # Determine presets to run.
    if args.all:
        presets = list(MODEL_PRESETS.keys())
    elif args.preset:
        presets = args.preset
    else:
        presets = ["opt_125m"]

    # Single preset, or internal subprocess call: run in-process.
    if len(presets) == 1 or args._json_stdout:
        if args._json_stdout:
            # Subprocess mode — run the single preset and emit JSON.
            r = benchmark_single_model(presets[0], args.steady_state_runs)
            print(json.dumps(r))
            return

        r = benchmark_single_model(presets[0], args.steady_state_runs)
        print_results_table([r])
        if args.json:
            write_json_report([r], args.json)
        return

    # Multiple presets: use subprocess isolation between models.
    all_results = []
    for name in presets:
        print(f"Benchmarking {name} ...")
        r = _run_preset_in_subprocess(name)
        if r is not None:
            all_results.append(r)

    if all_results:
        print_results_table(all_results)
        if args.json:
            write_json_report(all_results, args.json)
    else:
        print("No benchmarks succeeded.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
