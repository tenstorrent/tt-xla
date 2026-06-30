# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Binary search for the maximum number of least-sensitive weights that can be
cast to bfp_bf4 while keeping TOP1 p5 accuracy above a threshold.

Runs three baselines before the search:
 - all weights at bfp_bf8
 - all MLP weights at bfp_bf4
 - all weights at bfp_bf4

Usage:
  python lowering_search.py \\
    --model meta-llama/Llama-3.2-1B-Instruct \\
    --test tests/benchmark/test_llms.py::test_llama_3_2_1b \\
    --threshold 90
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass

from transformers import AutoModelForCausalLM


@dataclass
class SearchConfig:
    output_file: str
    results_file: str
    log_dir: str
    test: str
    threshold: float
    save_logs: bool = False


EXPERIMENTS_DIR = "mixed_precision_experiments"


def get_scores_path(model_name):
    model_short = model_name.split("/")[-1]
    return os.path.join(
        EXPERIMENTS_DIR,
        "sensitivity_scores",
        model_short,
        f"sensitivity_{model_short}.json",
    )


def get_output_path(model_name):
    model_short = model_name.split("/")[-1]
    matches = glob.glob(
        f"third_party/tt_forge_models/**/mixed_precision_configs/{model_short}.json",
        recursive=True,
    )
    if matches:
        return matches[0]
    return os.path.join(EXPERIMENTS_DIR, "configs", f"{model_short}.json")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Binary search for mixed-precision weight selection."
    )
    parser.add_argument(
        "--model", required=True, help="HuggingFace model name or local path"
    )
    parser.add_argument(
        "--scores",
        default=None,
        help="Path to sensitivity scores JSON (default: derived from --model)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to mixed precision config JSON to write (default: derived from --model)",
    )
    parser.add_argument(
        "--test",
        required=True,
        help="Pytest target, e.g. tests/benchmark/test_llms.py::test_llama_3_2_1b",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="Minimum TOP1 p5 accuracy percentage",
    )
    parser.add_argument("--results", default=None, help="Output markdown results file")
    parser.add_argument(
        "--save-logs", action="store_true", help="Save pytest output logs per iteration"
    )
    return parser.parse_args()


def load_param_sizes(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="meta")
    return {name: param.numel() for name, param in model.named_parameters()}


def write_config(config_path, override_weights):
    config = {"default": "bfp_bf8"}
    for w in override_weights:
        config[w] = "bfp_bf4"
    os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def run_test(test, log_file=None):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(os.getcwd(), "tests")
    cmd = [sys.executable, "-m", "pytest", test, "--accuracy-testing", "-s"]
    result = subprocess.run(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    if log_file is not None:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        with open(log_file, "w") as f:
            f.write(result.stdout)
    return parse_top1_p5(result.stdout)


def parse_top1_p5(output):
    match = re.search(r"TOP1.*?p5=([\d.]+)%", output)
    if match:
        return float(match.group(1))
    return None


# --- Markdown writers ---


def write_header(results_file, model, test, threshold, n):
    with open(results_file, "w") as f:
        f.write(f"# Mixed Precision Search: {model}\n\n")
        f.write("| | |\n|---|---|\n")
        f.write(f"| **Test** | `{test}` |\n")
        f.write(f"| **Threshold** | {threshold}% TOP1 p5 |\n")
        f.write(f"| **Total weights** | {n} |\n\n")


def write_baselines_header(results_file):
    with open(results_file, "a") as f:
        f.write("## Baselines\n\n")
        f.write("| Config | Weights at bfp_bf4 | TOP1 p5 |\n")
        f.write("|--------|--------------------|----------|\n")


def append_baseline(results_file, label, n_lower, n_total, top1_p5):
    with open(results_file, "a") as f:
        pct = n_lower / n_total * 100
        top1_str = f"{top1_p5:.2f}%" if top1_p5 is not None else "N/A"
        f.write(f"| {label} | {n_lower}/{n_total} ({pct:.1f}%) | {top1_str} |\n")


def write_final(results_file, best_k, n, top1_at_best, bfp4_numel, total_numel):
    n_default = n - best_k
    estimated_saving_pct = bfp4_numel / (2 * total_numel) * 100
    top1_str = f"{top1_at_best:.2f}%" if top1_at_best is not None else "N/A"
    with open(results_file, "a") as f:
        f.write("\n## Final Config\n\n")
        f.write(f"**Best k = {best_k}** — accuracy TOP1 p5={top1_str}\n\n")
        f.write("| Dtype | Weights | Share |\n")
        f.write("|-------|---------|-------|\n")
        f.write(f"| `bfp_bf8` | {n_default} | {n_default / n * 100:.1f}% |\n")
        f.write(f"| `bfp_bf4` | {best_k} | {best_k / n * 100:.1f}% |\n\n")
        f.write(
            f"**Estimated weight memory vs all-bfp_bf8 baseline:** ~{estimated_saving_pct:.1f}% smaller\n"
        )


def run_baselines(weights, mlp_weights, n, config):
    baselines = [
        ("all bfp_bf8", []),
        ("all MLP bfp_bf4", mlp_weights),
        ("all bfp_bf4", weights),
    ]
    write_baselines_header(config.results_file)
    print("Running baselines...")
    last_top1 = None
    for label, bw in baselines:
        write_config(config.output_file, bw)
        log_file = (
            os.path.join(config.log_dir, f"baseline_{label.replace(' ', '_')}.log")
            if config.save_logs
            else None
        )
        top1_p5 = run_test(config.test, log_file)
        top1_str = f"{top1_p5:.2f}%" if top1_p5 is not None else "N/A"
        print(f"  {label}: TOP1 p5={top1_str}")
        append_baseline(config.results_file, label, len(bw), n, top1_p5)
        last_top1 = top1_p5
    return last_top1


def binary_search(weights, sizes, n, total_numel, config, threshold):
    print("\nRunning binary search...")
    lo, hi = 0, n - 1
    best_k = 0
    top1_at_best = None
    iteration = 0

    while lo <= hi:
        iteration += 1
        k = (lo + hi) // 2

        write_config(config.output_file, weights[-k:] if k > 0 else [])
        log_file = (
            os.path.join(config.log_dir, f"iter_{iteration:03d}_k{k}.log")
            if config.save_logs
            else None
        )
        top1_p5 = run_test(config.test, log_file)

        if top1_p5 is None:
            print(f"  ERROR: Could not parse TOP1 p5. Check {log_file}")
            sys.exit(1)

        passed = top1_p5 >= threshold
        print(
            f"  [{iteration}] k={k}/{n} ({k / n * 100:.1f}% at bfp_bf4) → TOP1 p5={top1_p5:.2f}% {'PASS' if passed else 'FAIL'}"
        )

        if passed:
            best_k = k
            top1_at_best = top1_p5
            lo = k + 1
        else:
            hi = k - 1

    print(f"\nSearch complete. Best k={best_k} ({best_k / n * 100:.1f}% of weights)")
    bfp4_weights = weights[-best_k:] if best_k > 0 else []
    bfp4_numel = sum(sizes.get(w, 0) for w in bfp4_weights)
    write_config(config.output_file, bfp4_weights)
    write_final(config.results_file, best_k, n, top1_at_best, bfp4_numel, total_numel)


def resolve_paths(args):
    model_short = args.model.split("/")[-1]
    scores_file = args.scores or get_scores_path(args.model)
    output_file = args.output or get_output_path(args.model)
    results_file = args.results or os.path.join(
        EXPERIMENTS_DIR, "results", f"lowering_{model_short}.md"
    )
    log_dir = os.path.join(EXPERIMENTS_DIR, "search_logs", model_short)
    return model_short, scores_file, output_file, results_file, log_dir


def load_weights(scores_file, model_name):
    with open(scores_file) as f:
        weights = list(json.load(f).keys())
    sizes = load_param_sizes(model_name)
    mlp_weights = [w for w in weights if "mlp" in w.lower()]
    return weights, sizes, mlp_weights, len(weights), sum(sizes.values())


def run(weights, sizes, mlp_weights, n, total_numel, config):
    all_bfp4_top1 = run_baselines(weights, mlp_weights, n, config)
    threshold = config.threshold

    if all_bfp4_top1 is not None and all_bfp4_top1 >= config.threshold:
        print(
            f"\nAll-bfp_bf4 passed ({all_bfp4_top1:.2f}% ≥ {config.threshold}%). Skipping search."
        )
        write_final(config.results_file, n, n, all_bfp4_top1, total_numel, total_numel)
        return

    binary_search(weights, sizes, n, total_numel, config, threshold)


def main():
    args = parse_args()
    model_short, scores_file, output_file, results_file, log_dir = resolve_paths(args)
    weights, sizes, mlp_weights, n, total_numel = load_weights(scores_file, args.model)
    config = SearchConfig(
        output_file=output_file,
        results_file=results_file,
        log_dir=log_dir,
        test=args.test,
        threshold=args.threshold,
        save_logs=args.save_logs,
    )

    os.makedirs(os.path.dirname(os.path.abspath(results_file)), exist_ok=True)
    print(f"Model: {model_short} ({n} weights, threshold={args.threshold}% TOP1 p5)")
    print(f"Results: {results_file}\n")
    write_header(results_file, model_short, args.test, args.threshold, n)

    run(weights, sizes, mlp_weights, n, total_numel, config)

    print(f"Config written to: {output_file}")
    print(f"Results written to: {results_file}")


if __name__ == "__main__":
    main()
