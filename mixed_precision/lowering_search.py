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
import json
import os
import re
import subprocess
import sys

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
        help="Minimum TOP1 p5 accuracy (e.g. 80 for 80%%)",
    )
    parser.add_argument("--results", default=None, help="Output markdown results file")
    return parser.parse_args()


def load_scores(scores_file):
    with open(scores_file) as f:
        data = json.load(f)
    weights = []
    sizes = {}
    for name, value in data.items():
        # Strip _orig_mod. prefix added by torch.compile wrapping
        clean = name.replace("_orig_mod.", "")
        weights.append(clean)
        if isinstance(value, dict):
            sizes[clean] = value["numel"]
    return weights, sizes


def write_config(config_path, override_weights, lower_dtype, default_dtype):
    """Write config with override_weights set to lower_dtype, everything else at default_dtype."""
    config = {"default": default_dtype}
    for w in override_weights:
        config[w] = lower_dtype
    os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def run_test(test, log_file):
    os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(os.getcwd(), "tests")
    cmd = [sys.executable, "-m", "pytest", test, "--accuracy-testing", "-s"]
    with open(log_file, "w") as log:
        result = subprocess.run(
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        log.write(result.stdout)
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
        f.write(f"| | |\n|---|---|\n")
        f.write(f"| **Test** | `{test}` |\n")
        f.write(f"| **Threshold** | {threshold}% TOP1 p5 |\n")
        f.write(f"| **Total weights** | {n} |\n\n")


def write_baselines_header(results_file):
    with open(results_file, "a") as f:
        f.write(f"## Baselines\n\n")
        f.write(f"| Config | Weights at bfp_bf4 | TOP1 p5 |\n")
        f.write(f"|--------|--------------------|---------|---|\n")


def append_baseline(results_file, label, n_lower, n_total, top1_p5):
    with open(results_file, "a") as f:
        pct = n_lower / n_total * 100
        top1_str = f"{top1_p5:.2f}%" if top1_p5 is not None else "N/A"
        f.write(f"| {label} | {n_lower}/{n_total} ({pct:.1f}%) | {top1_str} |\n")


def write_search_header(results_file):
    with open(results_file, "a") as f:
        f.write(f"\n## Search Steps\n\n")
        f.write(f"| Iter | k | % at bfp_bf4 | TOP1 p5 | Result |\n")
        f.write(f"|------|---|--------------|---------|--------|\n")


def append_step(results_file, iteration, k, n, top1_p5, passed):
    with open(results_file, "a") as f:
        result = "PASS" if passed else "FAIL"
        f.write(
            f"| {iteration} | {k} | {k / n * 100:.1f}% | {top1_p5:.2f}% | {result} |\n"
        )


def write_final(
    results_file, best_k, n, top1_at_best, bfp4_numel=None, total_numel=None
):
    n_default = n - best_k
    if bfp4_numel is not None and total_numel:
        estimated_saving_pct = bfp4_numel / (2 * total_numel) * 100
        saving_note = ""
    else:
        estimated_saving_pct = best_k / (2 * n) * 100
        saving_note = " _(assumes uniform parameter count per weight)_"
    top1_str = f"{top1_at_best:.2f}%" if top1_at_best is not None else "N/A"
    with open(results_file, "a") as f:
        f.write(f"\n## Final Config\n\n")
        f.write(f"**Best k = {best_k}** — accuracy TOP1 p5={top1_str}\n\n")
        f.write(f"| Dtype | Weights | Share |\n")
        f.write(f"|-------|---------|-------|\n")
        f.write(f"| `bfp_bf8` | {n_default} | {n_default / n * 100:.1f}% |\n")
        f.write(f"| `bfp_bf4` | {best_k} | {best_k / n * 100:.1f}% |\n\n")
        f.write(
            f"**Estimated weight memory vs all-bfp_bf8 baseline:** ~{estimated_saving_pct:.1f}% smaller{saving_note}\n"
        )
        f.write(f"_(bfp_bf4 = 0.5× bfp_bf8 bits)_\n")


def run_baselines(weights, mlp_weights, n, output_file, results_file, log_dir, test):
    """Run three fixed baselines and return the all-bfp4 TOP1 p5."""
    baselines = [
        ("all bfp_bf8", []),
        ("all MLP bfp_bf4", mlp_weights),
        ("all bfp_bf4", weights),
    ]
    write_baselines_header(results_file)
    print("Running baselines...")
    all_bfp4_top1 = None
    for label, bw in baselines:
        write_config(output_file, bw, "bfp_bf4", "bfp_bf8")
        log_file = os.path.join(log_dir, f"baseline_{label.replace(' ', '_')}.log")
        top1_p5 = run_test(test, log_file)
        top1_str = f"{top1_p5:.2f}%" if top1_p5 is not None else "N/A"
        print(f"  {label}: TOP1 p5={top1_str}")
        append_baseline(results_file, label, len(bw), n, top1_p5)
        if label == "all bfp_bf4":
            all_bfp4_top1 = top1_p5
    return all_bfp4_top1


def binary_search(
    weights, sizes, n, total_numel, output_file, results_file, log_dir, test, threshold
):
    """Binary search for the largest k where the least-sensitive k weights can be bfp4."""
    write_search_header(results_file)
    print("\nRunning binary search...")
    lo, hi = 0, n - 1
    best_k = 0
    top1_at_best = None
    iteration = 0

    while lo <= hi:
        iteration += 1
        k = (lo + hi) // 2

        write_config(output_file, weights[-k:] if k > 0 else [], "bfp_bf4", "bfp_bf8")
        log_file = os.path.join(log_dir, f"iter_{iteration:03d}_k{k}.log")
        top1_p5 = run_test(test, log_file)

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

        append_step(results_file, iteration, k, n, top1_p5, passed)

    print(f"\nSearch complete. Best k={best_k} ({best_k / n * 100:.1f}% of weights)")
    bfp4_weights = weights[-best_k:] if best_k > 0 else []
    bfp4_numel = sum(sizes.get(w, 0) for w in bfp4_weights) if sizes else None
    write_config(output_file, bfp4_weights, "bfp_bf4", "bfp_bf8")
    write_final(
        results_file,
        best_k,
        n,
        top1_at_best,
        bfp4_numel=bfp4_numel,
        total_numel=total_numel,
    )


def main():
    args = parse_args()

    scores_file = args.scores if args.scores else get_scores_path(args.model)
    output_file = args.output if args.output else get_output_path(args.model)
    weights, sizes = load_scores(scores_file)
    n = len(weights)
    total_numel = sum(sizes.values()) if sizes else None
    model_short = args.model.split("/")[-1]
    mlp_weights = [w for w in weights if "mlp" in w.lower()]

    results_file = args.results
    if results_file is None:
        results_file = os.path.join(
            EXPERIMENTS_DIR, "results", f"lowering_{model_short}.md"
        )
    os.makedirs(os.path.dirname(os.path.abspath(results_file)), exist_ok=True)
    log_dir = os.path.join(EXPERIMENTS_DIR, "search_logs", model_short)

    print(f"Model: {model_short} ({n} weights, threshold={args.threshold}% TOP1 p5)")
    print(f"Results: {results_file}\n")

    write_header(results_file, model_short, args.test, args.threshold, n)

    all_bfp4_top1 = run_baselines(
        weights, mlp_weights, n, output_file, results_file, log_dir, args.test
    )

    if all_bfp4_top1 is not None and all_bfp4_top1 >= args.threshold:
        print(
            f"\nTest passed with all weights lowered to bfp_bf4 ({all_bfp4_top1:.2f}% ≥ {args.threshold}%). Skipping search."
        )
        write_final(
            results_file,
            n,
            n,
            all_bfp4_top1,
            bfp4_numel=total_numel,
            total_numel=total_numel,
        )
        print(f"Config written to: {output_file}")
        print(f"Results written to: {results_file}")
        return

    binary_search(
        weights,
        sizes,
        n,
        total_numel,
        output_file,
        results_file,
        log_dir,
        args.test,
        args.threshold,
    )
    print(f"Config written to: {output_file}")
    print(f"Results written to: {results_file}")


if __name__ == "__main__":
    main()
