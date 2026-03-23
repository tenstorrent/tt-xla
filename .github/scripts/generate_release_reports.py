# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import csv
import json

# ── Model Coverage ──────────────────────────────────────────────────────


def make_arch_slug(arch):
    return arch.lower().replace(" ", "_").replace(".", "_")


with open("coverage_nightly.json") as f:
    coverage_data = json.load(f)
with open("coverage_weekly.json") as f:
    coverage_data += json.load(f)
with open("coverage_weekly_training.json") as f:
    coverage_data += json.load(f)

# Write coverage CSV
with open("model_coverage.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(
        [
            "Model task",
            "Model architecture",
            "Model variant",
            "Inference",
            "Training",
            "n150",
            "n300",
            "p150",
            "Single device",
            "Data parallel",
            "Tensor parallel",
            "Model source",
        ]
    )
    for row in coverage_data:
        arch = row["Model architecture"]
        url = f"https://github.com/tenstorrent/tt-forge-models/tree/main/{make_arch_slug(arch)}"
        w.writerow(
            [
                row["Model task"],
                arch,
                row["Model variant"],
                row["Inference"],
                row["Training"],
                row["n150"],
                row["n300"],
                row["p150"],
                row["Single device"],
                row["Data parallel"],
                row["Tensor parallel"],
                url,
            ]
        )

# Write coverage MD
with open("model_coverage.md", "w") as f:
    f.write("## Model coverage\n")
    f.write(
        "> _Info:_  Full list of supported models is available in the assets section.\n\n"
    )
    f.write(
        "| Model task | Model architecture | Model variant | Inference | Training | n150 | n300 | p150 | Single device | Data parallel | Tensor parallel | Model source |\n"
    )
    f.write(
        "| --- | --- | --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | --- |\n"
    )
    for row in coverage_data:
        arch = row["Model architecture"]
        url = f"https://github.com/tenstorrent/tt-forge-models/tree/main/{make_arch_slug(arch)}"
        f.write(
            f"| {row['Model task']} | {arch} | {row['Model variant']} "
            f"| {row['Inference']} | {row['Training']} "
            f"| {row['n150']} | {row['n300']} | {row['p150']} "
            f"| {row['Single device']} | {row['Data parallel']} | {row['Tensor parallel']} "
            f"| [View Source]({url}) |\n"
        )

# ── Model Performance ───────────────────────────────────────────────────

with open("perf.json") as f:
    perf_data = json.load(f)

llms = [r for r in perf_data if r.get("ttft (ms)") not in (None, "", 0)]
non_llms = [r for r in perf_data if r.get("ttft (ms)") in (None, "")]

# LLM CSV
with open("model_performance_llms.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Model", "Token/sec/user", "Batch", "Token/sec", "ttft (ms)"])
    for r in llms:
        w.writerow(
            [
                r["Model"],
                round(float(r["Tokens/sec/user"]), 2),
                r["Batch"],
                round(float(r["Tokens/sec"]), 2),
                round(float(r["ttft (ms)"]), 2),
            ]
        )

# LLM MD
with open("model_performance_llms.md", "w") as f:
    f.write("## LLM Performance\n\n")
    f.write("| Model | Token/sec/user | Batch | Token/sec | ttft (ms) |\n")
    f.write("| --- | --- | --- | --- | --- |\n")
    for r in llms:
        f.write(
            f"| {r['Model']} "
            f"| {round(float(r['Tokens/sec/user']), 2)} "
            f"| {r['Batch']} "
            f"| {round(float(r['Tokens/sec']), 2)} "
            f"| {round(float(r['ttft (ms)']), 2)} |\n"
        )

# Non-LLM CSV
with open("model_performance_non_llms.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Model", "Batch", "Sample/sec"])
    for r in non_llms:
        w.writerow(
            [
                r["Model"],
                r["Batch"],
                round(float(r["Tokens/sec"]), 2),
            ]
        )

# Non-LLM MD
with open("model_performance_non_llms.md", "w") as f:
    f.write("## Non-LLM Performance\n\n")
    f.write("| Model | Batch | Sample/sec |\n")
    f.write("| --- | --- | --- |\n")
    for r in non_llms:
        f.write(
            f"| {r['Model']} "
            f"| {r['Batch']} "
            f"| {round(float(r['Tokens/sec']), 2)} |\n"
        )

print("All reports generated successfully")
