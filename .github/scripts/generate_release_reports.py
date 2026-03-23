# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import csv
import json

# Models listed here are intentionally omitted from all performance report outputs.
EXCLUDED_PERF_MODELS = [
    "Resnet 50 HF",
    "meta-llama/Llama-3.2-3B",
]


def extract_segment(cell: str):
    """Extract the model path segment from a full_test_name value.

    Looks for the part inside [...] up to the first '-':
      test_all_models_torch[efficientnet/pytorch-B0-...] -> efficientnet/pytorch
    Falls back to everything inside [...] if no '-' is found:
      test_torch_whisper_inference[whisper_base] -> whisper_base
    """
    start = cell.find("[")
    if start == -1:
        return None
    dash = cell.find("-", start + 1)
    if dash != -1:
        extracted = cell[start + 1 : dash].strip()
        if extracted:
            return extracted
    end_bracket = cell.find("]", start + 1)
    if end_bracket != -1:
        extracted = cell[start + 1 : end_bracket].strip()
        if extracted:
            return extracted
    return None


def model_source_url(full_test_name: str):
    segment = extract_segment(full_test_name)
    if segment:
        return f"https://github.com/tenstorrent/tt-forge-models/tree/main/{segment}"
    return ""


def sample_sec(r):
    return round(float(r["Tokens/sec"]) / int(r["Batch"]), 2)


def emoji(value: str):
    if value == "+":
        return "✅"
    if value == "-":
        return "❌"
    return value


with open("coverage_nightly.json") as f:
    coverage_data = json.load(f)
with open("coverage_weekly.json") as f:
    coverage_data += json.load(f)
with open("coverage_weekly_training.json") as f:
    coverage_data += json.load(f)

COVERAGE_HEADERS = [
    "Model task",
    "Model architecture",
    "Model variant",
    "Model framework",
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

# Coverage CSV — +/- as plain characters, URL as raw string
with open("model_coverage.csv", "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(COVERAGE_HEADERS)
    for row in coverage_data:
        url = model_source_url(row["full_test_name"])
        w.writerow(
            [
                row["Model task"],
                row["Model architecture"],
                row["Model variant"],
                row["Model framework"],
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

# Coverage MD — +/- as ✅/❌, URL as [View Source](...), capped at 80 rows
with open("model_coverage.md", "w") as f:
    f.write("## Model coverage\n")
    f.write(
        "> _Info:_  Full list of supported models is available in the assets section.\n\n"
    )
    f.write(
        "| Model task | Model architecture | Model variant | Model framework | Inference | Training | n150 | n300 | p150 | Single device | Data parallel | Tensor parallel | Model source |\n"
    )
    f.write(
        "| --- | --- | --- | --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | --- |\n"
    )
    for row in coverage_data[:80]:
        url = model_source_url(row["full_test_name"])
        source = f"[View Source]({url})" if url else "Source not available"
        f.write(
            f"| {row['Model task']} | {row['Model architecture']} | {row['Model variant']} "
            f"| {row['Model framework']} "
            f"| {emoji(row['Inference'])} | {emoji(row['Training'])} "
            f"| {emoji(row['n150'])} | {emoji(row['n300'])} | {emoji(row['p150'])} "
            f"| {emoji(row['Single device'])} | {emoji(row['Data parallel'])} | {emoji(row['Tensor parallel'])} "
            f"| {source} |\n"
        )

# ── Model Performance ───────────────────────────────────────────────────────

with open("perf.json") as f:
    perf_data = json.load(f)

llms = [
    r
    for r in perf_data
    if r.get("ttft (ms)") not in (None, "", 0)
    and r["Model"] not in EXCLUDED_PERF_MODELS
]
non_llms = [
    r
    for r in perf_data
    if r.get("ttft (ms)") in (None, "") and r["Model"] not in EXCLUDED_PERF_MODELS
]

# LLM CSV
with open("model_performance_llms.csv", "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n")
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
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["Model", "Batch", "Sample/sec"])
    for r in non_llms:
        w.writerow([r["Model"], r["Batch"], sample_sec(r)])

# Non-LLM MD
with open("model_performance_non_llms.md", "w") as f:
    f.write("## Non-LLM Performance\n\n")
    f.write("| Model | Batch | Sample/sec |\n")
    f.write("| --- | --- | --- |\n")
    for r in non_llms:
        f.write(f"| {r['Model']} | {r['Batch']} | {sample_sec(r)} |\n")

print("All reports generated successfully")
