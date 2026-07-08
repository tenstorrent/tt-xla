# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

SECONDS_PER_MB = 120


def estimate_seconds(size_bytes: int) -> int:
    return math.ceil((size_bytes * SECONDS_PER_MB / (1024 * 1024)))


def is_n150_job_dir(job_dir: Path) -> bool:
    parts = job_dir.name.split("-")
    return len(parts) >= 5 and parts[4] == "n150"


def collect_model_directories(root: Path, model_filter: str) -> tuple[list[dict], int]:
    all_model_dirs = [
        model_dir
        for job_dir in sorted(root.iterdir())
        if job_dir.is_dir() and is_n150_job_dir(job_dir)
        for model_dir in sorted(job_dir.iterdir())
        if model_dir.is_dir()
    ]
    models = []

    for model_dir in all_model_dirs:
        if model_filter and model_filter not in model_dir.name:
            continue

        irs_dir = model_dir / "irs"
        size_bytes = 0
        for file_path in irs_dir.rglob("*") if irs_dir.is_dir() else []:
            if file_path.is_file():
                size_bytes += file_path.stat().st_size

        models.append(
            {
                "name": model_dir.name,
                "relative_path": model_dir.relative_to(root).as_posix(),
                "size_bytes": size_bytes,
                "estimated_seconds": estimate_seconds(size_bytes),
            }
        )

    return models, len(all_model_dirs)


def build_assignment(root: Path, model_filter: str, target_job_seconds: int) -> dict:
    models, total_models = collect_model_directories(root, model_filter)

    if not models:
        return {
            "total_models": total_models,
            "matching_count": 0,
            "job_count": 0,
            "jobs": [],
        }

    total_estimated_seconds = sum(model["estimated_seconds"] for model in models)
    requested_jobs = max(
        1, math.ceil(total_estimated_seconds / max(target_job_seconds, 1))
    )
    job_count = min(requested_jobs, len(models))

    models.sort(
        key=lambda model: (
            model["estimated_seconds"],
            model["size_bytes"],
            model["relative_path"],
        ),
        reverse=True,
    )

    jobs = [
        {"job_index": job_index, "estimated_seconds": 0, "models": []}
        for job_index in range(job_count)
    ]

    for job, model in zip(jobs, models[:job_count]):
        job["models"].append(model)
        job["estimated_seconds"] += model["estimated_seconds"]

    for model in models[job_count:]:
        job = min(
            jobs,
            key=lambda entry: (
                entry["estimated_seconds"],
                len(entry["models"]),
                entry["job_index"],
            ),
        )
        job["models"].append(model)
        job["estimated_seconds"] += model["estimated_seconds"]

    return {
        "total_models": total_models,
        "matching_count": len(models),
        "job_count": job_count,
        "requested_jobs": requested_jobs,
        "target_job_seconds": target_job_seconds,
        "total_estimated_seconds": total_estimated_seconds,
        "jobs": jobs,
    }


def format_summary(assignment: dict, model_filter: str) -> str:
    lines = [
        f"Models matching filter '{model_filter}': {assignment['matching_count']}/{assignment['total_models']}"
    ]

    if assignment["job_count"]:
        lines.append(
            f"Planned {assignment['job_count']} processing jobs for {assignment['total_estimated_seconds']}s estimated total runtime"
        )
        for job in assignment["jobs"]:
            model_names = ", ".join(model["name"] for model in job["models"])
            lines.append(
                f"Job {job['job_index']}: {len(job['models'])} model(s), {job['estimated_seconds']}s estimated, models: {model_names}"
            )

    return "\n".join(lines)


def build_job_matrix(assignment: dict) -> dict:
    return {
        "include": [
            {
                "job_index": job["job_index"],
                "estimated_seconds": job["estimated_seconds"],
                "models_json": json.dumps(
                    [model["relative_path"] for model in job["models"]],
                    separators=(",", ":"),
                ),
            }
            for job in assignment["jobs"]
        ]
    }


def write_github_outputs(assignment: dict, github_output: Path) -> None:
    matrix = build_job_matrix(assignment)
    with github_output.open("a", encoding="utf-8") as output_file:
        if assignment["job_count"] > 0:
            output_file.write("skip_ir_processing=false\n")
            output_file.write(f"ir_count={assignment['job_count']}\n")
            output_file.write(
                f"job_matrix={json.dumps(matrix, separators=(',', ':'))}\n"
            )
        else:
            output_file.write("skip_ir_processing=true\n")
            output_file.write("ir_count=0\n")
            output_file.write('job_matrix={"include":[]}\n')


def write_assignment(assignment: dict, output_path: Path | None) -> None:
    serialized = json.dumps(assignment)
    if output_path is None:
        print(serialized)
        return

    output_path.write_text(serialized, encoding="utf-8")


def load_assignment(assignment_file: Path) -> dict:
    return json.loads(assignment_file.read_text(encoding="utf-8"))


def find_job(assignment: dict, job_index: int) -> dict:
    for job in assignment["jobs"]:
        if job["job_index"] == job_index:
            return job
    raise ValueError(f"Job index {job_index} not present in assignment")


def materialize_job(
    root: Path, job_index: int, models_json: str, estimated_seconds: int
) -> None:
    models = json.loads(models_json)
    target_root = Path(f"{root}_job_{job_index}")

    print(f"Estimated runtime for this job: {estimated_seconds}s")
    print(f"Assigned model directories: {models}")

    target_root.mkdir(parents=True, exist_ok=True)
    for relative_path in models:
        source_dir = root / relative_path
        destination_dir = target_root / relative_path
        if not source_dir.is_dir():
            raise ValueError(f"Assigned model directory not found: {source_dir}")
        destination_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source_dir), str(destination_dir))

    shutil.rmtree(root)
    shutil.move(str(target_root), str(root))


def command_plan(args: argparse.Namespace) -> int:
    assignment = build_assignment(args.root, args.model_filter, args.target_job_seconds)
    write_assignment(assignment, args.output)

    if args.print_summary:
        print(format_summary(assignment, args.model_filter), file=sys.stderr)

    if args.github_output is not None:
        write_github_outputs(assignment, args.github_output)

    return 0


def command_summary(args: argparse.Namespace) -> int:
    assignment = load_assignment(args.assignment_file)
    print(format_summary(assignment, args.model_filter))
    return 0


def command_matrix(args: argparse.Namespace) -> int:
    assignment = load_assignment(args.assignment_file)
    print(json.dumps(build_job_matrix(assignment), separators=(",", ":")))
    return 0


def command_materialize(args: argparse.Namespace) -> int:
    materialize_job(args.root, args.job_index, args.models_json, args.estimated_seconds)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--root", type=Path, required=True)
    plan_parser.add_argument("--model-filter", default="")
    plan_parser.add_argument("--target-job-seconds", type=int, required=True)
    plan_parser.add_argument("--output", type=Path)
    plan_parser.add_argument("--github-output", type=Path)
    plan_parser.add_argument("--print-summary", action="store_true")
    plan_parser.set_defaults(func=command_plan)

    summary_parser = subparsers.add_parser("summary")
    summary_parser.add_argument("--assignment-file", type=Path, required=True)
    summary_parser.add_argument("--model-filter", default="")
    summary_parser.set_defaults(func=command_summary)

    matrix_parser = subparsers.add_parser("matrix")
    matrix_parser.add_argument("--assignment-file", type=Path, required=True)
    matrix_parser.set_defaults(func=command_matrix)

    materialize_parser = subparsers.add_parser("materialize")
    materialize_parser.add_argument("--root", type=Path, required=True)
    materialize_parser.add_argument("--job-index", type=int, required=True)
    materialize_parser.add_argument("--models-json", required=True)
    materialize_parser.add_argument("--estimated-seconds", type=int, required=True)
    materialize_parser.set_defaults(func=command_materialize)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.func(args)
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
