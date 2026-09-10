# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import concurrent.futures
import json
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path

SECONDS_PER_MB = 120

# Max model/op names listed inline per job in the human-readable plan summary.
SUMMARY_MODEL_LIMIT = 25

# Extraction is CPU-bound MLIR parsing and its cost grows superlinearly with the
# op count of a module, so a handful of large models dominate the wall time. It
# parallelizes cleanly per IR file. Capped rather than using every core because
# each worker holds a fully parsed module in memory.
DEFAULT_EXTRACT_JOBS = min(os.cpu_count() or 1, 8)

# How often to print extraction progress, in files.
EXTRACT_PROGRESS_EVERY = 25


def estimate_seconds(size_bytes: int) -> int:
    return math.ceil((size_bytes * SECONDS_PER_MB / (1024 * 1024)))


def is_n150_job_dir(job_dir: Path) -> bool:
    parts = job_dir.name.split("-")
    return len(parts) >= 5 and parts[4] == "n150"


def get_job_machine(job_dir: Path) -> str:
    parts = job_dir.name.split("-")
    if len(parts) >= 5:
        return parts[4]
    return ""


def iter_model_dirs(root: Path):
    for job_dir in sorted(root.iterdir()):
        if not job_dir.is_dir():
            continue
        for model_dir in sorted(job_dir.iterdir()):
            if model_dir.is_dir():
                yield model_dir


def get_irs_size_bytes(model_dir: Path) -> int:
    irs_dir = model_dir / "irs"
    size_bytes = 0
    for file_path in irs_dir.rglob("*") if irs_dir.is_dir() else []:
        if file_path.is_file():
            size_bytes += file_path.stat().st_size
    return size_bytes


def choose_model_to_keep(entries: list[dict], preferred_machine: str) -> dict:
    preferred = [entry for entry in entries if entry["machine"] == preferred_machine]
    candidates = preferred if preferred else entries
    return max(
        candidates,
        key=lambda entry: (entry["size_bytes"], entry["relative_path"]),
    )


def remove_duplicate_models(root: Path, preferred_machine: str) -> tuple[int, int]:
    models_by_name: dict[str, list[dict]] = {}

    for model_dir in iter_model_dirs(root):
        job_dir = model_dir.parent
        entry = {
            "model_dir": model_dir,
            "machine": get_job_machine(job_dir),
            "relative_path": model_dir.relative_to(root).as_posix(),
            "size_bytes": get_irs_size_bytes(model_dir),
        }
        models_by_name.setdefault(model_dir.name, []).append(entry)

    duplicate_model_count = 0
    removed_dir_count = 0

    for model_name, entries in models_by_name.items():
        if len(entries) <= 1:
            continue

        duplicate_model_count += 1
        keep = choose_model_to_keep(entries, preferred_machine)
        print(
            f"Deduplicating model '{model_name}': keeping {keep['relative_path']} "
            f"(machine={keep['machine']})"
        )

        for entry in entries:
            if entry is keep:
                continue
            print(
                f"Removing duplicate model directory {entry['relative_path']} "
                f"(machine={entry['machine']})"
            )
            shutil.rmtree(entry["model_dir"])
            removed_dir_count += 1

    return duplicate_model_count, removed_dir_count


def collect_model_directories(
    root: Path, model_filter: str, seconds_per_model: int = 0
) -> tuple[list[dict], int]:
    all_model_dirs = [model_dir for model_dir in iter_model_dirs(root)]
    models = []

    for model_dir in all_model_dirs:
        if model_filter and model_filter not in model_dir.name:
            continue

        size_bytes = get_irs_size_bytes(model_dir)

        models.append(
            {
                "name": model_dir.name,
                "relative_path": model_dir.relative_to(root).as_posix(),
                "size_bytes": size_bytes,
                # Size is a good cost proxy for whole-model IR dumps, but not for
                # single-op modules: those are ~1KB each while their cost is
                # dominated by compile+execute. `seconds_per_model` lets the
                # unique-ops flow plan by op count instead.
                "estimated_seconds": (
                    seconds_per_model
                    if seconds_per_model > 0
                    else estimate_seconds(size_bytes)
                ),
            }
        )

    return models, len(all_model_dirs)


def build_assignment(
    root: Path,
    model_filter: str,
    target_job_seconds: int,
    seconds_per_model: int = 0,
) -> dict:
    models, total_models = collect_model_directories(
        root, model_filter, seconds_per_model
    )

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
            names = [model["name"] for model in job["models"]]
            # The unique-ops flow puts hundreds of single-op dirs in one job, so
            # cap the inline list to keep CI logs readable.
            shown = names[:SUMMARY_MODEL_LIMIT]
            model_names = ", ".join(shown)
            if len(names) > len(shown):
                model_names += f", ... (+{len(names) - len(shown)} more)"
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
    assignment = build_assignment(
        args.root,
        args.model_filter,
        args.target_job_seconds,
        args.seconds_per_model,
    )
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


def match_and_extract_model_name(file_path: Path, ir_file_prefix: str) -> str | None:
    """
    Check if a file matches the IR file prefix pattern and extract its model name.

    Mirrors the matching logic in tests/op_by_op/op_by_op_test.py so this pre-pass
    selects exactly the same files the test would.
    """
    if not ir_file_prefix:
        return file_path.parent.name

    parts = ir_file_prefix.split("/")
    file_prefix = parts[-1]
    dir_parts = parts[:-1]

    if not file_path.name.startswith(file_prefix):
        return None

    if not dir_parts:
        return file_path.parent.name

    path_parts = list(file_path.parts)
    for i in range(len(path_parts) - len(dir_parts)):
        if path_parts[i : i + len(dir_parts)] == dir_parts:
            if i > 0:
                return path_parts[i - 1]
            return dir_parts[0]

    return None


def _sanitize_op_dirname(name: str) -> str:
    """Sanitize an op name (e.g. 'stablehlo.add') into a filesystem-safe dir name."""
    return re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_") or "op"


def _extract_ops_from_file(task: tuple) -> dict:
    """
    Extract and locally-deduplicate the ops of a single IR file.

    Runs in a worker process, so it returns plain picklable data and imports the
    tt-mlir bindings itself. Deduplicating within the file before returning keeps
    the amount of module text shipped back to the parent small -- most ops in a
    model IR are repeats of one another.

    Ops whose module text cannot be built are reported as failures and, matching
    the serial path, are not recorded as seen, so an identical later op retries.
    """
    file_str, model_name, whitelist, blacklist = task

    from op_by_op_infra.workflow import extract_ops_from_module

    try:
        module = Path(file_str).read_text(encoding="utf-8", errors="replace")
    except (OSError, IOError) as error:
        return {
            "records": [],
            "failures": [{"model": model_name, "file": file_str, "error": str(error)}],
            "total_ops": 0,
        }

    try:
        ops = extract_ops_from_module(module, origin_model=model_name)
    except Exception as error:  # noqa: BLE001 - continue-on-failure by design
        return {
            "records": [],
            "failures": [{"model": model_name, "file": file_str, "error": str(error)}],
            "total_ops": 0,
        }

    records: list[dict] = []
    failures: list[dict] = []
    total_ops = 0
    local_seen: set = set()

    for op in ops:
        # op_name can be an MLIR StringAttr rather than a plain str; coerce so
        # filtering, dir-name sanitizing, and JSON serialization all work.
        op_name = str(op.op_name)
        if whitelist:
            if op_name not in whitelist:
                continue
        elif blacklist:
            if op_name in blacklist:
                continue

        total_ops += 1
        key = op.op_string

        if key and key in local_seen:
            continue

        try:
            module_str = op.as_module_str()
        except Exception as error:  # noqa: BLE001 - skip un-serializable ops
            failures.append(
                {"model": model_name, "op_name": op_name, "error": str(error)}
            )
            continue

        if key:
            local_seen.add(key)
        records.append(
            {
                "op_name": op_name,
                "op_string": key,
                "origin_models": list(op.origin_model),
                "module_str": module_str,
            }
        )

    return {"records": records, "failures": failures, "total_ops": total_ops}


def extract_unique_ops(
    root: Path,
    ir_file_prefix: str,
    output_root: Path,
    whitelist: list[str] | None,
    blacklist: list[str] | None,
    model_filter: str = "",
    jobs: int = DEFAULT_EXTRACT_JOBS,
    time_budget_seconds: int = 0,
) -> dict:
    """
    Extract ops from every matched IR under ``root``, deduplicate globally by
    op_string (same semantics as filter_and_deduplicate_ops in op_by_op_test.py),
    and write each unique op as a standalone single-op MLIR module under
    ``output_root`` in the layout the planner/test already understand
    (``unique_ops/<op_dir>/<prefix_dirs>/<file_prefix>_<idx>.mlir``).

    Returns a stats dict including the per-op manifest.
    """
    # Imported lazily: this is the only command that needs the explorer/tt-mlir
    # python bindings, so `plan`/`dedupe`/`materialize` still run on runners
    # without the wheel installed. The workers import it themselves; doing it here
    # too fails fast in the parent with one clear error if the wheel is missing,
    # rather than the same ImportError once per worker.
    from op_by_op_infra.workflow import extract_ops_from_module  # noqa: F401

    whitelist = whitelist or []
    blacklist = blacklist or []

    all_mlir_files = sorted(root.rglob("*.mlir"))
    matched = [
        (f, model_name)
        for f in all_mlir_files
        if (model_name := match_and_extract_model_name(f, ir_file_prefix)) is not None
    ]

    # Applied here rather than in `plan`: after extraction the planner sees op
    # dirs, not model dirs, so a model filter would no longer match anything.
    if model_filter:
        matched = [(f, name) for f, name in matched if model_filter in name]

    total_ops = 0
    extraction_failures: list[dict] = []
    seen: dict[str, dict] = {}  # op_string -> unique record
    unique_records: list[dict] = []

    tasks = [
        (str(ir_file), model_name, whitelist, blacklist)
        for ir_file, model_name in matched
    ]
    workers = max(1, min(jobs, len(tasks) or 1))
    started = time.monotonic()
    files_done = 0
    files_skipped = 0
    budget_exhausted = False

    print(
        f"Extracting ops from {len(tasks)} IR file(s) using {workers} worker(s)"
        + (f", time budget {time_budget_seconds}s" if time_budget_seconds else ""),
        file=sys.stderr,
    )

    # `map` yields results in submission order, so the resulting op indices (and
    # therefore the op directory names) are identical to the serial ordering and
    # do not depend on which worker finished first.
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        results = executor.map(_extract_ops_from_file, tasks)

        for (ir_file, model_name), result in zip(matched, results):
            files_done += 1
            total_ops += result["total_ops"]

            for failure in result["failures"]:
                if "file" in failure:
                    print(
                        f"WARNING: failed to extract ops from {failure['file']}: "
                        f"{failure['error']}",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"WARNING: could not build module for op "
                        f"'{failure['op_name']}' from model "
                        f"'{failure['model']}': {failure['error']}",
                        file=sys.stderr,
                    )
            extraction_failures.extend(result["failures"])

            for candidate in result["records"]:
                key = candidate["op_string"]

                if key and key in seen:
                    record = seen[key]
                    if model_name and model_name not in record["origin_models"]:
                        record["origin_models"].append(model_name)
                    continue

                record = {
                    "op_name": candidate["op_name"],
                    "origin_models": candidate["origin_models"],
                    "module_str": candidate["module_str"],
                }
                unique_records.append(record)
                if key:
                    seen[key] = record

            elapsed = time.monotonic() - started
            if files_done % EXTRACT_PROGRESS_EVERY == 0 or files_done == len(tasks):
                rate = files_done / elapsed if elapsed else 0
                remaining = (len(tasks) - files_done) / rate if rate else 0
                print(
                    f"  extracted {files_done}/{len(tasks)} files "
                    f"({len(unique_records)} unique ops so far, "
                    f"{elapsed / 60:.1f}min elapsed, ~{remaining / 60:.1f}min left)",
                    file=sys.stderr,
                )

            # Stopping with a partial op set beats being killed by the job
            # timeout with nothing uploaded, but it silently narrows coverage --
            # so say exactly how much was dropped.
            if time_budget_seconds and elapsed > time_budget_seconds:
                files_skipped = len(tasks) - files_done
                budget_exhausted = True
                print(
                    f"WARNING: extraction time budget of {time_budget_seconds}s "
                    f"exhausted after {files_done}/{len(tasks)} files. "
                    f"SKIPPING {files_skipped} remaining IR file(s); the unique-op "
                    f"set is incomplete and ops only present in those files will "
                    f"not be tested this run.",
                    file=sys.stderr,
                )
                for future_task in tasks[files_done:]:
                    extraction_failures.append(
                        {
                            "model": future_task[1],
                            "file": future_task[0],
                            "error": "skipped: extraction time budget exhausted",
                        }
                    )
                executor.shutdown(wait=False, cancel_futures=True)
                break

    # Write unique ops in a layout the existing planner + op_by_op_test consume as-is.
    prefix_parts = (
        ir_file_prefix.split("/") if ir_file_prefix else ["irs", "shlo_compiler"]
    )
    sub_dirs = prefix_parts[:-1] or ["irs"]
    file_prefix = prefix_parts[-1] if ir_file_prefix else "shlo_compiler"

    unique_root = output_root / "unique_ops"
    manifest: list[dict] = []
    for idx, record in enumerate(unique_records):
        op_dir_name = f"op_{idx:05d}_{_sanitize_op_dirname(record['op_name'])}"
        irs_dir = unique_root / op_dir_name
        for part in sub_dirs:
            irs_dir = irs_dir / part
        irs_dir.mkdir(parents=True, exist_ok=True)
        out_file = irs_dir / f"{file_prefix}_{idx:05d}.mlir"
        out_file.write_text(record["module_str"], encoding="utf-8")
        manifest.append(
            {
                "index": idx,
                "op_name": record["op_name"],
                "op_dir": op_dir_name,
                "file": out_file.relative_to(output_root).as_posix(),
                "origin_models": record["origin_models"],
                "origin_model_count": len(record["origin_models"]),
            }
        )

    return {
        "root": str(root),
        "output_root": str(output_root),
        "model_filter": model_filter,
        "matched_files": len(matched),
        "matched_models": len({name for _, name in matched}),
        "files_processed": files_done,
        "files_skipped": files_skipped,
        "extraction_complete": not budget_exhausted,
        "extract_jobs": workers,
        "extract_seconds": round(time.monotonic() - started, 1),
        "total_ops_after_filter": total_ops,
        "unique_ops": len(unique_records),
        "duplicate_ops_eliminated": total_ops - len(unique_records),
        "extraction_failures": extraction_failures,
        "manifest": manifest,
    }


def command_extract_unique_ops(args: argparse.Namespace) -> int:
    stats = extract_unique_ops(
        args.root,
        args.ir_file_prefix,
        args.output_root,
        args.whitelist.split(",") if args.whitelist else None,
        args.blacklist.split(",") if args.blacklist else None,
        args.model_filter,
        args.jobs,
        args.time_budget_seconds,
    )

    if args.manifest is not None:
        args.manifest.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    total = stats["total_ops_after_filter"]
    unique = stats["unique_ops"]
    reduction = (1 - unique / total) * 100 if total else 0.0
    print(
        f"Matched IR files: {stats['matched_files']} "
        f"across {stats['matched_models']} model(s)\n"
        f"Files processed: {stats['files_processed']}"
        + (
            f" (INCOMPLETE: {stats['files_skipped']} skipped on time budget)"
            if not stats["extraction_complete"]
            else ""
        )
        + f"\nExtraction wall time: {stats['extract_seconds']}s "
        f"on {stats['extract_jobs']} worker(s)\n"
        f"Total ops (after filter): {total}\n"
        f"Unique ops: {unique}\n"
        f"Duplicate ops eliminated: {stats['duplicate_ops_eliminated']} "
        f"({reduction:.1f}% reduction)\n"
        f"Extraction failures: {len(stats['extraction_failures'])}\n"
        f"Unique-op modules written under: {args.output_root / 'unique_ops'}",
        file=sys.stderr,
    )
    return 0


def command_dedupe(args: argparse.Namespace) -> int:
    duplicate_model_count, removed_dir_count = remove_duplicate_models(
        args.root, args.preferred_machine
    )
    print(
        f"Duplicate model groups: {duplicate_model_count}; removed directories: {removed_dir_count}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--root", type=Path, required=True)
    plan_parser.add_argument("--model-filter", default="")
    plan_parser.add_argument("--target-job-seconds", type=int, required=True)
    plan_parser.add_argument(
        "--seconds-per-model",
        type=int,
        default=0,
        help="Flat per-model-dir runtime estimate in seconds. Use for the "
        "unique-ops flow, where each dir holds one tiny single-op module and "
        "size is not a usable cost proxy. 0 (default) keeps size-based estimates.",
    )
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

    dedupe_parser = subparsers.add_parser("dedupe")
    dedupe_parser.add_argument("--root", type=Path, required=True)
    dedupe_parser.add_argument("--preferred-machine", default="n150")
    dedupe_parser.set_defaults(func=command_dedupe)

    extract_parser = subparsers.add_parser(
        "extract-unique-ops",
        help="Extract ops from all model IRs, deduplicate globally, and write "
        "one standalone MLIR module per unique op.",
    )
    extract_parser.add_argument("--root", type=Path, required=True)
    extract_parser.add_argument("--ir-file-prefix", default="irs/shlo_compiler")
    extract_parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory to write unique-op modules into (consumable by "
        "`plan` and op_by_op_test via --folder).",
    )
    extract_parser.add_argument("--model-filter", default="")
    extract_parser.add_argument("--whitelist", default="")
    extract_parser.add_argument("--blacklist", default="")
    extract_parser.add_argument(
        "--manifest",
        type=Path,
        help="Optional path to write the JSON stats + per-op manifest.",
    )
    extract_parser.add_argument(
        "--jobs",
        type=int,
        default=DEFAULT_EXTRACT_JOBS,
        help=(
            "Worker processes used to extract ops in parallel "
            f"(default: {DEFAULT_EXTRACT_JOBS}). 1 runs serially."
        ),
    )
    extract_parser.add_argument(
        "--time-budget-seconds",
        type=int,
        default=0,
        help=(
            "Stop extracting after this many seconds and continue with the ops "
            "found so far, loudly reporting how many IR files were skipped. "
            "Default 0 means no budget. Set this below the job timeout so a slow "
            "run still uploads a usable op set instead of being killed. This is a "
            "soft bound, checked as each file's result is consumed: already-running "
            "workers are allowed to finish, so the overshoot can be as large as the "
            "slowest single IR file (tens of minutes for the biggest models). Leave "
            "headroom accordingly."
        ),
    )
    extract_parser.set_defaults(func=command_extract_unique_ops)

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
