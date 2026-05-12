#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Run single-layer LLM/encoder benchmark tests and collect their TTIR/TTNN
MLIRs + per-test PCC/perf into a single JSON.

For each test:
  1. Invoke pytest with --num-layers 1, --max-output-tokens 2,
     --output-file <per-test JSON>.
  2. Locate the matching exported MLIRs in `modules/` (g0=prefill, g1=decode)
     and copy them into the output dir under canonical names.
  3. Read the per-test JSON for samples_per_sec / ttft / PCC / device info.

Bad-PCC tests fail naturally (the test writes the JSON, then raises
AssertionError) — the runner sees a non-zero pytest exit and marks the test
failed while still reading the metrics that were captured pre-assertion.

The aggregated `measured_<device_type>.json` is rewritten after every
completed test so a Ctrl-C / device-reset error mid-sweep leaves a
partial-but-valid JSON behind. Nonzero exit is reserved for
DEVICE_RESET_REQUIRED, which the wrapper treats as "reset and resume".
"""

import argparse
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

import pytest

# Pytest-output substrings that mean "the device hung; reset before any
# further test."
CRITICAL_ERRORS = [
    "Read unexpected run_mailbox value from core",
    "Timeout waiting for Ethernet core service remote IO request",
]

# Sentinel exit code: regen.sh catches this and runs safe_reset.sh + retries
# with --continue.
DEVICE_RESET_REQUIRED = 42

MAX_OUTPUT_TOKENS = 2
PYTEST_TIMEOUT_S = 1200


def _handle_critical_error() -> None:
    print(
        "\nDEVICE ERROR: TT device needs reset.\n"
        "  scripts/safe_reset.sh --force   # then re-run with --continue\n"
    )
    sys.exit(DEVICE_RESET_REQUIRED)


def _resolve_paths() -> tuple[Path, Path, Path]:
    single_layer_dir = Path(__file__).resolve().parent
    tt_xla_dir = single_layer_dir.parent
    repo_root = tt_xla_dir.parents[1]
    return single_layer_dir, tt_xla_dir, repo_root


def _ensure_import_paths(repo_root: Path, tt_xla_dir: Path) -> None:
    for path in (repo_root, tt_xla_dir):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


# --------------------------------------------------------------------------- #
# MLIR discovery
# --------------------------------------------------------------------------- #


def _extract_run_id(stem: str) -> Optional[str]:
    match = re.search(r"_run([0-9a-zA-Z]+)", stem)
    return match.group(1) if match else None


def _graph_index(stem: str) -> Optional[int]:
    match = re.search(r"_g(\d+)_", stem)
    return int(match.group(1)) if match else None


def _find_mlirs(
    export_path: Path, model_name: str, prefix: str, min_mtime: Optional[float] = None
) -> list[Path]:
    if not export_path.exists():
        return []
    name_pattern = re.compile(
        rf"^{re.escape(prefix)}_{re.escape(model_name)}_1lyr_bs\d+_isl\d+"
    )
    matches = []
    for path in export_path.rglob("*.mlir"):
        if not name_pattern.search(path.stem):
            continue
        if min_mtime is not None and path.stat().st_mtime < min_mtime:
            continue
        matches.append(path)
    return matches


def _select_latest_run(paths: list[Path]) -> list[Path]:
    if not paths:
        return []
    run_groups: dict[Optional[str], list[Path]] = {}
    for path in paths:
        run_groups.setdefault(_extract_run_id(path.stem), []).append(path)
    return max(
        run_groups.values(),
        key=lambda paths_in_group: max(p.stat().st_mtime for p in paths_in_group),
    )


def _select_g0_g1(
    paths: list[Path], group: str
) -> tuple[list[Path], dict[Path, str]]:
    """Strict g0=prefill, g1=decode. Encoders only have g0."""
    if not paths:
        return [], {}
    by_index: dict[int, list[Path]] = {}
    for path in paths:
        idx = _graph_index(path.stem)
        if idx is None:
            continue
        by_index.setdefault(idx, []).append(path)

    expected_indices = [0] if group == "encoder" else [0, 1]
    selected: list[Path] = []
    labels: dict[Path, str] = {}
    for idx in expected_indices:
        candidates = by_index.get(idx, [])
        if not candidates:
            continue
        path = max(candidates, key=lambda p: p.stat().st_mtime)
        selected.append(path)
        labels[path] = "prefill" if idx == 0 else "decode"
    return selected, labels


def _parse_mlir_stem(
    stem: str, model_name: str, prefix: str
) -> tuple[Optional[str], Optional[str]]:
    match = re.search(
        rf"^{re.escape(prefix)}_{re.escape(model_name)}_1lyr_bs(\d+)_isl(\d+)", stem
    )
    if not match:
        return None, None
    return match.group(1), match.group(2)


def _normalize_mlir_name(
    mlir_path: Path,
    group: str,
    model_name: str,
    prefix: str,
    graph_label: Optional[str],
) -> str:
    bs, isl = _parse_mlir_stem(mlir_path.stem, model_name, prefix)
    assert bs is not None and isl is not None, f"Expected bs/isl in {mlir_path.stem}"
    base = f"{model_name}_1lyr_bs{bs}"
    if group == "encoder":
        suffix = f"encoder_isl{isl}"
    elif graph_label == "decode":
        suffix = "decode"
    elif graph_label == "prefill":
        suffix = f"prefill_isl{isl}"
    else:
        raise ValueError(
            f"Cannot normalize {mlir_path.name}: missing g0/g1 marker; got "
            f"label={graph_label!r}"
        )
    ttnn_suffix = "_ttnn" if prefix == "ttnn" else ""
    return f"{base}_{suffix}{ttnn_suffix}.mlir"


def _copy_mlirs(
    mlir_paths: list[Path],
    output_dir: Path,
    group: str,
    model_name: str,
    prefix: str,
    graph_labels: dict[Path, str],
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for mlir_path in mlir_paths:
        target_name = _normalize_mlir_name(
            mlir_path, group, model_name, prefix, graph_labels.get(mlir_path)
        )
        target_path = output_dir / target_name
        if target_path.exists():
            target_path.unlink()
        shutil.copy2(mlir_path, target_path)
        copied.append(target_path)
    return copied


# --------------------------------------------------------------------------- #
# Pytest invocation + failure-message extraction
# --------------------------------------------------------------------------- #


_NANOBIND_NOISE_PREFIXES = (
    "nanobind:",
    " - leaked instance",
    " - leaked type",
    " - leaked function",
    " - ... skipped remainder",
    "See https://nanobind.readthedocs.io",
    "<frozen importlib._bootstrap>",
    "sys:1: DeprecationWarning",
    "-- Docs: https://docs.pytest.org",
)


def _is_noise_line(line: str) -> bool:
    stripped = line.lstrip()
    if not stripped:
        return True
    if any(stripped.startswith(p.lstrip()) for p in _NANOBIND_NOISE_PREFIXES):
        return True
    if set(stripped) <= {"=", "-", " "}:
        return True
    return False


_PYTEST_FOOTER_RE = re.compile(r"^\S+:\d+:\s+\w[\w.]*\s*$")
_MAX_DETAIL_LINES = 20


def _extract_pytest_failure_signal(output: str) -> tuple[Optional[str], Optional[str]]:
    """(one_line, detail) extracted from pytest output.

    one_line: `FAILED tests/... - <reason>` from the short summary.
    detail: `E ...` lines + per-test footer from the FAILURES block, capped.
    """
    lines = output.splitlines()
    summary_idx = failures_idx = warnings_idx = result_idx = None
    for idx, line in enumerate(lines):
        s = line.strip()
        if summary_idx is None and "short test summary info" in s and s.startswith("="):
            summary_idx = idx
        if failures_idx is None and re.fullmatch(r"=+ FAILURES =+", s):
            failures_idx = idx
        if warnings_idx is None and re.fullmatch(r"=+ warnings summary =+", s):
            warnings_idx = idx
        if result_idx is None and re.fullmatch(
            r"=+ \d+ (failed|error|errors|passed)[^=]*=+", s
        ):
            result_idx = idx

    one_line: Optional[str] = None
    if summary_idx is not None:
        end = result_idx if result_idx and result_idx > summary_idx else len(lines)
        for ln in lines[summary_idx + 1 : end]:
            stripped = ln.strip()
            if stripped.startswith(("FAILED ", "ERROR ")):
                one_line = stripped
                break

    detail_lines: list[str] = []
    if failures_idx is not None:
        candidates = [
            i
            for i in (warnings_idx, summary_idx, result_idx)
            if i is not None and i > failures_idx
        ]
        end = min(candidates) if candidates else len(lines)
        for ln in lines[failures_idx + 1 : end]:
            stripped = ln.strip()
            if not stripped or _is_noise_line(ln):
                continue
            if stripped.startswith("E "):
                detail_lines.append(stripped)
                if one_line is None:
                    one_line = stripped
            elif _PYTEST_FOOTER_RE.match(stripped):
                detail_lines.append(stripped)
        if len(detail_lines) > _MAX_DETAIL_LINES:
            suppressed = len(detail_lines) - _MAX_DETAIL_LINES
            detail_lines = detail_lines[:_MAX_DETAIL_LINES] + [
                f"... ({suppressed} more line{'s' if suppressed != 1 else ''} suppressed)"
            ]

    if detail_lines or one_line is not None:
        detail = "\n".join(detail_lines) if detail_lines else None
        return one_line, detail

    cleaned = [ln for ln in lines if not _is_noise_line(ln)]
    if not cleaned:
        return None, None
    tail = cleaned[-15:]
    return tail[-1].strip(), "\n".join(tail)


def _run_pytest(
    test_file: str,
    test_name: str,
    per_test_json: Path,
) -> tuple[bool, Optional[str], Optional[str]]:
    test_path = f"{test_file}::{test_name}"
    cmd = [
        sys.executable, "-m", "pytest", "-x", "-v", test_path,
        "--num-layers", "1",
        "--max-output-tokens", str(MAX_OUTPUT_TOKENS),
        "--output-file", str(per_test_json),
    ]
    pytest_env = os.environ.copy()
    # Wide terminal so pytest doesn't truncate the FAILED summary line.
    pytest_env.setdefault("COLUMNS", "200")
    pytest_env.setdefault("LINES", "200")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=PYTEST_TIMEOUT_S, env=pytest_env
        )
    except subprocess.TimeoutExpired:
        return False, "Test timed out", None
    except Exception as exc:
        if any(critical in str(exc) for critical in CRITICAL_ERRORS):
            _handle_critical_error()
        return False, f"{type(exc).__name__}: {exc}", None

    output = "\n".join(t for t in (result.stdout or "", result.stderr or "") if t)
    if result.returncode == 0:
        return True, None, None
    if result.returncode == getattr(pytest.ExitCode, "SKIPPED", 5):
        return False, "Test was skipped", None
    if any(critical in output for critical in CRITICAL_ERRORS):
        _handle_critical_error()

    one_line, detail = _extract_pytest_failure_signal(output)
    if one_line is None and detail is None:
        return False, "Test failed (no informative output captured)", None
    return False, one_line or "Test failed", detail


# --------------------------------------------------------------------------- #
# Test name discovery & filtering
# --------------------------------------------------------------------------- #


def _is_test_failed(test_file: Path, test_name: str) -> bool:
    """Tests prefixed with `# FAILED` in the source are skipped."""
    if not test_file.exists():
        return False
    lines = test_file.read_text().splitlines()
    for idx, line in enumerate(lines):
        if line.strip().startswith(f"def {test_name}("):
            prev_line = lines[idx - 1].strip() if idx > 0 else ""
            return prev_line.startswith("# FAILED")
    return False


def _test_names(
    module_name: str, skip: set[str], require_param: Optional[str] = None
) -> list[str]:
    module = __import__(module_name)
    names = []
    for name, func in inspect.getmembers(module, inspect.isfunction):
        if not name.startswith("test_") or name in skip:
            continue
        if require_param and require_param not in inspect.signature(func).parameters:
            continue
        names.append(name)
    return names


# --------------------------------------------------------------------------- #
# Per-test JSON parsing
# --------------------------------------------------------------------------- #


def _measurement_field(
    measurements: list[dict], name: str, field: str = "value"
) -> Optional[float]:
    for m in measurements:
        if m.get("measurement_name") == name:
            v = m.get(field)
            if v is None:
                return None
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
    return None


def _extract_metrics(per_test_json: Path) -> dict:
    if not per_test_json.exists():
        return {}
    try:
        data = json.loads(per_test_json.read_text())
    except (json.JSONDecodeError, OSError):
        return {}

    measurements = data.get("measurements") or []
    device_info = data.get("device_info") or {}
    return {
        "samples_per_sec": _measurement_field(measurements, "samples_per_sec"),
        "ttft_ms": _measurement_field(measurements, "ttft"),
        "prefill_pcc": _measurement_field(measurements, "prefill_pcc"),
        "decode_pcc": _measurement_field(measurements, "decode_pcc"),
        "prefill_pcc_target": _measurement_field(measurements, "prefill_pcc", "target"),
        "decode_pcc_target": _measurement_field(measurements, "decode_pcc", "target"),
        "device_type": device_info.get("device_type"),
        "device_count": device_info.get("device_count"),
        "arch": device_info.get("arch"),
        "mesh_shape": device_info.get("mesh_shape"),
    }


# --------------------------------------------------------------------------- #
# Resume support: skip tests whose MLIRs are already on disk
# --------------------------------------------------------------------------- #


def _has_existing_mlirs(
    test_name: str, output_dir: Path, group: str, ttnn_dir: Path
) -> bool:
    """True if every expected MLIR (TTIR + TTNN, prefill+decode or encoder)
    is already present for this test."""
    expected = ["encoder"] if group == "encoder" else ["prefill", "decode"]
    model_key = (test_name[5:] if test_name.startswith("test_") else test_name).lower()

    def _has(d: Path, kind: str) -> bool:
        if not d.exists():
            return False
        for f in d.glob("*.mlir"):
            stem = f.stem.lower()
            if stem.startswith(f"{model_key}_") and "1lyr" in stem and kind in stem:
                return True
        return False

    return all(_has(output_dir, k) and _has(ttnn_dir, k) for k in expected)


# --------------------------------------------------------------------------- #
# Test execution
# --------------------------------------------------------------------------- #


def _run_tests(
    group: str,
    test_file_path: Path,
    test_names: list[str],
    export_path: Path,
    output_dir: Path,
    ttnn_output_dir: Path,
    include_tests: set[str],
    resume: bool,
    per_test_json_dir: Path,
    results: list[dict],
    on_test_done: Optional[Callable[[], None]] = None,
) -> None:
    """Run each test; append one result dict per executed test to `results`
    and call `on_test_done` after each so the caller can flush to disk."""
    test_file = str(test_file_path)

    include_lower = {t.lower() for t in include_tests}
    for name in test_names:
        if _is_test_failed(test_file_path, name):
            continue
        model_name = name[5:] if name.startswith("test_") else name
        if include_lower and model_name.lower() not in include_lower:
            continue
        if resume and _has_existing_mlirs(name, output_dir, group, ttnn_output_dir):
            continue

        per_test_json = per_test_json_dir / f"{name}.json"
        if per_test_json.exists():
            per_test_json.unlink()

        status = "ok"
        error = None
        test_start_time = time.time()
        success, error_msg, error_detail = _run_pytest(test_file, name, per_test_json)

        if not success:
            if error_msg and "num_layers override requested but ModelLoader does not support it" in error_msg:
                status, error = "unsupported", error_msg
            elif error_msg and "Test was skipped" in error_msg:
                status, error = "skipped", error_msg
            else:
                status, error = "failed", error_msg

        ttir_paths = _find_mlirs(export_path, model_name, "ttir", min_mtime=test_start_time)
        ttnn_paths = _find_mlirs(export_path, model_name, "ttnn", min_mtime=test_start_time)
        ttir_paths = _select_latest_run(ttir_paths)
        ttnn_paths = _select_latest_run(ttnn_paths)
        ttir_paths, ttir_labels = _select_g0_g1(ttir_paths, group)
        ttnn_paths, ttnn_labels = _select_g0_g1(ttnn_paths, group)
        copied_paths = _copy_mlirs(ttir_paths, output_dir, group, model_name, "ttir", ttir_labels)
        copied_ttnn_paths = _copy_mlirs(ttnn_paths, ttnn_output_dir, group, model_name, "ttnn", ttnn_labels)

        metrics = _extract_metrics(per_test_json)

        icon = "✅" if status == "ok" else ("⚠️" if status in {"skipped", "unsupported"} else "❌")
        print(f"{icon} Finished {group}::{name} -> {status}")
        sps = metrics.get("samples_per_sec")
        if sps is not None:
            ppc = metrics.get("prefill_pcc")
            dpc = metrics.get("decode_pcc")
            ppc_s = f"{ppc:.4f}" if ppc is not None else "n/a"
            dpc_s = f"{dpc:.4f}" if dpc is not None else "n/a"
            print(f"    samples/sec={sps:.3f}  prefill_pcc={ppc_s}  decode_pcc={dpc_s}")
        if status == "failed":
            if error_detail:
                print("    Failure detail:")
                for line in error_detail.splitlines():
                    print(f"      {line}")
            elif error:
                print(f"    Failure: {error}")

        results.append({
            "group": group,
            "test": name,
            "status": status,
            "error": error,
            "ttir": [str(p) for p in copied_paths],
            "ttnn": [str(p) for p in copied_ttnn_paths],
            "metrics": metrics,
        })
        if on_test_done is not None:
            on_test_done()


# --------------------------------------------------------------------------- #
# Aggregation & flushing
# --------------------------------------------------------------------------- #


def _aggregate_results(results: list[dict]) -> dict:
    device_type = arch = device_count = None
    tests = {}
    for result in results:
        metrics = result.get("metrics") or {}
        if device_type is None and metrics.get("device_type"):
            device_type = metrics.get("device_type")
            arch = metrics.get("arch")
            device_count = metrics.get("device_count")
        tests[result["test"]] = {
            "group": result["group"],
            "status": result["status"],
            "error": result.get("error"),
            "samples_per_sec": metrics.get("samples_per_sec"),
            "ttft_ms": metrics.get("ttft_ms"),
            "prefill_pcc": metrics.get("prefill_pcc"),
            "prefill_pcc_target": metrics.get("prefill_pcc_target"),
            "decode_pcc": metrics.get("decode_pcc"),
            "decode_pcc_target": metrics.get("decode_pcc_target"),
        }
    return {
        "device_type": device_type,
        "arch": arch,
        "device_count": device_count,
        "tests": tests,
    }


def _flush_results(results: list[dict], output_dir: Path, state: dict) -> Path:
    """Write aggregated results JSON; safe to call after every test.

    The path is auto-derived from device_type (measured_<device>.json). It can
    change once the first test reports a device_type (measured_unknown.json ->
    measured_llmbox.json); we delete the previous file when that happens so no
    stale partial is left behind.
    """
    aggregated = _aggregate_results(results)
    device_type = aggregated.get("device_type") or "unknown"
    path = output_dir / f"measured_{device_type}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(aggregated, indent=2))
    prev = state.get("path")
    if prev is not None and prev != path:
        try:
            prev.unlink()
        except (OSError, FileNotFoundError):
            pass
    state["path"] = path
    return path


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run single-layer pytest tests, export TTIRs and capture perf/PCC."
    )
    parser.add_argument(
        "--test", action="append", default=[],
        help=(
            "Comma-separated list of EXACT model names to run (case-insensitive, "
            "test name minus the 'test_' prefix). May be repeated. Empty = all."
        ),
    )
    parser.add_argument(
        "--continue", dest="resume", action="store_true",
        help="Resume: skip tests whose output MLIRs already exist.",
    )
    parser.add_argument("--ttirs-output-dir", type=Path, default=None)
    parser.add_argument("--ttnn-output-dir", type=Path, default=None)
    args = parser.parse_args()

    single_layer_dir, tt_xla_dir, repo_root = _resolve_paths()
    _ensure_import_paths(repo_root, tt_xla_dir)
    os.chdir(tt_xla_dir)

    export_path = (tt_xla_dir / "modules").resolve()
    output_dir = (args.ttirs_output_dir or (single_layer_dir / "generated" / "ttir")).resolve()
    ttnn_output_dir = (args.ttnn_output_dir or (single_layer_dir / "generated" / "ttnn")).resolve()
    include_tests = {s.strip() for arg in args.test for s in arg.split(",") if s.strip()}

    print(f"single-layer runner: out={output_dir} ttnn={ttnn_output_dir}")
    if include_tests:
        print(f"tests: {', '.join(sorted(include_tests))}")
    if args.resume:
        print("resume: skipping tests with existing MLIRs")

    groups = [
        ("llm",     tt_xla_dir / "test_llms.py",
         _test_names("test_llms", skip={"test_llm", "test_llm_tp"})),
        ("encoder", tt_xla_dir / "test_encoders.py",
         _test_names("test_encoders", skip={"test_encoder"}, require_param="num_layers")),
    ]

    with tempfile.TemporaryDirectory(prefix="single_layer_per_test_json_") as per_test_dir_str:
        per_test_dir = Path(per_test_dir_str)
        results: list[dict] = []
        flush_state: dict = {"path": None}

        def flush() -> None:
            _flush_results(results, output_dir, flush_state)

        for group, test_file, names in groups:
            _run_tests(
                group, test_file, names,
                export_path, output_dir, ttnn_output_dir,
                include_tests, args.resume, per_test_dir, results, flush,
            )

    # Final flush covers the "every test was filtered out" case where the
    # per-test callback never fired.
    results_json = _flush_results(results, output_dir, flush_state)
    print(f"wrote {results_json} ({len(results)} tests)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
