#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Extract TTIR, TTNN, and post-create-d2m-subgraphs IR for a hardcoded set of
LLM benchmark tests, without requiring the full pipeline to complete.

Mechanism
---------
Stage snapshots (TTIR, TTNN) come from the existing ``export_path`` compile
option already wired in llm_benchmark.py.  The ``create-d2m-subgraphs`` pass
runs *inside* the TTIR-to-TTNN pipeline, so if that pipeline fails after the
pass the TTNN snapshot is never written.

To guarantee per-pass granularity the script enables ``TTXLA_LOGGER_LEVEL=VERBOSE``
in the pytest subprocess.  At that log level the PJRT plugin calls
``pm.enableIRPrinting()`` on the TTIR-to-TTNN pass manager, which writes
before/after IR for every pass to stderr.  The script captures stderr to a
per-model log file and then extracts the sections tagged
``IR Dump After TTNNCreateD2MSubgraphs``, saving each one as a separate
``d2m/d2m_after_<graph>.mlir`` file.  This works even when the pipeline fails
after that pass.

Early-exit mode (default)
-------------------------
With ``--early-exit`` (on by default) the script monitors the stderr log and
the ``modules/irs/`` directory while the subprocess runs.  Once compilation
activity stops (no new IR files or log growth for ``--early-exit-grace``
seconds) the subprocess is killed, avoiding the post-compilation warmup,
execution, and PCC-check phases entirely.  Use ``--no-early-exit`` to let
each test run to completion.

Output layout (per model)
--------------------------
  <output_dir>/<model_name>/
    ttir/   - TTIR stage snapshots (input to the TTIR->TTNN pipeline)
    ttnn/   - TTNN stage snapshots (only present when full pipeline completes)
    d2m/    - IR captured immediately after create-d2m-subgraphs

Usage
-----
On hardware:
    python tests/benchmark/scripts/extract_d2m_ir.py [--output-dir DIR]

Compile-only (no hardware):
    TT_COMPILE_ONLY_SYSTEM_DESC=/path/to/system.ttsys \\
        python tests/benchmark/scripts/extract_d2m_ir.py
"""

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import pytest

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HARDCODED_TESTS: list[str] = [
    "test_mistral_small_24b_instruct_2501_tp",
]

CRITICAL_ERRORS = [
    "Read unexpected run_mailbox value from core",
    "Timeout waiting for Ethernet core service remote IO request",
]

DEFAULT_TIMEOUT_SECONDS = 1800
DEFAULT_NUM_LAYERS = 1
DEFAULT_MAX_OUTPUT_TOKENS = 1
DEFAULT_EARLY_EXIT_GRACE = 120

# Marker written by MLIR's pm.enableIRPrinting() to stderr.
_D2M_AFTER_MARKER = "IR Dump After TTNNCreateD2MSubgraphs"
_IR_DUMP_HEADER = "// -----// IR Dump "

# Markers in stderr that indicate the TTIR-to-TTNN pipeline finished (or failed)
# for a given graph. Used for early-exit heuristic.
_PIPELINE_DONE_MARKERS = [
    "ERR| Failed to run TTIRToTTNNCommon pipeline",
    "ERR| Failed to run TTIRToTTNNRuntime pipeline",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _handle_critical_error() -> None:
    print(
        f"\n{'='*60}\n"
        f"DEVICE ERROR: TT device needs reset\n"
        f"{'='*60}\n"
        f"\n"
        f"  >>> Run: tt-smi -r\n"
        f"  >>> Then re-run the script.\n"
        f"\n"
        f"{'='*60}\n"
    )
    sys.exit(1)


def _resolve_paths() -> tuple[Path, Path, Path]:
    """Return ``(benchmark_dir, repo_root, test_file)``."""
    scripts_dir = Path(__file__).resolve().parent
    benchmark_dir = scripts_dir.parent
    repo_root = benchmark_dir.parents[1]
    test_file = benchmark_dir / "test_llms.py"
    return benchmark_dir, repo_root, test_file


def _model_name_from_test(test_name: str) -> str:
    return test_name[5:] if test_name.startswith("test_") else test_name


def _extract_error_line(output_lines: list[str]) -> str:
    """Return the most useful single-line error summary from pytest output."""

    def _is_frame_or_noise(s: str) -> bool:
        return s.startswith('File "') or len(s) > 300

    for line in output_lines:
        s = line.strip()
        if s.startswith("E ") or s.startswith("E\t"):
            candidate = s[1:].strip()
            if ": " in candidate and not _is_frame_or_noise(candidate):
                return candidate

    exception_starts = (
        "ValueError:",
        "RuntimeError:",
        "AssertionError:",
        "KeyError:",
        "TypeError:",
        "Error:",
    )
    for line in output_lines:
        s = line.strip()
        if any(s.startswith(p) for p in exception_starts) and not _is_frame_or_noise(s):
            return s

    for line in reversed(output_lines):
        s = line.strip()
        if s.startswith("FAILED ") and " - " in s and not _is_frame_or_noise(s):
            return s.split(" - ", 1)[1]

    for line in reversed(output_lines):
        s = line.strip()
        if s and not _is_frame_or_noise(s):
            return s

    return "Test failed"


def _count_ir_files(export_path: Path, min_mtime: float) -> int:
    """Count new IR files (ttir + ttnn, excluding ttnn_runtime) in the irs/ dir."""
    irs_dir = export_path / "irs"
    if not irs_dir.exists():
        return 0
    count = 0
    for path in irs_dir.iterdir():
        if not path.name.endswith(".mlir"):
            continue
        if path.name.startswith("ttnn_runtime"):
            continue
        if not (path.name.startswith("ttir_") or path.name.startswith("ttnn_")):
            continue
        try:
            if path.stat().st_mtime >= min_mtime:
                count += 1
        except FileNotFoundError:
            pass
    return count


def _stderr_log_size(log_path: Path) -> int:
    """Return current size of the stderr log, or 0 if it doesn't exist."""
    try:
        return log_path.stat().st_size
    except (FileNotFoundError, OSError):
        return 0


def _kill_process_tree(proc: subprocess.Popen) -> None:
    """Kill the subprocess and its children."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, OSError):
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
        proc.wait(timeout=5)


def _run_pytest(
    test_file: Path,
    test_name: str,
    num_layers: int,
    max_output_tokens: int,
    timeout: int,
    cwd: Path,
    repo_root: Path,
    stderr_log_file: Optional[Path] = None,
    early_exit: bool = False,
    early_exit_grace: int = DEFAULT_EARLY_EXIT_GRACE,
    export_path: Optional[Path] = None,
) -> tuple[bool, Optional[str], Optional[str]]:
    """Run pytest for a single test.

    When ``stderr_log_file`` is provided the subprocess runs with
    ``TTXLA_LOGGER_LEVEL=VERBOSE`` and its stderr is written to that file.

    When ``early_exit`` is True, the subprocess is killed once no new IR
    activity (new files in ``export_path/irs/`` or growth in the stderr log)
    has been observed for ``early_exit_grace`` seconds.  This avoids waiting
    for the post-compilation execution and PCC-check phases.

    Returns ``(success, error_summary, error_tail)``.
    """
    test_path = f"{test_file}::{test_name}"
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-x",
        "-v",
        "-s",
        "--confcutdir",
        str(cwd),
        test_path,
        "--optimization-level",
        "1",
        "--num-layers",
        str(num_layers),
        "--enable-d2m-subgraphs",
        "--max-output-tokens",
        str(max_output_tokens),
        "--output-file",
        "",
    ]

    env = os.environ.copy()
    extra_paths = [str(cwd), str(repo_root)]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        [p for p in extra_paths if p] + ([existing] if existing else [])
    )
    if stderr_log_file is not None:
        env["TTXLA_LOGGER_LEVEL"] = "VERBOSE"

    stderr_fh = None
    killed_early = False
    start_time = time.time()

    try:
        if stderr_log_file is not None:
            stderr_log_file.parent.mkdir(parents=True, exist_ok=True)
            stderr_fh = open(stderr_log_file, "w", errors="replace")

        if not early_exit:
            # Blocking mode: wait for the subprocess to finish.
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=stderr_fh if stderr_fh is not None else subprocess.PIPE,
                text=True,
                timeout=timeout,
                env=env,
                cwd=str(cwd),
            )
            returncode = result.returncode
            stdout_text = result.stdout or ""
        else:
            # Early-exit mode: poll for IR activity, kill when idle.
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=stderr_fh if stderr_fh is not None else subprocess.PIPE,
                text=True,
                env=env,
                cwd=str(cwd),
                preexec_fn=os.setsid,
            )

            last_activity_time = time.time()
            prev_ir_count = 0
            prev_log_size = 0
            # Don't start the grace countdown until we've seen at least one
            # IR file or some stderr log growth (i.e., compilation has started).
            compilation_started = False
            poll_interval = 5

            while True:
                ret = proc.poll()
                if ret is not None:
                    break

                elapsed = time.time() - start_time
                if elapsed > timeout:
                    _kill_process_tree(proc)
                    return False, f"Test timed out after {timeout}s", None

                # Check for new IR activity.
                cur_ir_count = (
                    _count_ir_files(export_path, start_time) if export_path else 0
                )
                cur_log_size = (
                    _stderr_log_size(stderr_log_file) if stderr_log_file else 0
                )

                if cur_ir_count > prev_ir_count or cur_log_size > prev_log_size:
                    last_activity_time = time.time()
                    if cur_ir_count > prev_ir_count or cur_log_size > 1_000_000:
                        compilation_started = True
                    prev_ir_count = cur_ir_count
                    prev_log_size = cur_log_size

                idle_seconds = time.time() - last_activity_time
                if compilation_started and idle_seconds >= early_exit_grace:
                    elapsed_total = time.time() - start_time
                    print(
                        f"    [early-exit] No new IR for {early_exit_grace}s "
                        f"(ir_files={cur_ir_count}, log={cur_log_size // 1_000_000}MB), "
                        f"killing subprocess after {elapsed_total:.0f}s total"
                    )
                    _kill_process_tree(proc)
                    killed_early = True
                    break

                time.sleep(poll_interval)

            stdout_text = ""
            if proc.stdout:
                try:
                    stdout_text = proc.stdout.read() or ""
                except Exception:
                    pass
            returncode = proc.returncode if proc.returncode is not None else -1

    except subprocess.TimeoutExpired:
        return False, f"Test timed out after {timeout}s", None
    except Exception as exc:
        error_msg = str(exc)
        if any(critical in error_msg for critical in CRITICAL_ERRORS):
            _handle_critical_error()
        return False, f"{type(exc).__name__}: {error_msg}", None
    finally:
        if stderr_fh is not None:
            stderr_fh.close()

    # Merge a portion of stderr for critical-error scanning.
    stderr_sample = ""
    if stderr_log_file and stderr_log_file.exists():
        try:
            with open(stderr_log_file, errors="replace") as f:
                stderr_sample = f.read(500_000)
        except OSError:
            pass
    elif not early_exit:
        stderr_sample = result.stderr or ""

    if killed_early:
        return True, "(killed early — IR collected)", None

    combined = stdout_text + "\n" + stderr_sample
    if returncode == 0:
        return True, None, None
    skipped_code = getattr(pytest.ExitCode, "SKIPPED", 5)
    if returncode == skipped_code:
        return False, "Test was skipped", None
    if any(critical in combined for critical in CRITICAL_ERRORS):
        _handle_critical_error()

    output_lines = [line for line in combined.split("\n") if line.strip()]
    tail = "\n".join(output_lines[-30:]) if output_lines else None
    error_msg = _extract_error_line(output_lines)
    return False, error_msg, tail


def _extract_d2m_sections(log_file: Path, output_dir: Path) -> list[Path]:
    """Parse the verbose IR log and save each post-create-d2m-subgraphs section.

    MLIR's pm.enableIRPrinting() writes lines like:
        // -----// IR Dump After TTNNCreateD2MSubgraphs (...) //----- //
    followed by the full module text, then the next section header.

    Each section found is written to ``output_dir/d2m_after_g<N>.mlir``.
    Returns a list of paths that were written.
    """
    if not log_file.exists():
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    section_buf: list[str] = []
    in_section = False
    graph_idx = 0

    def _flush(buf: list[str], idx: int) -> Optional[Path]:
        if not buf:
            return None
        text = "".join(buf).strip()
        if not text:
            return None
        out_path = output_dir / f"d2m_after_g{idx}.mlir"
        out_path.write_text(text + "\n", encoding="utf-8")
        return out_path

    with open(log_file, errors="replace") as fh:
        for line in fh:
            if _IR_DUMP_HEADER in line:
                if in_section:
                    path = _flush(section_buf, graph_idx)
                    if path:
                        saved.append(path)
                        graph_idx += 1
                    section_buf = []
                    in_section = False

                if _D2M_AFTER_MARKER in line:
                    in_section = True
                    section_buf = [line]
            elif in_section:
                section_buf.append(line)

    if in_section:
        path = _flush(section_buf, graph_idx)
        if path:
            saved.append(path)

    return saved


def _find_ir_files(
    export_path: Path,
    stage_prefix: str,
    min_mtime: float,
    exclude_prefixes: tuple[str, ...] = (),
) -> list[Path]:
    """Return ``*.mlir`` files in ``<export_path>/irs/`` for one stage."""
    irs_dir = export_path / "irs"
    if not irs_dir.exists():
        return []
    matches: list[Path] = []
    for path in irs_dir.glob(f"{stage_prefix}_*.mlir"):
        if exclude_prefixes and any(
            path.stem.startswith(prefix) for prefix in exclude_prefixes
        ):
            continue
        try:
            if path.stat().st_mtime < min_mtime:
                continue
        except FileNotFoundError:
            continue
        matches.append(path)
    return sorted(matches, key=lambda p: p.stat().st_mtime)


def _copy_irs(ir_paths: list[Path], output_dir: Path) -> list[Path]:
    """Copy IR files into ``output_dir``, disambiguating name collisions."""
    if not ir_paths:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for path in ir_paths:
        target = output_dir / path.name
        if target.exists():
            stem, suffix = target.stem, target.suffix
            counter = 2
            while True:
                candidate = output_dir / f"{stem}_v{counter}{suffix}"
                if not candidate.exists():
                    target = candidate
                    break
                counter += 1
        shutil.copy2(path, target)
        copied.append(target)
    return copied


def _print_summary(results: list[dict]) -> None:
    print("\n" + "=" * 80)
    print("D2M IR Extraction Summary")
    print("=" * 80)
    for result in results:
        status = result["status"]
        icon = {"ok": "PASS", "partial": "WARN"}.get(status, "FAIL")
        ttir_count = len(result["ttir"])
        ttnn_count = len(result["ttnn"])
        d2m_count = len(result["d2m"])
        print(
            f"[{icon}] {result['test']}: status={status} "
            f"ttir={ttir_count} ttnn={ttnn_count} d2m={d2m_count}"
        )
        if result["error"]:
            print(f"        error: {result['error']}")
    print("=" * 80)
    n_total = len(results)
    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_partial = sum(1 for r in results if r["status"] == "partial")
    n_failed = sum(1 for r in results if r["status"] == "failed")
    print(f"Total: {n_total} | ok: {n_ok} | partial: {n_partial} | failed: {n_failed}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compile a hardcoded list of LLM benchmark tests and extract "
            "TTIR, TTNN, and post-create-d2m-subgraphs IR files. "
            "Each test runs in its own subprocess so failures are isolated. "
            "Verbose PJRT logging is enabled to guarantee d2m IR capture even "
            "when the full TTIR-to-TTNN pipeline does not complete."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("d2m_ir_output"),
        help="Directory to copy extracted IR files into (default: d2m_ir_output).",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=DEFAULT_NUM_LAYERS,
        help=f"Override model layer count for faster compilation (default: {DEFAULT_NUM_LAYERS}).",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        help=f"Cap generation steps (default: {DEFAULT_MAX_OUTPUT_TOKENS}).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Per-test timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS}).",
    )
    parser.add_argument(
        "--tests",
        nargs="+",
        default=None,
        help="Override the hardcoded test list (space-separated test function names).",
    )
    parser.add_argument(
        "--keep-logs",
        action="store_true",
        default=False,
        help="Keep the verbose PJRT stderr log files after extraction (default: delete them).",
    )
    early_exit_group = parser.add_mutually_exclusive_group()
    early_exit_group.add_argument(
        "--early-exit",
        action="store_true",
        default=True,
        help="Kill the subprocess once compilation IR stops appearing (default: on).",
    )
    early_exit_group.add_argument(
        "--no-early-exit",
        action="store_false",
        dest="early_exit",
        help="Let each test run to completion (disables early-exit).",
    )
    parser.add_argument(
        "--early-exit-grace",
        type=int,
        default=DEFAULT_EARLY_EXIT_GRACE,
        help=(
            f"Seconds of no new IR activity before killing the subprocess "
            f"(default: {DEFAULT_EARLY_EXIT_GRACE})."
        ),
    )
    args = parser.parse_args()

    benchmark_dir, repo_root, test_file = _resolve_paths()
    if not test_file.exists():
        print(f"ERROR: test file not found: {test_file}", file=sys.stderr)
        return 2

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    export_path = (benchmark_dir / "modules").resolve()

    tests = args.tests if args.tests else HARDCODED_TESTS
    if not tests:
        print("ERROR: no tests selected.", file=sys.stderr)
        return 2

    print(f"Running {len(tests)} test(s).")
    print(f"  test_file        : {test_file}")
    print(f"  cwd (pytest)     : {benchmark_dir}")
    print(f"  export_path      : {export_path}")
    print(f"  output_dir       : {output_dir}")
    print(f"  num_layers       : {args.num_layers}")
    print(f"  max_output_tokens: {args.max_output_tokens}")
    print(f"  timeout          : {args.timeout}s")
    print(f"  early_exit       : {args.early_exit} (grace={args.early_exit_grace}s)")
    print(f"  verbose logging  : TTXLA_LOGGER_LEVEL=VERBOSE (stderr -> per-model log)")
    if os.environ.get("TT_COMPILE_ONLY_SYSTEM_DESC"):
        print(
            f"  compile-only     : TT_COMPILE_ONLY_SYSTEM_DESC="
            f"{os.environ['TT_COMPILE_ONLY_SYSTEM_DESC']}"
        )
    print()

    results: list[dict] = []
    for idx, test_name in enumerate(tests, 1):
        model_name = _model_name_from_test(test_name)
        model_output_dir = output_dir / model_name
        stderr_log = model_output_dir / "pjrt_verbose.log"

        print(f"[{idx}/{len(tests)}] {test_name} -> compiling ...")
        start_time = time.time()

        success, error_msg, error_tail = _run_pytest(
            test_file=test_file,
            test_name=test_name,
            num_layers=args.num_layers,
            max_output_tokens=args.max_output_tokens,
            timeout=args.timeout,
            cwd=benchmark_dir,
            repo_root=repo_root,
            stderr_log_file=stderr_log,
            early_exit=args.early_exit,
            early_exit_grace=args.early_exit_grace,
            export_path=export_path,
        )

        ttir_paths = _find_ir_files(export_path, "ttir", min_mtime=start_time)
        ttnn_paths = _find_ir_files(
            export_path,
            "ttnn",
            min_mtime=start_time,
            exclude_prefixes=("ttnn_runtime",),
        )
        copied_ttir = _copy_irs(ttir_paths, model_output_dir / "ttir")
        copied_ttnn = _copy_irs(ttnn_paths, model_output_dir / "ttnn")

        copied_d2m = _extract_d2m_sections(stderr_log, model_output_dir / "d2m")
        if not args.keep_logs and stderr_log.exists():
            stderr_log.unlink()

        if success:
            status = "ok"
            err = None
        elif copied_ttir or copied_ttnn or copied_d2m:
            status = "partial"
            err = error_msg
        else:
            status = "failed"
            err = error_msg

        if status == "failed" and error_tail:
            print("    --- tail of pytest output ---")
            for line in error_tail.splitlines():
                print(f"    {line}")
            print("    --- end tail ---")

        elapsed = time.time() - start_time
        print(
            f"    [{status}] ttir={len(copied_ttir)} ttnn={len(copied_ttnn)} "
            f"d2m={len(copied_d2m)} ({elapsed:.0f}s) -> {model_output_dir}"
        )
        if err:
            print(f"    note: {err}")

        results.append(
            {
                "test": test_name,
                "status": status,
                "error": err,
                "ttir": [str(p) for p in copied_ttir],
                "ttnn": [str(p) for p in copied_ttnn],
                "d2m": [str(p) for p in copied_d2m],
            }
        )

    _print_summary(results)
    return 0 if all(r["status"] == "ok" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
