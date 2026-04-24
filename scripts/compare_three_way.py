#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Three-way A/B/C comparison for a single HuggingFace model.

Compares:

  A. tt-xla native
     torch.compile(model, backend="tt")  --  no quetzal rewrite pass.

  B. tt-xla with the quetzal FX pre-pass enabled
     torch.compile(model, backend="tt", options={"tt_quetzal_rewrite_passes": "all"})
     which maps to TT_TORCH_QUETZAL_REWRITE_PASSES=all for subprocesses.

  C. tt-quetzalcoatlus alternate flow
     compile_hf.py produces a standalone TTNN generated.py; run_export.py runs
     it on device. Entirely separate compiler path, bypasses tt-xla.

All three runs share the same CPU reference output (cpu_ref.pt) and weights
(weights.pt) produced by compile_hf.py, so PCC is apples-to-apples across
variants.

Outputs a JSON artifact under --output-dir and prints a summary table.

Assumptions about the environment:
  - XLA Python venv at $XLA_PY (default /localdev/nkapre/venvs/xla/bin/python).
  - tt-quetzalcoatlus checkout at $QUETZAL_DIR (default /localdev/nkapre/tt-quetzalcoatlus).
  - This tt-xla checkout contains the overlay to source tt_torch from (and is
    where run_quetzal_rewrite_passes is defined).
  - LD_LIBRARY_PATH preconfigured to pick up the wheel's libprotobuf and
    libpython3.12 (the caller sets this once; we pass it through).

The orchestrator drives three subprocesses because torch_xla's global state
makes running multiple tt-xla compile sessions in one process unsafe.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import re
from collections import Counter
from pathlib import Path
from typing import Any


IR_TOKENS = (
    "tenstorrent.gelu_tanh",
    "tenstorrent.gelu",
    "tenstorrent.scaled_dot_product_attention",
    "ttir.gelu",
    "ttnn.gelu",
    "ttir.scaled_dot_product_attention",
    "ttnn.scaled_dot_product_attention",
    "stablehlo.dot_general",
    "stablehlo.multiply",
    "stablehlo.tanh",
    "stablehlo.power",
    "ttnn.matmul",
    "ttnn.softmax",
)


def count_ir_tokens(export_dir: Path) -> dict[str, int]:
    """Scan every .mlir under export_dir for the tokens we care about."""
    counts: Counter[str] = Counter()
    if not export_dir.exists():
        return {}
    pattern = re.compile(r"(" + "|".join(re.escape(t) for t in IR_TOKENS) + r")")
    for p in export_dir.rglob("*.mlir"):
        try:
            text = p.read_text(errors="replace")
        except OSError:
            continue
        for m in pattern.findall(text):
            counts[m] += 1
    return dict(counts)


def xla_export_dir(
    quetzal_dir: str, model_id: str, seq_len: int, batch_size: int, layer_only: bool,
) -> Path:
    safe = model_id.replace("/", "_")
    scope = "layer_only" if layer_only else "full"
    return (
        Path(quetzal_dir)
        / f"compiled_S{seq_len}_B{batch_size}"
        / safe
        / scope
        / "prefill"
        / "xla_export"
    )


def wipe_export(export_dir: Path) -> None:
    """Remove xla_export/ contents so the next run's MLIR dumps stand alone."""
    if export_dir.exists():
        for child in export_dir.rglob("*"):
            if child.is_file() or child.is_symlink():
                try:
                    child.unlink()
                except OSError:
                    pass


DEFAULTS = {
    "xla_py": os.environ.get("XLA_PY", "/localdev/nkapre/venvs/xla/bin/python"),
    "quetzal_dir": os.environ.get("QUETZAL_DIR", "/localdev/nkapre/tt-quetzalcoatlus"),
    "tt_xla_dir": str(Path(__file__).resolve().parents[1]),
}


def run_subprocess(
    cmd: list[str],
    cwd: str,
    extra_env: dict[str, str] | None = None,
    timeout: int = 1800,
) -> tuple[int, str, str]:
    """Run a subprocess, return (returncode, stdout, stderr). Both streams are captured."""
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    env.setdefault("PYTHONUNBUFFERED", "1")
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        return 124, exc.stdout or "", (exc.stderr or "") + f"\nTIMEOUT after {timeout}s"


def parse_json_tail(stdout: str) -> dict[str, Any] | None:
    """Find the last JSON object on stdout — child scripts print one terminal line."""
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def build_common_env(xla_py: str, tt_xla_dir: str, overlay_dir: str) -> dict[str, str]:
    """Env bits every XLA-path subprocess needs:

    - LD_LIBRARY_PATH reaching the wheel's libprotobuf and our uv libpython
    - TT_METAL_RUNTIME_ROOT so our overlay's pjrt_plugin_tt/__init__.py can
      find the wheel's bundled tt-metal (only relevant if the overlay path is
      also importing pjrt_plugin_tt, which the quetzal scripts don't do)
    """
    # Keep the venv bin path intact — do NOT .resolve() it here; that would
    # follow the python3.12 symlink out of the venv into uv's managed install
    # root, and site-packages would compute to the wrong directory.
    xla_py_path = Path(xla_py)
    site = xla_py_path.parent.parent / "lib" / "python3.12" / "site-packages"
    pjrt_libs = site / "pjrt_plugin_tt.libs"
    pjrt_lib64 = site / "pjrt_plugin_tt" / "lib64"
    tt_metal_bundle = site / "pjrt_plugin_tt" / "tt-metal"

    # libpython3.12.so.1.0 is needed at dlopen time for torch_xla's C ext.
    # uv's managed python distribution ships it next to the python binary.
    # Walk the symlink chain to find it.
    libpython_dir = ""
    real_py = xla_py_path.resolve()
    uv_root = real_py.parent.parent  # .../cpython-3.12.12-...
    candidate = uv_root / "lib"
    if (candidate / "libpython3.12.so.1.0").exists():
        libpython_dir = str(candidate)

    parts = [p for p in (str(pjrt_libs), str(pjrt_lib64), libpython_dir) if p]
    ld = ":".join(parts)
    prior = os.environ.get("LD_LIBRARY_PATH", "")
    if prior:
        ld = f"{ld}:{prior}"

    env = {
        "LD_LIBRARY_PATH": ld,
        "TT_METAL_RUNTIME_ROOT": str(tt_metal_bundle),
        # Our overlay is first so tt_torch resolves to this checkout (the one
        # with quetzal_rewrite.py / quetzal_analysis.py). Child scripts that
        # insert paths themselves will prepend — that's fine; they already
        # know what they need.
        "PYTHONPATH": overlay_dir,
    }
    return env


def prepare_overlay(tt_xla_dir: str, overlay_dir: str) -> None:
    """Create the selective-overlay directory that symlinks only tt_torch.

    Rationale: we want the branch's tt_torch (with quetzal code) to shadow the
    installed tt_torch. We do NOT want to shadow pjrt_plugin_tt or
    torch_plugin_tt, because those have wheel-bundled .so / tt-metal assets.
    """
    overlay = Path(overlay_dir)
    overlay.mkdir(parents=True, exist_ok=True)
    target = overlay / "tt_torch"
    src = Path(tt_xla_dir) / "python_package" / "tt_torch"
    if target.is_symlink() or target.exists():
        target.unlink()
    target.symlink_to(src)


def stage_a_b_command(
    xla_py: str, quetzal_dir: str, model_id: str, seq_len: int, batch_size: int,
    layer_only: bool, n_runs: int, opt_level: int, weight_dtype: str,
) -> list[str]:
    return [
        xla_py,
        str(Path(quetzal_dir) / "scripts" / "run_xla.py"),
        model_id,
        "--seq-len", str(seq_len),
        "--batch-size", str(batch_size),
        "--n-runs", str(n_runs),
        "--opt-level", str(opt_level),
        "--weight-dtype", weight_dtype,
        *(["--layer-only"] if layer_only else []),
    ]


def stage_compile_hf_command(
    xla_py: str, quetzal_dir: str, model_id: str, seq_len: int, batch_size: int,
    layer_only: bool,
) -> list[str]:
    # run_xla.py reads artifacts from compiled_S{S}_B{B}/...; compile_hf.py's
    # default --output-dir is "compiled" (no S/B). Pin it explicitly so both
    # scripts see the same tree.
    out = f"compiled_S{seq_len}_B{batch_size}"
    return [
        xla_py,
        str(Path(quetzal_dir) / "scripts" / "compile_hf.py"),
        model_id,
        "--seq-len", str(seq_len),
        "--batch-size", str(batch_size),
        "--output-dir", out,
        "--jsonl",
        *(["--layer-only"] if layer_only else []),
    ]


def stage_c_export_command(
    xla_py: str, quetzal_dir: str, generated_py: Path, timeout: int,
) -> list[str]:
    return [
        xla_py,
        str(Path(quetzal_dir) / "scripts" / "run_export.py"),
        str(generated_py),
        "--timeout", str(timeout),
    ]


def find_generated_py(
    quetzal_dir: str, model_id: str, seq_len: int, batch_size: int, layer_only: bool,
) -> Path:
    safe = model_id.replace("/", "_")
    scope = "layer_only" if layer_only else "full"
    return (
        Path(quetzal_dir)
        / f"compiled_S{seq_len}_B{batch_size}"
        / safe
        / scope
        / "prefill"
        / "generated.py"
    )


def summarize(results: dict[str, Any]) -> str:
    def fmt(v: Any, width: int, spec: str = "") -> str:
        if v is None:
            s = "-"
        elif isinstance(v, float):
            s = f"{v:{spec}}" if spec else str(v)
        else:
            s = str(v)
        return s[:width].ljust(width)

    hdr = f"{'variant':<8} {'status':<8} {'runtime_ms':>12} {'pcc':>10} {'ir_tokens':>10}  notes"
    lines = [hdr, "-" * len(hdr)]
    for key in ("A", "B", "C"):
        r = results.get(key) or {}
        status = r.get("status", "SKIP")
        rt = r.get("runtime_ms")
        pcc = r.get("pcc")
        # Prefer ir_token_counts (scavenged from MLIR dumps by the orchestrator);
        # fall back to hlo_op_breakdown if run_xla captured anything.
        ir = None
        ir_counts = r.get("ir_token_counts") or {}
        if ir_counts:
            ir = sum(ir_counts.values())
        elif isinstance(r.get("hlo_op_breakdown"), dict):
            ir = r.get("hlo_ops_total") or sum(r["hlo_op_breakdown"].values())
        note = r.get("error") or r.get("note") or ""
        lines.append(
            f"{key:<8} {fmt(status, 8)} {fmt(rt, 12, '.3f' if isinstance(rt, (int, float)) else '')} "
            f"{fmt(pcc, 10, '.6f' if isinstance(pcc, (int, float)) else '')} "
            f"{fmt(ir, 10)}  {note[:60]}"
        )
    return "\n".join(lines)


def composite_deltas(
    a: dict[str, Any], b: dict[str, Any]
) -> list[tuple[str, int, int, int]]:
    """For each tracked IR token, show A → B counts."""
    a_counts = (a or {}).get("ir_token_counts") or (a or {}).get("hlo_op_breakdown") or {}
    b_counts = (b or {}).get("ir_token_counts") or (b or {}).get("hlo_op_breakdown") or {}
    out = []
    for t in IR_TOKENS:
        av, bv = a_counts.get(t, 0), b_counts.get(t, 0)
        if av or bv:
            out.append((t, av, bv, bv - av))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_id", help="HuggingFace model id, e.g. meta-llama/Llama-3.1-8B")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--layer-only", action="store_true", default=True,
                        help="(Default.) Compile first decoder layer only.")
    parser.add_argument("--full", dest="layer_only", action="store_false",
                        help="Compile the full (possibly truncated) model.")
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--opt-level", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--weight-dtype", default="bfp_bf8", choices=["bf16", "bfp_bf8"])
    parser.add_argument("--output-dir", default="/tmp/quetzal-threeway",
                        help="Where summary.json and logs are written.")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="Per-subprocess timeout (seconds).")
    parser.add_argument("--skip-c", action="store_true",
                        help="Skip the quetzalcoatlus export flow (only A/B).")
    parser.add_argument("--xla-py", default=DEFAULTS["xla_py"])
    parser.add_argument("--quetzal-dir", default=DEFAULTS["quetzal_dir"])
    args = parser.parse_args()

    tt_xla_dir = DEFAULTS["tt_xla_dir"]
    out_dir = Path(args.output_dir) / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = str(out_dir / "_overlay")
    prepare_overlay(tt_xla_dir, overlay_dir)

    common_env = build_common_env(args.xla_py, tt_xla_dir, overlay_dir)

    results: dict[str, Any] = {
        "model_id": args.model_id,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "layer_only": args.layer_only,
        "n_runs": args.n_runs,
        "opt_level": args.opt_level,
        "weight_dtype": args.weight_dtype,
        "output_dir": str(out_dir),
    }

    # ── Step 0: compile_hf.py to produce cpu_ref.pt + weights.pt + generated.py
    # (generated.py only needed for variant C; cpu_ref.pt + weights.pt are used
    # by variants A and B too, so this step is required.)
    print(f"[compile_hf] starting for {args.model_id} ...", flush=True)
    rc, so, se = run_subprocess(
        stage_compile_hf_command(
            args.xla_py, args.quetzal_dir, args.model_id,
            args.seq_len, args.batch_size, args.layer_only,
        ),
        cwd=args.quetzal_dir,
        extra_env=common_env,
        timeout=args.timeout,
    )
    (out_dir / "compile_hf.stdout.log").write_text(so)
    (out_dir / "compile_hf.stderr.log").write_text(se)
    compile_hf_json = parse_json_tail(so)
    results["compile_hf"] = {"returncode": rc, "last_json": compile_hf_json}
    # compile_hf.py uses status "OK" in --jsonl mode, not "PASS" like the XLA
    # scripts. Trust the generated.py + cpu_ref.pt artifacts as the real signal.
    generated_py = find_generated_py(
        args.quetzal_dir, args.model_id,
        args.seq_len, args.batch_size, args.layer_only,
    )
    cpu_ref = generated_py.parent / "cpu_ref.pt"
    if rc != 0 or not generated_py.exists() or not cpu_ref.exists():
        status = (compile_hf_json or {}).get("status", "unknown")
        err = (compile_hf_json or {}).get("error", "")
        print(
            f"[compile_hf] FAILED (rc={rc}, status={status}, err={err[:120]}). "
            f"Missing: gen={generated_py.exists()} cpu_ref={cpu_ref.exists()}. "
            f"See {out_dir}/compile_hf.*.log",
            flush=True,
        )
        (out_dir / "summary.json").write_text(json.dumps(results, indent=2, default=str))
        return 2

    export_dir = xla_export_dir(
        args.quetzal_dir, args.model_id,
        args.seq_len, args.batch_size, args.layer_only,
    )

    # ── Variant A: tt-xla native ────────────────────────────────────────────
    print("[A] tt-xla native ...", flush=True)
    wipe_export(export_dir)
    env_a = dict(common_env)
    env_a.pop("TT_TORCH_QUETZAL_REWRITE_PASSES", None)
    rc, so, se = run_subprocess(
        stage_a_b_command(
            args.xla_py, args.quetzal_dir, args.model_id,
            args.seq_len, args.batch_size, args.layer_only, args.n_runs,
            args.opt_level, args.weight_dtype,
        ),
        cwd=args.quetzal_dir,
        extra_env=env_a,
        timeout=args.timeout,
    )
    (out_dir / "A.stdout.log").write_text(so)
    (out_dir / "A.stderr.log").write_text(se)
    a_json = parse_json_tail(so) or {"status": "FAIL", "error": f"no JSON; rc={rc}"}
    a_json["returncode"] = rc
    a_json["ir_token_counts"] = count_ir_tokens(export_dir)
    # Snapshot the A IR dumps so they don't get clobbered by B.
    a_snapshot = out_dir / "A_mlir"
    if export_dir.exists():
        shutil.copytree(export_dir, a_snapshot, dirs_exist_ok=True)
    results["A"] = a_json

    # ── Variant B: tt-xla + quetzal pre-pass ────────────────────────────────
    print("[B] tt-xla + quetzal pre-pass ...", flush=True)
    wipe_export(export_dir)
    env_b = dict(common_env)
    env_b["TT_TORCH_QUETZAL_REWRITE_PASSES"] = "all"
    rc, so, se = run_subprocess(
        stage_a_b_command(
            args.xla_py, args.quetzal_dir, args.model_id,
            args.seq_len, args.batch_size, args.layer_only, args.n_runs,
            args.opt_level, args.weight_dtype,
        ),
        cwd=args.quetzal_dir,
        extra_env=env_b,
        timeout=args.timeout,
    )
    (out_dir / "B.stdout.log").write_text(so)
    (out_dir / "B.stderr.log").write_text(se)
    b_json = parse_json_tail(so) or {"status": "FAIL", "error": f"no JSON; rc={rc}"}
    b_json["returncode"] = rc
    b_json["ir_token_counts"] = count_ir_tokens(export_dir)
    b_snapshot = out_dir / "B_mlir"
    if export_dir.exists():
        shutil.copytree(export_dir, b_snapshot, dirs_exist_ok=True)
    results["B"] = b_json

    # ── Variant C: quetzalcoatlus export flow ───────────────────────────────
    if not args.skip_c:
        generated_py = find_generated_py(
            args.quetzal_dir, args.model_id,
            args.seq_len, args.batch_size, args.layer_only,
        )
        if not generated_py.exists():
            results["C"] = {
                "status": "FAIL",
                "error": f"generated.py missing at {generated_py}",
            }
        else:
            print(f"[C] quetzalcoatlus export flow ({generated_py.name}) ...", flush=True)
            rc, so, se = run_subprocess(
                stage_c_export_command(
                    args.xla_py, args.quetzal_dir, generated_py, args.timeout,
                ),
                cwd=args.quetzal_dir,
                extra_env=common_env,
                timeout=args.timeout,
            )
            (out_dir / "C.stdout.log").write_text(so)
            (out_dir / "C.stderr.log").write_text(se)
            c_json = parse_json_tail(so) or {"status": "FAIL", "error": f"no JSON; rc={rc}"}
            c_json["returncode"] = rc
            results["C"] = c_json
    else:
        results["C"] = {"status": "SKIP", "note": "--skip-c"}

    # ── Table + composite deltas ────────────────────────────────────────────
    print()
    print(summarize(results))
    print()
    deltas = composite_deltas(results.get("A", {}), results.get("B", {}))
    if deltas:
        print("Composite-op deltas A → B (stablehlo / tenstorrent tokens):")
        for tok, a_c, b_c, d in deltas:
            sign = "+" if d > 0 else ""
            print(f"  {tok:<46} {a_c:>6} -> {b_c:<6}  ({sign}{d})")
    else:
        print("No HLO op counts captured from either A or B (see logs).")

    (out_dir / "summary.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nArtifacts: {out_dir}")
    print(f"Summary:   {out_dir}/summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
