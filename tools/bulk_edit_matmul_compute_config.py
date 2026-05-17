#!/usr/bin/env python3
"""Rewrite the `compute_config` on every `ttnn.matmul` in dumped TTNN IR.

Reads `modules/irs/ttnn_*_g*_<ts>.mlir` (final TTNN stage; ignores
`ttnn_runtime_*`), rewrites the `compute_config = #ttnn.device_compute_kernel_config<...>`
attribute to `<math_fidelity = lofi, fp32_dest_acc_en = false>` on lines that
contain `"ttnn.matmul"`, and writes the result to
`flatbuffers/<model_name>_g<N>.mlir` (deterministic name, no timestamp). Other
ops that carry `compute_config` are left alone.

Run from the repo root after running a benchmark with `export_path=modules`.
"""
import re
from pathlib import Path

CC_PATTERN = re.compile(
    r"compute_config = #ttnn\.device_compute_kernel_config<[^>]+>"
)
REPLACEMENT = (
    "compute_config = #ttnn.device_compute_kernel_config<"
    "math_fidelity = lofi, fp32_dest_acc_en = false>"
)


def rewrite_matmul_compute_configs(text: str) -> tuple[str, int]:
    n_subs = 0
    out_lines = []
    for line in text.splitlines(keepends=True):
        if '"ttnn.matmul"' in line:
            new, k = CC_PATTERN.subn(REPLACEMENT, line)
            n_subs += k
            out_lines.append(new)
        else:
            out_lines.append(line)
    return "".join(out_lines), n_subs


def main() -> None:
    src = Path("modules/irs")
    dst = Path("flatbuffers")
    dst.mkdir(exist_ok=True)

    pattern = "ttnn_*_g*_*.mlir"
    files = [p for p in src.glob(pattern) if "_runtime_" not in p.name]
    if not files:
        raise SystemExit(f"No files matching {src}/{pattern} (excluding _runtime_).")

    for mlir in sorted(files):
        # Strip leading "ttnn_" and the trailing "_<timestamp>" off the stem.
        stem = re.sub(r"_\d+$", "", mlir.stem[len("ttnn_"):])
        out_path = dst / f"{stem}.mlir"
        new_text, n_subs = rewrite_matmul_compute_configs(mlir.read_text())
        out_path.write_text(new_text)
        n_matmul = new_text.count('"ttnn.matmul"')
        print(f"{mlir.name} -> {out_path.name}   "
              f"({n_subs}/{n_matmul} matmul compute_configs rewritten)")


if __name__ == "__main__":
    main()
