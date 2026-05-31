# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ONNX → StableHLO MLIR via onnx-mlir (Workstream 2 tools)."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OnnxBridgeConfig:
    """Paths to onnx-mlir frontend tools (from tools/onnx/env.sh)."""

    onnx_mlir: Path
    onnx_mlir_opt: Path

    @classmethod
    def from_env(cls, repo_root: Path | None = None) -> "OnnxBridgeConfig":
        root = repo_root or _default_repo_root()
        default_bin = root / "tools" / "onnx" / "build" / "install" / "bin"
        return cls(
            onnx_mlir=Path(
                os.environ.get("TT_ONNX_MLIR", default_bin / "onnx-mlir")
            ),
            onnx_mlir_opt=Path(
                os.environ.get("TT_ONNX_MLIR_OPT", default_bin / "onnx-mlir-opt")
            ),
        )

    def validate(self) -> None:
        for tool in (self.onnx_mlir, self.onnx_mlir_opt):
            if not tool.is_file() or not os.access(tool, os.X_OK):
                raise FileNotFoundError(
                    f"onnx-mlir tool not found or not executable: {tool}. "
                    "Run tools/onnx/build_onnx_mlir.sh and source tools/onnx/env.sh."
                )


@dataclass(frozen=True)
class BridgeArtifacts:
    onnx_path: Path
    work_dir: Path
    onnx_dialect_mlir: Path
    stablehlo_mlir: Path

    @property
    def stablehlo_text(self) -> str:
        return self.stablehlo_mlir.read_text(encoding="utf-8")


class OnnxBridge:
    """Wrap onnx-mlir CLI steps from tools/onnx/smoke_test.sh."""

    def __init__(self, config: OnnxBridgeConfig | None = None) -> None:
        self._config = config or OnnxBridgeConfig.from_env()
        self._config.validate()

    def convert(
        self,
        onnx_path: str | Path,
        work_dir: str | Path,
        *,
        basename: str | None = None,
    ) -> BridgeArtifacts:
        onnx_path = Path(onnx_path).resolve()
        work_dir = Path(work_dir).resolve()
        work_dir.mkdir(parents=True, exist_ok=True)

        if not onnx_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        stem = basename or onnx_path.stem
        onnx_ir_base = work_dir / stem
        onnx_dialect_mlir = Path(f"{onnx_ir_base}.onnx.mlir")
        stablehlo_mlir = work_dir / f"{stem}.stablehlo.mlir"

        # -o must be a basename without extension; onnx-mlir appends ".onnx.mlir".
        subprocess.run(
            [
                str(self._config.onnx_mlir),
                "--EmitONNXIR",
                str(onnx_path),
                "-o",
                str(onnx_ir_base),
            ],
            check=True,
        )
        if not onnx_dialect_mlir.is_file():
            raise RuntimeError(
                f"Expected ONNX dialect IR at {onnx_dialect_mlir} after onnx-mlir."
            )

        subprocess.run(
            [
                str(self._config.onnx_mlir_opt),
                str(onnx_dialect_mlir),
                "--convert-onnx-to-stablehlo",
                "-o",
                str(stablehlo_mlir),
            ],
            check=True,
        )
        if not stablehlo_mlir.is_file():
            raise RuntimeError(
                f"Expected StableHLO IR at {stablehlo_mlir} after onnx-mlir-opt."
            )

        return BridgeArtifacts(
            onnx_path=onnx_path,
            work_dir=work_dir,
            onnx_dialect_mlir=onnx_dialect_mlir,
            stablehlo_mlir=stablehlo_mlir,
        )


def _default_repo_root() -> Path:
    # python_package/tt_onnx/bridge.py → repo root is two levels up from python_package.
    return Path(__file__).resolve().parents[2]
