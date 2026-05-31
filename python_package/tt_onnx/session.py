# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""High-level ONNXSession API: ONNX → compile → execute on TT."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import jax
import numpy as np
import onnx

from .bridge import BridgeArtifacts, OnnxBridge, OnnxBridgeConfig
from .compiler import CompileArtifacts, compile_stablehlo_mlir, get_tt_device
from .feed_utils import prepare_feed
from .runtime import run_loaded_executable


@dataclass
class SessionConfig:
    work_dir: Path | None = None
    bridge: OnnxBridgeConfig | None = None
    compile_options: dict[str, str] = field(default_factory=dict)
    device: str = "tt"


class ONNXSession:
    """
    Compile an ONNX model once and run it on a TT device.

    Example:
        session = ONNXSession("model.onnx")
        outputs = session.run({"A": arr_a, "B": arr_b})
    """

    def __init__(
        self,
        onnx_path: str | Path,
        *,
        config: SessionConfig | None = None,
        device: str | None = None,
        compile_options: dict[str, str] | None = None,
        work_dir: str | Path | None = None,
    ) -> None:
        cfg = config or SessionConfig()
        if device is not None:
            cfg.device = device
        if compile_options is not None:
            cfg.compile_options = compile_options
        if work_dir is not None:
            cfg.work_dir = Path(work_dir)

        self._onnx_path = Path(onnx_path).resolve()
        self._config = cfg
        self._model = onnx.load(self._onnx_path)
        initializer_names = {init.name for init in self._model.graph.initializer}
        self._input_names = [
            value_info.name
            for value_info in self._model.graph.input
            if value_info.name not in initializer_names
        ]
        self._output_names = [
            value_info.name for value_info in self._model.graph.output
        ]

        work = cfg.work_dir or (
            Path(__file__).resolve().parents[2]
            / "tools"
            / "onnx"
            / "build"
            / "tt_onnx"
            / self._onnx_path.stem
        )
        bridge = OnnxBridge(cfg.bridge)
        self._bridge_artifacts: BridgeArtifacts = bridge.convert(
            self._onnx_path, work
        )

        compile_opts = {
            "mlir_input_format": "auto",
            "export_path": str(work / "export"),
            "export_model_name": self._onnx_path.stem,
            **cfg.compile_options,
        }

        self._device = get_tt_device() if cfg.device == "tt" else jax.devices(cfg.device)[0]
        self._executable, self._compile_artifacts = compile_stablehlo_mlir(
            self._bridge_artifacts.stablehlo_text,
            compile_opts,
            device=self._device,
        )

    @property
    def bridge_artifacts(self) -> BridgeArtifacts:
        return self._bridge_artifacts

    @property
    def compile_artifacts(self) -> CompileArtifacts:
        return self._compile_artifacts

    @property
    def input_names(self) -> list[str]:
        return list(self._input_names)

    @property
    def output_names(self) -> list[str]:
        return list(self._output_names)

    def run(self, feed: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        missing = [name for name in self._input_names if name not in feed]
        if missing:
            raise KeyError(f"Missing ONNX inputs: {missing}")

        prepared = prepare_feed(self._model, feed)
        ordered = [prepared[name] for name in self._input_names]
        outputs = run_loaded_executable(self._executable, self._device, ordered)
        if len(outputs) != len(self._output_names):
            raise RuntimeError(
                f"Expected {len(self._output_names)} outputs, got {len(outputs)}"
            )
        return dict(zip(self._output_names, outputs))
