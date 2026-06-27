# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Portions (c) 2025 Tenstorrent AI ULC

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

ENV_VAR_REQUEST_CAPTURE_JSON = "TTXLA_VLLM_REQUEST_CAPTURE_JSON"

CAPTURE_MODE_COMPACT = "compact"
CAPTURE_MODE_VERBOSE = "verbose"


class RequestCaptureRecorder:
    def __init__(self, startup_path: Path, events_path: Path, mode: str):
        self.startup_path = startup_path
        self.events_path = events_path
        self.mode = mode
        self.startup_path.parent.mkdir(parents=True, exist_ok=True)
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        self.startup_payload: dict[str, Any] = {
            "version": 1,
            "pid": os.getpid(),
            "mode": mode,
            "startup": None,
        }
        self.events_payload: dict[str, Any] = {
            "version": 1,
            "pid": os.getpid(),
            "mode": mode,
            "events": [],
        }

    @classmethod
    def from_env(cls) -> "RequestCaptureRecorder | None":
        raw_value = os.getenv(ENV_VAR_REQUEST_CAPTURE_JSON)
        if raw_value is None or raw_value.strip() == "":
            return None

        mode_raw = raw_value.strip().lower()
        if mode_raw == CAPTURE_MODE_VERBOSE:
            mode = CAPTURE_MODE_VERBOSE
        elif mode_raw == CAPTURE_MODE_COMPACT:
            mode = CAPTURE_MODE_COMPACT
        else:
            return None

        run_dir_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_pid{os.getpid()}"
        cwd = Path.cwd().resolve()
        root_with_generated = cwd
        if not (cwd / "generated").exists():
            for parent in (cwd, *cwd.parents):
                if (parent / "generated").exists():
                    root_with_generated = parent
                    break

        run_dir = root_with_generated / "generated" / "vllm" / run_dir_name

        startup_path = run_dir / "startup.json"
        events_path = run_dir / "events.json"
        return cls(startup_path=startup_path, events_path=events_path, mode=mode)

    @property
    def is_verbose(self) -> bool:
        return self.mode == CAPTURE_MODE_VERBOSE

    def record_startup(self, startup: dict[str, Any]) -> None:
        self.startup_payload["startup"] = self._serialize(startup)
        self._flush_startup()

    def record_event(self, event_type: str, payload: dict[str, Any]) -> None:
        events = self.events_payload["events"]
        assert isinstance(events, list)
        events.append(
            {
                "index": len(events),
                "type": event_type,
                "payload": self._serialize(payload),
            }
        )
        self._flush_events()

    def _flush_startup(self) -> None:
        tmp_path = self.startup_path.with_suffix(f"{self.startup_path.suffix}.tmp")
        tmp_path.write_text(
            json.dumps(self.startup_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp_path.replace(self.startup_path)

    def _flush_events(self) -> None:
        tmp_path = self.events_path.with_suffix(f"{self.events_path.suffix}.tmp")
        tmp_path.write_text(
            json.dumps(self.events_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp_path.replace(self.events_path)

    def _serialize(
        self, value: Any, depth: int = 0, seen: set[int] | None = None
    ) -> Any:
        if seen is None:
            seen = set()

        if value is None or isinstance(value, (bool, int, float, str)):
            return value

        if isinstance(value, (torch.dtype, torch.device, Path)):
            return str(value)

        if isinstance(value, np.generic):
            return value.item()

        if isinstance(value, torch.Tensor):
            if value.device.type == "cpu" and value.numel() <= 1024:
                return value.tolist()
            return {
                "type": "torch.Tensor",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
            }

        if isinstance(value, np.ndarray):
            if value.size <= 1024:
                return value.tolist()
            return {
                "type": "numpy.ndarray",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }

        if depth >= 6:
            return f"<{type(value).__name__}>"

        if isinstance(value, Mapping):
            return {
                str(key): self._serialize(item, depth + 1, seen)
                for key, item in sorted(value.items(), key=lambda item: str(item[0]))
            }

        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            return [self._serialize(item, depth + 1, seen) for item in value]

        if isinstance(value, set):
            return [self._serialize(item, depth + 1, seen) for item in sorted(value)]

        value_id = id(value)
        if value_id in seen:
            return f"<recursive:{type(value).__name__}>"

        seen.add(value_id)
        try:
            if hasattr(value, "__dict__"):
                data = {
                    key: val
                    for key, val in vars(value).items()
                    if not key.startswith("_") and not callable(val)
                }
                return {
                    "type": type(value).__name__,
                    "value": self._serialize(data, depth + 1, seen),
                }
        finally:
            seen.discard(value_id)

        return str(value)
