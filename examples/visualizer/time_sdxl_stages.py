# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Time each stage of the SDXL-Lightning demo.

`examples/pytorch/sdxl_lightning.py` marks its stages and denoising steps with
loguru `[STAGE]`/`[STEP]` lines, which the plugin's log level hides. This runs the
demo with a stand-in logger that prints them with elapsed and per-stage times,
which separates compilation from steady-state execution.

Run from the repository root:
    python examples/visualizer/time_sdxl_stages.py
"""

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "examples" / "pytorch"))

import sdxl_lightning
import torch_xla.runtime as xr

OUTPUT_PATH = "sdxl_lightning_output.png"


class StageTimer:
    """Stands in for the demo's module-level logger, timing the lines it emits."""

    def __init__(self, inner):
        self._inner = inner
        self._start = time.perf_counter()
        self._previous = self._start

    def info(self, message, *args, **kwargs):
        now = time.perf_counter()
        elapsed = now - self._start
        delta = now - self._previous
        print(f"[T+{elapsed:8.2f}s  +{delta:7.2f}s] {message}", flush=True)
        self._previous = now

    def __getattr__(self, name):
        return getattr(self._inner, name)


def main(output_path: str = OUTPUT_PATH):
    xr.set_device_type("TT")
    sdxl_lightning.logger = StageTimer(sdxl_lightning.logger)

    output_file = Path(output_path)
    if output_file.exists():
        output_file.unlink()

    sdxl_lightning.run_sdxl_lightning(output_path=output_path)
    sdxl_lightning.logger.info(f"Output image saved to {output_path}")


if __name__ == "__main__":
    main()
