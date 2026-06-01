# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Precompute CPU golden (``Layer0LnAttnNoDep``) for TTNN PCC.

From tt-metal repo root (``python_env`` needs **transformers 5.5.x**; no ``torch_xla``)::

  export JANUS_TTXLA_ROOT=/path/to/31_may_yyz/tt-xla
  export JANUS_LAYER0_FIXTURE_DIR=.../janus_logs/layer0_tensors/Pro_1B
  python janus_layer0_ln_attn_no_dep_codegen/cpu_reference/generate_golden.py

Or precompute in **tt-xla venv** and only load the ``.pt`` in tt-metal::

  cd tt-xla && source venv/bin/activate
  export JANUS_LAYER0_FIXTURE_DIR=.../janus_logs/layer0_tensors/Pro_1B
  python -c "
from pathlib import Path; import sys, torch
r=Path('.').resolve(); sys.path[:0]=[str(r), str(r/'examples/pytorch/codegen/python')]
from janus_layer0_build import run_forward_stacked
g=run_forward_stacked('Pro_1B')
out=Path('../31_may_tt_metal/tt-metal/janus_layer0_ln_attn_no_dep_codegen/cpu_reference/golden/stacked_stages_pro_1b.pt')
out.parent.mkdir(parents=True, exist_ok=True); torch.save(g, out); print(out, g.shape)
"
"""

from __future__ import annotations

import sys
from pathlib import Path

_CODEGEN_ROOT = Path(__file__).resolve().parent.parent
if str(_CODEGEN_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODEGEN_ROOT))

from cpu_reference.layer0_cpu import load_or_compute_cpu_golden


def _require_transformers_551() -> None:
    import transformers

    if not transformers.__version__.startswith("5.5."):
        raise SystemExit(
            f"Need transformers 5.5.x (tt-xla codegen env); got {transformers.__version__}"
        )


def main() -> None:
    _require_transformers_551()
    golden = load_or_compute_cpu_golden(refresh=True)
    print(f"CPU golden shape={tuple(golden.shape)} dtype={golden.dtype}")


if __name__ == "__main__":
    main()
