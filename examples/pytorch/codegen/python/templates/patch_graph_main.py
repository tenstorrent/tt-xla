# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Post-process generated ``graph_0/main.py`` for tt-metal."""

from __future__ import annotations

from pathlib import Path

_COMPARE_BLOCK = '''

def compare_layer0_ln_attn_stages(ttnn_outputs):
    """TTNN vs ``Layer0LnAttnNoDep`` CPU golden (same module as codegen)."""
    from cpu_reference.layer0_cpu import compare_layer0_ln_attn_stages as _compare

    return _compare(ttnn_outputs)
'''

_BOOTSTRAP_PREFIX = '''import importlib.util
import sys
from pathlib import Path

_GRAPH_DIR = Path(__file__).resolve().parent
_CODEGEN_ROOT = _GRAPH_DIR.parent
for _path in (_GRAPH_DIR, _CODEGEN_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _load_graph_module(module_name: str, filename: str):
    path = _GRAPH_DIR / filename
    spec = importlib.util.spec_from_file_location(f"codegen_graph_{module_name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"codegen_graph_{module_name}"] = mod
    spec.loader.exec_module(mod)
    return mod


'''


def patch_graph_main(main_path: Path) -> None:
    text = main_path.read_text()

    if "_load_graph_module" not in text:
        text = text.replace(
            "import ttnn\nimport utils\nimport ttir_cpu\nimport torch\n",
            _BOOTSTRAP_PREFIX
            + "import ttnn\nimport torch\n\n"
            + "ttir_cpu = _load_graph_module(\"ttir_cpu\", \"ttir_cpu.py\")\n"
            + "utils = _load_graph_module(\"utils\", \"utils.py\")\n\n",
        )

    if "device = utils.DeviceGetter.get_device((1, 1))" not in text:
        text = text.replace(
            "def consteval__main(ce_cache, weights):\n    if not ce_cache:\n        main_const_eval_0_0 = main_const_eval_0(",
            "def consteval__main(ce_cache, weights):\n    if not ce_cache:\n        device = utils.DeviceGetter.get_device((1, 1))\n        main_const_eval_0_0 = main_const_eval_0(",
        )
        text = text.replace(
            "            ]\n        )\n        ce_cache[\"main_const_eval_0\"]",
            "            ],\n            device,\n        )\n        ce_cache[\"main_const_eval_0\"]",
            1,
        )
        text = text.replace(
            "main_const_eval_1_0 = main_const_eval_1()",
            "main_const_eval_1_0 = main_const_eval_1(device)",
        )
        text = text.replace(
            "main_const_eval_2_0 = main_const_eval_2(\n            [weights[\"L__self___rotary_emb_inv_freq\"]]\n        )",
            "main_const_eval_2_0 = main_const_eval_2(\n            [weights[\"L__self___rotary_emb_inv_freq\"]],\n            device,\n        )",
        )

    if "compare_layer0_ln_attn_stages" not in text:
        text = text.replace(
            "\ndef main():\n    load_activations_for__main_0",
            _COMPARE_BLOCK + "\n\ndef main():\n    load_activations_for__main_0",
        )

    if "compare_layer0_ln_attn_stages(ttnn_outputs)" not in text:
        for old, new in (
            (
                "_main_0 = _main(load_activations_for__main_0, load_weights_for__main_0)\n"
                "    return _main_0",
                "ttnn_outputs = _main(load_activations_for__main_0, load_weights_for__main_0)\n"
                "    compare_layer0_ln_attn_stages(ttnn_outputs)\n"
                "    return ttnn_outputs",
            ),
            (
                "ttnn_outputs = _main(load_activations_for__main_0, load_weights_for__main_0)\n"
                "    return ttnn_outputs",
                "ttnn_outputs = _main(load_activations_for__main_0, load_weights_for__main_0)\n"
                "    compare_layer0_ln_attn_stages(ttnn_outputs)\n"
                "    return ttnn_outputs",
            ),
        ):
            if old in text:
                text = text.replace(old, new, 1)
                break

    main_path.write_text(text)
