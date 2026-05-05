"""Verify that streaming weight loaders cover every model parameter.

Builds a fresh skeleton, loads via the same path streaming code uses
(embed + top_level + per-layer block), and reports any state_dict keys
that were NOT covered by any loader. Excludes registered non-persistent
buffers (kv_cache, freqs_cis, etc.) which are not in the checkpoint.

Run:
    source venv/activate
    STREAM_NUM_LAYERS=4 python streaming/_verify_weight_coverage.py
"""
from __future__ import annotations

import os
from typing import Set

import torch

from tests.torch.models.deepseek_v4 import weight_loader
from third_party.tt_forge_models.deepseek_v4.modified_model import (
    model_decode_opt as mdo,
)


NUM_LAYERS = int(os.environ.get("STREAM_NUM_LAYERS", "4"))


def main():
    args = weight_loader.load_config_args()
    args.n_mtp_layers = 0
    args.max_batch_size = 1
    args.n_layers = NUM_LAYERS
    args.compress_ratios = args.compress_ratios[:NUM_LAYERS]
    args.max_seq_len = 128

    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = mdo.Transformer(args).eval()
    finally:
        torch.set_default_dtype(prev)

    # All Parameter keys (mandatory) and persistent buffer keys (also expected
    # in checkpoint) — we shouldn't miss either.
    all_params: Set[str] = set(name for name, _ in model.named_parameters())
    persistent_buffers: Set[str] = set()
    non_persistent_buffers: Set[str] = set()
    for module_name, module in model.named_modules():
        for buf_name, _ in module.named_buffers(recurse=False):
            full = f"{module_name}.{buf_name}" if module_name else buf_name
            if buf_name in module._non_persistent_buffers_set:
                non_persistent_buffers.add(full)
            else:
                persistent_buffers.add(full)

    expected_in_checkpoint = all_params | persistent_buffers

    print(f"[verify] NUM_LAYERS={NUM_LAYERS}", flush=True)
    print(f"[verify] {len(all_params)} parameters, "
          f"{len(persistent_buffers)} persistent buffers, "
          f"{len(non_persistent_buffers)} non-persistent buffers", flush=True)

    # Now apply our streaming loaders and track which keys get filled.
    covered: Set[str] = set()

    def load_and_track(target_module, sd, prefix=""):
        """Apply state_dict to a submodule; collect the corresponding
        full keys into `covered`."""
        target_module.load_state_dict(sd, strict=False, assign=True)
        for k in sd.keys():
            full = f"{prefix}{k}" if prefix else k
            covered.add(full)

    # Embed (separate loader).
    embed_sd = weight_loader.load_embed_state_dict()
    load_and_track(model.embed, embed_sd, prefix="embed.")
    print(f"[verify] embed_sd keys: {list(embed_sd.keys())}", flush=True)

    # Top-level (head/norm/hc_head_*).
    top_sd = weight_loader.load_top_level_state_dict()
    load_and_track(model, top_sd)
    print(f"[verify] top_sd keys: {list(top_sd.keys())}", flush=True)

    # Per-layer.
    for layer_id in range(NUM_LAYERS):
        block_sd = weight_loader.load_block_state_dict(layer_id)
        prefix = f"layers.{layer_id}."
        stripped = {(k[len(prefix):] if k.startswith(prefix) else k): v
                    for k, v in block_sd.items()}
        load_and_track(model.layers[layer_id], stripped, prefix=prefix)
        if layer_id == 0:
            print(f"[verify] layer 0 block_sd keys (first 10): "
                  f"{list(stripped.keys())[:10]}", flush=True)

    # Diff.
    missing = expected_in_checkpoint - covered
    extra = covered - expected_in_checkpoint

    print(f"\n[verify] === COVERAGE REPORT ===", flush=True)
    print(f"[verify] Expected in checkpoint: {len(expected_in_checkpoint)}", flush=True)
    print(f"[verify] Covered by loaders:    {len(covered)}", flush=True)
    print(f"[verify] Missing (loaded NOTHING for): {len(missing)}", flush=True)
    print(f"[verify] Extra (loaded but not in model): {len(extra)}", flush=True)

    if missing:
        print(f"\n[verify] MISSING keys (top 30):", flush=True)
        for k in sorted(missing)[:30]:
            print(f"  - {k}", flush=True)
        if len(missing) > 30:
            print(f"  ... and {len(missing) - 30} more", flush=True)

    if extra:
        print(f"\n[verify] EXTRA keys (top 30):", flush=True)
        for k in sorted(extra)[:30]:
            print(f"  + {k}", flush=True)


if __name__ == "__main__":
    main()
