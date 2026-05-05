# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""End-to-end streaming inference for DeepSeek-V4-Flash.

Mirrors `tests/torch/models/deepseek_v4/test_deepseek_v4_full_e2e.py`
but uses the streaming loader to keep host RAM under ~30 GB instead of
~280 GB.

Run:
    source venv/activate
    python streaming/run_streaming.py

See ./README.md, ./DESIGN.md, ./MEMORY_BUDGET.md.
"""

from __future__ import annotations

import gc
import os
from typing import Tuple

import numpy as np
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh

from infra.utilities.torch_multichip_utils import enable_spmd
from tt_torch.sharding import sharding_constraint_hook
from tt_torch.weight_dtype import apply_weight_dtype_overrides

from tests.torch.models.deepseek_v4 import weight_loader

from streaming.streaming_loader import stream_load_transformer, _log_mem


# ----------------------------------------------------------------------------
# Same knobs as test_deepseek_v4_full_e2e.py — keep these in sync to make
# behavior diff'able with the e2e test.
# ----------------------------------------------------------------------------

PROMPT_LEN = int(os.environ.get("STREAM_PROMPT_LEN", "128"))
MAX_NEW_TOKENS = int(os.environ.get("STREAM_MAX_NEW_TOKENS", "64"))
BATCH_SIZE = int(os.environ.get("STREAM_BATCH_SIZE", "32"))
# Override via env for iterative bring-up:
#   STREAM_NUM_LAYERS=2 python streaming/run_streaming.py
NUM_LAYERS = int(os.environ.get("STREAM_NUM_LAYERS", "43"))
# When set, draw input_ids from realistic_inputs (cached natural-text
# tokenization, same source as test_transformer_prefill) instead of
# left-padded short tokenizer prompts. Default ON — left-padded short
# prompts give the model an out-of-distribution input (mostly pad) and
# tend to produce incoherent token output even when the streaming load
# itself is correct.
USE_REALISTIC_INPUTS = os.environ.get("STREAM_USE_REALISTIC", "1") == "1"

# Streaming-load only: stream-load model + apply weight dtype overrides,
# then exit. Useful to measure peak host RSS at large NUM_LAYERS without
# tripping device DRAM OOM during prefill compile/run.
#   STREAM_LOAD_ONLY=1 STREAM_NUM_LAYERS=43 python streaming/run_streaming.py
LOAD_ONLY = bool(int(os.environ.get("STREAM_LOAD_ONLY", "0")))

# see issue https://github.com/tenstorrent/tt-xla/issues/4444
torch._dynamo.config.cache_size_limit = 100

PROMPTS = [
    "How are you today?",
    "What is the capital of France?",
    "Explain machine learning briefly.",
    "Who painted the Mona Lisa?",
    "What is two plus two?",
    "Tell me a fun fact about space.",
    "What is photosynthesis?",
    "How does a transformer model work?",
    "What is the speed of light?",
    "Name three programming languages.",
    "What causes earthquakes?",
    "How do you make pasta from scratch?",
    "What is the largest planet in our solar system?",
    "Who wrote the play Hamlet?",
    "What is gravity?",
    "How does the internet work?",
    "What is the human brain made of?",
    "Tell me about black holes.",
    "What is DNA?",
    "How does a car engine work?",
    "What is the meaning of recursion?",
    "Who was Albert Einstein?",
    "What is climate change?",
    "How do plants grow?",
    "What is quantum entanglement?",
    "Tell me a short story about a robot.",
    "What is a relational database?",
    "How does Wi-Fi work?",
    "What is the Pythagorean theorem?",
    "Name three renewable energy sources.",
    "What was the French Revolution about?",
    "How do volcanoes form?",
]
assert BATCH_SIZE <= len(PROMPTS), f"BATCH_SIZE={BATCH_SIZE} > available prompts={len(PROMPTS)}"
PROMPTS = PROMPTS[:BATCH_SIZE]

# Per-block weight dtype overrides. The e2e test pins MoE experts to
# bfp_bf4 and attention Linear weights to bfp_bf8 for memory headroom.
# Disabled by default here — bf16 throughout for cleaner correctness
# checks. Set STREAM_WDTYPE_OVERRIDES=1 to re-enable the e2e overrides.
WDTYPE_OVERRIDES = os.environ.get("STREAM_WDTYPE_OVERRIDES", "0") == "1"
EXPERT_DTYPE = "bfp_bf4"
ATTN_DTYPE = "bfp_bf8"


def make_mesh() -> Tuple[Mesh, Tuple[int, int]]:
    # Auto-pick a 2D mesh based on visible TT devices. 32-device llmbox -> (4, 8);
    # 8-device dual-host -> (2, 4). Fall back to a row mesh otherwise.
    num_devices = xr.global_runtime_device_count()
    if num_devices == 32:
        mesh_shape = (4, 8)
    elif num_devices == 8:
        mesh_shape = (2, 4)
    else:
        mesh_shape = (1, num_devices)
    device_ids = np.array(range(num_devices))
    print(
        f"[mesh] num_devices={num_devices} mesh_shape={mesh_shape}", flush=True,
    )
    return Mesh(device_ids, mesh_shape, ("_axis_0", "_axis_1")), mesh_shape


def _print_decoded(tokenizer, prompts: list, generated: list) -> None:
    eos = tokenizer.eos_token_id
    bar = "=" * 72
    print(f"\n{bar}", flush=True)
    for i, (prompt, ids) in enumerate(zip(prompts, generated)):
        cut = next((k for k, t in enumerate(ids) if t == eos), len(ids))
        ids = ids[:cut]
        cont = tokenizer.decode(ids, skip_special_tokens=True)
        print(f"[row {i:02d}] prompt={prompt!r}")
        print(f"         ids={ids}")
        print(f"         cont={cont!r}")
    print(f"{bar}\n", flush=True)


def main() -> None:
    enable_spmd()
    xr.set_device_type("TT")
    torch.manual_seed(0)

    mesh, mesh_shape = make_mesh()
    bsz = BATCH_SIZE

    # --- Real config, MTP off, layer-prefix truncation --------------------
    args = weight_loader.load_config_args()
    args.n_mtp_layers = 0
    args.max_batch_size = bsz
    if NUM_LAYERS < args.n_layers:
        args.n_layers = NUM_LAYERS
        args.compress_ratios = args.compress_ratios[:NUM_LAYERS]
    # max_seq_len must satisfy two constraints:
    #   (a) >= PROMPT_LEN + MAX_NEW_TOKENS - 1 to fit the generated sequence
    #   (b) be a multiple of max(compress_ratios) AND >= 2 * max_cr so the
    #       Compressor.kv_cache (sized as max_seq_len // cr) has at least 2
    #       slots — slot 0 for the prefill compress write, slot 1+ for
    #       decode writes. Otherwise decode index_select trips OOB on
    #       cr=max_cr layers.
    max_cr = max(args.compress_ratios) if args.compress_ratios else 0
    needed = PROMPT_LEN + MAX_NEW_TOKENS
    if max_cr > 0:
        rounded = ((needed + max_cr - 1) // max_cr) * max_cr
        args.max_seq_len = max(rounded, 2 * max_cr)
    else:
        args.max_seq_len = ((needed + 31) // 32) * 32
    print(
        f"[args] n_layers={args.n_layers}, n_routed_experts={args.n_routed_experts}, "
        f"n_activated_experts={args.n_activated_experts}, "
        f"bsz={bsz}, max_seq_len={args.max_seq_len}, "
        f"compress_ratios={args.compress_ratios}",
        flush=True,
    )

    # --- Input ids -------------------------------------------------------
    # Two sources:
    #   - realistic_inputs: cached natural-text tokenization, every position
    #     is a real token (no padding). Same source as test_transformer_prefill.
    #     Gives the model an in-distribution input → coherent token output.
    #   - tokenizer + left-pad of short Q-style prompts: legacy path. Most
    #     positions are pad tokens, which is out-of-distribution and
    #     produces incoherent output even with a correct streaming load.
    from transformers import AutoTokenizer  # noqa: WPS433
    tokenizer = AutoTokenizer.from_pretrained(weight_loader.REPO_ID)
    if USE_REALISTIC_INPUTS:
        from tests.torch.models.deepseek_v4 import realistic_inputs
        prompt_ids, _ = realistic_inputs.get_realistic_inputs(
            layer_id=args.n_hash_layers,
            batch_size=bsz,
            seq_len=PROMPT_LEN,
        )
        prompt_ids = prompt_ids.contiguous()
        print(
            f"[input] using realistic_inputs (layer={args.n_hash_layers}) "
            f"shape={tuple(prompt_ids.shape)}",
            flush=True,
        )
    else:
        pad_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )
        prompt_rows = []
        for prompt in PROMPTS:
            ids = tokenizer(
                prompt, return_tensors="pt", add_special_tokens=False
            ).input_ids[0]
            if ids.shape[0] >= PROMPT_LEN:
                ids = ids[-PROMPT_LEN:]
            else:
                pad = torch.full(
                    (PROMPT_LEN - ids.shape[0],), pad_id, dtype=torch.long
                )
                ids = torch.cat([pad, ids], dim=0)
            prompt_rows.append(ids)
        prompt_ids = torch.stack(prompt_rows, dim=0).contiguous()
        assert prompt_ids.shape == (bsz, PROMPT_LEN)
        print(
            f"[tokenize] prompt_ids[0][-8:]={prompt_ids[0][-8:].tolist()}",
            flush=True,
        )

    # --- Streamed model build + load --------------------------------------
    _log_mem("pre-stream-load")
    model = stream_load_transformer(
        args, mesh, mesh_shape, cluster_axis=0, verbose=True,
    )
    _log_mem("post-stream-load")

    # --- Apply weight dtype overrides -------------------------------------
    # Disabled unless STREAM_WDTYPE_OVERRIDES=1. Re-enables the e2e test's
    # bfp_bf4 (experts) + bfp_bf8 (attn Linear) overrides for memory
    # headroom. Keep `mark_sharding` (done inside stream_load_transformer
    # per block) BEFORE this call so the sharding hint lands on raw
    # nn.Parameter storage rather than the parametrize custom_call output.
    if WDTYPE_OVERRIDES:
        weight_dtype_overrides = {
            # MoE routed experts (compound stacked across the 256 experts).
            "layers.*.ffn.mlp.experts.gate_proj": EXPERT_DTYPE,
            "layers.*.ffn.mlp.experts.up_proj":   EXPERT_DTYPE,
            "layers.*.ffn.mlp.experts.down_proj": EXPERT_DTYPE,
            # Attention Linear weights.
            "layers.*.attn.wq_a.weight": ATTN_DTYPE,
            "layers.*.attn.wq_b.weight": ATTN_DTYPE,
            "layers.*.attn.wkv.weight":  ATTN_DTYPE,
            "layers.*.attn.wo_a.weight": ATTN_DTYPE,
            "layers.*.attn.wo_b.weight": ATTN_DTYPE,
        }
        applied = apply_weight_dtype_overrides(model, weight_dtype_overrides)
        print(f"[wdtype] applied {len(applied)} overrides", flush=True)
        for path, dtype_str in applied:
            print(f"[wdtype]   {path} -> {dtype_str}", flush=True)
    else:
        print("[wdtype] no overrides (bf16 throughout). "
              "Set STREAM_WDTYPE_OVERRIDES=1 to enable bfp_bf4 experts + "
              "bfp_bf8 attn.", flush=True)

    _log_mem("post-wdtype-overrides")

    if LOAD_ONLY:
        print(
            "[stream] STREAM_LOAD_ONLY=1 set — skipping compile + decode loop. "
            "Peak host RSS measured above.",
            flush=True,
        )
        return

    hook = sharding_constraint_hook(model.head, mesh, (None, None))
    model.head.register_forward_hook(hook)

    # --- Compile + run ----------------------------------------------------
    compiled = torch.compile(model, backend="tt")

    generated: list[list[int]] = [[] for _ in range(bsz)]

    print("[prefill] compiling + running ...", flush=True)
    device = torch_xla.device()
    prompt_ids_tt = prompt_ids.to(device)
    xs.mark_sharding(prompt_ids_tt, mesh, ("_axis_0", None))
    # True prefill: start_pos=0, KV writes go to positions [0, PROMPT_LEN).
    # The prior pattern (sp=PROMPT_LEN) treats prefill as an offset-write
    # which gives an out-of-distribution activation pattern and noticeably
    # lower PCC vs CPU eager (~0.82 at layer 0).
    sp_tt = torch.tensor(0, dtype=torch.long).to(device)
    _log_mem("pre-prefill")
    prefill_logits = compiled(prompt_ids_tt, sp_tt)
    torch_xla.sync(wait=True)
    _log_mem("post-prefill-sync")
    next_ids = prefill_logits.detach().to("cpu").argmax(dim=-1)
    for i in range(bsz):
        generated[i].append(int(next_ids[i].item()))
    print(f"[prefill] first ids[:8]={next_ids[:8].tolist()}", flush=True)
    _log_mem("post-prefill-result")

    prev_token = next_ids.unsqueeze(1)
    for step in range(MAX_NEW_TOKENS - 1):
        start_pos = PROMPT_LEN + step
        prev_token_tt = prev_token.to(device)
        xs.mark_sharding(prev_token_tt, mesh, ("_axis_0", None))
        sp_tt = torch.tensor(start_pos, dtype=torch.long).to(device)
        decode_logits = compiled(prev_token_tt, sp_tt)

        next_ids = decode_logits.detach().to("cpu").argmax(dim=-1)
        for i in range(bsz):
            generated[i].append(int(next_ids[i].item()))
        print(
            f"[decode {step + 1:>2}] sp={start_pos}: "
            f"ids[:8]={next_ids[:8].tolist()}",
            flush=True,
        )
        prev_token = next_ids.unsqueeze(1)

    _print_decoded(tokenizer, PROMPTS, generated)


if __name__ == "__main__":
    main()
