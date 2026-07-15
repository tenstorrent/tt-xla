# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Exact-across-boundary incremental decode integration test for DeepSeek-V4 on
TT, for BOTH C4A (compress_ratio 4) and C128A (compress_ratio 128), on the real
DeepSeek-V4-Flash bf16 model through vLLM.

Background: the compressor pools every ``ratio`` tokens into one compressed slot.
During decode, a new slot must be formed the moment its ``ratio``-token block
completes (crossing a "compression boundary"); the earlier folded-state decode
froze the compressed cache at prefill and never formed new slots, so any decode
past a boundary silently attended a stale/garbage slot. This test verifies the
incremental decode forms the boundary-crossing slot correctly, on real weights.

Why a two-run consistency check (not a token/logit comparison): a newly-formed
slot summarizes the last ``ratio`` tokens, which for ``ratio < window`` the
per-token sliding window already attends, so the fix is numerically near-invisible
at the model output level. Instead we compare a SINGLE compressed slot two ways:

  Run A: prefill the prompt, then DECODE across the target layer's boundary so
         slot S completes DURING decode (from the rolling compressor state seeded
         at prefill + the tokens A generates).
  Run B: prefill ``[prompt + A's generated tokens]`` all at once, so slot S is
         formed by the PREFILL compressor over the SAME real tokens.

  exact-across-boundary  =>  slotA (decode) == slotB (prefill)   [PCC >= 0.99]
  folded-state decode    =>  slotA is prefill-time padding garbage != slotB.

The compressed VALUES cache is paged, so slot S is read via the layer's page table
(captured from the eager metadata build), robust to the physical block layout.

Isolation: reading the layer's bound cache + capturing wrapper/metadata requires
the engine IN-PROCESS (``VLLM_ENABLE_V1_MULTIPROCESSING=0``), which inits the
once-only XLA computation cache — so two params in one interpreter collide
("Computation cache has already been initialized"). Each param therefore re-execs
this file as a CLEAN subprocess (not os.fork/pytest-forked: the parent pytest
worker has already loaded the PJRT plugin), matching test_prefill_recompile.py.

C128A first appears at layer 3 in the real model, so it needs the 4-layer
checkpoint (native C128A @ layer 3); C4A uses the 3-layer checkpoint (native C4A @
layer 2). Build with ``build_vllm_bf16_checkpoint.py --n-layers {3,4}`` and point:

    DSV4_C4A_CKPT=/path/to/dsv4-bf16-3layer  \
    DSV4_C128A_CKPT=/path/to/dsv4-bf16-4layer \
    pytest -svv tests/integrations/vllm_plugin/generative/test_dsv4_boundary_decode.py

Each parametrization is skipped unless its checkpoint dir is set, so CI stays green
without the giant checkpoints. The worker patches ``compress_ratios`` (and prunes
the overridden layer's index keys for C128A) with a backup/restore, leaving the
checkpoint as found.
"""
import contextlib
import json
import os
import subprocess
import sys

import pytest

_C4A_CKPT = os.environ.get("DSV4_C4A_CKPT")
_C128A_CKPT = os.environ.get("DSV4_C128A_CKPT")

# per-case: ckpt, n_layers, ratio, compress_ratios, prompt_len, slot, n_decode, prune.
# n_decode reaches a COMPLETING position so the read slot has no in-progress partial:
# slot S completes at pos prompt_len-1+n_decode.
_CASES = {
    "c4a": (_C4A_CKPT, 3, 4, [0, 0, 4], 130, 32, 2, False),
    "c128a": (_C128A_CKPT, 4, 128, [0, 0, 0, 128], 125, 0, 3, True),
}

_WORKER_TIMEOUT = 2400  # 2 model loads' worth of headroom for a cold compile


@pytest.mark.nightly
@pytest.mark.bh_galaxy
@pytest.mark.parametrize("kind", ["c4a", "c128a"])
def test_dsv4_exact_across_boundary_decode(kind):
    ckpt, n_layers = _CASES[kind][0], _CASES[kind][1]
    if not ckpt or not os.path.isdir(ckpt):
        pytest.skip(
            f"set DSV4_{kind.upper()}_CKPT to a bf16 DSV4 {n_layers}-layer "
            f"checkpoint dir (build_vllm_bf16_checkpoint.py --n-layers {n_layers})"
        )
    env = {
        **os.environ,
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "TT_ROPE_CACHE_CAP": "4096",
    }
    try:
        proc = subprocess.run(
            [sys.executable, __file__, kind],
            env=env,
            capture_output=True,
            text=True,
            timeout=_WORKER_TIMEOUT,
        )
        stdout, stderr, rc = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as e:
        stdout, stderr, rc = e.stdout or "", e.stderr or "", None
    print(stdout, flush=True)
    if stderr:
        print(stderr, file=sys.stderr, flush=True)
    assert (
        rc == 0
    ), f"[{kind}] exact-across-boundary worker failed (exit={rc}); see output"


# --------------------------------------------------------------------------- #
# Worker body (runs in the isolated in-process child).
# --------------------------------------------------------------------------- #
def _pcc(a, b):
    import torch

    a, b = a.flatten().float(), b.flatten().float()
    va, vb = a - a.mean(), b - b.mean()
    return float((va @ vb) / (va.norm() * vb.norm() + 1e-12))


@contextlib.contextmanager
def _patched_config(ckpt, n_layers, ratios, prune):
    """Patch num_hidden_layers/compress_ratios (+ prune the overridden layer-2 C4A
    index keys for C128A) with a backup/restore, leaving the checkpoint as found."""
    cfg_p = f"{ckpt}/config.json"
    idx_p = f"{ckpt}/model.safetensors.index.json"
    cfg_bak = open(cfg_p).read()
    idx_bak = open(idx_p).read() if prune and os.path.isfile(idx_p) else None
    try:
        cfg = json.loads(cfg_bak)
        cfg["num_hidden_layers"] = n_layers
        cfg["compress_ratios"] = ratios
        json.dump(cfg, open(cfg_p, "w"), indent=2)
        if idx_bak is not None:
            idx = json.loads(idx_bak)
            wm = idx["weight_map"]
            for k in [
                k
                for k in wm
                if k.startswith("layers.2.") and ("compressor" in k or "indexer" in k)
            ]:
                del wm[k]
            json.dump(idx, open(idx_p, "w"), indent=2)
        yield
    finally:
        open(cfg_p, "w").write(cfg_bak)
        if idx_bak is not None:
            open(idx_p, "w").write(idx_bak)


def _read_slot(cache, page_table, slot, torch_xla):
    """Read compressed VALUES logical ``slot`` via the request page table. cache
    [nb, 1, cb, Hd]; slot -> physical block page_table[0, slot//cb], row slot%cb."""
    torch_xla.sync()
    c = cache.detach().to("cpu").float()
    cb = c.shape[2]
    blk = int(page_table[0, slot // cb])
    return c[blk, 0, slot % cb, :]


def _worker(kind):
    ckpt, n_layers, ratio, ratios, prompt_len, slot, n_decode, prune = _CASES[kind]
    import torch_xla
    import vllm
    import vllm_tt.attention_impls.attention_dsv4 as adsv4
    import vllm_tt.model_runner as mr
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    # capture wrapper instances (at construction) + the compressed page table (from
    # the eager metadata build) — both outside the compiled graph.
    wrappers = []
    page_tables = []
    _orig_init = adsv4.TTDeepseekV4MLAWrapper.__init__

    def _cap_init(self, *a, **k):
        _orig_init(self, *a, **k)
        wrappers.append(self)

    adsv4.TTDeepseekV4MLAWrapper.__init__ = _cap_init
    _orig_bcm = mr.TTModelRunner._build_compressed_metadata

    def _cap_bcm(self, *a, **k):
        comp_md = _orig_bcm(self, *a, **k)
        try:
            page_tables.append(comp_md.page_table.detach().to("cpu").clone())
        except Exception:
            page_tables.append(None)
        return comp_md

    mr.TTModelRunner._build_compressed_metadata = _cap_bcm

    with _patched_config(ckpt, n_layers, ratios, prune):
        llm = vllm.LLM(
            model=ckpt,
            max_num_batched_tokens=192,
            max_num_seqs=1,
            max_model_len=192,
            gpu_memory_utilization=0.06 if kind == "c4a" else 0.08,
            skip_tokenizer_init=True,
            additional_config={
                "min_context_len": 32,
                "enable_tensor_parallel": True,
                "mesh_shape": [2, 4],
                "flat_model_io": True,
            },
        )
        targets = [w for w in wrappers if getattr(w, "compress_ratio", 1) == ratio]
        assert targets, (
            f"no wrapper with compress_ratio={ratio}; got "
            f"{[getattr(w, 'compress_ratio', None) for w in wrappers]}"
        )
        w = targets[0]
        prompt_ids = list(range(1, prompt_len + 1))

        # Run A: prefill + decode across the boundary (slot formed by decode).
        page_tables.clear()
        outA = llm.generate(
            [TokensPrompt(prompt_token_ids=prompt_ids)],
            SamplingParams(max_tokens=n_decode + 1, temperature=0.0),
        )
        genA = list(outA[0].outputs[0].token_ids)
        slotA = _read_slot(w.mla_attn.kv_cache, page_tables[-1], slot, torch_xla)

        # Run B: prefill [prompt + tokens A fed at decode] all at once.
        page_tables.clear()
        llm.generate(
            [TokensPrompt(prompt_token_ids=prompt_ids + genA[:n_decode])],
            SamplingParams(max_tokens=1, temperature=0.0),
        )
        slotB = _read_slot(w.mla_attn.kv_cache, page_tables[-1], slot, torch_xla)

    pcc = _pcc(slotA, slotB)
    print(
        f"[{kind}] slot {slot}: A(decode) rms={slotA.norm():.3f} "
        f"B(prefill) rms={slotB.norm():.3f} PCC={pcc:.5f}",
        flush=True,
    )
    ok = (
        bool(slotA.isfinite().all() and slotB.isfinite().all())
        and float(slotA.norm()) > 1e-2
        and float(slotB.norm()) > 1e-2
        and pcc >= 0.99
    )
    if not ok:
        print(
            f"[{kind}] FAIL: decode-formed compressed slot {slot} != all-at-once "
            f"prefill (PCC {pcc:.4f}) -> incremental decode NOT exact across the "
            f"ratio-{ratio} boundary",
            flush=True,
        )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_worker(sys.argv[1]))
