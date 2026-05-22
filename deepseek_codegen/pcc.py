# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PCC driver for the EmitPy `main.py` under `deepseek_codegen/graph_0/`.

The codegen `_main` was patched to additionally return the pre-argmax logits
tensor (`ttnn_to_layout_267`) as out[9]. Outputs are now:

    out[0..5]  BF16 KV-cache deltas (per-layer / per-cache)
    out[6..8]  INT32 sampled-token / position outputs
    out[9]     BF16 logits, shape (batch=32, vocab=129280) per chip,
               stacked across the 4x8 mesh to (32, 32, 129280)

Two correctness signals are reported:

* Bootstrap baseline: `baseline_outputs.pt` is captured on the first run with
  `--save-baseline` (or whenever it's missing). Subsequent runs compare every
  KV-cache + token output against it. This catches accidental drift / refactor
  bugs (PCC=1.0 means "bit-identical to last run"). It is **not** a correctness
  signal against the reference model.
* Golden logits: `golden_logits.pt` is a (128, 1, 129280) FP32 PyTorch reference.
  Both the golden's 128 rows and the TTNN stack's 32 chips x 32 batch rows are
  mesh-/tile-replicated copies of one real (vocab,) logits vector. The harness
  picks the canonical row from each side (`chip[0].row[0]` vs `golden[0]`) and
  PCC-compares. This is the real correctness signal -- because of bf16
  quantization through the LM head and reduce_scatter the unmodified baseline
  already only reaches ~0.92 PCC vs golden, so the hard-fail floor is set to
  0.9 (an experiment is "as good as baseline" if it stays >= 0.9).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent  # .../deepseek_codegen/
GRAPH_DIR = THIS_DIR / "graph_0"
BASELINE_PATH = THIS_DIR / "baseline_outputs.pt"
GOLDEN_PATH = THIS_DIR / "golden_logits.pt"
REPO_ROOT = THIS_DIR.parent

GOLDEN_PCC_FLOOR = 0.9


def _import_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load benchmark compute_pcc under a private name so it can't shadow the
# generator's sibling utils.py inside main.py.
_bench_utils = _import_from_path(
    "_bench_utils", REPO_ROOT / "tests" / "benchmark" / "utils.py"
)
compute_pcc = _bench_utils.compute_pcc

# main.py uses `import utils` (sibling), `./tensors/...` relative paths.
sys.path.insert(0, str(GRAPH_DIR))
os.chdir(GRAPH_DIR)

import ttnn  # noqa: E402
import utils as gen_utils  # noqa: E402
import main as gen_main  # noqa: E402


def _to_torch(t) -> torch.Tensor:
    """Return a single CPU torch.Tensor by stacking per-shard data along dim 0."""
    if isinstance(t, torch.Tensor):
        return t.detach().cpu()
    shards = ttnn.get_device_tensors(t) if hasattr(ttnn, "get_device_tensors") else [t]
    pieces = [ttnn.to_torch(s).detach().cpu() for s in shards]
    if len(pieces) == 1:
        return pieces[0]
    return torch.stack(pieces, dim=0)


def _canonical_logits_row(logits_stack: torch.Tensor) -> tuple[torch.Tensor, str]:
    """Reduce a (num_shards, batch_per_shard, vocab) TTNN logits stack to a
    single canonical (vocab,) logits vector.

    Empirically (galaxy 4x8 decode-only test):
      * chips 0..7, 8..15, 16..23, 24..31 are pairwise replicated (cluster
        axis-1 all_gather on the vocab dim), so there are 4 unique batch
        shards.
      * within each chip, the 32 "batch" rows are also (mostly) identical -
        the test uses a single sequence, the batch=32 dim is tile-layout
        padding. Only the first row is the real per-shard logits; the trailing
        rows can carry padding noise.
      * golden_logits.pt is itself (128, 1, 129280) where all 128 rows are
        identical (one batch's logits replicated for the cluster).

    The canonical, structurally-meaningful comparison is therefore:
        chip[0].row[0]  vs  golden[0]
    which gives PCC ~ 0.9218 against an unmodified baseline (set by bf16
    quantization through the LM head and reduce_scatter).
    """
    assert logits_stack.dim() == 3, f"expected 3D stack, got {logits_stack.shape}"
    n, bps, vocab = logits_stack.shape
    chip0_row0 = logits_stack[0, 0].float()  # (vocab,)
    return chip0_row0, f"chip[0].row[0] from stack {tuple(logits_stack.shape)}"


def _golden_pcc(logits_stack: torch.Tensor) -> tuple[float, float, str, tuple]:
    """Compute logits-vs-golden PCC. Returns (pcc, max_diff, mode, shape)."""
    golden = torch.load(GOLDEN_PATH, map_location="cpu", weights_only=False).float()
    if golden.dim() == 3 and golden.shape[1] == 1:
        golden = golden.squeeze(1)  # (batch, vocab)
    # All golden rows are identical for this test; pick row 0.
    golden_row0 = golden[0]
    cand_row, mode = _canonical_logits_row(logits_stack)
    if cand_row.shape != golden_row0.shape:
        return float("nan"), float("nan"), (
            f"{mode}; shape-mismatch {tuple(cand_row.shape)} vs "
            f"{tuple(golden_row0.shape)}"
        ), tuple(cand_row.shape)
    pcc = compute_pcc(cand_row, golden_row0)
    max_diff = (cand_row - golden_row0).abs().max().item()
    return pcc, max_diff, mode, tuple(cand_row.shape)


def main() -> int:
    save = "--save-baseline" in sys.argv
    use_trace = "--trace" in sys.argv

    weights = gen_main.load_weights_for__main()
    if use_trace:
        device = gen_utils.DeviceGetter.get_device((4, 8))
        # 1) WARMUP — cold compile + populate ce_cache. _main deallocates args_0/args_1.
        activations_warmup = gen_main.load_activations_for__main()
        gen_main._main(activations_warmup, weights)
        ttnn.synchronize_device(device)
        print("warmup done")
        # 2) Re-load activations (the warmup call freed them).
        activations_trace = gen_main.load_activations_for__main()
        # 3) Capture trace — _main records all decode ops into the trace.
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        outputs = gen_main._main(activations_trace, weights)
        ttnn.end_trace_capture(device, tid, cq_id=0)
        ttnn.synchronize_device(device)
        print("trace captured")
        # 4) Execute trace — replays the captured device program; outputs are refilled in place.
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        print("trace executed")
    else:
        activations = gen_main.load_activations_for__main()
        outputs = gen_main._main(activations, weights)
    candidates = [_to_torch(o) for o in outputs]

    print(f"_main returned {len(candidates)} outputs:")
    for i, c in enumerate(candidates):
        print(f"  out[{i}]: shape={tuple(c.shape)} dtype={c.dtype}")

    # Identify logits output (last FP/BF tensor with vocab-sized trailing dim).
    logits_idx = None
    if len(candidates) >= 10:
        logits_idx = 9
    elif len(candidates) >= 1:
        # Fallback: last non-int tensor.
        for i in range(len(candidates) - 1, -1, -1):
            if candidates[i].dtype not in (torch.int32, torch.int64, torch.int16, torch.int8):
                logits_idx = i
                break

    # --- bootstrap baseline write ---
    if save or not BASELINE_PATH.is_file():
        torch.save(candidates, BASELINE_PATH)
        print(f"saved baseline to {BASELINE_PATH}")
        # Still print golden PCC if we have a logits output, so save-runs report something.
        print()
        if logits_idx is not None and GOLDEN_PATH.is_file():
            pcc, max_diff, mode, shape = _golden_pcc(candidates[logits_idx])
            print(
                f"golden logits PCC (out[{logits_idx}]): mode='{mode}' "
                f"shape={shape} PCC={pcc:.6f} max|Δ|={max_diff:.3e}"
            )
        return 0

    # --- compare against bootstrap baseline (KV+tokens) ---
    baseline = torch.load(BASELINE_PATH, map_location="cpu", weights_only=False)
    if len(baseline) != len(candidates):
        print(
            f"warning: baseline has {len(baseline)} outputs, candidate has "
            f"{len(candidates)} -- comparing the overlap"
        )

    print()
    print("=== vs bootstrap baseline (reproducibility check, soft) ===")
    worst_bootstrap = 1.0
    for i, (b, c) in enumerate(zip(baseline, candidates)):
        if b.shape != c.shape:
            print(f"out[{i}]: shape mismatch {tuple(b.shape)} vs {tuple(c.shape)}")
            worst_bootstrap = min(worst_bootstrap, 0.0)
            continue
        if b.dtype != c.dtype:
            b = b.float()
            c = c.float()
        if b.dtype in (torch.int32, torch.int64, torch.int16, torch.int8):
            eq = bool(torch.equal(b, c))
            ndiff = int((b != c).sum().item()) if not eq else 0
            print(
                f"out[{i}] [{b.dtype}] shape={tuple(b.shape)}: "
                f"exact_match={eq} (diff_elems={ndiff}/{b.numel()})"
            )
            continue
        pcc = compute_pcc(b, c)
        max_diff = (b.float() - c.float()).abs().max().item()
        print(
            f"out[{i}] [{b.dtype}] shape={tuple(b.shape)}: "
            f"PCC={pcc:.6f}  max|Δ|={max_diff:.3e}"
        )
        worst_bootstrap = min(worst_bootstrap, pcc)
    print(f"worst_bootstrap_pcc={worst_bootstrap:.6f} (informational)")

    # --- compare logits against golden (the real correctness signal) ---
    print()
    print(f"=== vs golden_logits.pt (correctness check, floor={GOLDEN_PCC_FLOOR}) ===")
    if logits_idx is None:
        print("no logits output found; skipping golden check")
        return 1
    if not GOLDEN_PATH.is_file():
        print(f"no golden file at {GOLDEN_PATH}; skipping golden check")
        return 1
    pcc, max_diff, mode, shape = _golden_pcc(candidates[logits_idx])
    print(f"mode='{mode}'")
    print(f"logits (out[{logits_idx}]) flat-shape={shape}: PCC={pcc:.6f} max|Δ|={max_diff:.3e}")

    if not (pcc == pcc):  # NaN
        print("FAIL: golden PCC is NaN (alignment broken?)")
        return 2
    if pcc < GOLDEN_PCC_FLOOR:
        print(f"FAIL: golden PCC {pcc:.6f} < floor {GOLDEN_PCC_FLOOR}")
        return 2
    print(f"PASS: golden PCC {pcc:.6f} >= floor {GOLDEN_PCC_FLOOR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
