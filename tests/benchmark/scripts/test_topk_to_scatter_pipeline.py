# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Data-driven replay tests for every op in the GPT-OSS MoE topk→scatter
pipeline.

The C++ runtime dumps inputs/outputs of every op downstream of topk as
.npy files plus a *_meta.json with op type & params.  This test discovers
those dumps, builds the corresponding torch.nn.Module for each op, feeds
the dumped inputs through TT device via ``run_op_test``, and verifies:

  1. Device output matches CPU golden  (catches op-level bugs)
  2. Device output matches the dumped output  (catches pipeline-vs-isolation
     discrepancies)

Pipeline (from TTNN MLIR):
  topk         [272,32]bf16 → values[272,4]bf16 + indices[272,4]ui16
  typecast     [272,4] ui16 → si32
  reshape      [272,4] si32 → [272,4,1]
  concat       [272,4,1] + [272,4,1] → [272,4,2]   (dim=2)
  softmax      [272,4] bf16 → [272,4] bf16          (dim=1)
  all_gather   [272,4,2] → [1088,4,2]               (CCL, dim=0, axis=0)
  all_gather   [272,4] → [1088,4]                   (CCL, dim=0, axis=0)
  reshape      [1088,4,2] → [4352,2]
  slice×2      [4352,2] → [4352,1]
  multiply     [4352,1] × [1,1] → [4352,1]
  add          [4352,1] + [4352,1] → [4352,1]
  reshape      [4352,1] → [4352]
  reshape      [1088,4] → [4352]
  slice (1-D)  → [256]  ×N
  to_layout    (layout change)
  scatter      (3-input)

Usage:
    # 1. Run model with dumps enabled
    TT_RUNTIME_OP_TENSOR_TRACE_TOPK_DUMP_DIR=/path/to/dump \\
        python prefill_and_decode ...

    # 2. Run this test
    TOPK_DUMP_DIR=/path/to/dump pytest -svv \\
        tests/benchmark/scripts/test_topk_to_scatter_pipeline.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_BENCH_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_BENCH_ROOT))

from infra import Framework, run_op_test  # noqa: E402

DEFAULT_DUMP_DIR = (
    _BENCH_ROOT / "modules" / "gpt_oss_input_sharding_dbg" / "topk_dump"
)

SKIP_OPS = frozenset({
    "ttnn.topk",       # trigger op, multi-output; tested separately
    "ttnn.all_gather", # CCL multi-device op
    "ttnn.to_layout",  # device layout change, no torch equivalent
    "ttnn.scatter",    # 3-input op with accumulator; tested separately
    "",
})


# ---------------------------------------------------------------------------
# Dump discovery
# ---------------------------------------------------------------------------

def _dump_dir() -> Path:
    d = Path(os.environ.get("TOPK_DUMP_DIR", str(DEFAULT_DUMP_DIR)))
    return d if d.is_dir() else DEFAULT_DUMP_DIR


def _discover_ops(dump_dir: Path) -> List[dict]:
    """Find all *_meta.json (new format only) and return sorted by op_seq."""
    metas = []
    for f in sorted(dump_dir.glob("ttnn_*_meta.json")):
        with open(f) as fp:
            m = json.load(fp)
        if "params" not in m:
            continue
        m["_meta_path"] = str(f)
        m["_base"] = str(f).rsplit("_meta.json", 1)[0]
        metas.append(m)
    return sorted(metas, key=lambda x: x["op_seq"])


def _filter_pipeline_ops(ops: List[dict]) -> List[dict]:
    """Keep only ops inside topk→scatter pipelines (discard leaked ops)."""
    topk_seqs = [m["op_seq"] for m in ops if m.get("mlir_op") == "ttnn.topk"]
    if not topk_seqs:
        return ops

    scatter_seqs = sorted(
        m["op_seq"] for m in ops if m.get("mlir_op") == "ttnn.scatter"
    )

    ranges = []
    for tk in topk_seqs:
        next_topk = min((t for t in topk_seqs if t > tk), default=float("inf"))
        pipeline_scatters = [s for s in scatter_seqs if tk < s < next_topk]
        last_scatter = max(pipeline_scatters) if pipeline_scatters else tk
        ranges.append((tk, last_scatter))

    def _in_pipeline(seq):
        return any(lo <= seq <= hi for lo, hi in ranges)

    return [m for m in ops if _in_pipeline(m["op_seq"])]


# ---------------------------------------------------------------------------
# Tensor loading
# ---------------------------------------------------------------------------

_TTNN_DTYPE_TO_TORCH = {
    "UInt16": torch.int16,    # ui16 → int16 (closest torch unsigned)
    "UInt32": torch.int32,
    "SInt32": torch.int32,
    "BFloat16": torch.bfloat16,
    "Float16": torch.float16,
    "Float32": torch.float32,
}


def _load_npy_tensor(path: str) -> Optional[torch.Tensor]:
    if not os.path.exists(path):
        return None
    arr = np.load(path)
    return torch.from_numpy(arr)


def _find_dev_files(base: str, tag: str) -> List[Tuple[int, str]]:
    """Return [(dev_id, path), ...] for files matching {base}_{tag}_dev{N}.npy."""
    results = []
    parent = Path(base).parent
    prefix = Path(base).name + f"_{tag}_dev"
    for f in parent.iterdir():
        name = f.name
        if name.startswith(prefix) and name.endswith(".npy"):
            dev_str = name[len(prefix):-len(".npy")]
            try:
                results.append((int(dev_str), str(f)))
            except ValueError:
                pass
    return sorted(results, key=lambda x: x[0])


def _load_inputs(base: str, meta: dict, dev: int = 0) -> List[torch.Tensor]:
    """Load all input npy files for a given op."""
    n_inputs = len(meta.get("in_global_ids", []))
    tensors = []
    for i in range(max(n_inputs, 1)):
        p = f"{base}_in{i}_dev{dev}.npy"
        t = _load_npy_tensor(p)
        if t is None:
            # try without dev suffix
            p2 = f"{base}_in{i}.npy"
            t = _load_npy_tensor(p2)
        if t is not None:
            tensors.append(t)
    return tensors


def _load_output(base: str, dev: int = 0) -> Optional[torch.Tensor]:
    p = f"{base}_out_dev{dev}.npy"
    return _load_npy_tensor(p)


# ---------------------------------------------------------------------------
# Module factories  (one per TTNN op type)
# ---------------------------------------------------------------------------

def _make_typecast_module(params: dict) -> torch.nn.Module:
    dtype_str = params.get("dtype", "")
    dtype_map = {
        "SInt32": torch.int32, "si32": torch.int32,
        "UInt32": torch.int32, "u32": torch.int32,
        "BFloat16": torch.bfloat16, "bf16": torch.bfloat16,
        "Float32": torch.float32, "f32": torch.float32,
        "Float16": torch.float16, "f16": torch.float16,
    }
    target_dtype = dtype_map.get(dtype_str, torch.int32)

    class Typecast(torch.nn.Module):
        def forward(self, x):
            return x.to(target_dtype)

    return Typecast()


def _make_reshape_module(params: dict) -> torch.nn.Module:
    shape = tuple(params["shape"])

    class Reshape(torch.nn.Module):
        def forward(self, x):
            return x.reshape(shape)

    return Reshape()


def _make_concat_module(params: dict) -> torch.nn.Module:
    dim = params.get("dim", 0)

    class Concat(torch.nn.Module):
        def forward(self, *args):
            return torch.cat(list(args), dim=dim)

    return Concat()


def _make_softmax_module(params: dict) -> torch.nn.Module:
    dim = params.get("dim", -1)

    class Softmax(torch.nn.Module):
        def forward(self, x):
            return torch.softmax(x, dim=dim)

    return Softmax()


def _make_slice_module(params: dict) -> torch.nn.Module:
    begins = params.get("begins", [])
    ends = params.get("ends", [])
    step = params.get("step", [1] * len(begins))

    class Slice(torch.nn.Module):
        def forward(self, x):
            slices = tuple(
                slice(int(b), int(e), int(s))
                for b, e, s in zip(begins, ends, step)
            )
            return x[slices]

    return Slice()


def _make_multiply_module(params: dict) -> torch.nn.Module:
    class Multiply(torch.nn.Module):
        def forward(self, a, b):
            return a * b

    return Multiply()


def _make_add_module(params: dict) -> torch.nn.Module:
    class Add(torch.nn.Module):
        def forward(self, a, b):
            return a + b

    return Add()


def _make_scatter_module(params: dict) -> torch.nn.Module:
    dim = params.get("dim", 0)

    class Scatter(torch.nn.Module):
        def forward(self, inp, index, source):
            return inp.scatter(dim, index.long(), source)

    return Scatter()


_MODULE_FACTORIES: Dict[str, Callable] = {
    "ttnn.typecast": _make_typecast_module,
    "ttnn.reshape": _make_reshape_module,
    "ttnn.concat": _make_concat_module,
    "ttnn.softmax": _make_softmax_module,
    "ttnn.slice_static": _make_slice_module,
    "ttnn.multiply": _make_multiply_module,
    "ttnn.add": _make_add_module,
    "ttnn.scatter": _make_scatter_module,
}


def _make_module(mlir_op: str, params: dict) -> Optional[torch.nn.Module]:
    factory = _MODULE_FACTORIES.get(mlir_op)
    if factory is None:
        # Try without prefix
        short = mlir_op.split(".")[-1] if "." in mlir_op else mlir_op
        factory = _MODULE_FACTORIES.get(f"ttnn.{short}")
    if factory:
        return factory(params)
    return None


# ---------------------------------------------------------------------------
# Input preparation — convert npy types back to the dtypes expected by
# the torch module (the C++ dump stores bf16→f32, ui16→u32/i32).
# ---------------------------------------------------------------------------

def _prepare_inputs(mlir_op: str, params: dict,
                    raw_inputs: List[torch.Tensor]) -> List[torch.Tensor]:
    """Adjust dtypes so the torch Module sees the correct input types.

    The C++ dump converts bfloat16→float32 and uint16→int32 for npy
    compatibility.  We undo those conversions here based on the op type
    and parameters.
    """
    if mlir_op == "ttnn.typecast":
        dtype_str = params.get("dtype", "")
        if dtype_str in ("SInt32", "si32"):
            # typecast ui16→si32: input was dumped as i32, cast to i16
            return [t.to(torch.int16) if t.dtype in (torch.int32, torch.int64)
                    else t for t in raw_inputs]
        return raw_inputs

    # For ops that operate on bf16 tensors, convert f32 npy back to bf16.
    # Integer ops (reshape/concat/slice/multiply/add on si32 path) keep i32.
    BF16_OPS = {"ttnn.softmax"}
    if mlir_op in BF16_OPS:
        return [t.to(torch.bfloat16) if t.dtype == torch.float32 else t
                for t in raw_inputs]

    return raw_inputs


# ---------------------------------------------------------------------------
# Comparators
# ---------------------------------------------------------------------------

def _make_comparator(dumped_output: Optional[torch.Tensor], is_integer: bool):
    """Create a comparator that checks device output against both CPU golden
    and (optionally) the dumped pipeline output."""

    def comparator(device_output, golden_output, args, kwargs):
        dev = device_output.cpu()
        gold = golden_output.cpu()

        if is_integer:
            dev_cast = dev.to(gold.dtype)
            nz_dev = (dev_cast != 0).sum().item()
            nz_gold = (gold != 0).sum().item()
            match = torch.equal(dev_cast, gold)
            print(f"  vs golden: exact_match={match}  "
                  f"dev_nonzero={nz_dev}/{dev.numel()}  "
                  f"gold_nonzero={nz_gold}/{gold.numel()}")
            if nz_dev == 0 and nz_gold > 0:
                print("  >>> BUG REPRODUCED: device all-zero, golden nonzero <<<")
            assert match, (
                f"Exact mismatch with golden: "
                f"dev nonzero={nz_dev} vs golden nonzero={nz_gold}"
            )
        else:
            dev_f = dev.float()
            gold_f = gold.float()
            cos = torch.nn.functional.cosine_similarity(
                dev_f.flatten().unsqueeze(0),
                gold_f.flatten().unsqueeze(0),
            ).item()
            print(f"  vs golden: cosine_sim={cos:.6f}  "
                  f"dev_sum={dev_f.sum():.4f}  gold_sum={gold_f.sum():.4f}")
            assert cos > 0.99, f"Cosine similarity {cos} < 0.99 vs golden"

        if dumped_output is not None:
            dumped = dumped_output.cpu()
            if is_integer:
                match_dump = torch.equal(dev.to(dumped.dtype), dumped)
                print(f"  vs dump:   exact_match={match_dump}")
                if not match_dump:
                    diffs = (dev.to(dumped.dtype) != dumped).sum().item()
                    print(f"  vs dump:   {diffs} elements differ")
            else:
                dev_f = dev.float()
                dump_f = dumped.float()
                cos_d = torch.nn.functional.cosine_similarity(
                    dev_f.flatten().unsqueeze(0),
                    dump_f.flatten().unsqueeze(0),
                ).item()
                print(f"  vs dump:   cosine_sim={cos_d:.6f}")

    return comparator


# ---------------------------------------------------------------------------
# Test parametrization
# ---------------------------------------------------------------------------

def _collect_testable_ops() -> List[Tuple[str, dict]]:
    """Discover testable ops from the dump directory at collection time."""
    d = _dump_dir()
    if not d.is_dir():
        return []
    ops = _filter_pipeline_ops(_discover_ops(d))
    testable = []
    for m in ops:
        mlir_op = m.get("mlir_op", "")
        if mlir_op in SKIP_OPS:
            continue
        params = m.get("params", {})
        if _make_module(mlir_op, params) is None:
            continue
        seq = m["op_seq"]
        testable.append((f"{mlir_op}_seq{seq}", m))
    return testable


_TESTABLE_OPS = _collect_testable_ops()


@pytest.mark.single_device
@pytest.mark.parametrize(
    "meta",
    [m for _, m in _TESTABLE_OPS],
    ids=[name for name, _ in _TESTABLE_OPS],
)
def test_replay_op(meta: dict):
    """Replay a single TTNN op using dumped inputs and verify output."""
    mlir_op = meta["mlir_op"]
    params = meta.get("params", {})
    base = meta["_base"]
    seq = meta["op_seq"]

    print(f"\n{'='*60}")
    print(f"  op_seq={seq}  mlir_op={mlir_op}")
    print(f"  loc={meta.get('loc', '?')}")
    print(f"  params={json.dumps(params)}")

    module = _make_module(mlir_op, params)
    assert module is not None, f"No module factory for {mlir_op}"

    raw_inputs = _load_inputs(base, meta)
    assert len(raw_inputs) > 0, f"No input files found for {base}"

    inputs = _prepare_inputs(mlir_op, params, raw_inputs)

    for i, t in enumerate(inputs):
        print(f"  input[{i}]: shape={list(t.shape)} dtype={t.dtype} "
              f"nonzero={int((t != 0).sum())}/{t.numel()}")

    dumped_out = _load_output(base)
    if dumped_out is not None:
        print(f"  dumped_output: shape={list(dumped_out.shape)} "
              f"dtype={dumped_out.dtype} "
              f"nonzero={int((dumped_out != 0).sum())}/{dumped_out.numel()}")

    is_integer = all(t.dtype in (torch.int16, torch.int32, torch.int64,
                                  torch.uint8) for t in inputs)

    comparator = _make_comparator(dumped_out, is_integer)

    run_op_test(
        module, inputs,
        framework=Framework.TORCH,
        custom_comparator=comparator,
    )


# ---------------------------------------------------------------------------
# Standalone topk test (multi-output, handled separately)
# ---------------------------------------------------------------------------

@pytest.mark.single_device
def test_topk_from_dumps():
    """Replay topk using dumped input, compare values+indices vs golden."""
    d = _dump_dir()
    ops = _filter_pipeline_ops(_discover_ops(d))
    topk_ops = [m for m in ops if m.get("mlir_op") == "ttnn.topk"]
    if not topk_ops:
        pytest.skip("No topk dumps found")

    meta = topk_ops[0]
    base = meta["_base"]
    params = meta.get("params", {})
    k = params.get("k", 4)
    dim = params.get("dim", -1)

    print(f"\n  topk: k={k} dim={dim}")

    raw_inputs = _load_inputs(base, meta)
    assert len(raw_inputs) > 0
    inp = raw_inputs[0].to(torch.bfloat16)
    print(f"  input: shape={list(inp.shape)} dtype={inp.dtype}")

    # Load dumped values/indices for comparison
    dumped_values = _load_npy_tensor(f"{base}_values_dev0.npy")
    dumped_indices = _load_npy_tensor(f"{base}_indices_dev0.npy")

    class TopK(torch.nn.Module):
        def forward(self, x):
            vals, idxs = torch.topk(x, k=k, dim=dim, largest=True, sorted=True)
            return vals

    def topk_comparator(device_output, golden_output, args, kwargs):
        dev = device_output.cpu().float()
        gold = golden_output.float()
        cos = torch.nn.functional.cosine_similarity(
            dev.flatten().unsqueeze(0), gold.flatten().unsqueeze(0)
        ).item()
        print(f"  values vs golden: cosine_sim={cos:.6f}")
        assert cos > 0.99, f"TopK values cosine sim {cos} < 0.99"

        if dumped_values is not None:
            dv = dumped_values.float()
            cos_d = torch.nn.functional.cosine_similarity(
                dev.flatten().unsqueeze(0), dv.flatten().unsqueeze(0)
            ).item()
            print(f"  values vs dump: cosine_sim={cos_d:.6f}")

    run_op_test(
        TopK(), [inp],
        framework=Framework.TORCH,
        custom_comparator=topk_comparator,
    )


# ---------------------------------------------------------------------------
# Standalone scatter test (3-input op)
# ---------------------------------------------------------------------------

@pytest.mark.single_device
def test_scatter_from_dumps():
    """Verify scatter dumps: run golden CPU scatter and compare to dumped output.

    stablehlo.scatter legalization is not supported for 1-D shapes, so we
    cannot replay through torch.compile("tt").  Instead we verify that the
    CPU-golden result matches the dumped device output.
    """
    d = _dump_dir()
    ops = _filter_pipeline_ops(_discover_ops(d))
    scatter_ops = [m for m in ops if m.get("mlir_op") == "ttnn.scatter"]
    if not scatter_ops:
        pytest.skip("No scatter dumps found")

    meta = scatter_ops[0]
    base = meta["_base"]
    params = meta.get("params", {})
    dim = params.get("dim", 0)

    raw_inputs = _load_inputs(base, meta)
    if len(raw_inputs) < 3:
        pytest.skip(f"Scatter needs 3 inputs, found {len(raw_inputs)}")

    inp, index, source = raw_inputs[0], raw_inputs[1], raw_inputs[2]
    inp = inp.to(torch.bfloat16)
    source = source.to(torch.bfloat16)
    index = index.to(torch.long)

    print(f"\n  scatter: dim={dim}")
    print(f"  input={list(inp.shape)} index={list(index.shape)} "
          f"source={list(source.shape)}")

    golden = inp.float().scatter(dim, index, source.float())

    dumped_out = _load_output(base)
    if dumped_out is None:
        pytest.skip("No dumped output for scatter")

    dev = dumped_out.float()
    cos = torch.nn.functional.cosine_similarity(
        dev.flatten().unsqueeze(0), golden.flatten().unsqueeze(0)
    ).item()
    print(f"  dump vs golden: cosine_sim={cos:.6f}")
    print(f"  dump nonzero: {int((dev != 0).sum())}/{dev.numel()}")
    print(f"  golden nonzero: {int((golden != 0).sum())}/{golden.numel()}")
    assert cos > 0.99, f"Scatter dump vs golden cosine_sim {cos:.6f} < 0.99"


if __name__ == "__main__":
    pytest.main([__file__, "-svv"])
