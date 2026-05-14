#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Load DeepSeek-OCR via the local forge ``ModelLoader``, run one forward, and verify finite tensors.

Two hook modes:

- ``major`` (default): after outputs of ``model`` (full inner stack), each direct child of
  ``model`` (e.g. ``sam_model``, ``vision_model``, ``projector``, ``embed_tokens``, ``layers``),
  each ``model.layers.<i>``, and ``lm_head``. Pure tensor ops inside ``DeepseekOCRModel.forward``
  (``cat`` / ``view`` / ``masked_scatter`` between those submodules) are not separate hooks unless
  you use ``--hook-mode all``.
- ``all``: forward hook on every submodule (verbose; use ``--only-failures`` to limit noise).

**Origin tracing** (default: on): registers a **forward pre-hook** on every submodule to catch the
first **inputs** that contain NaN/Inf, and augments the post-hook timeline so you can see whether
the first anomaly appears as **arguments into** a child (blame the caller / earlier op) or as
**output of** a submodule (that forward produced the first NaNs). Use ``--no-trace-origin`` to skip
pre-hooks and the global sequence (slightly faster).

Prerequisites match other OCR scripts (``easydict``, image cache for ``doc.png``, etc.).

Run from tt-xla repo root::

    PYTHONPATH=. python scripts/probe_deepseek_ocr_finite_forward.py
    PYTHONPATH=. python scripts/probe_deepseek_ocr_finite_forward.py --hook-mode all
    PYTHONPATH=. python scripts/probe_deepseek_ocr_finite_forward.py --no-trace-origin
    PYTHONPATH=. python scripts/probe_deepseek_ocr_finite_forward.py --device cuda --bfloat16
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from third_party.tt_forge_models.deepseek.deepseek_ocr.pytorch.loader import ModelLoader


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _iter_tensors(obj: Any, seen: Optional[Set[int]] = None) -> Iterator[torch.Tensor]:
    if seen is None:
        seen = set()
    if obj is None:
        return
    if torch.is_tensor(obj):
        yield obj
        return
    oid = id(obj)
    if oid in seen:
        return
    seen.add(oid)
    if isinstance(obj, (tuple, list)):
        for x in obj:
            yield from _iter_tensors(x, seen)
        return
    if isinstance(obj, dict):
        for x in obj.values():
            yield from _iter_tensors(x, seen)
        return
    if hasattr(obj, "__dataclass_fields__"):
        for fname in obj.__dataclass_fields__:
            yield from _iter_tensors(getattr(obj, fname, None), seen)
        return
    for attr in ("logits", "last_hidden_state", "hidden_states", "past_key_values"):
        if hasattr(obj, attr):
            yield from _iter_tensors(getattr(obj, attr), seen)


def _tensor_finite_report(t: torch.Tensor) -> Tuple[bool, int, int]:
    """Return (all_finite, finite_count, total_count)."""
    total = t.numel()
    if total == 0:
        return True, 0, 0
    fin = torch.isfinite(t)
    nfin = int(fin.sum().item())
    return nfin == total, nfin, total


def _finite_value_range_str(t: torch.Tensor) -> Tuple[str, str]:
    """Min/max over finite entries only; avoids Tensor.nanmin (missing on some builds / dtypes)."""
    with torch.no_grad():
        flat = t.detach().flatten()
        m = torch.isfinite(flat)
        if not bool(m.any().item()):
            return "nan", "nan"
        vals = flat[m].to(torch.float64)
        return f"{vals.min().item():.4g}", f"{vals.max().item():.4g}"


def _check_tensors(tensors: Iterable[torch.Tensor]) -> List[str]:
    problems: List[str] = []
    for idx, t in enumerate(tensors):
        ok, nfin, total = _tensor_finite_report(t)
        if not ok:
            n_nan = int(torch.isnan(t).sum().item())
            n_inf = int(torch.isinf(t).sum().item())
            vmin, vmax = _finite_value_range_str(t)
            problems.append(
                f"  tensor[{idx}] shape={tuple(t.shape)} dtype={t.dtype} "
                f"finite={nfin}/{total} nan={n_nan} inf={n_inf} min={vmin} max={vmax}"
            )
    return problems


@dataclass
class _HookState:
    failures: List[str] = field(default_factory=list)
    ok_count: int = 0


@dataclass
class _OriginState:
    """Monotonic hook timeline to locate first non-finite tensors."""

    seq: int = 0
    first_bad_input: Optional[Tuple[int, str, List[str]]] = None
    first_bad_output: Optional[Tuple[int, str, List[str]]] = None

    def next_seq(self) -> int:
        self.seq += 1
        return self.seq


def _major_hook_names(model: nn.Module) -> Set[str]:
    """Submodule names to hook for a high-level trace."""
    names: Set[str] = {"model"}
    for n, m in model.named_modules():
        if n == "model":
            for child_name, child in m.named_children():
                if isinstance(child, nn.ModuleList):
                    continue
                names.add(f"model.{child_name}")
            continue
        if re.fullmatch(r"model\.layers\.\d+", n):
            names.add(n)
        if n == "lm_head":
            names.add(n)
    return names


def _register_forward_pre_hook_compat(
    mod: nn.Module, hook_with_kwargs, hook_args_only
) -> torch.utils.hooks.RemovableHandle:
    try:
        return mod.register_forward_pre_hook(hook_with_kwargs, with_kwargs=True)
    except TypeError:
        return mod.register_forward_pre_hook(hook_args_only)


def _pre_hook_pair(origin: _OriginState, full_name: str) -> Tuple[Any, Any]:
    def pre_with_kwargs(_m: nn.Module, args: Any, kwargs: Any = None) -> None:
        seq = origin.next_seq()
        tensors_in = list(_iter_tensors(args))
        if kwargs:
            tensors_in.extend(_iter_tensors(kwargs))
        probs_in = _check_tensors(tensors_in)
        if probs_in and origin.first_bad_input is None:
            origin.first_bad_input = (seq, full_name, probs_in)

    def pre_args_only(_m: nn.Module, args: Any) -> None:
        seq = origin.next_seq()
        tensors_in = list(_iter_tensors(args))
        probs_in = _check_tensors(tensors_in)
        if probs_in and origin.first_bad_input is None:
            origin.first_bad_input = (seq, full_name, probs_in)

    return pre_with_kwargs, pre_args_only


def _register_hooks(
    model: nn.Module,
    *,
    mode: str,
    verbose: bool,
    state: _HookState,
    origin: Optional[_OriginState],
) -> List[torch.utils.hooks.RemovableHandle]:
    handles: List[torch.utils.hooks.RemovableHandle] = []
    allowed = _major_hook_names(model) if mode == "major" else None

    def make_post_hook(full_name: str):
        def post_hook(_mod: nn.Module, _inp: Any, outp: Any) -> None:
            if origin is not None:
                seq = origin.next_seq()
                tensors_o = list(_iter_tensors(outp))
                probs_o = _check_tensors(tensors_o)
                if probs_o and origin.first_bad_output is None:
                    origin.first_bad_output = (seq, full_name, probs_o)

            if allowed is not None and full_name not in allowed:
                return

            tensors = list(_iter_tensors(outp))
            if not tensors:
                if verbose:
                    print(f"OK  {full_name}  (no tensor outputs)", flush=True)
                return
            probs = _check_tensors(tensors)
            if probs:
                state.failures.append(f"[{full_name}]")
                state.failures.extend(probs)
            else:
                state.ok_count += 1
                if verbose:
                    shapes = ", ".join(str(tuple(t.shape)) for t in tensors[:3])
                    more = " …" if len(tensors) > 3 else ""
                    print(f"OK  {full_name}  tensors={len(tensors)}  {shapes}{more}", flush=True)

        return post_hook

    for name, mod in model.named_modules():
        if name == "":
            continue

        if origin is not None:
            pk, pa = _pre_hook_pair(origin, name)
            handles.append(_register_forward_pre_hook_compat(mod, pk, pa))

        if origin is not None or allowed is None or name in allowed:
            handles.append(mod.register_forward_hook(make_post_hook(name)))
    return handles


def _print_origin_report(origin: _OriginState) -> None:
    print("---", flush=True)
    print("NaN/Inf origin (global hook order: each pre- and post-hook advances one step)", flush=True)
    if origin.first_bad_input is None and origin.first_bad_output is None:
        print("  No submodule saw non-finite tensor inputs or outputs in hooked modules.", flush=True)
        return
    if origin.first_bad_input is not None:
        seq, name, lines = origin.first_bad_input
        print(f"  First non-finite INPUT to a submodule:  step={seq}  module={name!r}", flush=True)
        print("\n".join(lines), flush=True)
    if origin.first_bad_output is not None:
        seq, name, lines = origin.first_bad_output
        print(f"  First non-finite OUTPUT from a submodule: step={seq}  module={name!r}", flush=True)
        print("\n".join(lines), flush=True)
    if origin.first_bad_input is not None and origin.first_bad_output is not None:
        si, mi, _ = origin.first_bad_input
        so, mo, _ = origin.first_bad_output
        if si < so:
            print(
                "  Interpretation: hooks first saw non-finite **inputs** to "
                f"{mi!r} (step {si}) before the first bad **output** from {mo!r} (step {so}). "
                "That usually means NaNs/Inf were introduced in **Python/tensor ops between "
                "submodules** (no `nn.Module` forward), or via buffers/parameters, rather than "
                f"the forward of {mo!r} alone.",
                flush=True,
            )
        elif so < si:
            print(
                "  Interpretation: hooks first saw a non-finite **output** from "
                f"{mo!r} (step {so}) before non-finite **inputs** to {mi!r} (step {si}). "
                f"Treat {mo!r} (and its children, if any) as the primary suspect for where NaNs "
                "started.",
                flush=True,
            )
        else:
            print(
                "  Interpretation: first bad input and first bad output were recorded at the "
                "same step index; inspect both modules listed above.",
                flush=True,
            )
    elif origin.first_bad_output is not None:
        _, mo, _ = origin.first_bad_output
        print(
            f"  Interpretation: first signal was a bad **output** from {mo!r} "
            "(no earlier bad submodule inputs were recorded).",
            flush=True,
        )
    elif origin.first_bad_input is not None:
        _, mi, _ = origin.first_bad_input
        print(
            f"  Interpretation: first signal was bad **inputs** to {mi!r} "
            "(no bad submodule output was recorded — unusual).",
            flush=True,
        )


def _move_inputs(batch: dict, device: torch.device, float_dtype: torch.dtype) -> dict:
    out: dict = {}
    for k, v in batch.items():
        if k == "images":
            moved = []
            for crop, ori in v:
                moved.append(
                    (
                        crop.to(device=device, dtype=float_dtype),
                        ori.to(device=device, dtype=float_dtype),
                    )
                )
            out[k] = moved
        elif torch.is_tensor(v):
            if v.is_floating_point():
                out[k] = v.to(device=device, dtype=float_dtype)
            else:
                out[k] = v.to(device=device)
        else:
            out[k] = v
    return out


def _check_input_batch(batch: dict, tag: str) -> List[str]:
    problems: List[str] = []
    for k, v in batch.items():
        if k == "images":
            for i, pair in enumerate(v):
                for j, t in enumerate(pair):
                    probs = _check_tensors([t])
                    if probs:
                        problems.append(f"{tag} batch[{k!r}][{i}][{j}]")
                        problems.extend(probs)
        elif torch.is_tensor(v):
            probs = _check_tensors([v])
            if probs:
                problems.append(f"{tag} batch[{k!r}]")
                problems.extend(probs)
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument(
        "--hook-mode",
        choices=("major", "all"),
        default="major",
        help="major: vision stack + each decoder layer + lm_head; all: every submodule.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print OK for each hooked module (very noisy for hook-mode=all).",
    )
    parser.add_argument("--bfloat16", action="store_true", help="Use bfloat16 weights and floats.")
    parser.add_argument(
        "--trace-origin",
        action="store_true",
        default=True,
        help="Pre-hooks + ordered steps to find first non-finite inputs/outputs (default: on).",
    )
    parser.add_argument(
        "--no-trace-origin",
        action="store_false",
        dest="trace_origin",
        help="Disable origin tracing (fewer hooks).",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    float_dtype = torch.bfloat16 if args.bfloat16 else torch.float32

    os.chdir(_repo_root())

    print(
        f"device={device} dtype={float_dtype} hook_mode={args.hook_mode} "
        f"trace_origin={args.trace_origin}",
        flush=True,
    )

    loader = ModelLoader()
    inputs = loader.load_inputs(
        dtype_override=float_dtype if args.bfloat16 else torch.float32
    )
    inp_probs = _check_input_batch(inputs, "input")
    if inp_probs:
        print("NON-FINITE INPUTS:", flush=True)
        print("\n".join(inp_probs), flush=True)
        return 1

    model = loader.load_model(torch_dtype=float_dtype)
    model.eval()
    model.to(device)
    batch = _move_inputs(inputs, device, float_dtype)

    state = _HookState()
    origin = _OriginState() if args.trace_origin else None
    handles = _register_hooks(
        model,
        mode=args.hook_mode,
        verbose=args.verbose,
        state=state,
        origin=origin,
    )

    print("Running forward with finite hooks ...", flush=True)
    try:
        with torch.no_grad():
            out = model(**batch, return_dict=True, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    logits = out.logits
    log_probs = _check_tensors([logits])
    print("---", flush=True)
    print(f"post-forward logits shape={tuple(logits.shape)} dtype={logits.dtype}", flush=True)
    if log_probs:
        print("NON-FINITE logits:", flush=True)
        print("\n".join(log_probs), flush=True)
    else:
        print("logits: all finite", flush=True)

    if origin is not None:
        _print_origin_report(origin)

    if state.failures:
        print("---", flush=True)
        print(f"HOOK FAILURES ({len(state.failures)} lines):", flush=True)
        print("\n".join(state.failures), flush=True)

    if args.verbose:
        print(f"hook modules with finite tensor outputs: {state.ok_count}", flush=True)

    if inp_probs or log_probs or state.failures:
        print("RESULT: non-finite values detected.", flush=True)
        return 1
    print("RESULT: inputs, hooked activations, and logits are finite.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
