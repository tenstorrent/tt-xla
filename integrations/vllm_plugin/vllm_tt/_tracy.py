# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC

"""Lightweight Tracy profiler helpers for the vLLM TT plugin.

Uses ttnn._ttnn.profiler when available (full TT environment with Tracy
build); silently degrades to no-ops otherwise so normal runs are unaffected.

Usage:
    from . import _tracy

    _tracy.signpost("decode_start")

    with _tracy.zone("prepare_inputs"):
        ...

    _tracy.frame()   # mark end of one decode iteration
"""

import contextlib
from typing import Optional

_profiler_obj = None
_probe_done = False


def _profiler():
    global _profiler_obj, _probe_done
    if not _probe_done:
        _probe_done = True
        try:
            import ttnn  # noqa: PLC0415

            _profiler_obj = ttnn._ttnn.profiler
        except Exception:
            pass
    return _profiler_obj


def signpost(name: str, message: Optional[str] = None) -> None:
    """Emit a Tracy signpost (message marker visible in the timeline)."""
    p = _profiler()
    if p is None:
        return
    text = (
        f"`TT_SIGNPOST: {name}`"
        if message is None
        else f"`TT_SIGNPOST: {name}\n{message}`"
    )
    p.tracy_message(text)


def frame() -> None:
    """Emit a Tracy frame boundary — marks the end of one decode iteration."""
    p = _profiler()
    if p is not None:
        p.tracy_frame()


@contextlib.contextmanager
def zone(name: str, color: int = 0):
    """Context manager that wraps code in a named Tracy zone."""
    p = _profiler()
    if p is not None:
        p.start_tracy_zone(__file__, name, 0, color)
    try:
        yield
    finally:
        if p is not None:
            p.stop_tracy_zone(name, color)


def zone_enter(name: str, color: int = 0) -> None:
    """Standalone zone enter — pair with zone_exit. For use in forward hooks
    where a context manager is awkward."""
    p = _profiler()
    if p is not None:
        p.start_tracy_zone(__file__, name, 0, color)


def zone_exit(name: str, color: int = 0) -> None:
    """Standalone zone exit — pair with zone_enter."""
    p = _profiler()
    if p is not None:
        p.stop_tracy_zone(name, color)


# Mark zone primitives as opaque to torch.compile so forward hooks emitting
# zones survive graph compilation. Without this, dynamo either graph-breaks
# or skips them; with it, the calls are inlined as opaque ops in the graph
# and fire on every compiled execution.
try:
    import torch.compiler  # noqa: PLC0415

    torch.compiler.allow_in_graph(zone_enter)
    torch.compiler.allow_in_graph(zone_exit)
except Exception:
    pass
