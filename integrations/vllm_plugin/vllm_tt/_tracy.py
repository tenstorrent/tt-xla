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


_auto_marker_cached: Optional[tuple] = None


def _auto_markers():
    """Resolve (line_marker, func_marker, finish_zones, doPartial, doLine)
    from the tracy package, or None if tracy isn't running.

    Re-resolves on each call so that state changes between import and use
    (e.g. tracy_state.doPartial set by `tracy -p`) are picked up.
    """
    try:
        import tracy.tracy_state as tracy_state  # noqa: PLC0415
        from tracy.tracy_ttnn import (  # noqa: PLC0415
            tracy_marker_func,
            tracy_marker_line,
            finish_all_zones,
        )
    except Exception:
        return None
    return (
        tracy_marker_line,
        tracy_marker_func,
        finish_all_zones,
        tracy_state.doPartial,
        tracy_state.doLine,
    )


@contextlib.contextmanager
def auto_profile():
    """Enable Tracy auto Python instrumentation for the enclosed block only.

    Run the harness as ``tracy -p -l --no-device -m pytest …`` to make this
    active: ``-p`` keeps the global setprofile off so we can scope which
    sections are auto-profiled; ``-l`` switches to per-line marks.
    Without those flags this is a no-op, so it's safe to leave in place.

    Bypasses ``tracy.Profiler`` (whose ``doProfile`` check requires
    ``sys.gettrace()/getprofile() is None`` and silently fails when
    pytest/coverage have already set a hook). We capture and restore the
    previous trace/profile so we don't trample existing instrumentation.
    """
    markers = _auto_markers()
    if markers is None:
        yield
        return
    line_marker, func_marker, finish_zones, do_partial, do_line = markers
    if not do_partial:
        yield
        return

    import sys  # noqa: PLC0415

    prev_trace = sys.gettrace()
    prev_profile = sys.getprofile()
    if do_line:
        sys.settrace(line_marker)
    else:
        sys.setprofile(func_marker)
    try:
        yield
    finally:
        sys.settrace(prev_trace)
        sys.setprofile(prev_profile)
        try:
            finish_zones()
        except Exception:
            pass
