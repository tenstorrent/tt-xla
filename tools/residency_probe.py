# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Residency + compile + cache-tier instrumentation for staged pipelines.

DEV-ONLY. This module is untracked and must never reach a PR. Import it from
*inside a function*, never at module scope: a checkout that lacks ``tools/``
would otherwise fail with ModuleNotFoundError.

Emits three independent signals at every stage boundary, so "did this component
rebuild?" is measured rather than inferred from wall clock:

  [DRAM]  xm.get_memory_info(<explicit device>) -- device DRAM residency
  [CMPL]  torch_xla.debug.metrics counters      -- XLA compilations
  [TIER]  generated/inspector/programs_log.yaml -- tt-metal program build tier

THREE TIERS, not a cold/warm split:
  tier 1  first ever build, empty on-disk kernel cache   kernels AND program built
  tier 2  later residency, warm TT_METAL_CACHE           kernels reused, program REBUILT
  tier 3  repeat inside the SAME residency               nothing rebuilt -- the only warm

Two traps this module exists to avoid:

1. ``xm.get_memory_info()`` with no argument resolves to ``torch_xla.device(None)``,
   which under SPMD is the virtual device "SPMD:0" and trips an
   XLA_CHECK_NE(device, spmd_device_str) in PjRtComputationClient::GetMemoryInfo.
   That THROWS. A bare ``except Exception`` makes it indistinguishable from
   "not implemented", which is why every earlier log read ``dram=n/a``.
   We pass an explicit addressable device and classify the failure.

2. torch_xla/_dynamo/dynamo_bridge.py:650 calls metrics.clear_counters() and
   restores ONLY DynamoExtractCompiledGraph and UncachedCompile. Deltas built on
   CachedCompile / MarkStep / CompileTime CAN GO NEGATIVE across a stage
   boundary. Those fields are printed with a trailing '~' and are NOT
   load-bearing. Use `uncached` and `dynamo`.

Requires the dev-only PJRT_Device_MemoryStats implementation in
pjrt_implementation/src/api/device_instance.cc (issue #470 stays open).

MEASURED LIMITATION of that binding, do not chase it: torch_xla's MemoryInfo
TypedDict (xla_model.py:1506) carries ONLY bytes_used and bytes_limit. The C++
sets largest_free_block_bytes and the PJRT struct carries it, but _xla_memory_info
never plumbs it to Python -- so free_blk can only ever read n/a from here, no
matter what the plugin does. peak_bytes_used is absent from the dict too, and the
runtime never sets peak_bytes_in_use anyway.
"""

import os
import time
from contextlib import contextmanager

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.runtime as xr

GIB = float(1 << 30)

# Inspector duration_ns bands. A real JIT compile is 270-380 ms; an on-disk
# kernel-cache hit is ~0.19 ms. These bracket both with three orders of margin.
TIER1_NS = 1e8  # >= this is a real compile
TIER2_NS = 1e6  # <  this is an on-disk cache hit

# Cumulative values as of the previous snap(), so each line can show deltas.
_prev = {}

# Resolved once, then latched. See trap (1) above.
_DRAM = {"resolved": False, "dev": None, "reason": None, "detail": "", "peak": False}

# Byte offset into programs_log.yaml, so each snap parses only what was appended.
_INSPECTOR = {"path": None, "offset": 0}


# ---------------------------------------------------------------- DRAM


def _classify(exc):
    """Map an exception to (reason, detail) so failures render distinguishably."""
    text = str(exc)
    low = text.lower()
    if "unimplemented" in low or "not implemented" in low:
        return "UNIMPLEMENTED", text
    if "spmd" in low:
        return "SPMD-BLOCKED", text
    return "ERROR", text


def _resolve_dram_device():
    """Try every addressable device in turn; latch the first that answers.

    Returns the winning torch.device, or None with the reason recorded in _DRAM.
    """
    last = ("ERROR", "no addressable devices reported")
    for spec in xm.get_xla_supported_devices():  # ['xla:0', 'xla:1', ...]
        try:
            info = xm.get_memory_info(torch.device(spec))
        except Exception as exc:  # noqa: BLE001 -- classified, not swallowed
            last = _classify(exc)
            continue
        # peak_bytes_in_use is never set by the runtime, so this reads 0 and we
        # latch peak as unsupported rather than printing a misleading 0.000GiB.
        _DRAM["peak"] = info.get("peak_bytes_used", 0) > 0
        return torch.device(spec)
    _DRAM["reason"], _DRAM["detail"] = last
    return None


def dram(device=None):
    """Device DRAM for an EXPLICIT device. Never passes None to torch_xla.

    Returns a dict with used/limit/free_block/dev, or {"reason": ...} on failure.
    """
    if device is None:
        if not _DRAM["resolved"]:
            _DRAM["dev"] = _resolve_dram_device()
            _DRAM["resolved"] = True
        device = _DRAM["dev"]
        if device is None:
            return {"reason": _DRAM["reason"], "detail": _DRAM["detail"]}
    if not isinstance(device, torch.device):
        device = torch.device(str(device))
    try:
        info = xm.get_memory_info(device)
    except Exception as exc:  # noqa: BLE001
        reason, detail = _classify(exc)
        return {"reason": reason, "detail": detail}
    return {
        "used": info["bytes_used"],
        "limit": info["bytes_limit"],
        "free_block": info.get("largest_free_block_bytes"),
        "dev": str(device),
    }


def _fmt_dram(info):
    if "reason" in info:
        return f"{info['reason']:<14} {info['detail'][:80]}"
    used, limit = info["used"], info["limit"]
    pct = 100.0 * used / limit if limit else 0.0
    blk = info["free_block"]
    blk_s = f"{blk / GIB:8.3f}GiB" if blk is not None else "     n/a"
    peak_s = "n/s" if not _DRAM["peak"] else "set"
    return (
        f"used={used / GIB:8.3f}GiB limit={limit / GIB:8.3f}GiB ({pct:5.1f}%) "
        f"free_blk={blk_s} peak={peak_s} dev={info['dev']}"
    )


# ------------------------------------------------------------- COMPILE


def _counter(name):
    value = met.counter_value(name)
    return 0 if value is None else value


def _compile_seconds():
    """Accumulated CompileTime in seconds (the metric accumulates nanoseconds)."""
    data = met.metric_data("CompileTime")
    return data[1] / 1e9 if data else 0.0


def compile_stats():
    """Counters. Only `uncached` and `dynamo` survive dynamo's counter reset."""
    return {
        "graphs": xr.get_num_cached_compilation_graph(),
        "uncached": _counter("UncachedCompile"),
        "dynamo": _counter("DynamoExtractCompiledGraph"),
        # Below here: NOT load-bearing, deltas can go negative. See trap (2).
        "cached": _counter("CachedCompile"),
        "ctime": _compile_seconds(),
        # Direct evidence for #498: Serialize is implemented, DeserializeAndLoad
        # is stubbed, so entries are written and never read back.
        "pcache_hit": _counter("PersistentCacheHit"),
        "pcache_miss": _counter("PersistentCacheMiss"),
        "pcache_fail": _counter("PersistentCacheDeserializeFailure"),
    }


# ---------------------------------------------------- TT-METAL CACHE TIER


def _inspector_path():
    root = os.environ.get("TT_METAL_LOGS_PATH") or os.getcwd()
    return os.path.join(root, "generated", "inspector", "programs_log.yaml")


def inspector_delta():
    """Program builds recorded since the previous call.

    The file is a flat, append-only, flushed sequence of records:

        - program_compile_finished:
            id: 21
            duration_ns: 177912425

    so a line scanner is exact and no YAML parser is needed. Reading from a byte
    offset keeps each snap O(bytes appended); re-parsing the whole file every
    time would be O(n^2) and would itself perturb the measurement.
    """
    if _INSPECTOR["path"] is None:
        _INSPECTOR["path"] = _inspector_path()
    path = _INSPECTOR["path"]
    empty = {"progs": 0, "jit": 0, "diskhit": 0, "kernels": 0, "lo": None, "hi": None}
    if not os.path.exists(path):
        return None  # renders as NO-INSPECTOR
    with open(path, "r") as handle:
        handle.seek(_INSPECTOR["offset"])
        chunk = handle.read()
    if not chunk:
        return empty
    # Never parse a half-written trailing record.
    cut = chunk.rfind("\n")
    if cut == -1:
        return empty
    _INSPECTOR["offset"] += cut + 1
    chunk = chunk[: cut + 1]

    durations = []
    kernels = 0
    record = None
    for line in chunk.splitlines():
        if line.startswith("- ") and line.endswith(":"):
            record = line[2:-1]
            if record == "program_kernel_compile_finished":
                kernels += 1
        elif record == "program_compile_finished" and "duration_ns:" in line:
            durations.append(int(line.split("duration_ns:")[1].strip()))
    return {
        "progs": len(durations),
        "jit": sum(1 for d in durations if d >= TIER1_NS),
        "diskhit": sum(1 for d in durations if d < TIER2_NS),
        "kernels": kernels,
        "lo": min(durations) / 1e6 if durations else None,
        "hi": max(durations) / 1e6 if durations else None,
    }


def _verdict(insp, uncached_delta):
    """Classify the tier. TIER3 is INFERRED from absence, not read from the file:
    log_program_compile_already_exists is a deliberate no-op (logger.cpp:144-146),
    so an in-memory program-cache hit logs nothing at all."""
    if insp is None:
        return "NO-INSPECTOR"
    if insp["progs"] == 0:
        return "TIER3" if uncached_delta == 0 else "XLA-ONLY"
    if insp["jit"] and insp["diskhit"]:
        return f"TIER1/2-MIXED({insp['jit']}jit/{insp['diskhit']}hit)"
    if insp["jit"]:
        return "TIER1"
    if insp["diskhit"] == insp["progs"]:
        return "TIER2"
    return f"TIER?({insp['lo']:.1f}..{insp['hi']:.1f}ms)"


# ---------------------------------------------------------------- OUTPUT


def probe_init(tag=""):
    """One-time banner. The counter-reset warning is here so the next reader does
    not re-derive the negative-delta mystery a third time."""
    if not _DRAM["resolved"]:
        _DRAM["dev"] = _resolve_dram_device()
        _DRAM["resolved"] = True
    path = _inspector_path()
    kernels_yaml = os.path.join(os.path.dirname(path), "kernels.yaml")
    cache_key = "<unknown>"
    if os.path.exists(kernels_yaml):
        with open(kernels_yaml, "r") as handle:
            for line in handle:
                if "path:" in line:
                    cache_key = line.split("path:")[1].strip()
                    break
    dev = _DRAM["dev"]
    print(
        f"[PROBE-INIT] {tag}\n"
        f"[PROBE-INIT] dram_device={dev}  "
        f"{'resolved' if dev is not None else _DRAM['reason']}  "
        f"addressable={len(xm.get_xla_supported_devices())}\n"
        f"[PROBE-INIT] dram_peak={'set' if _DRAM['peak'] else 'UNSUPPORTED'} "
        f"(runtime never sets peak_bytes_in_use; #470 patch is dev-only)\n"
        f"[PROBE-INIT] inspector={path} exists={os.path.exists(path)}\n"
        f"[PROBE-INIT] kernel_cache_path={cache_key}\n"
        f"[PROBE-INIT] tt_metal_cache={os.environ.get('TT_METAL_CACHE', '<unset>')} "
        f"force_jit={os.environ.get('TT_METAL_FORCE_JIT_COMPILE', '0')}\n"
        f"[PROBE-INIT] WARNING cached/ctime deltas may be NEGATIVE: dynamo_bridge.py:650\n"
        f"[PROBE-INIT]          clears all counters, restores only UncachedCompile +\n"
        f"[PROBE-INIT]          DynamoExtractCompiledGraph. Fields marked '~' are not\n"
        f"[PROBE-INIT]          load-bearing; use uncached and dynamo.",
        flush=True,
    )


def snap(tag: str, *, sync: bool = True, extra: str = "", device=None) -> None:
    """Emit [DRAM] / [CMPL] / [TIER] for this instant, with deltas since the last snap."""
    if sync:
        torch_xla.sync()
        xm.wait_device_ops()

    now = compile_stats()
    delta = {k: now[k] - _prev.get(k, 0) for k in now}
    _prev.update(now)

    insp = inspector_delta()
    verdict = _verdict(insp, delta["uncached"])

    print(f"[DRAM] {tag:<38} {_fmt_dram(dram(device))}", flush=True)
    print(
        f"[CMPL] {tag:<38} "
        f"uncached={now['uncached']:<4}(+{delta['uncached']:<3}) "
        f"dynamo={now['dynamo']:<4}(+{delta['dynamo']:<3}) "
        f"graphs={now['graphs']:<4}(+{delta['graphs']:<3}) "
        f"pcache h/m/f={now['pcache_hit']}/{now['pcache_miss']}/{now['pcache_fail']} "
        f"ctime={now['ctime']:9.2f}s(+{delta['ctime']:.2f})~ "
        f"cached={now['cached']:<4}(+{delta['cached']})~",
        flush=True,
    )
    if insp is None:
        print(
            f"[TIER] {tag:<38} NO-INSPECTOR  programs_log.yaml absent at "
            f"{_INSPECTOR['path']}",
            flush=True,
        )
    else:
        band = (
            f"[{insp['lo']:6.1f}..{insp['hi']:6.1f}ms]"
            if insp["lo"] is not None
            else "[     -..     -ms]"
        )
        print(
            f"[TIER] {tag:<38} "
            f"progs=+{insp['progs']:<4} jit={insp['jit']:<4}{band} "
            f"diskhit={insp['diskhit']:<4} kern=+{insp['kernels']:<5} "
            f"verdict={verdict} {extra}",
            flush=True,
        )


def param_bytes(module) -> int:
    return sum(p.numel() * p.element_size() for p in module.parameters())


def snap_module(tag: str, module, **kwargs) -> None:
    """snap() plus the module's own weight footprint and placement."""
    devices = sorted({str(p.device) for p in module.parameters()})
    snap(
        tag,
        extra=f"weights={param_bytes(module) / GIB:.3f}GiB on {devices}",
        **kwargs,
    )


@contextmanager
def stage_timer(label: str):
    """Time a sub-step (host load / weight upload / forward) and print it alone."""
    t0 = time.perf_counter()
    yield
    print(f"[TIME] {label:<44} {time.perf_counter() - t0:9.2f}s", flush=True)
