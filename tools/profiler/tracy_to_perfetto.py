"""Convert tt-metal .tracy_artifacts to Perfetto Chrome Trace Event Format JSON."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence, TextIO


@dataclass(frozen=True)
class DeviceHeader:
    arch: str
    chip_freq_mhz: int


_HEADER_RE = re.compile(r"ARCH:\s*(\w+),\s*CHIP_FREQ\[MHz\]:\s*(\d+)")


def parse_device_header(stream: TextIO) -> DeviceHeader:
    """Parse the first metadata line of profile_log_device.csv."""
    first_line = stream.readline()
    match = _HEADER_RE.search(first_line)
    if not match:
        raise ValueError(f"profile_log_device.csv missing ARCH/CHIP_FREQ header: {first_line!r}")
    return DeviceHeader(arch=match.group(1), chip_freq_mhz=int(match.group(2)))


@dataclass(frozen=True)
class DeviceEvent:
    pcie_slot: int
    core_x: int
    core_y: int
    risc: str
    time_cycles: int
    run_host_id: int
    zone_name: str
    event_type: str  # "ZONE_START" or "ZONE_END"


def iter_device_events(path: Path) -> Iterator[DeviceEvent]:
    """Yield one DeviceEvent per data row in profile_log_device.csv."""
    with open(path, newline="") as f:
        f.readline()  # skip ARCH line
        reader = csv.reader(f)
        next(reader)  # skip column-name line
        for row in reader:
            yield DeviceEvent(
                pcie_slot=int(row[0]),
                core_x=int(row[1]),
                core_y=int(row[2]),
                risc=row[3],
                time_cycles=int(row[5]),
                run_host_id=int(row[7]),
                zone_name=row[10],
                event_type=row[11],
            )


@dataclass(frozen=True)
class Zone:
    pcie_slot: int
    core_x: int
    core_y: int
    risc: str
    run_host_id: int
    zone_name: str
    start_cycles: int
    end_cycles: int

    @property
    def duration_cycles(self) -> int:
        return self.end_cycles - self.start_cycles


def pair_zones(events: Iterable[DeviceEvent]) -> list[Zone]:
    """Pair ZONE_START / ZONE_END rows into Zones (FIFO stack per key)."""
    open_stacks: dict[tuple, list[DeviceEvent]] = defaultdict(list)
    zones: list[Zone] = []
    for ev in events:
        key = (ev.pcie_slot, ev.core_x, ev.core_y, ev.risc, ev.run_host_id, ev.zone_name)
        if ev.event_type == "ZONE_START":
            open_stacks[key].append(ev)
        elif ev.event_type == "ZONE_END":
            if not open_stacks[key]:
                raise ValueError(f"unmatched ZONE_END for {key}")
            start = open_stacks[key].pop()
            zones.append(Zone(
                pcie_slot=ev.pcie_slot,
                core_x=ev.core_x,
                core_y=ev.core_y,
                risc=ev.risc,
                run_host_id=ev.run_host_id,
                zone_name=ev.zone_name,
                start_cycles=start.time_cycles,
                end_cycles=ev.time_cycles,
            ))
        else:
            raise ValueError(f"unknown event_type {ev.event_type!r}")
    leftover = [k for k, v in open_stacks.items() if v]
    if leftover:
        raise ValueError(f"unmatched ZONE_START events for keys: {leftover}")
    return zones


@dataclass(frozen=True)
class OpInfo:
    op_code: str
    global_call_count: int
    compute_kernels: list[str]
    data_movement_kernels: list[str]


def _split_kernel_list(cell: str) -> list[str]:
    """Parse a ['x.cpp'; 'y.cpp'] cell from the ops CSV."""
    cell = cell.strip()
    if not cell or cell == "[]":
        return []
    inner = cell.strip("[]")
    return [piece.strip().strip("'\"") for piece in inner.split(";") if piece.strip()]


def load_ops(path: Path) -> dict[int, OpInfo]:
    """Index ops_perf_results_*.csv by GLOBAL CALL COUNT."""
    out: dict[int, OpInfo] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gcc = int(row["GLOBAL CALL COUNT"])
            out[gcc] = OpInfo(
                op_code=row["OP CODE"],
                global_call_count=gcc,
                compute_kernels=_split_kernel_list(row.get("COMPUTE KERNEL SOURCE", "")),
                data_movement_kernels=_split_kernel_list(row.get("DATA MOVEMENT KERNEL SOURCE", "")),
            )
    return out


def _basename_no_ext(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def classify_kernel(risc: str, op: Optional[OpInfo], fallback: str) -> tuple[str, str]:
    """Return (display_name, role) for a given RISC and op."""
    if op is None:
        return (fallback, "unknown")

    if risc.startswith("TRISC"):
        if op.compute_kernels:
            return (_basename_no_ext(op.compute_kernels[0]), "compute")
        return (fallback, "unknown")

    if risc in ("BRISC", "NCRISC"):
        prefix = "reader_" if risc == "BRISC" else "writer_"
        for src in op.data_movement_kernels:
            if _basename_no_ext(src).startswith(prefix):
                return (_basename_no_ext(src), prefix.rstrip("_"))
        # Fall back to list-order: BRISC = first, NCRISC = second
        idx = 0 if risc == "BRISC" else 1
        if idx < len(op.data_movement_kernels):
            base = _basename_no_ext(op.data_movement_kernels[idx])
            return (base, base)
        return (fallback, "unknown")

    return (fallback, "unknown")


_RISC_INDEX = {"BRISC": 0, "NCRISC": 1, "TRISC_0": 2, "TRISC_1": 3, "TRISC_2": 4, "ERISC": 5}


def _tid(core_x: int, core_y: int, risc: str) -> int:
    return core_x * 1_000_000 + core_y * 1_000 + _RISC_INDEX.get(risc, 9)


def zones_to_perfetto(
    zones: list[Zone],
    ops: dict[int, OpInfo],
    chip_freq_mhz: int,
) -> list[dict]:
    """Build Chrome Trace Event Format event list for Perfetto."""
    events: list[dict] = []
    seen_processes: set[int] = set()
    seen_threads: set[tuple[int, int]] = set()  # (pcie_slot, tid)

    cycles_to_us = 1.0 / chip_freq_mhz

    for z in zones:
        op = ops.get(z.run_host_id)
        is_fw = z.zone_name.endswith("-FW")
        if is_fw:
            display_name, role = z.zone_name, "fw"
            kernel_source = ""
        else:
            display_name, role = classify_kernel(z.risc, op, fallback=z.zone_name)
            kernel_source = ""
            if op is not None:
                if z.risc.startswith("TRISC") and op.compute_kernels:
                    kernel_source = op.compute_kernels[0]
                elif z.risc in ("BRISC", "NCRISC"):
                    prefix = "reader_" if z.risc == "BRISC" else "writer_"
                    matched = [s for s in op.data_movement_kernels if _basename_no_ext(s).startswith(prefix)]
                    if matched:
                        kernel_source = matched[0]
                    elif op.data_movement_kernels:
                        idx = 0 if z.risc == "BRISC" else 1
                        if idx < len(op.data_movement_kernels):
                            kernel_source = op.data_movement_kernels[idx]

        tid = _tid(z.core_x, z.core_y, z.risc)

        if z.pcie_slot not in seen_processes:
            events.append({
                "ph": "M", "name": "process_name", "pid": z.pcie_slot,
                "args": {"name": f"Chip {z.pcie_slot}"},
            })
            seen_processes.add(z.pcie_slot)

        thread_key = (z.pcie_slot, tid)
        if thread_key not in seen_threads:
            events.append({
                "ph": "M", "name": "thread_name", "pid": z.pcie_slot, "tid": tid,
                "args": {"name": f"({z.core_x},{z.core_y}) {z.risc}"},
            })
            seen_threads.add(thread_key)

        events.append({
            "ph": "X",
            "name": display_name,
            "cat": role,
            "ts": z.start_cycles * cycles_to_us,
            "dur": (z.end_cycles - z.start_cycles) * cycles_to_us,
            "pid": z.pcie_slot,
            "tid": tid,
            "args": {
                "op_code": op.op_code if op else "",
                "run_host_id": z.run_host_id,
                "risc": z.risc,
                "kernel_source": kernel_source,
                "zone_name": z.zone_name,
            },
        })

    return events


def _discover_inputs(reports_dir: Path) -> tuple[Path, Path]:
    device = reports_dir / "profile_log_device.csv"
    if not device.exists():
        raise FileNotFoundError(f"profile_log_device.csv not found in {reports_dir}")
    ops_matches = sorted(reports_dir.glob("ops_perf_results_*.csv"))
    if not ops_matches:
        raise FileNotFoundError(f"No ops_perf_results_*.csv found in {reports_dir}")
    if len(ops_matches) > 1:
        print(f"Warning: multiple ops CSVs found, using {ops_matches[-1].name}", file=sys.stderr)
    return device, ops_matches[-1]


def _write_trace(events: list[dict], out_path: Path) -> None:
    payload = {"traceEvents": events, "displayTimeUnit": "ns"}
    if out_path.suffix == ".gz":
        with gzip.open(out_path, "wt") as f:
            json.dump(payload, f)
    else:
        with open(out_path, "w") as f:
            json.dump(payload, f)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert tt-metal .tracy_artifacts/reports/<ts>/ to a Perfetto-loadable trace."
    )
    parser.add_argument("reports_dir", type=Path,
                        help="Path to the reports/<timestamp>/ directory")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output path (.json or .json.gz). Default: <reports_dir>/perfetto_kernel_timeline.json.gz")
    args = parser.parse_args(argv)

    device_csv, ops_csv = _discover_inputs(args.reports_dir)
    out = args.output or (args.reports_dir / "perfetto_kernel_timeline.json.gz")

    with open(device_csv) as f:
        header = parse_device_header(f)

    zones = pair_zones(iter_device_events(device_csv))
    ops = load_ops(ops_csv)
    events = zones_to_perfetto(zones, ops, chip_freq_mhz=header.chip_freq_mhz)
    _write_trace(events, out)

    print(f"Wrote {len(events)} events to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
