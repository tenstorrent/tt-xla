"""Convert tt-metal .tracy_artifacts to Perfetto Chrome Trace Event Format JSON."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, TextIO


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


def main() -> int:
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
