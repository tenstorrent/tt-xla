"""Convert tt-metal .tracy_artifacts to Perfetto Chrome Trace Event Format JSON."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TextIO


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


def main() -> int:
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
