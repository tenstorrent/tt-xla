from io import StringIO
from pathlib import Path

import pytest

from tools.profiler import tracy_to_perfetto
from tools.profiler.tracy_to_perfetto import DeviceEvent, iter_device_events, pair_zones, parse_device_header

FIXTURES = Path(__file__).parent / "fixtures"


def test_module_importable():
    assert callable(tracy_to_perfetto.main)


def test_parse_device_header_extracts_chip_freq_mhz():
    sample = (
        "ARCH: blackhole, CHIP_FREQ[MHz]: 1350, Max Compute Cores: 140\n"
        "PCIe slot, core_x, core_y, RISC processor type, timer_id, ...\n"
    )
    header = parse_device_header(StringIO(sample))
    assert header.chip_freq_mhz == 1350
    assert header.arch == "blackhole"


def test_iter_device_events_yields_one_per_row():
    events = list(iter_device_events(FIXTURES / "mini_device.csv"))
    assert len(events) == 16  # 4 zones x 4 events for op 32770 + 1 zone x 4 for op 32771

    first = events[0]
    assert first.pcie_slot == 2
    assert first.core_x == 1
    assert first.core_y == 2
    assert first.risc == "BRISC"
    assert first.time_cycles == 1000
    assert first.run_host_id == 32770
    assert first.zone_name == "BRISC-FW"
    assert first.event_type == "ZONE_START"


def test_pair_zones_produces_complete_intervals():
    events = list(iter_device_events(FIXTURES / "mini_device.csv"))
    zones = pair_zones(events)

    # 5 zones for op 32770 (BRISC-FW, BRISC-KERNEL, NCRISC-FW, NCRISC-KERNEL,
    # TRISC_0-FW, TRISC_0-KERNEL) → 6, plus 2 for op 32771 = 8 total
    assert len(zones) == 8

    brisc_fw_op0 = [
        z for z in zones
        if z.run_host_id == 32770 and z.risc == "BRISC" and z.zone_name == "BRISC-FW"
    ]
    assert len(brisc_fw_op0) == 1
    z = brisc_fw_op0[0]
    assert z.start_cycles == 1000
    assert z.end_cycles == 5100
    assert z.duration_cycles == 4100


def test_pair_zones_raises_on_unmatched_start():
    bad = [
        DeviceEvent(2, 1, 2, "BRISC", 1000, 1, "BRISC-FW", "ZONE_START"),
        # no ZONE_END
    ]
    with pytest.raises(ValueError, match="unmatched ZONE_START"):
        pair_zones(bad)
