from io import StringIO
from pathlib import Path

from tools.profiler import tracy_to_perfetto
from tools.profiler.tracy_to_perfetto import iter_device_events, parse_device_header

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
