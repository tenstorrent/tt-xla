from io import StringIO

from tools.profiler import tracy_to_perfetto
from tools.profiler.tracy_to_perfetto import parse_device_header


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
