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


from tools.profiler.tracy_to_perfetto import OpInfo, load_ops


def test_load_ops_indexes_by_global_call_count():
    ops = load_ops(FIXTURES / "mini_ops.csv")
    assert set(ops.keys()) == {32770, 32771}

    op = ops[32770]
    assert op.op_code == "PermuteDeviceOperation"
    assert op.compute_kernels == [
        "ttnn/cpp/ttnn/operations/data_movement/permute/device/kernels/compute/transpose_xw_rm_single_tile_size.cpp",
    ]
    assert sorted(op.data_movement_kernels) == [
        "ttnn/cpp/ttnn/operations/data_movement/permute/device/kernels/dataflow/reader_permute_interleaved_rm_blocked_generic.cpp",
        "ttnn/cpp/ttnn/operations/data_movement/permute/device/kernels/dataflow/writer_permute_interleaved_rm_blocked_generic.cpp",
    ]


def test_load_ops_handles_missing_source_lists():
    # When an op has no COMPUTE KERNEL SOURCE entry the cell is empty.
    from tools.profiler.tracy_to_perfetto import _split_kernel_list
    assert _split_kernel_list("") == []
    assert _split_kernel_list("[]") == []
    assert _split_kernel_list("['a.cpp']") == ["a.cpp"]
    assert _split_kernel_list("['a.cpp'; 'b.cpp']") == ["a.cpp", "b.cpp"]


from tools.profiler.tracy_to_perfetto import classify_kernel


def _make_op(compute=None, dm=None):
    return OpInfo(
        op_code="X",
        global_call_count=0,
        compute_kernels=compute or [],
        data_movement_kernels=dm or [],
    )


def test_classify_compute_kernel_for_trisc():
    op = _make_op(compute=["ttnn/.../compute/foo_kernel.cpp"])
    name, role = classify_kernel("TRISC_0", op, fallback="TRISC-KERNEL")
    assert role == "compute"
    assert name == "foo_kernel"


def test_classify_brisc_picks_reader_prefix():
    op = _make_op(dm=[
        "ttnn/.../dataflow/writer_perm.cpp",
        "ttnn/.../dataflow/reader_perm.cpp",
    ])
    name, role = classify_kernel("BRISC", op, fallback="BRISC-KERNEL")
    assert role == "reader"
    assert name == "reader_perm"


def test_classify_ncrisc_picks_writer_prefix():
    op = _make_op(dm=[
        "ttnn/.../dataflow/writer_perm.cpp",
        "ttnn/.../dataflow/reader_perm.cpp",
    ])
    name, role = classify_kernel("NCRISC", op, fallback="NCRISC-KERNEL")
    assert role == "writer"
    assert name == "writer_perm"


def test_classify_falls_back_to_list_order_when_no_prefix():
    op = _make_op(dm=["ttnn/.../dm0.cpp", "ttnn/.../dm1.cpp"])
    assert classify_kernel("BRISC", op, fallback="X") == ("dm0", "dm0")
    assert classify_kernel("NCRISC", op, fallback="X") == ("dm1", "dm1")


def test_classify_uses_fallback_when_op_missing():
    name, role = classify_kernel("BRISC", None, fallback="BRISC-KERNEL")
    assert role == "unknown"
    assert name == "BRISC-KERNEL"
