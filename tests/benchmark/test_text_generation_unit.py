# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""CPU-only unit tests for the shared text-generation / reporting helpers.

These pin the measurement-dict and output-JSON shapes that the LLM and vLLM
drivers now share, so the extraction stays byte-identical to the inline code it
replaced (the vLLM path isn't runnable on CPU here).
"""

import json

from text_generation import throughput_measurement, ttft_measurement
from utils import write_benchmark_json


def test_ttft_measurement_shape():
    assert ttft_measurement(12.5) == {
        "measurement_name": "ttft",
        "value": 12.5,
        "target": -1,
    }


def test_throughput_measurement_shape():
    assert throughput_measurement(3.0) == {
        "measurement_name": "samples_per_sec",
        "value": 3.0,
        "target": -1,
    }


def test_write_benchmark_json_stamps_and_dumps(tmp_path):
    results = {"model": "m", "config": {"existing": 1}}
    out = tmp_path / "out.json"

    write_benchmark_json(results, str(out), model_rawname="raw/name")

    # Stamping mutates the dict in place, then dumps it verbatim.
    assert results["project"] == "tt-forge/tt-xla"
    assert results["model_rawname"] == "raw/name"
    on_disk = json.loads(out.read_text())
    assert on_disk == results
    assert on_disk["config"]["existing"] == 1


def test_write_benchmark_json_custom_project(tmp_path):
    results = {}
    out = tmp_path / "out.json"
    write_benchmark_json(results, str(out), model_rawname="m", project="other/proj")
    assert json.loads(out.read_text())["project"] == "other/proj"
