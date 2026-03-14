# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum

from tests.runner.conftest import (
    _apply_model_group_markers,
    _normalize_group_marker_name,
)


class FakeModelGroup(Enum):
    RED = "red"
    PRIORITY = "priority"
    GENERALITY = "generality"
    VULCAN = "vulcan"

    def __str__(self):
        return self.value


class DummyItem:
    def __init__(self):
        self.marker_names = []

    def add_marker(self, mark):
        self.marker_names.append(mark.name)


def test_normalize_group_marker_name_uses_enum_value():
    assert _normalize_group_marker_name(FakeModelGroup.VULCAN) == "vulcan"


def test_red_group_gets_direct_and_nightly_markers():
    item = DummyItem()

    added = _apply_model_group_markers(item, FakeModelGroup.RED, [])

    assert added == ["red", "nightly"]
    assert item.marker_names == ["red", "nightly"]


def test_generality_group_gets_direct_and_weekly_markers():
    item = DummyItem()

    added = _apply_model_group_markers(item, FakeModelGroup.GENERALITY, [])

    assert added == ["generality", "weekly"]
    assert item.marker_names == ["generality", "weekly"]


def test_vulcan_group_gets_direct_marker_without_default_schedule_marker():
    item = DummyItem()

    added = _apply_model_group_markers(item, FakeModelGroup.VULCAN, [])

    assert added == ["vulcan"]
    assert item.marker_names == ["vulcan"]


def test_config_schedule_marker_overrides_default_schedule_policy():
    item = DummyItem()

    added = _apply_model_group_markers(item, FakeModelGroup.VULCAN, ["nightly"])

    assert added == ["vulcan"]
    assert item.marker_names == ["vulcan"]
