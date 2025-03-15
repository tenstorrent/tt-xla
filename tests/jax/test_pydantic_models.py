# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from datetime import datetime, timedelta
from pprint import pprint

from infra.ttmlir import Test

t = Test(
    test_start_ts=datetime.now(),
    test_end_ts=datetime.now() + timedelta(minutes=1),
    error_message="some error msg",
    success=True,
    skipped=False,
    full_test_name="full test name",
    tags={"tags": "some tags"},
    test_case_name="test case name",
    filepath="file path",
    category="category",
    group="red/priority/generality",
    owner="tt-xla",
)

pprint(t.model_dump())
# pprint(t.model_dump_json())
