# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Unit test for the teacher-forcing override helper (no device needed)."""

import pytest
from vllm_tt.vllm_utils import teacher_forced_token


@pytest.mark.push
@pytest.mark.cpu
def test_returns_none_when_extra_args_absent():
    assert teacher_forced_token(None, 0) is None
    assert teacher_forced_token({}, 0) is None
    assert teacher_forced_token({"other": 1}, 0) is None
    assert teacher_forced_token({"teacher_forcing_tokens": []}, 0) is None


@pytest.mark.push
@pytest.mark.cpu
def test_returns_ground_truth_token_at_step():
    tf = {"teacher_forcing_tokens": [10, 20, 30]}
    assert teacher_forced_token(tf, 0) == 10
    assert teacher_forced_token(tf, 1) == 20
    assert teacher_forced_token(tf, 2) == 30


@pytest.mark.push
@pytest.mark.cpu
def test_returns_none_when_step_out_of_range():
    tf = {"teacher_forcing_tokens": [10, 20]}
    assert teacher_forced_token(tf, 2) is None
    assert teacher_forced_token(tf, -1) is None
