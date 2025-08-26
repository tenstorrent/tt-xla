# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from tests.runner.test_utils import ModelStatus
from tests.utils import BringupStatus


test_config = {
    # [x] Passing Test
    "mnist/pytorch-full-inference": {
        "status": ModelStatus.EXPECTED_PASSING,
        "marker": "tests_to_run",
    },
    # [x] PCC Failing test
    "openpose/v2/pytorch-full-inference": {
        "assert_pcc": True,
        "status": ModelStatus.EXPECTED_PASSING,
    },
    # [x]XFAIL pcc failing test
    "mobilenetv2/pytorch-mobilenet_v2-full-inference": {
        "required_pcc": 1.00,
        "status": ModelStatus.KNOWN_FAILURE_XFAIL,
        "marker": "tests_to_run",
    },
    # XFAIL passing test no reason given
    "monodepth2/pytorch-mono+stereo_640x192-full-inference": {
        "assert_pcc": False,  # PCC observed: 0.001758846541901752 (below 0.99 threshold)
        "status": ModelStatus.KNOWN_FAILURE_XFAIL,
        "marker": "tests_to_run",
    },
    # XFAIL test as unknown bringup status
    "vgg/pytorch-vgg11-full-inference": {
        "status": ModelStatus.KNOWN_FAILURE_XFAIL,
        "reason": "Testing unknown bringup status",
        "bringup_status": BringupStatus.UNKNOWN,
        "marker": "tests_to_run",
    },
    # Skip test marked as unknown bringup status
    "deepseek/pytorch-full-inference": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "reason": "Fix KILLED",
        "bringup_status": BringupStatus.UNKNOWN,
        "marker": "tests_to_run",
    },
    # Skip test
    "segformer/pytorch-mit_b0-full-inference": {
        "status": ModelStatus.NOT_SUPPORTED_SKIP,
        "reason": "Doesn't fit on device",
        "bringup_status": BringupStatus.FAILED_FE_COMPILATION,
        "marker": "tests_to_run",
    },
    # Another passing test
    "wide_resnet/pytorch-wide_resnet50_2-full-inference": {
        "required_pcc": 0.98,
        "status": ModelStatus.EXPECTED_PASSING,
        "marker": "tests_to_run",
    }
    # tests/runner/test_models.py::test_all_models[mnist/pytorch-full-inference]
    # tests/runner/test_models.py::test_all_models[openpose/v2/pytorch-full-inference]
    # tests/runner/test_models.py::test_all_models[mobilenetv2/pytorch-mobilenet_v2-full-inference]
    # tests/runner/test_models.py::test_all_models[monodepth2/pytorch-mono+stereo_640x192-full-inference]
    # tests/runner/test_models.py::test_all_models[vgg/pytorch-vgg11-full-inference]
    # tests/runner/test_models.py::test_all_models[deepseek/pytorch-full-inference]
    # tests/runner/test_models.py::test_all_models[segformer/pytorch-mit_b0-full-inference]
    # tests/runner/test_models.py::test_all_models[wide_resnet/pytorch-wide_resnet50_2-full-inference]
    # # Unknown tests
    # tests/runner/test_models.py::test_all_models[bi_lstm_crf/pytorch-full-inference] # Fails
    # tests/runner/test_models.py::test_all_models[mobilenetv2/pytorch-mobilenet_v2_torchvision-full-inference] # Passes
}
