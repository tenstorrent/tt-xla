# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Dynamic Torch model tester for CPU-vs-CUDA validation."""

import torch

from .dynamic_torch_model_tester import DynamicTorchModelTester


class DynamicTorchCudaModelTester(DynamicTorchModelTester):
    """Torch tester that validates CUDA outputs against CPU golden outputs."""

    def _test_inference(self, request=None):
        assert torch.cuda.is_available(), "CUDA device not available"

        cpu_res = self._run_on_cpu(self._workload)
        cuda_res = self._device_runner.run_on_cuda_device(self._workload)

        return (self._compare(cuda_res, cpu_res),)
