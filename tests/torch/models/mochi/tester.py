# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tester for Mochi VAE model."""

from typing import Any, Dict, Sequence

from infra import ComparisonConfig, Model, RunMode, TorchModelTester
from infra.evaluators.evaluator import ComparisonResult

from third_party.tt_forge_models.mochi.pytorch import ModelLoader

from .model_utils import MochiVAEWrapper, calculate_expected_output_shape


class MochiVAETester(TorchModelTester):
    """
    Tester for Mochi VAE decoder.

    NOTE: This tester skips CPU comparison because bfloat16 is not well-supported
    on CPU. It only validates that the model compiles and
    runs successfully on TT hardware.

    This is a test that ensures:
    1. Model loads correctly from HuggingFace
    2. Compilation succeeds with TT backend
    3. Execution completes without errors
    4. Output has expected shape
    """

    def __init__(
        self,
        variant_name,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
        **kwargs,
    ) -> None:
        self._model_loader = ModelLoader(variant_name, subfolder="vae")
        super().__init__(comparison_config, run_mode, **kwargs)

    def _get_model(self) -> Model:
        vae_model = self._model_loader.load_model()
        enable_tiling = self._model_loader._variant_config.enable_tiling
        if not enable_tiling:
            vae_model = vae_model.decoder
        return MochiVAEWrapper(vae_model, enable_tiling)

    def _get_input_activations(self) -> Dict | Sequence[Any]:
        return self._model_loader.load_inputs(vae_type="decoder")

    def test(self):
        """
        Override test() to skip CPU comparison.

        Only validates compilation and execution on TT device.
        This is necessary because bfloat16 Conv3D operations hang on
        CPU without AVX-512_BF16 hardware support and ttnn.conv3d has
        constraint that it works on bfloat16 only.

        Returns:
            Output tensor from TT device execution
        """
        self._compile_for_tt_device(self._workload)

        output = self._device_runner.run_on_tt_device(self._workload)
        expected_shape = calculate_expected_output_shape(self._input_activations.shape)

        assert (
            output.shape == expected_shape
        ), f"Output shape {output.shape} does not match expected shape {expected_shape}"
