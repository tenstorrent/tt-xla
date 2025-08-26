# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# XLA frontend provider


from typing import Any, Dict, Sequence
from loguru import logger

from common.setup.frontend.xla import forge_property_utils

import torch
from infra import ComparisonConfig, Model, RunMode, TorchModelTester

from ...datatypes import VerifyConfig


class SweepsTester(TorchModelTester):
    """Tester for Sweeps model."""

    def __init__(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
    ) -> None:
        self.model = model
        self.inputs = inputs
        super().__init__(comparison_config, run_mode)

    # @override
    def _get_model(self) -> Model:
        return self.model

    def _get_forward_method_args(self) -> Sequence[Any]:
        return self.inputs

    # @override
    def _get_input_activations(self) -> Dict | Sequence[Any]:
        inputs = {}
        return inputs


class XlaFrontend:
    @classmethod
    def to_pt_tensors(cls, tensors):
        pass

    @classmethod
    def to_forge_tensors(cls, tensors):
        pass

    @classmethod
    def is_forge_module(cls, model) -> bool:
        return False

    @classmethod
    def verify(cls, verify_config):
        model = verify_config.model
        inputs = verify_config.inputs
        sweeps_tester = SweepsTester(model, inputs)
        sweeps_tester.test()

    @classmethod
    def verify_torch(cls, verify_config: VerifyConfig):
        """
        Verify the pytorch model with the given inputs
        """
        framework_model: torch.nn.Module = verify_config.model
        inputs = verify_config.inputs
        if not verify_config.enabled:
            logger.warning("Verification is disabled")
            return None

        # 1st step: run forward pass for the networks
        fw_out = framework_model(*inputs)

        fw_out = [fw_out] if isinstance(fw_out, torch.Tensor) else fw_out
        return fw_out

    @classmethod
    def get_pcc(cls):
        return forge_property_utils.forge_property_handler_var.get().get("tags.pcc")

    @classmethod
    def get_atol(cls):
        return forge_property_utils.forge_property_handler_var.get().get("tags.atol")

    @classmethod
    def record_test_tags(
        cls,
        operator: str,
        input_source: str = None,
        input_shape: str = None,
        dev_data_format: str = None,
        math_fidelity: str = None,
        kwargs: str = None,
    ):
        forge_property_utils.record_sweeps_test_tags(
            operator=operator,
            input_source=input_source,
            input_shape=input_shape,
            dev_data_format=dev_data_format,
            math_fidelity=math_fidelity,
            kwargs=kwargs,
        )

    @classmethod
    def record_expected_failing_reason(
        cls,
        expected_failing_reason: str = "",
        expected_failing_reason_desc: str = "",
        expected_component: str = "",
    ):
        forge_property_utils.record_sweeps_expected_failing_reason(
            expected_failing_reason=expected_failing_reason,
            expected_failing_reason_desc=expected_failing_reason_desc,
            expected_component=expected_component,
        )

    @classmethod
    def record_detected_failing_reason(
        cls,
        detected_failing_reason: str = "",
        detected_failing_reason_desc: str = "",
        detected_component: str = "",
    ):
        forge_property_utils.record_sweeps_detected_failing_reason(
            detected_failing_reason=detected_failing_reason,
            detected_failing_reason_desc=detected_failing_reason_desc,
            detected_component=detected_component,
        )
