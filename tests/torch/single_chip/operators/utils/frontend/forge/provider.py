# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Forge frontend provider

from loguru import logger

import os
import torch

from forge import compile
from forge import forge_property_utils
from forge import tensor
from forge import ForgeModule
from forge.verify import verify

from ...datatypes import DataFormat
from ...datatypes import MathFidelity
from ...datatypes import InputSourceFlag
from ...datatypes import VerifyConfig
from ...datatypes import ValueChecker
from ...datatypes import AutomaticValueChecker
from ...datatypes import AllCloseValueChecker

from forge._C import DataFormat as ForgeDataFormat
from forge._C import MathFidelity as ForgeMathFidelity
from forge.config import CompilerConfig as ForgeCompilerConfig
from forge.verify.config import VerifyConfig as ForgeVerifyConfig
from forge.verify.value_checkers import ValueChecker as ForgeValueChecker
from forge.verify.value_checkers import AutomaticValueChecker as ForgeAutomaticValueChecker
from forge.verify.value_checkers import AllCloseValueChecker as ForgeAllCloseValueChecker


class ForgeConverterUtils:
    @classmethod
    def to_forge_data_format(cls, dev_data_format: DataFormat) -> ForgeDataFormat:
        if dev_data_format is None:
            return dev_data_format
        return getattr(ForgeDataFormat, dev_data_format.name)

    @classmethod
    def to_forge_math_fidelity(cls, math_fidelity: MathFidelity) -> ForgeMathFidelity:
        if math_fidelity is None:
            return math_fidelity
        return getattr(ForgeMathFidelity, math_fidelity.name)

    @classmethod
    def to_forge_compiler_config(cls, verify_config: VerifyConfig) -> ForgeCompilerConfig:
        compiler_cfg = ForgeCompilerConfig()
        if verify_config.model_dtype is not None and verify_config.model_dtype == torch.bfloat16:
            # Override default data format for bfloat16
            compiler_cfg.default_df_override = ForgeDataFormat.Float16_b
        if verify_config.input_source_flag:
            CompilerUtils.set_input_source(verify_config.input_source_flag.value, compiler_cfg)
        if verify_config.math_fidelity:
            compiler_cfg.default_math_fidelity = cls.to_forge_math_fidelity(verify_config.math_fidelity)
        return compiler_cfg

    @classmethod
    def to_forge_value_checker(cls, value_checker: ValueChecker) -> ForgeValueChecker:
        if value_checker is None:
            return None
        if isinstance(value_checker, AutomaticValueChecker):
            return ForgeAutomaticValueChecker(
                atol=value_checker.atol,
                rtol=value_checker.rtol,
            )
        elif isinstance(value_checker, AllCloseValueChecker):
            return ForgeAllCloseValueChecker(
                atol=value_checker.atol,
                rtol=value_checker.rtol,
            )
        else:
            raise ValueError(f"Unsupported value checker type: {type(value_checker)}")

    @classmethod
    def to_forge_verify_config(cls, sweeps_verify_config: VerifyConfig) -> ForgeVerifyConfig:
        verify_config = ForgeVerifyConfig()
        if sweeps_verify_config.value_checker:
            verify_config.value_checker = cls.to_forge_value_checker(sweeps_verify_config.value_checker)
        return verify_config


class CompilerUtils:
    """Utility functions for Forge compiler configuration"""

    @staticmethod
    def set_input_source(input_source_flag: InputSourceFlag, compiler_cfg: ForgeCompilerConfig):
        """Set compiler configuration for input source"""
        # Not existing in the compiler, after global config removal
        # compiler_cfg.input_queues_on_host = input_source_flag.input_queues_on_host
        # if input_source_flag.set_default_dram_parameters:
        #     compiler_cfg.default_dram_parameters = input_source_flag.default_dram_parameters

        # NOP since we don't use this flag in the compiler, currently.

    @staticmethod
    def set_math_fidelity(math_fidelity: MathFidelity, compiler_cfg: ForgeCompilerConfig):
        """Set compiler configuration for math fidelity"""
        # Currently not respected/supported in the compiler
        compiler_cfg.default_math_fidelity = math_fidelity


class DeviceUtils:
    """Utility functions for Forge verification"""

    @staticmethod
    def warm_reset():
        reset_command = "/home/software/syseng/wh/tt-smi -lr all wait -er"
        os.system(reset_command)


class ForgeFrontend:
    @classmethod
    def to_pt_tensors(cls, tensors):
        return tensor.to_pt_tensors(tensors)

    @classmethod
    def to_forge_tensors(cls, tensors):
        return tensor.to_forge_tensors(tensors)

    @classmethod
    def is_forge_module(cls, model) -> bool:
        return isinstance(model, ForgeModule)

    @classmethod
    def setup(cls, verify_config: VerifyConfig):
        if verify_config.warm_reset:
            DeviceUtils.warm_reset()

    @classmethod
    def verify(cls, verify_config: VerifyConfig):
        cls.setup(verify_config)
        model = verify_config.model
        inputs = verify_config.inputs
        compiler_cfg = ForgeConverterUtils.to_forge_compiler_config(verify_config)
        verify_config = ForgeConverterUtils.to_forge_verify_config(verify_config)
        compiled_model = compile(model, sample_inputs=inputs, compiler_cfg=compiler_cfg)
        verify(inputs, model, compiled_model, verify_config)

    @classmethod
    def verify_torch(cls, verify_config: VerifyConfig):
        """
        Verify the pytorch model with the given inputs
        """
        cls.setup(verify_config)
        framework_model: torch.nn.Module = verify_config.model
        inputs = verify_config.inputs
        verify_config = ForgeConverterUtils.to_forge_verify_config(verify_config)
        if verify_config is None:
            verify_config = ForgeVerifyConfig()
        if not verify_config.enabled:
            logger.warning("Verification is disabled")
            return None

        # 0th step: input checks

        # Check if inputs are of the correct type
        if not inputs:
            raise ValueError("Input tensors must be provided")
        for input_tensor in inputs:
            if not isinstance(input_tensor, verify_config.supported_tensor_types):
                raise TypeError(
                    f"Input tensor must be of type {verify_config.supported_tensor_types}, but got {type(input_tensor)}"
                )

        if not isinstance(framework_model, verify_config.framework_model_types):
            raise TypeError(
                f"Framework model must be of type {verify_config.framework_model_types}, but got {type(framework_model)}"
            )

        # 1st step: run forward pass for the networks
        fw_out = framework_model(*inputs)

        # 2nd step: apply preprocessing (push tensors to cpu, perform any reshape if necessary,
        #  cast from tensorflow tensors to pytorch tensors if needed)
        if not isinstance(fw_out, torch.Tensor):
            fw_out = cls.to_pt_tensors(fw_out)

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
