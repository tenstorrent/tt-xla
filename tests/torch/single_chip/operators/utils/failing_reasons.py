# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Failing reasons definition


import re

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Callable, Optional, Generator
from loguru import logger

from .pytest import PyTestUtils


@dataclass
class ExceptionData:
    # operator: str
    class_name: str
    message: str
    error_log: str


MessageCheckerType = Callable[[str], bool]


@dataclass
class ExceptionCheck:
    # operators: List[str]
    class_name: Optional[str] = None
    component: Optional["ComponentChecker"] = None
    message: List[MessageCheckerType] = field(default_factory=list)
    error_log: List[MessageCheckerType] = field(default_factory=list)

    def __contains__(self, ex: ExceptionData) -> bool:
        return self.check(ex)

    def check(self, ex: ExceptionData) -> bool:
        if self.class_name:
            if ex.class_name != self.class_name:
                return False
        if self.component is not None:
            if not ex in self.component:
                return False
        for message_check in self.message:
            if not message_check(ex.message):
                return False
        for message_check in self.error_log:
            if not message_check(ex.error_log):
                return False
        return True


@dataclass
class FailingReason:
    description: str
    checks: List[ExceptionCheck] = field(default_factory=list)

    def __post_init__(self):
        self.checks = [
            check for check in self.checks if check.component is None or check.component != ComponentChecker.NONE.value
        ]
        if len(self.checks) == 0:
            logger.trace(f"FailingReason '{self.description}' has no checks defined, it will not be used.")
        elif len(self.checks) > 1:
            logger.trace(f"FailingReason '{self.description}' has multiple ({len(self.checks)}) checks defined.")

    @property
    def component_checker(self) -> Optional["ComponentChecker"]:
        for check in self.checks:
            component = check.component
            if component is None or component == ComponentChecker.NONE.value:
                continue
            return component
        return None

    @property
    def component_checker_description(self) -> Optional[str]:
        component_checker = self.component_checker
        return component_checker.description if component_checker else None

    def __contains__(self, ex: ExceptionData) -> bool:
        return self.check(ex)

    def check(self, ex: ExceptionData) -> bool:
        for check in self.checks:
            if ex in check:
                return True
        return False

    def __repr__(self) -> str:
        return f"FailingReason(description={self.description!r})"


class MessageChecker:
    @staticmethod
    def contains(message: str) -> bool:
        return lambda ex_message: message in ex_message

    @staticmethod
    def starts_with(message: str) -> bool:
        return lambda ex_message: ex_message.startswith(message)

    @staticmethod
    def equals(message: str) -> bool:
        return lambda ex_message: ex_message == message

    @staticmethod
    def regex(pattern: str) -> bool:
        return lambda ex_message: re.search(pattern, ex_message) is not None

    @staticmethod
    def any(*checkers: MessageCheckerType) -> bool:
        """Check if any of the checkers match the message (or)."""
        return lambda ex_message: any(checker(ex_message) for checker in checkers)

    @staticmethod
    def neg(checker: MessageCheckerType) -> bool:
        """Negate the checker function (not)."""
        return lambda ex_message: not checker(ex_message)

    @staticmethod
    def last_line(checker: MessageCheckerType) -> str:
        return lambda ex_message: checker(ex_message.splitlines()[-1] if ex_message else ex_message)


M = MessageChecker


class ComponentChecker(Enum):
    def __repr__(self) -> str:
        return self.name

    NONE = FailingReason(
        # A helper component to identify checks that are not used anymore
        description="None",
        checks=[
            ExceptionCheck(
                error_log=[
                    M.contains("A non existing line in the error log"),
                ],
            ),
        ],
    )

    # Obsoleted binary names not used anymore:
    #   - lib/_ttnn.so
    #   - lib/libTTMLIRCompiler.so

    METAL = FailingReason(
        description="Metal",
        checks=[
            ExceptionCheck(
                error_log=[
                    M.contains("lib/libtt_metal.so"),
                    M.contains("lib/_ttnncpp.so"),
                    M.contains("lib/libTTMLIRRuntime.so"),
                    M.contains("forge/_C.so"),
                    M.last_line(M.contains("forge/compiled_graph_state.py:")),
                ],
            ),
        ],
    )

    TTNN = FailingReason(
        description="TTNN",
        checks=[
            ExceptionCheck(
                error_log=[
                    M.neg(M.contains("lib/libtt_metal.so")),
                    M.contains("lib/_ttnncpp.so"),
                    M.contains("lib/libTTMLIRRuntime.so"),
                    M.contains("forge/_C.so"),
                    M.last_line(M.contains("forge/compiled_graph_state.py:")),
                ],
            ),
        ],
    )

    MLIR = FailingReason(
        description="MLIR",
        checks=[
            ExceptionCheck(
                error_log=[
                    M.neg(M.contains("lib/libtt_metal.so")),
                    M.neg(M.contains("lib/_ttnncpp.so")),
                    M.contains("lib/libTTMLIRRuntime.so"),
                    M.contains("forge/_C.so"),
                    M.last_line(M.contains("forge/compiled_graph_state.py:")),
                ],
            ),
        ],
    )

    FORGE = FailingReason(
        description="Forge",
        checks=[
            ExceptionCheck(
                error_log=[
                    M.neg(M.contains("lib/libtt_metal.so")),
                    M.neg(M.contains("lib/_ttnncpp.so")),
                    M.neg(M.contains("lib/libTTMLIRRuntime.so")),
                    M.neg(M.contains("forge/_C.so")),  # Python code
                    M.any(
                        M.last_line(M.contains("forge/verify/compare.py:")),
                        M.last_line(M.contains("forge/verify/value_checkers.py:")),
                        M.last_line(M.contains("forge/verify/verify.py:")),
                        M.last_line(M.contains("forge/op/common.py:")),
                        M.last_line(M.contains("forge/op/eval/interface.py:")),
                        M.last_line(M.contains("forge/compile.py:")),
                        M.last_line(M.contains("forge/compiled_graph_state.py:")),
                        M.last_line(M.contains("forge/op/eval/forge/convolution.py:")),
                        M.last_line(M.contains("forge/op/eval/forge/tm.py:")),
                        M.last_line(M.contains("forge/op/eval/forge/embedding.py:")),
                        M.last_line(M.contains("test/operators/utils/compat.py:")),  # pcc error levels
                        M.last_line(M.contains("test/operators/pytorch/")),
                        # Fail with pytorch also. TODO: check if tests are correct
                        M.last_line(M.contains("torch/nn/modules/conv.py:")),
                        M.last_line(M.contains("torch/nn/functional.py")),
                        M.last_line(M.contains("forge/op/eval/forge/pooling.py")),
                        M.last_line(M.contains("forge/tensor.py:")),
                    ),
                ],
            ),
            ExceptionCheck(
                error_log=[
                    M.neg(M.contains("lib/libtt_metal.so")),
                    M.neg(M.contains("lib/_ttnncpp.so")),
                    M.neg(M.contains("lib/libTTMLIRRuntime.so")),
                    M.contains("forge/_C.so"),  # C code
                    M.any(
                        M.last_line(M.contains("forge/verify/compare.py:")),
                        M.last_line(M.contains("forge/verify/value_checkers.py:")),
                        M.last_line(M.contains("forge/op/eval/interface.py:")),
                        M.last_line(M.contains("forge/compile.py:")),
                        M.last_line(M.contains("forge/op/common.py")),
                    ),
                ],
            ),
        ],
    )

    TVM = FailingReason(
        description="Tvm",
        checks=[
            ExceptionCheck(
                error_log=[
                    M.neg(M.contains("lib/libtt_metal.so")),
                    M.neg(M.contains("lib/_ttnncpp.so")),
                    M.neg(M.contains("lib/libTTMLIRRuntime.so")),
                    M.neg(M.contains("forge/_C.so")),
                    M.any(
                        M.last_line(M.contains("/tvm/relay/frontend/pytorch.py:")),
                        M.last_line(M.contains("/tvm/relay/expr_functor.py:")),
                        M.last_line(M.contains("/tvm/_ffi/base.py:")),
                        M.last_line(M.contains("forge/tvm_calls/relay/op/forge_passes.py:")),
                        M.last_line(M.contains("forge/tvm_to_python.py:")),
                    ),
                ],
            ),
        ],
    )


class FailingReasonsFinder:
    @classmethod
    def build_ex_data(cls, exception_value: Exception, exception_traceback: str) -> ExceptionData:
        """Convert exception to ExceptionData object

        Args:
            exception_value (Exception): Raised exception
            exception_traceback (str): Exception traceback

        Returns:
            ExceptionData: Exception data object
        """
        ex_class_name = f"{type(exception_value).__module__}.{type(exception_value).__name__}"
        ex_class_name = ex_class_name.replace("builtins.", "")
        ex_message = f"{exception_value}"
        exception_traceback = PyTestUtils.remove_colors(exception_traceback)
        ex_data = ExceptionData(
            class_name=ex_class_name,
            message=ex_message,
            error_log=exception_traceback,
        )
        return ex_data

    @classmethod
    def find_reason_by_ex_data(cls, ex: ExceptionData) -> Optional["FailingReasons"]:
        reasons = list(cls.find_reasons_by_ex_data(ex))
        if not reasons:
            return None
        if len(reasons) > 1:
            logger.warning(f"Multiple reasons found: {reasons} for ex: {ex}")
        return reasons[0]

    @classmethod
    def find_reasons_by_ex_data(cls, ex: ExceptionData) -> Generator["FailingReasons", None, None]:
        for failing_reason in FailingReasons:
            # Checking if exception data matches the failing reason
            if ex in failing_reason.value:
                yield failing_reason


class FailingReasons(Enum):
    def __repr__(self) -> str:
        return self.name

    @classmethod
    def find_by_description(cls, desc: str) -> Optional["FailingReasons"]:
        """Find failing reason by description."""
        failing_reasons = [xfail_reason for xfail_reason in FailingReasons if xfail_reason.value.description == desc]
        if len(failing_reasons) == 0:
            return None
        elif len(failing_reasons) > 1:
            raise ValueError(f"Multiple xfail reasons {failing_reasons} found for description: {desc}")
        return failing_reasons[0]

    UNCLASSIFIED = FailingReason(
        description="Unclassified error",
    )

    NOT_SUPPORTED_IN_TORCH = FailingReason(
        description="Not supported in Torch",
        checks=[
            ExceptionCheck(
                class_name="RuntimeError",
                component=ComponentChecker.FORGE.value,
                message=[
                    # layer_norm RuntimeError: "LayerNormKernelImpl" not implemented for 'Half'
                    M.regex(r"\".*\" not implemented for '.*'"),
                ],
                error_log=[
                    M.any(
                        M.last_line(M.contains("torch/nn/functional.py")),
                    ),
                ],
            ),
        ],
    )

    NOT_IMPLEMENTED_DATA_FORMAT_OLD = FailingReason(
        description="Data format is not implemented old",
        checks=[
            ExceptionCheck(
                class_name="RuntimeError",
                component=ComponentChecker.NONE.value,  # TODO remove
                message=[
                    M.contains("Unsupported data type"),
                ],
            ),
            ExceptionCheck(
                class_name="RuntimeError",
                component=ComponentChecker.NONE.value,  # TODO remove
                message=[
                    M.contains("/forge/csrc/passes/lower_to_mlir.cpp:"),
                ],
            ),
            ExceptionCheck(
                class_name="RuntimeError",
                # component=ComponentChecker.FORGE.value,
                component=ComponentChecker.NONE.value,  # TODO remove
                # bitwise_and	RuntimeError: "bitwise_and_cpu" not implemented for 'Float'
                # bitwise_left_shift	RuntimeError: "lshift_cpu" not implemented for 'Float'
                # bitwise_not	RuntimeError: "bitwise_not_cpu" not implemented for 'Float'
                # bitwise_or	RuntimeError: "bitwise_or_cpu" not implemented for 'Float'
                # bitwise_right_shift	RuntimeError: "rshift_cpu" not implemented for 'Float'
                # bitwise_xor	RuntimeError: "bitwise_xor_cpu" not implemented for 'Float'
                # conv2d	RuntimeError: "slow_conv2d_cpu" not implemented for 'Half'
                # matmul	RuntimeError: "bmm" not implemented for 'Half'
                # softmax	RuntimeError: "softmax_lastdim_kernel_impl" not implemented for 'Char'
                # softmax	RuntimeError: "softmax_lastdim_kernel_impl" not implemented for 'Half'
                # softmax	RuntimeError: "softmax_lastdim_kernel_impl" not implemented for 'Int'
                # softmax	RuntimeError: "softmax_lastdim_kernel_impl" not implemented for 'Long'
                # >       return self.operator(x, y)
                # E       RuntimeError: "bmm" not implemented for 'Half'
                # forge/test/operators/pytorch/matmul/test_matmul.py:48: RuntimeError
                message=[
                    M.regex(r"\".*\" not implemented for '.*'"),
                ],
                error_log=[
                    M.any(
                        M.last_line(M.contains("test/operators/pytorch/")),
                        M.last_line(M.contains("torch/nn/modules/conv.py:")),
                    ),
                ],
            ),
        ],
    )

    UNSUPPORTED_DATA_FORMAT = FailingReason(
        description="Data format is not supported",
        checks=[
            ExceptionCheck(
                # matmul	RuntimeError: TT_FATAL @ tt-metal/ttnn/cpp/ttnn/operations/matmul/device/matmul_op.cpp:1542: is_floating_point(input_tensor_a.get_dtype())
                # >       self.runtime_model_state.run_program(ProgramType.Forward, self.inputs)
                # E       RuntimeError: TT_FATAL @ /proj_sw/user_dev/vbrkic/src_bgd/ttforge/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/matmul_op.cpp:1542: is_floating_point(input_tensor_a.get_dtype())
                # E       info:
                # E       Unsupported data format
                # E       backtrace:
                # E        --- ttnn::operations::matmul::Matmul::validate(std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > const&, std::vector<std::optional<tt::tt_metal::Tensor const>, std::allocator<std::optional<tt::tt_metal::Tensor const> > > const&, std::vector<std::optional<tt::tt_metal::Tensor>, std::allocator<std::optional<tt::tt_metal::Tensor> > > const&) const
                # E        --- tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >::validate_on_program_cache_miss(tt::tt_metal::operation::DeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > > const&, tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >::tensor_args_t const&)
                # E        --- void ttnn::device_operation::detail::create_and_cache_mesh_workload<ttnn::device_operation::MeshDeviceOperationAdapter<tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > > > >(tt::stl::StrongType<unsigned char, ttnn::QueueIdTag>, ttnn::device_operation::MeshDeviceOperationAdapter<tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > > >::operation_attributes_t const&)
                # E        --- tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >::tensor_return_value_t ttnn::device_operation::detail::invoke<tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > > >(tt::stl::StrongType<unsigned char, ttnn::QueueIdTag>, tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >::operation_attributes_t const&, tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >::tensor_args_t const&)
                # E        --- std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > tt::tt_metal::operation::run<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >(tt::tt_metal::operation::DeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >&&, std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > const&, std::vector<std::optional<tt::tt_metal::Tensor const>, std::allocator<std::optional<tt::tt_metal::Tensor const> > > const&, std::vector<std::optional<tt::tt_metal::Tensor>, std::allocator<std::optional<tt::tt_metal::Tensor> > > const&, tt::stl::StrongType<unsigned char, ttnn::QueueIdTag>)
                # E        --- ttnn::operations::matmul::matmul(tt::tt_metal::Tensor const&, tt::tt_metal::Tensor const&, std::optional<tt::tt_metal::Tensor const> const&, ttnn::operations::matmul::Matmul const&, tt::stl::StrongType<unsigned char, ttnn::QueueIdTag>, std::optional<tt::tt_metal::Tensor> const&)
                # E        --- ttnn::operations::matmul::bound_matmul(tt::tt_metal::Tensor const&, tt::tt_metal::Tensor const&, std::optional<tt::tt_metal::Tensor const> const&, ttnn::operations::matmul::Matmul const&, unsigned char const&, std::optional<tt::tt_metal::Tensor>&)
                # E        --- tt::runtime::ttnn::operations::matmul::run(tt::target::ttnn::MatmulOp const*, tt::runtime::ttnn::ProgramContext&)
                # E        --- tt::runtime::ttnn::ProgramExecutor::runOperation(tt::target::ttnn::Operation const*)
                # E        --- tt::runtime::ttnn::ProgramExecutor::execute()
                # E        --- tt::runtime::ttnn::runProgram(std::shared_ptr<tt::tt_metal::distributed::MeshDevice>, tt::runtime::Binary, unsigned int, std::vector<tt::runtime::Tensor, std::allocator<tt::runtime::Tensor> >&)
                # E        --- tt::runtime::ttnn::submit(tt::runtime::Device, tt::runtime::Binary, unsigned int, std::vector<tt::runtime::Tensor, std::allocator<tt::runtime::Tensor> >&)
                # E        --- tt::runtime::submit(tt::runtime::Device, tt::runtime::Binary, unsigned int, std::vector<tt::runtime::Tensor, std::allocator<tt::runtime::Tensor> >&)
                # E        --- tt::run_program(tt::runtime::Binary&, int, std::vector<tt::Tensor, std::allocator<tt::Tensor> >&)
                # E        --- tt::ModelState::run_program(tt::ProgramType, std::vector<tt::Tensor, std::allocator<tt::Tensor> >)
                # forge/forge/compiled_graph_state.py:310: RuntimeError
                class_name="RuntimeError",
                component=ComponentChecker.TTNN.value,
                message=[
                    M.starts_with("TT_FATAL"),
                    M.contains("Unsupported data format"),
                ],
                error_log=[
                    M.last_line(M.contains("forge/compiled_graph_state.py:")),
                ],
            ),
            ExceptionCheck(
                class_name="RuntimeError",
                component=ComponentChecker.NONE.value,  # TODO remove
                message=[
                    M.contains("Input tensors must have the same data type, but got {} and {}"),
                ],
            ),
        ],
    )

    DATA_MISMATCH_OLD = FailingReason(
        description="Verification failed due to data mismatch old",
        checks=[
            # sum	AssertionError: PCC check failed
            # transpose	AssertionError: PCC check failed
            # E           AssertionError: PCC check failed
            # forge/test/operators/utils/compat.py:319: AssertionError
            ExceptionCheck(
                class_name="AssertionError",
                # component=ComponentChecker.FORGE.value,
                component=ComponentChecker.NONE.value,  # TODO remove
                message=[
                    M.equals("PCC check failed"),
                ],
                error_log=[
                    M.last_line(M.contains("test/operators/utils/compat.py:")),
                ],
            ),
            ExceptionCheck(
                class_name="AssertionError",
                component=ComponentChecker.NONE.value,  # TODO remove
                message=[
                    M.starts_with("Data mismatch"),
                ],
            ),
        ],
    )

    DATA_MISMATCH_ALL_CLOSE = FailingReason(
        description="Data mismatch (all close)",
        checks=[
            # add	ValueError: Data mismatch -> AllCloseValueChecker (all_close):
            # add	ValueError: Data mismatch -> AutomaticValueChecker (compare_with_golden): framework_model=tensor([],
            # clamp	ValueError: Data mismatch -> AllCloseValueChecker (all_close):
            # E           ValueError: Data mismatch -> AllCloseValueChecker (all_close):
            # E           - Tensor mismatch. Required rtol=0.01, atol=0.01
            # E           - Observed maximum relative diff: 19.0, maximum absolute diff: 572239.6875
            # E           - Framework output: (torch.Size([1, 4]))
            # E           tensor([[-196356.5625, -371447.5312,  259605.8906, -542121.8125]])
            # E           - Compiled model output: (torch.Size([1, 4]))
            # E           tensor([[ 10908.6982,  20635.9746, -14422.5498,  30117.8789]])
            # forge/forge/verify/value_checkers.py:56: ValueError
            # E           ValueError: Data mismatch -> AutomaticValueChecker (compare_with_golden): framework_model=tensor([[[7.1363e+04,        inf, 6.9049e+00, 8.3854e+00, 2.7964e+01,
            # forge/forge/verify/value_checkers.py:39: ValueError
            ExceptionCheck(
                class_name="ValueError",
                component=ComponentChecker.FORGE.value,
                message=[
                    M.any(
                        M.starts_with("Data mismatch -> AllCloseValueChecker (all_close)"),
                    ),
                ],
                error_log=[
                    M.any(
                        M.last_line(M.contains("forge/verify/value_checkers.py:")),
                    ),
                ],
            ),
        ],
    )

    # Should not happen
    DATA_MISMATCH_AUTOMATIC = FailingReason(
        description="Data mismatch (automatic)",
        checks=[
            # forge/forge/verify/value_checkers.py:39: ValueError
            ExceptionCheck(
                class_name="ValueError",
                component=ComponentChecker.FORGE.value,
                message=[
                    M.any(
                        M.starts_with("Data mismatch -> AutomaticValueChecker (compare_with_golden)"),
                    ),
                    M.neg(
                        M.any(
                            M.starts_with(
                                "Data mismatch -> AutomaticValueChecker (compare_with_golden): pcc is in invalid low range:"
                            ),
                            M.starts_with(
                                "Data mismatch -> AutomaticValueChecker (compare_with_golden): pcc is in invalid medium range:"
                            ),
                            M.starts_with(
                                "Data mismatch -> AutomaticValueChecker (compare_with_golden): pcc is in invalid high range:"
                            ),
                        ),
                    ),
                ],
                error_log=[
                    M.any(
                        M.last_line(M.contains("forge/verify/value_checkers.py:")),
                        M.last_line(M.contains("test/operators/utils/compat.py:")),
                    ),
                ],
            ),
        ],
    )

    DATA_MISMATCH_PCC_LOW_RANGE = FailingReason(
        description="Data mismatch PCC in low range",
        checks=[
            # >                   raise ValueError(
            #                         f"Data mismatch -> AutomaticValueChecker (compare_with_golden): pcc is in invalid low range: {pcc} <= 0.85"
            # E                       ValueError: Data mismatch -> AutomaticValueChecker (compare_with_golden): pcc is in invalid low range: 0.736095037069272 <= 0.85
            # forge/test/operators/utils/compat.py:326: ValueError
            ExceptionCheck(
                class_name="ValueError",
                component=ComponentChecker.FORGE.value,
                message=[
                    M.any(
                        M.starts_with(
                            "Data mismatch -> AutomaticValueChecker (compare_with_golden): pcc is in invalid low range:"
                        ),
                    ),
                ],
                error_log=[
                    M.any(
                        M.last_line(M.contains("test/operators/utils/compat.py:")),
                    ),
                ],
            ),
        ],
    )

    DATA_MISMATCH_PCC_MEDIUM_RANGE = FailingReason(
        description="Data mismatch PCC in medium range",
        checks=[
            # forge/test/operators/utils/compat.py:326: ValueError
            ExceptionCheck(
                class_name="ValueError",
                component=ComponentChecker.FORGE.value,
                message=[
                    M.any(
                        M.starts_with(
                            "Data mismatch -> AutomaticValueChecker (compare_with_golden): pcc is in invalid medium range:"
                        ),
                    ),
                ],
                error_log=[
                    M.any(
                        M.last_line(M.contains("test/operators/utils/compat.py:")),
                    ),
                ],
            ),
        ],
    )

    DATA_MISMATCH_PCC_HIGH_RANGE = FailingReason(
        description="Data mismatch PCC in high range",
        checks=[
            # forge/test/operators/utils/compat.py:326: ValueError
            ExceptionCheck(
                class_name="ValueError",
                component=ComponentChecker.FORGE.value,
                message=[
                    M.any(
                        M.starts_with(
                            "Data mismatch -> AutomaticValueChecker (compare_with_golden): pcc is in invalid high range:"
                        ),
                    ),
                ],
                error_log=[
                    M.any(
                        M.last_line(M.contains("test/operators/utils/compat.py:")),
                    ),
                ],
            ),
        ],
    )

    VERIFICATION_FAILED_SPECIAL_VALUES = FailingReason(
        description="Verification failed due to special values",
        checks=[
            # div	RuntimeError: TT_ASSERT @ tt-forge-fe/forge/csrc/verif/verif_ops.cpp:361: !has_special_values(a)
            # >       if not verif.all_close(fw_out, co_out, rtol=self.rtol, atol=self.atol):
            # E       RuntimeError: TT_ASSERT @ /proj_sw/user_dev/vbrkic/src_bgd/ttforge/tt-forge-fe/forge/csrc/verif/verif_ops.cpp:362: !has_special_values(b)
            # E       info:
            # E       Tensor b contains NaN/Inf values
            # E       backtrace:
            # E        --- tt::all_close(at::Tensor const&, at::Tensor const&, double, double)
            # forge/forge/verify/value_checkers.py:54: RuntimeError
            ExceptionCheck(
                class_name="RuntimeError",
                component=ComponentChecker.FORGE.value,
                message=[
                    M.any(
                        M.contains("Tensor a contains NaN/Inf values"),
                        M.contains("Tensor b contains NaN/Inf values"),
                    ),
                    M.contains("verif_ops.cpp"),
                    M.any(
                        M.contains("!has_special_values(a)"),
                        M.contains("!has_special_values(b)"),
                    ),
                ],
                error_log=[
                    M.last_line(M.contains("forge/verify/value_checkers.py:")),
                ],
            ),
            # RuntimeError: TT_ASSERT @ /proj_sw/user_dev/vbrkic/src_bgd/ttforge/tt-forge-fe/forge/csrc/verif/verif_ops.cpp:549: has_special_values(cov) == false
            # >           pcc = verif.calculate_tensor_pcc(a, b)
            # E           RuntimeError: TT_ASSERT @ /__w/tt-forge-fe/tt-forge-fe/forge/csrc/verif/verif_ops.cpp:549: has_special_values(cov) == false
            # E           info:
            # E           Covariance matrix contains NaN/Inf values - possibly due to an overflow
            # E           backtrace:
            # E            --- tt::calculate_tensor_pcc(at::Tensor const&, at::Tensor const&)
            # /opt/ttforge-toolchain/venv/lib/python3.10/site-packages/forge/verify/compare.py:194: RuntimeError
            ExceptionCheck(
                class_name="RuntimeError",
                component=ComponentChecker.FORGE.value,
                message=[
                    M.contains("Covariance matrix contains NaN/Inf values - possibly due to an overflow"),
                    M.contains("verif_ops.cpp"),
                    M.contains("has_special_values(cov) == false"),
                ],
                error_log=[
                    M.last_line(M.contains("forge/verify/compare.py:")),
                ],
            ),
        ],
    )

    VERIFICATION_FAILED_PCC_NAN = FailingReason(
        description="Verification failed due to PCC being NaN",
        checks=[
            # clamp	AssertionError: PCC is nan, but tensors are not equal
            # pow	AssertionError: PCC is nan, but tensors are not equal
            # >               assert False, "PCC is nan, but tensors are not equal"
            # E               AssertionError: PCC is nan, but tensors are not equal
            # forge/forge/verify/compare.py:202: AssertionError
            ExceptionCheck(
                class_name="AssertionError",
                component=ComponentChecker.FORGE.value,
                message=[
                    M.equals("PCC is nan, but tensors are not equal"),
                ],
                error_log=[
                    M.last_line(M.contains("forge/verify/compare.py:")),
                ],
            ),
        ],
    )

    ALL_CLOSE_FOR_INT_TYPE_OLD = FailingReason(
        description="All close for integer type old",
        checks=[
            # max	AssertionError: AllCloseValueChecker (all_close): all_close doesn't make sense for integer/bool types
            # >       assert fw_out.dtype not in [
            #             torch.int32,
            #             torch.int64,
            #             torch.bool,
            #         ], f"AllCloseValueChecker (all_close): all_close doesn't make sense for integer/bool types"
            # E       AssertionError: AllCloseValueChecker (all_close): all_close doesn't make sense for integer/bool types
            # forge/forge/verify/value_checkers.py:48: AssertionError
            ExceptionCheck(
                class_name="AssertionError",
                # component=ComponentChecker.FORGE.value,
                component=ComponentChecker.NONE.value,  # TODO remove
                message=[
                    M.equals("AllCloseValueChecker (all_close): all_close doesn't make sense for integer/bool types"),
                ],
                error_log=[
                    M.last_line(M.contains("forge/verify/value_checkers.py:")),
                ],
            ),
        ],
    )

    DTYPE_MISMATCH = FailingReason(
        description="Dtype mismatch",
        checks=[
            # >           raise ValueError(f"Dtype mismatch: framework_model.dtype={fw_dtype}, compiled_model.dtype={co_dtype}")
            # E           ValueError: Dtype mismatch: framework_model.dtype=torch.int32, compiled_model.dtype=torch.float32
            # /opt/ttforge-toolchain/venv/lib/python3.10/site-packages/forge/verify/verify.py:260: ValueError
            # forge/forge/verify/verify.py:281: ValueError
            ExceptionCheck(
                class_name="ValueError",
                component=ComponentChecker.FORGE.value,
                message=[
                    M.starts_with("Dtype mismatch"),
                    M.regex("Dtype mismatch: framework_model.dtype=torch\..*, compiled_model.dtype=torch\..*"),
                ],
                error_log=[
                    M.last_line(M.contains("forge/verify/verify.py:")),
                ],
            ),
        ],
    )

    WEIGHT_TYPE_MISMATCH_OLD = FailingReason(
        description="Weight type mismatch old",
        checks=[
            # conv2d	RuntimeError: Input type (CPUBFloat16Type) and weight type (torch.FloatTensor) should be the same or input should be a MKLDNN tensor and weight is a dense tensor
            # E       RuntimeError: Input type (CPUBFloat16Type) and weight type (torch.FloatTensor) should be the same or input should be a MKLDNN tensor and weight is a dense tensor
            # ../ttforge-toolchain/venv/lib/python3.10/site-packages/torch/nn/modules/conv.py:952: RuntimeError
            ExceptionCheck(
                class_name="RuntimeError",
                # component=ComponentChecker.FORGE.value,
                component=ComponentChecker.NONE.value,  # TODO remove
                message=[
                    M.starts_with(
                        "Input type (CPUBFloat16Type) and weight type (torch.FloatTensor) should be the same or input should be a MKLDNN tensor and weight is a dense tensor"
                    ),
                ],
                error_log=[
                    M.last_line(M.contains("torch/nn/modules/conv.py:")),
                ],
            ),
            # Tensor 1 - data type mismatch: expected BFloat16, got Float32
            # Tensor 1 - data type mismatch: expected Float32, got BFloat16
            # Tensor 2 - data type mismatch: expected UInt32, got Float32
            # >       self.runtime_model_state.run_program(ProgramType.Forward, self.inputs)
            # E       RuntimeError: Tensor 2 - data type mismatch: expected Int32, got Float32
            # forge/forge/compiled_graph_state.py:310: RuntimeError
            ExceptionCheck(
                class_name="RuntimeError",
                # component=ComponentChecker.FORGE.value,
                component=ComponentChecker.NONE.value,  # TODO remove
                message=[
                    # M.contains("data type mismatch"),
                    M.regex("Tensor .* - data type mismatch: expected .*, got .*"),
                ],
                error_log=[
                    M.last_line(M.contains("forge/compiled_graph_state.py:")),
                ],
            ),
            #                  Always |    FATAL | DataType mismatch, expected INT32, got FLOAT32
            # DEBUG_ASSERT @ /proj_sw/user_dev/vbrkic/src_bgd/ttforge/tt-forge-fe/third_party/tt-mlir/runtime/lib/ttnn/debug/debug_apis.cpp:30: expectedDataType == actualDataType
            # backtrace:
            #  --- tt::runtime::ttnn::debug::checkTensorRefMatchesTTNNTensor(tt::target::ttnn::TensorRef const*, tt::tt_metal::Tensor const&)
            #  --- tt::runtime::ttnn::ProgramTensorPool::insertTTNNTensorAndValidate(tt::target::ttnn::TensorRef const*, tt::tt_metal::Tensor const&, bool)
            #  --- /proj_sw/user_dev/vbrkic/src_bgd/ttforge/tt-forge-fe/third_party/tt-mlir/build/install/lib/libTTMLIRRuntime.so(+0x2c1385) [0x7f0acaabd385]
            #  --- tt::runtime::ttnn::operations::eltwise::unary::run(tt::target::ttnn::EltwiseUnaryCompositeOp const*, tt::runtime::ttnn::ProgramContext&)
            #  --- tt::runtime::ttnn::ProgramExecutor::runOperation(tt::target::ttnn::Operation const*)
            #  --- tt::runtime::ttnn::ProgramExecutor::execute()
            #  --- tt::runtime::ttnn::runProgram(std::shared_ptr<tt::tt_metal::distributed::MeshDevice>, tt::runtime::Binary, unsigned int, std::vector<tt::runtime::Tensor, std::allocator<tt::runtime::Tensor> >&)
            #  --- tt::runtime::ttnn::submit(tt::runtime::Device, tt::runtime::Binary, unsigned int, std::vector<tt::runtime::Tensor, std::allocator<tt::runtime::Tensor> >&)
            #  --- /proj_sw/user_dev/vbrkic/src_bgd/ttforge/tt-forge-fe/third_party/tt-mlir/build/install/lib/libTTMLIRRuntime.so(+0x195cfa) [0x7f0aca991cfa]
            #  --- /proj_sw/user_dev/vbrkic/src_bgd/ttforge/tt-forge-fe/third_party/tt-mlir/build/install/lib/libTTMLIRRuntime.so(+0x19324c) [0x7f0aca98f24c]
            #  --- tt::runtime::submit(tt::runtime::Device, tt::runtime::Binary, unsigned int, std::vector<tt::runtime::Tensor, std::allocator<tt::runtime::Tensor> >&)
            #  --- tt::run_program(tt::runtime::Binary&, int, std::vector<tt::Tensor, std::allocator<tt::Tensor> >&)
            #  --- tt::ModelState::run_program(tt::ProgramType, std::vector<tt::Tensor, std::allocator<tt::Tensor> >)
            #  --- /localdev/vbrkic/src/forge/tt-forge-fe/forge/forge/_C.so(+0xa7e5f6) [0x7f0ad669f5f6]
            #  --- /localdev/vbrkic/src/forge/tt-forge-fe/forge/forge/_C.so(+0xa7e59f) [0x7f0ad669f59f]
            #  --- /localdev/vbrkic/src/forge/tt-forge-fe/forge/forge/_C.so(+0xa7e521) [0x7f0ad669f521]
            #  --- /localdev/vbrkic/src/forge/tt-forge-fe/forge/forge/_C.so(+0xa7e472) [0x7f0ad669f472]
            #  --- /localdev/vbrkic/src/forge/tt-forge-fe/forge/forge/_C.so(+0xa7e389) [0x7f0ad669f389]
            #  --- /localdev/vbrkic/src/forge/tt-forge-fe/forge/forge/_C.so(+0x62a0ee) [0x7f0ad624b0ee]
            #
            # # >       self.runtime_model_state.run_program(ProgramType.Forward, self.inputs)
            # # E       RuntimeError: Fatal error
            # # forge/forge/compiled_graph_state.py:310: RuntimeError
            ExceptionCheck(
                class_name="RuntimeError",
                component=ComponentChecker.NONE.value,  # TODO remove
                # component=ComponentChecker.FORGE.value,
                # component=ComponentChecker.TTNN.value,
                # TODO: introduce console output check
                # TODO: remove duplicate of Fatal error
                # execution_log=[
                #     M.contains("DataType mismatch, expected INT32, got FLOAT32"),
                # ],
                message=[
                    M.equals("Fatal error"),
                ],
                error_log=[
                    M.last_line(M.contains("forge/compiled_graph_state.py:")),
                ],
            ),
        ],
    )

    WRONG_SCALAR_TYPE_OLD = FailingReason(
        description="Wrong scalar type old",
        checks=[
            # RuntimeError: expected scalar type Char but found Float
            # RuntimeError: expected scalar type Int but found Float
            # RuntimeError: expected scalar type Long but found Float
            # E       RuntimeError: expected scalar type Char but found Float
            # ../ttforge-toolchain/venv/lib/python3.10/site-packages/torch/nn/modules/conv.py:456: RuntimeError
            ExceptionCheck(
                class_name="RuntimeError",
                # component=ComponentChecker.FORGE.value,
                component=ComponentChecker.NONE.value,  # TODO remove
                message=[
                    M.regex("expected scalar type .* but found Float"),
                ],
                error_log=[
                    M.last_line(M.contains("torch/nn/modules/conv.py:")),
                ],
            ),
        ],
    )

    UNSUPPORTED_EXPONENT_VALUE = FailingReason(
        description="Unsupported exponent value",
        checks=[
            # >               assert False, f"Exponent value {dec} is not yet supported."
            # E               AssertionError: Exponent value 0.25999999046325684 is not yet supported.
            # forge/forge/tvm_calls/relay/op/forge_passes.py:704: AssertionError
            ExceptionCheck(
                class_name="AssertionError",
                component=ComponentChecker.TVM.value,
                message=[
                    M.regex("Exponent value .* is not yet supported"),
                ],
                error_log=[
                    M.last_line(M.contains("forge/tvm_calls/relay/op/forge_passes.py:")),
                ],
            ),
        ],
    )

    # # TODO
    # UNSUPPORTED_SPECIAL_CASE = FailingReason(
    #     description="Unsupported special case",
    #     checks=[
    #     ]
    # )

    WEIGHT_BIAS_SHAPE_MISMATCH = FailingReason(
        description="Weight and bias shape mismatch",
        checks=[
            # ExceptionCheck(
            #     class_name="RuntimeError",
            #     message=[
            #         M.contains("normalized_index >= 0 and normalized_index < rank"),
            #     ]
            # ),
            # conv2d	RuntimeError: Given weight of size [1, 1, 2, 28], expected bias to be 1-dimensional with 1 elements, but got bias of size [1, 1, 1, 1] instead
            # E       RuntimeError: Given weight of size [1, 1, 2, 28], expected bias to be 1-dimensional with 1 elements, but got bias of size [1, 1, 1, 1] instead
            # forge/forge/op/eval/forge/convolution.py:69: RuntimeError
            ExceptionCheck(
                class_name="RuntimeError",
                component=ComponentChecker.FORGE.value,
                # Given weight of size [10, 2, 1, 1], expected bias to be 1-dimensional with 10 elements, but got bias of size [1, 1, 1, 10] instead
                message=[
                    M.regex(
                        "Given weight of size .*, expected bias to be .*-dimensional with .* elements, but got bias of size .* instead"
                    ),
                ],
                error_log=[
                    M.last_line(M.contains("forge/op/eval/forge/convolution.py:")),
                ],
            ),
            # conv2d	TypeError: 'NotImplementedType' object is not callable
            # forge/forge/op/eval/forge/__init__.py:169: in is_eltwise_binary
            #     return module_name_or_cls(op_type).is_eltwise_binary()
            # _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
            # self = pad{mode: replicate, pad_len: 8, padding: [0, 0, 0, 0, 4, 4, 4, 4], value: 0.000000e+00}
            #     def is_eltwise_binary(self) -> bool:
            # >       raise NotImplemented()
            # E       TypeError: 'NotImplementedType' object is not callable
            # forge/forge/op/eval/interface.py:112: TypeError
            ExceptionCheck(
                class_name="TypeError",
                # component=ComponentChecker.FORGE.value,
                component=ComponentChecker.NONE.value,  # TODO remove
                message=[
                    M.equals("'NotImplementedType' object is not callable"),
                ],
                error_log=[
                    M.last_line(M.contains("forge/op/eval/interface.py:")),
                ],
            ),
        ],
    )

    NOT_IMPLEMENTED_ATEN = FailingReason(
        description="Not implemented aten operator",
        checks=[
            # >           raise NotImplementedError(msg)
            # E           NotImplementedError: The following operators are not implemented: ['aten::exp2']
            # tvm/relay/frontend/pytorch.py:5319: NotImplementedError
            ExceptionCheck(
                class_name="NotImplementedError",
                component=ComponentChecker.TVM.value,
                message=[
                    M.starts_with("The following operators are not implemented: ['aten::"),
                ],
                error_log=[
                    M.last_line(M.contains("tvm/relay/frontend/pytorch.py:")),
                ],
            ),
        ],
    )

    LOWERING_UNSUPPORTED_OPERATION = FailingReason(
        description="Lowering unsupported operation",
        checks=[
            # arctan	RuntimeError: Found Unsupported operations while lowering from TTForge to TTIR in forward graph
            # arctan2	RuntimeError: Found Unsupported operations while lowering from TTForge to TTIR in forward graph
            # atan	RuntimeError: Found Unsupported operations while lowering from TTForge to TTIR in forward graph
            # atan2	RuntimeError: Found Unsupported operations while lowering from TTForge to TTIR in forward graph
            # >       context.compiled_binary = forge._C.run_mlir_compiler(forge_module, compiler_cfg.mlir_config, forge_property_handler)
            # E       RuntimeError: Found Unsupported operations while lowering from TTForge to TTIR in forward graph
            # forge/forge/compile.py:1015: RuntimeError
            ExceptionCheck(
                class_name="RuntimeError",
                component=ComponentChecker.FORGE.value,
                message=[
                    M.starts_with("Found Unsupported operations while lowering from TTForge to TTIR in forward graph"),
                ],
                error_log=[
                    M.last_line(M.contains("forge/compile.py:")),
                ],
            ),
            ExceptionCheck(
                class_name="RuntimeError",
                # component=ComponentChecker.FORGE.value,
                component=ComponentChecker.NONE.value,  # TODO remove
                message=[
                    M.starts_with("Unsupported operation for lowering from TTForge to TTIR:"),
                ],
            ),
        ],
    )

    UNSUPPORTED_OP_TYPES = FailingReason(
        description="Unsupported op types",
        checks=[
            # ExceptionCheck(
            #     class_name="RuntimeError",
            #     message=[
            #         M.contains(" not implemented for "),
            #     ],
            # ),
            # tan	AssertionError: Encountered unsupported op types. Check error logs for more details
            # E               AssertionError: Encountered unsupported op types. Check error logs for more details
            # forge/forge/tvm_to_python.py:2240: AssertionError
            ExceptionCheck(
                class_name="AssertionError",
                component=ComponentChecker.TVM.value,
                message=[
                    M.equals("Encountered unsupported op types. Check error logs for more details"),
                ],
                error_log=[
                    M.last_line(M.contains("forge/tvm_to_python.py:")),
                ],
            ),
        ],
    )

    NOT_IMPLEMENTED_OLD = FailingReason(
        description="Not implemented operator old",
        checks=[
            ExceptionCheck(
                class_name="RuntimeError",
                component=ComponentChecker.NONE.value,  # TODO remove
                message=[
                    # tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/data_movement/concat/device/concat_device_operation.cpp:47: !in_ref.get_shape().has_tile_padding(this->dim)
                    M.contains("!in_ref.get_shape().has_tile_padding(this->dim)"),
                ],
            ),
            # floor	RuntimeError: TT_THROW @ /proj_sw/user_dev/vbrkic/src_bgd/ttforge/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/eltwise/binary/device/broadcast_height_and_
            # >       self.runtime_model_state.run_program(ProgramType.Forward, self.inputs)
            # E       RuntimeError: TT_THROW @ /proj_sw/user_dev/vbrkic/src_bgd/ttforge/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/eltwise/binary/device/broadcast_height_and_width_multi_core_program_factory.cpp:27: tt::exception
            # E       info:
            # E       BinaryOpType cannot be mapped to BcastOpMath
            # E       backtrace:
            # E        --- ttnn::operations::binary::BinaryDeviceOperation::BroadcastHeightAndWidthMultiCore::create(ttnn::operations::binary::BinaryDeviceOperation::operation_attributes_t const&, ttnn::operations::binary::BinaryDeviceOperation::tensor_args_t const&, tt::tt_metal::Tensor&)
            # E        --- ttnn::operations::binary::BinaryDeviceOperation::tensor_return_value_t ttnn::device_operation::detail::launch_on_single_device<ttnn::operations::binary::BinaryDeviceOperation>(tt::stl::StrongType<unsigned char, ttnn::QueueIdTag>, ttnn::operations::binary::BinaryDeviceOperation::operation_attributes_t const&, ttnn::operations::binary::BinaryDeviceOperation::tensor_args_t const&)
            # E        --- ttnn::operations::binary::BinaryDeviceOperation::tensor_return_value_t ttnn::device_operation::detail::invoke<ttnn::operations::binary::BinaryDeviceOperation>(tt::stl::StrongType<unsigned char, ttnn::QueueIdTag>, ttnn::operations::binary::BinaryDeviceOperation::operation_attributes_t const&, ttnn::operations::binary::BinaryDeviceOperation::tensor_args_t const&)
            # E        --- tt::runtime::ttnn::operations::eltwise::binary::run(tt::target::ttnn::EltwiseBinaryOp const*, tt::runtime::ttnn::ProgramContext&)
            # E        --- tt::runtime::ttnn::ProgramExecutor::runOperation(tt::target::ttnn::Operation const*)
            # E        --- tt::runtime::ttnn::ProgramExecutor::execute()
            # E        --- tt::runtime::ttnn::runProgram(std::shared_ptr<tt::tt_metal::distributed::MeshDevice>, tt::runtime::Binary, unsigned int, std::vector<tt::runtime::Tensor, std::allocator<tt::runtime::Tensor> >&)
            # E        --- tt::runtime::ttnn::submit(tt::runtime::Device, tt::runtime::Binary, unsigned int, std::vector<tt::runtime::Tensor, std::allocator<tt::runtime::Tensor> >&)
            # E        --- tt::runtime::submit(tt::runtime::Device, tt::runtime::Binary, unsigned int, std::vector<tt::runtime::Tensor, std::allocator<tt::runtime::Tensor> >&)
            # E        --- tt::run_program(tt::runtime::Binary&, int, std::vector<tt::Tensor, std::allocator<tt::Tensor> >&)
            # E        --- tt::ModelState::run_program(tt::ProgramType, std::vector<tt::Tensor, std::allocator<tt::Tensor> >)
            # forge/forge/compiled_graph_state.py:310: RuntimeError
            ExceptionCheck(
                class_name="RuntimeError",
                # component=ComponentChecker.TTNN.value,
                component=ComponentChecker.NONE.value,  # TODO remove
                message=[
                    M.contains("info:\nBinaryOpType cannot be mapped to BcastOpMath"),
                ],
                error_log=[
                    M.last_line(M.contains("forge/compiled_graph_state.py:")),
                ],
            ),
        ],
    )

    SOURCE_MAPS_NOT_POPULATED = FailingReason(
        description="Source maps are not populated",
        checks=[
            # sum	tvm._ffi.base.TVMError: Traceback (most recent call last):
            # E       tvm._ffi.base.TVMError: Traceback (most recent call last):
            # E         12: _ZN3tvm7runtime13PackedFuncObj
            # E         11: tvm::runtime::TypedPackedFunc<tvm::RelayExpr (tvm::runtime::Array<tvm::relay::DFPatternCallback, void>, tvm::RelayExpr, tvm::IRModule)>::AssignTypedLambda<tvm::RelayExpr (*)(tvm::runtime::Array<tvm::relay::DFPatternCallback, void>, tvm::RelayExpr, tvm::IRModule)>(tvm::RelayExpr (*)(tvm::runtime::Array<tvm::relay::DFPatternCallback, void>, tvm::RelayExpr, tvm::IRModule), std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}::operator()(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*) const
            # E         10: tvm::relay::RewritePatterns(tvm::runtime::Array<tvm::relay::DFPatternCallback, void>, tvm::RelayExpr, tvm::IRModule)
            # E         9: tvm::relay::PatternRewriter::Rewrite(tvm::runtime::Array<tvm::relay::DFPatternCallback, void> const&, tvm::RelayExpr const&)
            # E         8: tvm::relay::InferTypeWithModule(tvm::RelayExpr const&, tvm::IRModule const&)
            # E         7: tvm::transform::Pass::operator()(tvm::IRModule) const
            # E         6: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
            # E         5: tvm::transform::ModulePassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
            # E         4: tvm::runtime::PackedFuncObj::Extractor<tvm::runtime::PackedFuncSubObj<tvm::runtime::TypedPackedFunc<tvm::IRModule (tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::relay::transform::InferType()::{lambda(tvm::IRModule, tvm::transform::PassContext const&)#1}>(tvm::relay::transform::InferType()::{lambda(tvm::IRModule, tvm::transform::PassContext const&)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}> >::Call(tvm::runtime::PackedFuncObj const*, tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)
            # E         3: tvm::DiagnosticContext::Render()
            # E         2: tvm::DiagnosticRenderer::Render(tvm::DiagnosticContext const&)
            # E         1: tvm::runtime::PackedFuncObj::Extractor<tvm::runtime::PackedFuncSubObj<tvm::runtime::TypedPackedFunc<void (tvm::DiagnosticContext)>::AssignTypedLambda<tvm::TerminalRenderer(std::ostream&)::{lambda(tvm::DiagnosticContext const&)#1}>(tvm::TerminalRenderer(std::ostream&)::{lambda(tvm::DiagnosticContext const&)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}> >::Call(tvm::runtime::PackedFuncObj const*, tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)
            # E         0: tvm::ReportAt(tvm::DiagnosticContext const&, std::ostream&, tvm::Span const&, tvm::Diagnostic const&)
            # E         File "/__w/tt-forge-fe/tt-forge-fe/third_party/tvm/src/ir/diagnostic.cc", line 267
            # E       TVMError: The source maps are not populated for this module. Please use `tvm.relay.transform.AnnotateSpans` to attach source maps for error reporting.
            # E       Error: tensor type `Tensor[(1, 1, 1), float32]` has 3 dimensions, while `float32` has 0 dimensions
            # /localdev/kmilanovic/src/forge/forge_wheels/venv/lib/python3.10/site-packages/tvm/_ffi/base.py:479: TVMError
            ExceptionCheck(
                class_name="tvm._ffi.base.TVMError",
                component=ComponentChecker.TVM.value,
                message=[
                    M.contains("Traceback (most recent call last)"),
                ],
                error_log=[
                    M.contains(
                        "TVMError: The source maps are not populated for this module. Please use `tvm.relay.transform.AnnotateSpans` to attach source maps for error reporting."
                    ),
                    M.regex("tensor type .* has .* dimensions, while .* has .* dimensions"),
                    M.last_line(M.contains("tvm/_ffi/base.py:")),
                ],
            ),
        ],
    )

    OUT_OF_MEMORY = FailingReason(
        description="Out of Memory",
        checks=[
            # # INFO     | forge.compiled_graph_state:__call__:247  Running model forward on device...
            # # Always | FATAL    | Out of Memory: Not enough space to allocate 896204800 B DRAM buffer across 12 banks, where each bank needs to store 74686464 B
            # >       self.runtime_model_state.run_program(ProgramType.Forward, self.inputs)
            # E       RuntimeError: TT_THROW @ /proj_sw/user_dev/vbrkic/src_bgd/ttforge/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/impl/allocator/bank_manager.cpp:141: tt::exception
            # E       info:
            # E       Out of Memory: Not enough space to allocate 4063232000 B DRAM buffer across 12 banks, where each bank needs to store 338604032 B
            # E       backtrace:
            # E        --- tt::tt_metal::BankManager::allocate_buffer(unsigned long, unsigned long, bool, CoreRangeSet const&, std::optional<unsigned int>)
            # E        --- tt::tt_metal::Allocator::allocate_buffer(tt::tt_metal::Buffer*)
            # E        --- tt::tt_metal::Buffer::allocate_impl()
            # E        --- tt::tt_metal::Buffer::create_buffer(tt::tt_metal::IDevice*, unsigned long, unsigned long, tt::tt_metal::BufferType, tt::tt_metal::TensorMemoryLayout, std::optional<tt::tt_metal::ShardSpecBuffer> const&, std::optional<tt::tt_metal::BufferDistributionSpec> const&, std::optional<bool>, std::optional<tt::stl::StrongType<unsigned char, tt::tt_metal::SubDeviceIdTag> >)
            # E        --- tt::tt_metal::Buffer::create(tt::tt_metal::IDevice*, unsigned long, unsigned long, tt::tt_metal::BufferType, tt::tt_metal::TensorMemoryLayout, std::optional<tt::tt_metal::ShardSpecBuffer> const&, std::optional<bool>, std::optional<tt::stl::StrongType<unsigned char, tt::tt_metal::SubDeviceIdTag> >)
            # E        --- tt::tt_metal::distributed::MeshBuffer::create(std::variant<tt::tt_metal::distributed::ReplicatedBufferConfig, tt::tt_metal::distributed::ShardedBufferConfig> const&, tt::tt_metal::distributed::DeviceLocalBufferConfig const&, tt::tt_metal::distributed::MeshDevice*, std::optional<unsigned long>)
            # E        --- tt::tt_metal::tensor_impl::allocate_mesh_buffer_on_device(tt::tt_metal::distributed::MeshDevice*, tt::tt_metal::TensorSpec const&)
            # E        --- tt::tt_metal::Tensor tt::tt_metal::tensor_impl::to_device_mesh_tensor<float>(tt::tt_metal::Tensor const&, tt::tt_metal::distributed::MeshDevice*, tt::tt_metal::MemoryConfig const&, tt::stl::StrongType<unsigned char, ttnn::QueueIdTag>)
            # E        --- auto tt::tt_metal::tensor_impl::to_device_mesh_tensor_wrapper<tt::tt_metal::Tensor const&, tt::tt_metal::distributed::MeshDevice*&, tt::tt_metal::MemoryConfig const&, tt::stl::StrongType<unsigned char, ttnn::QueueIdTag>&>(tt::tt_metal::Tensor const&, tt::tt_metal::distributed::MeshDevice*&, tt::tt_metal::MemoryConfig const&, tt::stl::StrongType<unsigned char, ttnn::QueueIdTag>&)
            # E        --- tt::tt_metal::tensor_ops::tensor_to_device(tt::tt_metal::Tensor const&, tt::tt_metal::distributed::MeshDevice*, tt::tt_metal::MemoryConfig const&, tt::stl::StrongType<unsigned char, ttnn::QueueIdTag>)
            # E        --- tt::tt_metal::Tensor::to_device(tt::tt_metal::distributed::MeshDevice*, tt::tt_metal::MemoryConfig const&, tt::stl::StrongType<unsigned char, ttnn::QueueIdTag>) const
            # E        --- ttnn::operations::core::to_device(tt::tt_metal::Tensor const&, tt::tt_metal::distributed::MeshDevice*, std::optional<tt::tt_metal::MemoryConfig> const&, tt::stl::StrongType<unsigned char, ttnn::QueueIdTag>)
            # E        --- tt::runtime::ttnn::LayoutConverter::toDeviceIfNeeded(tt::tt_metal::Tensor const&, std::optional<std::reference_wrapper<tt::tt_metal::distributed::MeshDevice> >, bool)
            # E        --- tt::runtime::ttnn::LayoutConverter::handleHostInputLayoutNoTypecast(tt::tt_metal::Tensor const&, std::optional<std::reference_wrapper<tt::tt_metal::distributed::MeshDevice> >)
            # E        --- tt::runtime::ttnn::LayoutConverter::convertHostTensorLayout(tt::tt_metal::Tensor const&, std::optional<std::reference_wrapper<tt::tt_metal::distributed::MeshDevice> >)
            # E        --- tt::runtime::ttnn::LayoutConverter::convertTensorLayout(tt::tt_metal::Tensor const&, std::optional<std::reference_wrapper<tt::tt_metal::distributed::MeshDevice> >)
            # E        --- tt::runtime::ttnn::toLayout(tt::runtime::Tensor, tt::runtime::Device, tt::runtime::Layout, std::optional<bool>)
            # E        --- tt::runtime::toLayout(tt::runtime::Tensor, tt::runtime::Device, tt::runtime::Layout, std::optional<bool>)
            # E        --- tt::TensorImpl::to_device(unsigned long, tt::runtime::Layout&)
            # E        --- tt::Tensor::to_device(unsigned long, tt::runtime::Layout&)
            # E        --- tt::ModelState::run_program(tt::ProgramType, std::vector<tt::Tensor, std::allocator<tt::Tensor> >)
            # forge/forge/compiled_graph_state.py:310: RuntimeError
            ExceptionCheck(
                class_name="RuntimeError",
                component=ComponentChecker.METAL.value,
                message=[
                    M.any(
                        M.regex(
                            # Out of Memory: Not enough space to allocate 1261568 B L1 buffer across 2 banks, where each bank needs to store 630784 B
                            "Out of Memory: Not enough space to allocate .* B L1 buffer across .* banks, where each bank needs to store .* B"
                        ),
                        M.regex(
                            "Out of Memory: Not enough space to allocate .* B L1_SMALL buffer across .* banks, where each bank needs to store .* B"
                        ),
                        M.regex(
                            "Out of Memory: Not enough space to allocate .* B DRAM buffer across .* banks, where each bank needs to store .* B"
                        ),
                    ),
                ],
                error_log=[
                    M.last_line(M.contains("forge/compiled_graph_state.py:")),
                ],
            ),
            # ExceptionCheck(
            #     class_name="RuntimeError",
            #     message=[
            #         M.contains("tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/impl/allocator/allocator.cpp:143"),
            #     ]
            # ),
            # ExceptionCheck(
            #     class_name="RuntimeError",
            #     message=[
            #         M.contains("tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/impl/allocator/allocator.cpp:145"),
            #     ]
            # ),
        ],
    )

    CIRCULAR_BUFFER_EXCEEDS_L1 = FailingReason(
        description="Circular buffer exceeds L1 cache size",
        checks=[
            # concatenate	RuntimeError: TT_THROW @ /__w/tt-forge-fe/tt-metal/tt_metal/impl/program/program.cpp:939: tt::exception
            # >       self.runtime_model_state.run_program(ProgramType.Forward, self.inputs)
            # E       RuntimeError: TT_THROW @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/impl/program/program.cpp:939: tt::exception
            # E       info:
            # E       Statically allocated circular buffers on core range [(x=0,y=0) - (x=7,y=7)] grow to 2140832 B which is beyond max L1 size of 1499136 B
            # E       backtrace:
            # E        --- tt::tt_metal::detail::ProgramImpl::validate_circular_buffer_region(tt::tt_metal::IDevice const*)
            # E        --- tt::tt_metal::distributed::MeshWorkloadImpl::compile(tt::tt_metal::distributed::MeshDevice*)
            # E        --- tt::tt_metal::distributed::EnqueueMeshWorkload(tt::tt_metal::distributed::MeshCommandQueue&, tt::tt_metal::distributed::MeshWorkload&, bool)
            # E        --- tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >::tensor_return_value_t ttnn::device_operation::detail::invoke<tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > > >(ttsl::StrongType<unsigned char, ttnn::QueueIdTag>, tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >::operation_attributes_t const&, tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >::tensor_args_t const&)
            # E        --- std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > tt::tt_metal::operation::run<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >(tt::tt_metal::operation::DeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >&&, std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > const&, std::vector<std::optional<tt::tt_metal::Tensor const>, std::allocator<std::optional<tt::tt_metal::Tensor const> > > const&, std::vector<std::optional<tt::tt_metal::Tensor>, std::allocator<std::optional<tt::tt_metal::Tensor> > > const&, ttsl::StrongType<unsigned char, ttnn::QueueIdTag>)
            # E        --- ttnn::operations::data_movement::detail::pad_impl(ttsl::StrongType<unsigned char, ttnn::QueueIdTag>, tt::tt_metal::Tensor const&, std::span<unsigned int const, 18446744073709551615ul>, std::span<unsigned int const, 18446744073709551615ul>, float, bool, std::optional<tt::tt_metal::MemoryConfig> const&)
            # E        --- ttnn::operations::data_movement::detail::pad_impl(ttsl::StrongType<unsigned char, ttnn::QueueIdTag>, tt::tt_metal::Tensor const&, ttsl::SmallVector<ttnn::operations::data_movement::PadSpecDim, 8ul>, float, bool, std::optional<tt::tt_metal::MemoryConfig> const&)
            # E        --- ttnn::operations::data_movement::detail::invoke_rm(ttsl::StrongType<unsigned char, ttnn::QueueIdTag>, tt::tt_metal::Tensor const&, ttsl::SmallVector<ttnn::operations::data_movement::PadSpecDim, 8ul> const&, float, bool, std::optional<tt::tt_metal::MemoryConfig> const&)
            # E        --- ttnn::operations::data_movement::ExecutePad::invoke(ttsl::StrongType<unsigned char, ttnn::QueueIdTag>, tt::tt_metal::Tensor const&, ttsl::SmallVector<ttnn::operations::data_movement::PadSpecDim, 8ul> const&, float, bool, std::optional<tt::tt_metal::MemoryConfig> const&)
            # E        --- ttnn::operations::data_movement::ExecutePad::invoke(ttsl::StrongType<unsigned char, ttnn::QueueIdTag>, tt::tt_metal::Tensor const&, ttsl::SmallVector<std::pair<unsigned int, unsigned int>, 8ul> const&, float, bool, std::optional<tt::tt_metal::MemoryConfig> const&)
            # E        --- ttnn::operations::data_movement::pad_to_tile_vol(ttsl::StrongType<unsigned char, ttnn::QueueIdTag>, tt::tt_metal::Tensor const&, float, bool, std::optional<tt::tt_metal::MemoryConfig> const&)
            # E        --- ttnn::operations::data_movement::ConcatOperation::invoke(ttsl::StrongType<unsigned char, ttnn::QueueIdTag>, std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > const&, int, std::optional<tt::tt_metal::MemoryConfig> const&, std::optional<tt::tt_metal::Tensor> const&, unsigned int)
            # E        --- tt::runtime::ttnn::operations::data_movement::run(tt::target::ttnn::ConcatOp const*, tt::runtime::ttnn::ProgramContext&)
            # E        --- tt::runtime::ttnn::ProgramExecutor::execute()
            # E        --- tt::runtime::ttnn::submit(tt::runtime::Device, tt::runtime::Binary, unsigned int, std::vector<tt::runtime::Tensor, std::allocator<tt::runtime::Tensor> >&)
            # E        --- tt::runtime::submit(tt::runtime::Device, tt::runtime::Binary, unsigned int, std::vector<tt::runtime::Tensor, std::allocator<tt::runtime::Tensor> >&)
            # E        --- tt::run_program(tt::runtime::Binary&, int, std::vector<tt::Tensor, std::allocator<tt::Tensor> >&)
            # E        --- tt::ModelState::run_program(tt::ProgramType, std::vector<tt::Tensor, std::allocator<tt::Tensor> >)
            # forge/compiled_graph_state.py:316: RuntimeError
            ExceptionCheck(
                class_name="RuntimeError",
                component=ComponentChecker.METAL.value,
                message=[
                    M.starts_with("TT_THROW"),
                    M.contains("Statically allocated circular buffers on core range"),
                ],
                error_log=[
                    M.last_line(M.contains("forge/compiled_graph_state.py:")),
                ],
            ),
        ],
    )

    CIRCULAR_BUFFER_CLASH = FailingReason(
        description="Circular buffer clash",
        checks=[
            # conv2d	RuntimeError: TT_THROW @ tt-metal/tt_metal/impl/program/program.cpp:791: tt::exception
            # >       self.runtime_model_state.run_program(ProgramType.Forward, self.inputs)
            # E       RuntimeError: TT_THROW @ /proj_sw/user_dev/vbrkic/src_bgd/ttforge/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/impl/program/program.cpp:791: tt::exception
            # E       info:
            # E       Statically allocated circular buffers in program 6914 clash with L1 buffers on core range [(x=0,y=0) - (x=7,y=7)]. L1 buffer allocated at 1030272 and static circular buffer region ends at 1473568
            # E       backtrace:
            # E        --- tt::tt_metal::detail::ProgramImpl::validate_circular_buffer_region(tt::tt_metal::IDevice const*)
            # E        --- tt::tt_metal::detail::ValidateCircularBufferRegion(tt::tt_metal::Program const&, tt::tt_metal::IDevice const*)
            # E        --- tt::tt_metal::EnqueueProgram(tt::tt_metal::CommandQueue&, tt::tt_metal::Program&, bool)
            # E        --- tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >::tensor_return_value_t ttnn::device_operation::detail::invoke<tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > > >(tt::stl::StrongType<unsigned char, ttnn::QueueIdTag>, tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >::operation_attributes_t const&, tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >::tensor_args_t const&)
            # E        --- std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > tt::tt_metal::operation::run<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >(tt::tt_metal::operation::DeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >&&, std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > const&, std::vector<std::optional<tt::tt_metal::Tensor const>, std::allocator<std::optional<tt::tt_metal::Tensor const> > > const&, std::vector<std::optional<tt::tt_metal::Tensor>, std::allocator<std::optional<tt::tt_metal::Tensor> > > const&, tt::stl::StrongType<unsigned char, ttnn::QueueIdTag>)
            # E        --- std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > tt::tt_metal::operation::run_without_autoformat<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >(tt::tt_metal::operation::DeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >&&, std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > const&, std::vector<std::optional<tt::tt_metal::Tensor const>, std::allocator<std::optional<tt::tt_metal::Tensor const> > > const&, std::vector<std::optional<tt::tt_metal::Tensor>, std::allocator<std::optional<tt::tt_metal::Tensor> > > const&, tt::stl::StrongType<unsigned char, ttnn::QueueIdTag>)
            # E        --- void tt::tt_metal::operation::launch_op_func<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >(std::function<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > (std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > const&, std::vector<std::optional<tt::tt_metal::Tensor const>, std::allocator<std::optional<tt::tt_metal::Tensor const> > > const&, std::vector<std::optional<tt::tt_metal::Tensor>, std::allocator<std::optional<tt::tt_metal::Tensor> > > const&)> const&, std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> >, std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> >&, std::vector<std::optional<tt::tt_metal::Tensor const>, std::allocator<std::optional<tt::tt_metal::Tensor const> > >, std::vector<std::optional<tt::tt_metal::Tensor>, std::allocator<std::optional<tt::tt_metal::Tensor> > >)
            # E        --- tt::runtime::ttnn::operations::conv::run(tt::target::ttnn::Conv2dOp const*, tt::runtime::ttnn::ProgramContext&)
            # E        --- tt::runtime::ttnn::runProgram(tt::tt_metal::distributed::MeshDevice&, tt::runtime::Binary, unsigned int, std::vector<tt::runtime::Tensor, std::allocator<tt::runtime::Tensor> >&)
            # E        --- tt::runtime::ttnn::submit(tt::runtime::Device, tt::runtime::Binary, unsigned int, std::vector<tt::runtime::Tensor, std::allocator<tt::runtime::Tensor> >&)
            # E        --- tt::runtime::submit(tt::runtime::Device, tt::runtime::Binary, unsigned int, std::vector<tt::runtime::Tensor, std::allocator<tt::runtime::Tensor> >&)
            # E        --- tt::run_program(tt::runtime::Binary&, int, std::vector<tt::Tensor, std::allocator<tt::Tensor> >&)
            # E        --- tt::ModelState::run_program(tt::ProgramType, std::vector<tt::Tensor, std::allocator<tt::Tensor> >)
            # forge/forge/compiled_graph_state.py:310: RuntimeError
            ExceptionCheck(
                class_name="RuntimeError",
                component=ComponentChecker.METAL.value,
                message=[
                    M.starts_with("TT_THROW"),
                    M.any(
                        M.regex("tt-metal/tt_metal/impl/program/program.cpp:\\d+: tt::exception"),
                    ),
                ],
                error_log=[
                    M.regex(
                        "Statically allocated circular buffers in program .* clash with L1 buffers on core range .*. L1 buffer allocated at .* and static circular buffer region ends at .*"
                    ),
                    M.last_line(M.contains("forge/compiled_graph_state.py:")),
                ],
            ),
        ],
    )

    CIRCULAR_MODE_NOT_SUPPORTED = FailingReason(
        description="Circular mode not supported",
        checks=[
            # conv2d	ValueError: circular mode for torch.nn.functional.pad are not supported in TVM
            # >           raise ValueError("circular mode for torch.nn.functional.pad are not supported in TVM")
            # E           ValueError: circular mode for torch.nn.functional.pad are not supported in TVM
            # third_party/tvm/python/tvm/relay/frontend/pytorch.py:2406: ValueError
            ExceptionCheck(
                class_name="ValueError",
                component=ComponentChecker.TVM.value,
                message=[
                    M.starts_with("circular mode for torch.nn.functional.pad are not supported in TVM"),
                ],
                error_log=[
                    M.any(
                        M.last_line(M.contains("/tvm/relay/frontend/pytorch.py:")),
                    ),
                ],
            ),
        ],
    )

    ATTRIBUTE_ERROR = FailingReason(
        description="Attribute error",
        checks=[
            # square	AttributeError: 'int' object has no attribute 'is_integer'
            # E       AttributeError: 'int' object has no attribute 'is_integer'
            # forge/forge/tvm_calls/relay/op/forge_passes.py:693: AttributeError
            ExceptionCheck(
                class_name="AttributeError",
                component=ComponentChecker.TVM.value,
                message=[
                    M.equals("'int' object has no attribute 'is_integer'"),
                ],
                error_log=[
                    M.last_line(M.contains("forge/tvm_calls/relay/op/forge_passes.py:")),
                ],
            ),
        ],
    )

    COMPILATION_FAILED = FailingReason(
        description="Model compilation failed",
    )

    COMPILATION_FAILED_OLD1 = FailingReason(
        description="Model compilation failed old 1",
        checks=[
            ExceptionCheck(
                class_name="RuntimeError",
                component=ComponentChecker.NONE.value,  # TODO remove
                message=[
                    M.regex(
                        "tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/core/core.cpp:\\d+: tt::exception"
                    ),
                ],
            ),
        ],
    )

    FAILED_MLIR_VERIFICATION = FailingReason(
        description="MLIR module failed verification",
        checks=[
            # forge/forge/compile.py:1015: RuntimeError
            # >       context.compiled_binary = forge._C.run_mlir_compiler(forge_module, compiler_cfg.mlir_config, forge_property_handler)
            # E       RuntimeError: Generated MLIR module failed verification.
            # forge/forge/compile.py:1015: RuntimeError
            ExceptionCheck(
                class_name="RuntimeError",
                component=ComponentChecker.FORGE.value,
                message=[
                    M.starts_with("Generated MLIR module failed verification"),
                ],
                error_log=[
                    M.last_line(M.contains("forge/compile.py:")),
                ],
            ),
        ],
    )

    COMPILATION_FAILED_OLD2 = FailingReason(
        description="Model compilation failed old2",
        checks=[
            # clamp	RuntimeError: Fatal error
            # >       self.runtime_model_state.run_program(ProgramType.Forward, self.inputs)
            # E       RuntimeError: Fatal error
            # forge/forge/compiled_graph_state.py:310: RuntimeError
            ExceptionCheck(
                class_name="RuntimeError",
                # component=ComponentChecker.FORGE.value,
                component=ComponentChecker.NONE.value,  # TODO remove
                message=[
                    M.equals("Fatal error"),
                ],
                error_log=[
                    M.contains(">       self.runtime_model_state.run_program(ProgramType.Forward, self.inputs)"),
                    M.last_line(M.contains("forge/compiled_graph_state.py:")),
                ],
            ),
            # E       RuntimeError: Unable to cast Python instance to C++ type (#define PYBIND11_DETAILED_ERROR_MESSAGES or compile in debug mode for details)
            # forge/compile.py:745: RuntimeError
            ExceptionCheck(
                class_name="RuntimeError",
                # component=ComponentChecker.FORGE.value,
                component=ComponentChecker.NONE.value,  # TODO remove
                message=[
                    M.contains(
                        "Unable to cast Python instance to C++ type (#define PYBIND11_DETAILED_ERROR_MESSAGES or compile in debug mode for details)"
                    ),
                ],
                error_log=[
                    M.last_line(M.contains("forge/compile.py:")),
                ],
            ),
        ],
    )

    INFERENCE_FAILED_OLD1 = FailingReason(
        description="Inference failed old 1",
        checks=[
            ExceptionCheck(
                class_name="AttributeError",
                component=ComponentChecker.NONE.value,  # TODO remove
                message=[
                    M.equals("'TransposeTM' object has no attribute 'z_dim_slice' (via OpType cpp underlying class)"),
                ],
            ),
            ExceptionCheck(
                class_name="RuntimeError",
                component=ComponentChecker.NONE.value,  # TODO remove
                message=[
                    M.contains(
                        "tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/data_movement/tilize/device/tilize_op.cpp"
                    ),
                ],
            ),
            ExceptionCheck(
                class_name="RuntimeError",
                component=ComponentChecker.NONE.value,  # TODO remove
                message=[
                    M.contains(
                        "Statically allocated circular buffers on core range [(x=0,y=0) - (x=7,y=7)] grow to 28100144 B which is beyond max L1 size of 1499136 B"
                    ),
                ],
            ),
        ],
    )

    NORMALIZED_INDEX_OUT_OF_BOUNDS = FailingReason(
        description="Normalized index out of bounds",
        checks=[
            ExceptionCheck(
                # repeat_interleave	RuntimeError: TT_FATAL @ /proj_sw/user_dev/vbrkic/src_bgd/ttforge/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/common/shape.cpp:57: normalized_index >= 0 and normalized_i
                # >       self.runtime_model_state.run_program(ProgramType.Forward, self.inputs)
                # E       RuntimeError: TT_FATAL @ /proj_sw/user_dev/vbrkic/src_bgd/ttforge/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/common/shape.cpp:57: normalized_index >= 0 and normalized_index < rank
                # E       info:
                # E       Index is out of bounds for the rank, should be between 0 and 0 however is 18446744073709551615
                # E       backtrace:
                # E        --- tt::tt_metal::Shape::get_normalized_index(long) const
                # E        --- ttnn::operations::data_movement::ExecuteTranspose::invoke(tt::stl::StrongType<unsigned char, ttnn::QueueIdTag>, tt::tt_metal::Tensor const&, long const&, long const&, std::optional<tt::tt_metal::MemoryConfig> const&, std::optional<float> const&)
                # E        --- ttnn::operations::data_movement::ExecuteTranspose::invoke(tt::tt_metal::Tensor const&, long const&, long const&, std::optional<tt::tt_metal::MemoryConfig> const&, std::optional<float> const&)
                # E        --- ttnn::operations::data_movement::ExecuteRepeatInterleave::invoke(tt::tt_metal::Tensor const&, unsigned int, int, std::optional<tt::tt_metal::MemoryConfig> const&)
                # E        --- tt::runtime::ttnn::operations::data_movement::run(tt::target::ttnn::RepeatInterleaveOp const*, tt::runtime::ttnn::ProgramContext&)
                # E        --- tt::runtime::ttnn::ProgramExecutor::runOperation(tt::target::ttnn::Operation const*)
                # E        --- tt::runtime::ttnn::ProgramExecutor::execute()
                # E        --- tt::runtime::ttnn::runProgram(std::shared_ptr<tt::tt_metal::distributed::MeshDevice>, tt::runtime::Binary, unsigned int, std::vector<tt::runtime::Tensor, std::allocator<tt::runtime::Tensor> >&)
                # E        --- tt::runtime::ttnn::submit(tt::runtime::Device, tt::runtime::Binary, unsigned int, std::vector<tt::runtime::Tensor, std::allocator<tt::runtime::Tensor> >&)
                # E        --- tt::runtime::submit(tt::runtime::Device, tt::runtime::Binary, unsigned int, std::vector<tt::runtime::Tensor, std::allocator<tt::runtime::Tensor> >&)
                # E        --- tt::run_program(tt::runtime::Binary&, int, std::vector<tt::Tensor, std::allocator<tt::Tensor> >&)
                # E        --- tt::ModelState::run_program(tt::ProgramType, std::vector<tt::Tensor, std::allocator<tt::Tensor> >)
                # forge/forge/compiled_graph_state.py:310: RuntimeError
                class_name="RuntimeError",
                component=ComponentChecker.METAL.value,
                message=[
                    M.contains("Index is out of bounds for the rank, should be between 0 and 0 however is"),
                ],
                error_log=[
                    M.contains("normalized_index >= 0 and normalized_index < rank"),
                    M.last_line(M.contains("forge/compiled_graph_state.py:")),
                ],
            ),
        ],
    )

    INFERENCE_FAILED_OLD2 = FailingReason(
        description="Inference failed old 2",
        checks=[
            ExceptionCheck(
                class_name="RuntimeError",
                component=ComponentChecker.NONE.value,  # TODO remove
                message=[
                    M.contains(
                        "293 unique+common runtime args targeting kernel reader_concat_stick_layout_interleaved_start_id on (x=0,y=0) are too large. Max allowable is 256"
                    ),
                ],
            ),
            ExceptionCheck(
                class_name="RuntimeError",
                component=ComponentChecker.NONE.value,  # TODO remove
                message=[
                    M.contains("mat1 and mat2 must have the same dtype, but got Int and Float"),
                ],
            ),
        ],
    )

    MICROBATCHING_UNSUPPORTED = FailingReason(
        description="Higher microbatch size is not supported",
        checks=[
            # linear	RuntimeError: The expanded size of the tensor (10) must match the existing size (21) at non-singleton dimension 0. Target sizes: [10]. Tensor sizes: [21]
            # >           return torch.broadcast_to(tensor, target_shape)
            # E           RuntimeError: The expanded size of the tensor (11) must match the existing size (983) at non-singleton dimension 0.  Target sizes: [11].  Tensor sizes: [983]
            # forge/forge/op/eval/forge/tm.py:137: RuntimeError
            ExceptionCheck(
                class_name="RuntimeError",
                component=ComponentChecker.FORGE.value,
                message=[
                    M.regex(
                        "The expanded size of the tensor .* must match the existing size .* at non-singleton dimension .*.  Target sizes: .*.  Tensor sizes: .*"
                    ),
                ],
                error_log=[
                    M.contains(">           return torch.broadcast_to(tensor, target_shape)"),
                    M.last_line(M.contains("forge/op/eval/forge/tm.py:")),
                ],
            ),
        ],
    )

    UNSUPPORTED_AXIS = FailingReason(
        description="Unsupported axis parameter",
        checks=[
            # softmax	RuntimeError: TT_FATAL @ /__w/tt-forge-fe/tt-metal/ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/moreh_softmax_device_operation.cpp:94: input.dtype() == DataType::BFLOAT16 || input.dtype() == DataType::BFLOAT8_B
            # >       self.runtime_model_state.run_program(ProgramType.Forward, self.inputs)
            # E       RuntimeError: TT_FATAL @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/moreh_softmax_device_operation.cpp:94: input.dtype() == DataType::BFLOAT16 || input.dtype() == DataType::BFLOAT8_B
            # E       info:
            # E       Inputs must be of bfloat16 or bfloat8_b type
            # E       backtrace:
            # E        --- ttnn::operations::moreh::moreh_softmax::MorehSoftmaxOperation::tensor_return_value_t ttnn::device_operation::detail::launch_on_single_device<ttnn::operations::moreh::moreh_softmax::MorehSoftmaxOperation>(ttsl::StrongType<unsigned char, ttnn::QueueIdTag>, ttnn::operations::moreh::moreh_softmax::MorehSoftmaxOperation::operation_attributes_t const&, ttnn::operations::moreh::moreh_softmax::MorehSoftmaxOperation::tensor_args_t const&)
            # E        --- ttnn::operations::moreh::moreh_softmax::MorehSoftmaxOperation::tensor_return_value_t ttnn::device_operation::detail::invoke<ttnn::operations::moreh::moreh_softmax::MorehSoftmaxOperation>(ttsl::StrongType<unsigned char, ttnn::QueueIdTag>, ttnn::operations::moreh::moreh_softmax::MorehSoftmaxOperation::operation_attributes_t const&, ttnn::operations::moreh::moreh_softmax::MorehSoftmaxOperation::tensor_args_t const&)
            # E        --- ttnn::operations::normalization::ExecuteSoftmax::invoke(tt::tt_metal::Tensor const&, int, std::optional<tt::tt_metal::MemoryConfig> const&, std::optional<std::variant<ttnn::GrayskullComputeKernelConfig, ttnn::WormholeComputeKernelConfig> const>, bool)
            # E        --- tt::runtime::ttnn::operations::normalization::run(tt::target::ttnn::SoftmaxOp const*, tt::runtime::ttnn::ProgramContext&)
            # E        --- tt::runtime::ttnn::ProgramExecutor::execute()
            # E        --- tt::runtime::ttnn::submit(tt::runtime::Device, tt::runtime::Binary, unsigned int, std::vector<tt::runtime::Tensor, std::allocator<tt::runtime::Tensor> >&)
            # E        --- tt::runtime::submit(tt::runtime::Device, tt::runtime::Binary, unsigned int, std::vector<tt::runtime::Tensor, std::allocator<tt::runtime::Tensor> >&)
            # E        --- tt::run_program(tt::runtime::Binary&, int, std::vector<tt::Tensor, std::allocator<tt::Tensor> >&)
            # E        --- tt::ModelState::run_program(tt::ProgramType, std::vector<tt::Tensor, std::allocator<tt::Tensor> >)
            # /opt/ttforge-toolchain/venv/lib/python3.10/site-packages/forge/compiled_graph_state.py:316: RuntimeError
            ExceptionCheck(
                class_name="RuntimeError",
                component=ComponentChecker.TTNN.value,
                message=[
                    M.contains("Inputs must be of bfloat16 or bfloat8_b type"),
                    M.contains("input.dtype() == DataType::BFLOAT16 || input.dtype() == DataType::BFLOAT8_B"),
                ],
                error_log=[
                    M.last_line(M.contains("forge/compiled_graph_state.py:")),
                ],
            ),
        ],
    )

    CONV2D_VALIDATE_ARGS = FailingReason(
        description="Validating Conv2d dilation args",
        checks=[
            #     def populate_conv2d_args(graph, nid, compiler_cfg):
            #         ...
            #         dilation = [int(dilation) for dilation in node["attrs"]["dilation"][0]]
            # >       assert all([dim == dilation[0] for dim in dilation])
            # E       AssertionError
            # forge/forge/tvm_to_python.py:567: AssertionError
            ExceptionCheck(
                class_name="AssertionError",
                component=ComponentChecker.TVM.value,
                message=[],
                error_log=[
                    M.contains("def populate_conv2d_args("),
                    M.contains(">       assert all([dim == dilation[0] for dim in dilation])"),
                    M.last_line(M.contains("forge/tvm_to_python.py:")),
                ],
            ),
        ],
    )

    BUGGY_SHAPE_OLD = FailingReason(
        description="Buggy shape old",
        checks=[
            ExceptionCheck(
                # max	ValueError: Shape mismatch: framework_model.shape=torch.Size([a, b]), compiled_model.shape=torch.Size([a, b])
                # >                   raise ValueError(f"Shape mismatch: framework_model.shape={fw.shape}, compiled_model.shape={co.shape}")
                # E                   ValueError: Shape mismatch: framework_model.shape=torch.Size([1, 10000]), compiled_model.shape=torch.Size([10000, 1])
                # forge/forge/verify/verify.py:481: ValueError
                class_name="ValueError",
                # component=ComponentChecker.FORGE.value,
                component=ComponentChecker.NONE.value,  # TODO remove
                message=[
                    M.regex("Shape mismatch: framework_model.shape=torch.Size.*, compiled_model.shape=torch.Size.*"),
                ],
                error_log=[
                    M.last_line(M.contains("forge/verify/verify.py:")),
                ],
            ),
        ],
    )

    UNSUPPORTED_DIMENSION = FailingReason(
        description="Unsupported dimension",
    )

    NORMALIZATION_ONLY_LAST_DIM = FailingReason(
        description="Normalization only over last dimension",
        checks=[
            # layer_norm	AssertionError: Support only normalization over last one dimension.
            # >       assert ndims == 1, "Support only normalization over last one dimension."
            # E       AssertionError: Support only normalization over last one dimension.
            # third_party/tvm/python/tvm/relay/frontend/pytorch.py:1662: AssertionError
            ExceptionCheck(
                class_name="AssertionError",
                component=ComponentChecker.TVM.value,
                message=[
                    M.starts_with("Support only normalization over last one dimension."),
                ],
                error_log=[
                    M.contains(">       assert ndims == 1"),
                    M.last_line(M.contains("/tvm/relay/frontend/pytorch.py:")),
                ],
            ),
        ],
    )

    # UNSUPPORTED_PARAMETER_VALUE = FailingReason(
    #     description="Unsupported parameter value",
    # )

    # UNSUPPORTED_TYPE_FOR_VALIDATION = FailingReason(
    #     description="Verification failed due to unsupported type in verify_module",
    # )

    # # "Fatal python error - xfail does not work; UserWarning: resource_tracker: There appear to be 26 leaked semaphore objects to clean up at shutdown"
    # # "Fatal python error - xfail does not work. Error message: Fatal Python error: Segmentation fault; UserWarning: resource_tracker: There appear to be 26 leaked semaphore objects to clean up at shutdown"
    # SEMAPHORE_LEAK = "Semaphore leak"
    # SEMAPHORE_LEAK = FailingReason(
    #     description="Semaphore leak",
    # )

    RELAY_OP_SHAPE_CHECK = FailingReason(
        description="Relay operator shape check",
        checks=[
            # matmul	tvm.error.InternalError: Traceback (most recent call last):
            # E       tvm.error.InternalError: Traceback (most recent call last):
            # E         8: tvm::runtime::PackedFuncObj::Extractor<tvm::runtime::PackedFuncSubObj<tvm::runtime::TypedPackedFunc<tvm::IRModule (tvm::transform::Pass, tvm::IRModule)>::AssignTypedLambda<tvm::transform::__mk_TVM9::{lambda(tvm::transform::Pass, tvm::IRModule)#1}>(tvm::transform::__mk_TVM9::{lambda(tvm::transform::Pass, tvm::IRModule)#1}, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}> >::Call(tvm::runtime::PackedFuncObj const*, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, tvm::runtime::TVMRetValue)
            # E         7: tvm::transform::Pass::operator()(tvm::IRModule) const
            # E         6: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
            # E         5: tvm::transform::ModulePassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
            # E         4: tvm::runtime::PackedFuncObj::Extractor<tvm::runtime::PackedFuncSubObj<tvm::runtime::TypedPackedFunc<tvm::IRModule (tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::relay::transform::InferType()::{lambda(tvm::IRModule, tvm::transform::PassContext const&)#1}>(tvm::relay::transform::InferType()::{lambda(tvm::IRModule, tvm::transform::PassContext const&)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}> >::Call(tvm::runtime::PackedFuncObj const*, tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)
            # E         3: tvm::relay::TypeInferencer::Infer(tvm::GlobalVar, tvm::relay::Function)
            # E         2: tvm::relay::TypeSolver::Solve()
            # E         1: tvm::runtime::PackedFuncObj::Extractor<tvm::runtime::PackedFuncSubObj<tvm::runtime::TypedPackedFunc<bool (tvm::runtime::Array<tvm::Type, void> const&, int, tvm::Attrs const&, tvm::TypeReporter const&)>::AssignTypedLambda<bool (*)(tvm::runtime::Array<tvm::Type, void> const&, int, tvm::Attrs const&, tvm::TypeReporter const&)>(bool (*)(tvm::runtime::Array<tvm::Type, void> const&, int, tvm::Attrs const&, tvm::TypeReporter const&))::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}> >::Call(tvm::runtime::PackedFuncObj const*, tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)
            # E         0: bool tvm::relay::MatmulRel<tvm::relay::MatmulAttrs>(tvm::runtime::Array<tvm::Type, void> const&, int, tvm::Attrs const&, tvm::TypeReporter const&)
            # E         File "/localdev/vbrkic/src/forge/tt-forge-fe/third_party/tvm/src/relay/op/nn/nn.h", line 109
            # E       InternalError: Check failed: (static_cast<int>(tensor_b->shape.size()) == 2) is false:
            # third_party/tvm/python/tvm/_ffi/base.py:479: InternalError
            ExceptionCheck(
                class_name="tvm.error.InternalError",
                component=ComponentChecker.TVM.value,
                error_log=[
                    M.last_line(M.contains("/tvm/_ffi/base.py:")),
                    M.contains(
                        "E       InternalError: Check failed: (static_cast<int>(tensor_b->shape.size()) == 2) is false:"
                    ),
                ],
            ),
        ],
    )

    CANNOT_SQUEEZE_AXIS_NOT_ONE = FailingReason(
        description="Cannot squeeze axis with dimension not equal to 1",
        checks=[
            # squeeze	tvm.error.InternalError: Traceback (most recent call last):
            # >       raise py_err
            # E       tvm.error.InternalError: Traceback (most recent call last):
            # E         8: tvm::runtime::PackedFuncObj::Extractor<tvm::runtime::PackedFuncSubObj<tvm::runtime::TypedPackedFunc<tvm::IRModule (tvm::transform::Pass, tvm::IRModule)>::AssignTypedLambda<tvm::transform::__mk_TVM9::{lambda(tvm::transform::Pass, tvm::IRModule)#1}>(tvm::transform::__mk_TVM9::{lambda(tvm::transform::Pass, tvm::IRModule)#1}, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}> >::Call(tvm::runtime::PackedFuncObj const*, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, tvm::runtime::TVMRetValue)
            # E         7: tvm::transform::Pass::operator()(tvm::IRModule) const
            # E         6: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
            # E         5: tvm::transform::ModulePassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
            # E         4: tvm::runtime::PackedFuncObj::Extractor<tvm::runtime::PackedFuncSubObj<tvm::runtime::TypedPackedFunc<tvm::IRModule (tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::relay::transform::InferType()::{lambda(tvm::IRModule, tvm::transform::PassContext const&)#1}>(tvm::relay::transform::InferType()::{lambda(tvm::IRModule, tvm::transform::PassContext const&)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}> >::Call(tvm::runtime::PackedFuncObj const*, tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)
            # E         3: tvm::relay::TypeInferencer::Infer(tvm::GlobalVar, tvm::relay::Function)
            # E         2: tvm::relay::TypeSolver::Solve()
            # E         1: tvm::runtime::PackedFuncObj::Extractor<tvm::runtime::PackedFuncSubObj<tvm::runtime::TypedPackedFunc<bool (tvm::runtime::Array<tvm::Type, void> const&, int, tvm::Attrs const&, tvm::TypeReporter const&)>::AssignTypedLambda<bool (*)(tvm::runtime::Array<tvm::Type, void> const&, int, tvm::Attrs const&, tvm::TypeReporter const&)>(bool (*)(tvm::runtime::Array<tvm::Type, void> const&, int, tvm::Attrs const&, tvm::TypeReporter const&))::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}> >::Call(tvm::runtime::PackedFuncObj const*, tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)
            # E         0: tvm::relay::SqueezeRel(tvm::runtime::Array<tvm::Type, void> const&, int, tvm::Attrs const&, tvm::TypeReporter const&)
            # E         File "/localdev/vbrkic/src/forge/tt-forge-fe/third_party/tvm/src/relay/op/tensor/transform.cc", line 2507
            # E       InternalError: Check failed: *axis_ptr == 1 (2 vs. 1) : cannot squeeze axis with dimension not equal to 1
            # third_party/tvm/python/tvm/_ffi/base.py:479: InternalError
            ExceptionCheck(
                class_name="tvm.error.InternalError",
                component=ComponentChecker.TVM.value,
                message=[
                    M.starts_with("Traceback (most recent call last):"),
                ],
                error_log=[
                    M.regex("Check failed: .* : cannot squeeze axis with dimension not equal to 1"),
                    M.contains("third_party/tvm/src/relay/op/tensor/transform.cc"),
                    M.contains(">       raise py_err"),
                    M.last_line(M.contains("/tvm/_ffi/base.py:")),
                ],
            ),
        ],
    )

    TENSOR_TRANSFORM_SHAPE_CHECK = FailingReason(
        description="Tensor transform shape check",
        checks=[
            # reshape	tvm.error.InternalError: Traceback (most recent call last):
            # >       raise py_err
            # E       tvm.error.InternalError: Traceback (most recent call last):
            # E         9: tvm::runtime::PackedFuncObj::Extractor<tvm::runtime::PackedFuncSubObj<tvm::runtime::TypedPackedFunc<tvm::IRModule (tvm::transform::Pass, tvm::IRModule)>::AssignTypedLambda<tvm::transform::__mk_TVM9::{lambda(tvm::transform::Pass, tvm::IRModule)#1}>(tvm::transform::__mk_TVM9::{lambda(tvm::transform::Pass, tvm::IRModule)#1}, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}> >::Call(tvm::runtime::PackedFuncObj const*, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, tvm::runtime::TVMRetValue)
            # E         8: tvm::transform::Pass::operator()(tvm::IRModule) const
            # E         7: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
            # E         6: tvm::transform::ModulePassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
            # E         5: tvm::runtime::PackedFuncObj::Extractor<tvm::runtime::PackedFuncSubObj<tvm::runtime::TypedPackedFunc<tvm::IRModule (tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::relay::transform::InferType()::{lambda(tvm::IRModule, tvm::transform::PassContext const&)#1}>(tvm::relay::transform::InferType()::{lambda(tvm::IRModule, tvm::transform::PassContext const&)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}> >::Call(tvm::runtime::PackedFuncObj const*, tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)
            # E         4: tvm::relay::TypeInferencer::Infer(tvm::GlobalVar, tvm::relay::Function)
            # E         3: tvm::relay::TypeSolver::Solve()
            # E         2: tvm::runtime::PackedFuncObj::Extractor<tvm::runtime::PackedFuncSubObj<tvm::runtime::TypedPackedFunc<bool (tvm::runtime::Array<tvm::Type, void> const&, int, tvm::Attrs const&, tvm::TypeReporter const&)>::AssignTypedLambda<bool (*)(tvm::runtime::Array<tvm::Type, void> const&, int, tvm::Attrs const&, tvm::TypeReporter const&)>(bool (*)(tvm::runtime::Array<tvm::Type, void> const&, int, tvm::Attrs const&, tvm::TypeReporter const&))::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}> >::Call(tvm::runtime::PackedFuncObj const*, tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)
            # E         1: tvm::relay::ReshapeRel(tvm::runtime::Array<tvm::Type, void> const&, int, tvm::Attrs const&, tvm::TypeReporter const&)
            # E         0: tvm::relay::InferNewShape(tvm::runtime::Array<tvm::PrimExpr, void> const&, tvm::Attrs const&, bool)
            # E         File "/localdev/vbrkic/src/forge/tt-forge-fe/third_party/tvm/src/relay/op/tensor/transform.cc", line 698
            # E       InternalError: Check failed: src_idx < ishape.size() (2 vs. 1) :
            # third_party/tvm/python/tvm/_ffi/base.py:479: InternalError
            ExceptionCheck(
                class_name="tvm.error.InternalError",
                component=ComponentChecker.TVM.value,
                error_log=[
                    M.last_line(M.contains("/tvm/_ffi/base.py:")),
                    M.regex("E       InternalError: Check failed: src_idx < ishape.size\\(\\) \\(\\d+ vs\\. \\d+\\)"),
                ],
            ),
        ],
    )

    UNHANDLED_CASE_NONE = FailingReason(
        description="Unhandled case NoneType",
        checks=[
            # layer_norm	Exception: warning unhandled case:
            # >           raise Exception(f"warning unhandled case: {type(expr)}")
            # E           Exception: warning unhandled case: <class 'NoneType'>
            # third_party/tvm/python/tvm/relay/expr_functor.py:79: Exception
            ExceptionCheck(
                class_name="Exception",
                component=ComponentChecker.TVM.value,
                message=[
                    M.starts_with("warning unhandled case: <class 'NoneType'>"),
                ],
                error_log=[
                    M.last_line(M.contains("/tvm/relay/expr_functor.py:")),
                ],
            ),
        ],
    )

    # # # RuntimeError: Fatal Python error: Segmentation fault
    # SEG_FAULT = FailingReason(
    #     description="Inference failed due to seg fault",
    # )

    # # RuntimeError: Fatal Python error: Aborted
    FATAL_ERROR = FailingReason(
        description="Fatal error occured",
    )

    HIGH_MEMORY = FailingReason(
        description="High memory usage",
    )

    # UNSUPPORTED_INPUT_SOURCE = FailingReason(
    #     description="Unsupported input source",
    # )

    INFERENCE_FROZE = FailingReason(
        description="Inference froze without error message",
    )

    TTNN_RUNTIME_OLD = FailingReason(
        description="TTNN runtime error old",
        checks=[
            # cumsum	RuntimeError: TT_ASSERT @ ttnn/cpp/ttnn/operations/moreh/moreh_cumsum/device/moreh_cumsum_program_factory.cpp:23: dim == 0 || dim == 1
            # >       self.runtime_model_state.run_program(ProgramType.Forward, self.inputs)
            # E       RuntimeError: TT_ASSERT @ /proj_sw/user_dev/vbrkic/src_bgd/ttforge/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/moreh/moreh_cumsum/device/moreh_cumsum_program_factory.cpp:23: dim == 0 || dim == 1
            # E       backtrace:
            # E        --- ttnn::operations::moreh::moreh_cumsum::MorehCumsumDeviceOperation::ProgramFactory::create(ttnn::operations::moreh::moreh_cumsum::MorehCumsumDeviceOperation::operation_attributes_t const&, ttnn::operations::moreh::moreh_cumsum::MorehCumsumDeviceOperation::tensor_args_t const&, tt::tt_metal::Tensor&)
            # E        --- ttnn::operations::moreh::moreh_cumsum::MorehCumsumDeviceOperation::tensor_return_value_t ttnn::device_operation::detail::launch_on_single_device<ttnn::operations::moreh::moreh_cumsum::MorehCumsumDeviceOperation>(tt::stl::StrongType<unsigned char, ttnn::QueueIdTag>, ttnn::operations::moreh::moreh_cumsum::MorehCumsumDeviceOperation::operation_attributes_t const&, ttnn::operations::moreh::moreh_cumsum::MorehCumsumDeviceOperation::tensor_args_t const&)
            # E        --- ttnn::operations::moreh::moreh_cumsum::MorehCumsumDeviceOperation::tensor_return_value_t ttnn::device_operation::detail::invoke<ttnn::operations::moreh::moreh_cumsum::MorehCumsumDeviceOperation>(tt::stl::StrongType<unsigned char, ttnn::QueueIdTag>, ttnn::operations::moreh::moreh_cumsum::MorehCumsumDeviceOperation::operation_attributes_t const&, ttnn::operations::moreh::moreh_cumsum::MorehCumsumDeviceOperation::tensor_args_t const&)
            # E        --- ttnn::operations::moreh::moreh_cumsum::MorehCumsum::invoke(tt::tt_metal::Tensor const&, long, std::optional<tt::tt_metal::Tensor> const&, std::optional<tt::tt_metal::MemoryConfig> const&, std::optional<std::variant<ttnn::GrayskullComputeKernelConfig, ttnn::WormholeComputeKernelConfig> > const&)
            # E        --- void tt::tt_metal::operation::launch_op_func<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >(std::function<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > (std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > const&, std::vector<std::optional<tt::tt_metal::Tensor const>, std::allocator<std::optional<tt::tt_metal::Tensor const> > > const&, std::vector<std::optional<tt::tt_metal::Tensor>, std::allocator<std::optional<tt::tt_metal::Tensor> > > const&)> const&, std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> >, std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> >&, std::vector<std::optional<tt::tt_metal::Tensor const>, std::allocator<std::optional<tt::tt_metal::Tensor const> > >, std::vector<std::optional<tt::tt_metal::Tensor>, std::allocator<std::optional<tt::tt_metal::Tensor> > >)
            # E        --- tt::runtime::ttnn::operations::moreh::run(tt::target::ttnn::MorehCumSumOp const*, tt::runtime::ttnn::ProgramContext&)
            # E        --- tt::runtime::ttnn::runProgram(tt::tt_metal::distributed::MeshDevice&, tt::runtime::Binary, unsigned int, std::vector<tt::runtime::Tensor, std::allocator<tt::runtime::Tensor> >&)
            # E        --- tt::runtime::ttnn::submit(tt::runtime::Device, tt::runtime::Binary, unsigned int, std::vector<tt::runtime::Tensor, std::allocator<tt::runtime::Tensor> >&)
            # E        --- tt::runtime::submit(tt::runtime::Device, tt::runtime::Binary, unsigned int, std::vector<tt::runtime::Tensor, std::allocator<tt::runtime::Tensor> >&)
            # E        --- tt::run_program(tt::runtime::Binary&, int, std::vector<tt::Tensor, std::allocator<tt::Tensor> >&)
            # E        --- tt::ModelState::run_program(tt::ProgramType, std::vector<tt::Tensor, std::allocator<tt::Tensor> >)
            # forge/forge/compiled_graph_state.py:310: RuntimeError
            ExceptionCheck(
                class_name="RuntimeError",
                # component=ComponentChecker.METAL.value,
                component=ComponentChecker.NONE.value,  # TODO remove
                message=[
                    M.starts_with("TT_ASSERT"),
                    M.regex(
                        "tt-metal/ttnn/cpp/ttnn/operations/moreh/moreh_cumsum/device/moreh_cumsum_program_factory.cpp:\\d+: dim == 0 || dim == 1"
                    ),
                ],
                error_log=[
                    M.contains("ttnn::operations::moreh::moreh_cumsum::MorehCumsumDeviceOperation"),
                    M.last_line(M.contains("forge/compiled_graph_state.py:")),
                ],
            ),
            # max	RuntimeError: TT_FATAL @ ttnn/cpp/ttnn/operations/data_movement/fill_pad/device/fill_pad_op.cpp:18: detail::data_type_to_size.count(input_tensor_a.get_dtype())	UNCLASSIFIED	6
            # sum	RuntimeError: TT_FATAL @ ttnn/cpp/ttnn/operations/data_movement/fill_pad/device/fill_pad_op.cpp:18: detail::data_type_to_size.count(input_tensor_a.get_dtype())	UNCLASSIFIED	36
            # >       self.runtime_model_state.run_program(ProgramType.Forward, self.inputs)
            # E       RuntimeError: TT_FATAL @ /proj_sw/user_dev/vbrkic/src_bgd/ttforge/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/data_movement/fill_pad/device/fill_pad_op.cpp:18: detail::data_type_to_size.count(input_tensor_a.get_dtype())
            # E       info:
            # E       Unsupported datatype
            # E       backtrace:
            # E        --- ttnn::operations::data_movement::FillPad::validate(std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > const&) const
            # E        --- tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >::validate_on_program_cache_miss(tt::tt_metal::operation::DeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > > const&, tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >::tensor_args_t const&)
            # E        --- tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >::tensor_return_value_t ttnn::device_operation::detail::invoke<tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > > >(tt::stl::StrongType<unsigned char, ttnn::QueueIdTag>, tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >::operation_attributes_t const&, tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >::tensor_args_t const&)
            # E        --- std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > tt::tt_metal::operation::run<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >(tt::tt_metal::operation::DeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >&&, std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > const&, std::vector<std::optional<tt::tt_metal::Tensor const>, std::allocator<std::optional<tt::tt_metal::Tensor const> > > const&, std::vector<std::optional<tt::tt_metal::Tensor>, std::allocator<std::optional<tt::tt_metal::Tensor> > > const&, tt::stl::StrongType<unsigned char, ttnn::QueueIdTag>)
            # E        --- std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > tt::tt_metal::operation::run_without_autoformat<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >(tt::tt_metal::operation::DeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >&&, std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > const&, std::vector<std::optional<tt::tt_metal::Tensor const>, std::allocator<std::optional<tt::tt_metal::Tensor const> > > const&, std::vector<std::optional<tt::tt_metal::Tensor>, std::allocator<std::optional<tt::tt_metal::Tensor> > > const&, tt::stl::StrongType<unsigned char, ttnn::QueueIdTag>)
            # E        --- ttnn::operations::data_movement::FillPadOperation::invoke(tt::stl::StrongType<unsigned char, ttnn::QueueIdTag>, tt::tt_metal::Tensor const&, float, std::optional<tt::tt_metal::MemoryConfig> const&)
            # E        --- void tt::tt_metal::operation::launch_op_func<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >(std::function<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > (std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > const&, std::vector<std::optional<tt::tt_metal::Tensor const>, std::allocator<std::optional<tt::tt_metal::Tensor const> > > const&, std::vector<std::optional<tt::tt_metal::Tensor>, std::allocator<std::optional<tt::tt_metal::Tensor> > > const&)> const&, std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> >, std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> >&, std::vector<std::optional<tt::tt_metal::Tensor const>, std::allocator<std::optional<tt::tt_metal::Tensor const> > >, std::vector<std::optional<tt::tt_metal::Tensor>, std::allocator<std::optional<tt::tt_metal::Tensor> > >)
            # E        --- tt::runtime::ttnn::operations::reduction::run(tt::target::ttnn::ReductionOp const*, tt::runtime::ttnn::ProgramContext&)
            # E        --- tt::runtime::ttnn::runProgram(tt::tt_metal::distributed::MeshDevice&, tt::runtime::Binary, unsigned int, std::vector<tt::runtime::Tensor, std::allocator<tt::runtime::Tensor> >&)
            # E        --- tt::runtime::ttnn::submit(tt::runtime::Device, tt::runtime::Binary, unsigned int, std::vector<tt::runtime::Tensor, std::allocator<tt::runtime::Tensor> >&)
            # E        --- tt::runtime::submit(tt::runtime::Device, tt::runtime::Binary, unsigned int, std::vector<tt::runtime::Tensor, std::allocator<tt::runtime::Tensor> >&)
            # E        --- tt::run_program(tt::runtime::Binary&, int, std::vector<tt::Tensor, std::allocator<tt::Tensor> >&)
            # E        --- tt::ModelState::run_program(tt::ProgramType, std::vector<tt::Tensor, std::allocator<tt::Tensor> >)
            # forge/forge/compiled_graph_state.py:310: RuntimeError
            ExceptionCheck(
                class_name="RuntimeError",
                # component=ComponentChecker.METAL.value,
                component=ComponentChecker.NONE.value,  # TODO remove
                message=[
                    M.starts_with("TT_FATAL"),
                    M.regex(
                        "tt-metal/ttnn/cpp/ttnn/operations/data_movement/fill_pad/device/fill_pad_op.cpp:\\d+: detail::data_type_to_size.count\\(input_tensor_a.get_dtype\\(\\)\\)"
                    ),
                ],
                error_log=[
                    M.contains("Unsupported datatype"),
                    M.contains("forge/compiled_graph_state.py:"),
                ],
            ),
            # clamp	RuntimeError: TT_THROW @ ttnn/cpp/ttnn/operations/creation.hpp:182: tt::exception
            # >       self.runtime_model_state.run_program(ProgramType.Forward, self.inputs)
            # E       RuntimeError: TT_THROW @ /proj_sw/user_dev/vbrkic/src_bgd/ttforge/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/creation.hpp:182: tt::exception
            # E       info:
            # E       Unsupported DataType!
            # E       backtrace:
            # E        --- ttnn::operations::unary::ExecuteUnaryCompositeClamp::invoke(tt::tt_metal::Tensor const&, std::optional<float>, std::optional<float>, std::optional<tt::tt_metal::MemoryConfig> const&)
            # E        --- void tt::tt_metal::operation::launch_op_func<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >(std::function<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > (std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > const&, std::vector<std::optional<tt::tt_metal::Tensor const>, std::allocator<std::optional<tt::tt_metal::Tensor const> > > const&, std::vector<std::optional<tt::tt_metal::Tensor>, std::allocator<std::optional<tt::tt_metal::Tensor> > > const&)> const&, std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> >, std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> >&, std::vector<std::optional<tt::tt_metal::Tensor const>, std::allocator<std::optional<tt::tt_metal::Tensor const> > >, std::vector<std::optional<tt::tt_metal::Tensor>, std::allocator<std::optional<tt::tt_metal::Tensor> > >)
            # E        --- tt::runtime::ttnn::operations::eltwise::unary::run(tt::target::ttnn::EltwiseUnaryCompositeOp const*, tt::runtime::ttnn::ProgramContext&)
            # E        --- tt::runtime::ttnn::runProgram(tt::tt_metal::distributed::MeshDevice&, tt::runtime::Binary, unsigned int, std::vector<tt::runtime::Tensor, std::allocator<tt::runtime::Tensor> >&)
            # E        --- tt::runtime::ttnn::submit(tt::runtime::Device, tt::runtime::Binary, unsigned int, std::vector<tt::runtime::Tensor, std::allocator<tt::runtime::Tensor> >&)
            # E        --- tt::runtime::submit(tt::runtime::Device, tt::runtime::Binary, unsigned int, std::vector<tt::runtime::Tensor, std::allocator<tt::runtime::Tensor> >&)
            # E        --- tt::run_program(tt::runtime::Binary&, int, std::vector<tt::Tensor, std::allocator<tt::Tensor> >&)
            # E        --- tt::ModelState::run_program(tt::ProgramType, std::vector<tt::Tensor, std::allocator<tt::Tensor> >)
            # forge/forge/compiled_graph_state.py:310: RuntimeError
            ExceptionCheck(
                class_name="RuntimeError",
                # component=ComponentChecker.TTNN.value,
                component=ComponentChecker.NONE.value,  # TODO remove
                message=[
                    M.starts_with("TT_THROW"),
                    M.any(
                        M.regex("tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/creation.hpp:.*: tt::exception"),
                    ),
                ],
                error_log=[
                    M.contains("Unsupported DataType!"),
                    M.last_line(M.contains("forge/compiled_graph_state.py:")),
                ],
            ),
        ],
    )

    UNSUPPORTED_INPUT_SHAPE_MATMUL = FailingReason(
        description="Non broadcast shape",
        checks=[
            # matmul	matmul	RuntimeError: TT_FATAL @ tt-metal/ttnn/cpp/ttnn/operations/matmul/device/matmul_op.cpp:1479: a_shape[i] == b_shape[i]
            # >       self.runtime_model_state.run_program(ProgramType.Forward, self.inputs)
            # E       RuntimeError: TT_FATAL @ /proj_sw/user_dev/vbrkic/src_bgd/ttforge/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/matmul_op.cpp:1538: a_shape[i] == b_shape[i]
            # E       info:
            # E       bmm (non-bcast matmul) expects input tensors of shapes BCMK*BCKN=BCMN or equivalent
            # E       backtrace:
            # E        --- ttnn::operations::matmul::Matmul::validate(std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > const&, std::vector<std::optional<tt::tt_metal::Tensor const>, std::allocator<std::optional<tt::tt_metal::Tensor const> > > const&, std::vector<std::optional<tt::tt_metal::Tensor>, std::allocator<std::optional<tt::tt_metal::Tensor> > > const&) const
            # E        --- tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >::validate_on_program_cache_miss(tt::tt_metal::operation::DeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > > const&, tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >::tensor_args_t const&)
            # E        --- void ttnn::device_operation::detail::create_and_cache_mesh_workload<ttnn::device_operation::MeshDeviceOperationAdapter<tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > > > >(tt::stl::StrongType<unsigned char, ttnn::QueueIdTag>, ttnn::device_operation::MeshDeviceOperationAdapter<tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > > >::operation_attributes_t const&)
            # E        --- tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >::tensor_return_value_t ttnn::device_operation::detail::invoke<tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > > >(tt::stl::StrongType<unsigned char, ttnn::QueueIdTag>, tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >::operation_attributes_t const&, tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >::tensor_args_t const&)
            # E        --- std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > tt::tt_metal::operation::run<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >(tt::tt_metal::operation::DeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >&&, std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > const&, std::vector<std::optional<tt::tt_metal::Tensor const>, std::allocator<std::optional<tt::tt_metal::Tensor const> > > const&, std::vector<std::optional<tt::tt_metal::Tensor>, std::allocator<std::optional<tt::tt_metal::Tensor> > > const&, tt::stl::StrongType<unsigned char, ttnn::QueueIdTag>)
            # E        --- ttnn::operations::matmul::matmul(tt::tt_metal::Tensor const&, tt::tt_metal::Tensor const&, std::optional<tt::tt_metal::Tensor const> const&, ttnn::operations::matmul::Matmul const&, tt::stl::StrongType<unsigned char, ttnn::QueueIdTag>, std::optional<tt::tt_metal::Tensor> const&)
            # E        --- ttnn::operations::matmul::bound_matmul(tt::tt_metal::Tensor const&, tt::tt_metal::Tensor const&, std::optional<tt::tt_metal::Tensor const> const&, ttnn::operations::matmul::Matmul const&, unsigned char const&, std::optional<tt::tt_metal::Tensor>&)
            # E        --- tt::runtime::ttnn::operations::matmul::run(tt::target::ttnn::MatmulOp const*, tt::runtime::ttnn::ProgramContext&)
            # E        --- tt::runtime::ttnn::ProgramExecutor::runOperation(tt::target::ttnn::Operation const*)
            # E        --- tt::runtime::ttnn::ProgramExecutor::execute()
            # E        --- tt::runtime::ttnn::runProgram(std::shared_ptr<tt::tt_metal::distributed::MeshDevice>, tt::runtime::Binary, unsigned int, std::vector<tt::runtime::Tensor, std::allocator<tt::runtime::Tensor> >&)
            # E        --- tt::runtime::ttnn::submit(tt::runtime::Device, tt::runtime::Binary, unsigned int, std::vector<tt::runtime::Tensor, std::allocator<tt::runtime::Tensor> >&)
            # E        --- tt::runtime::submit(tt::runtime::Device, tt::runtime::Binary, unsigned int, std::vector<tt::runtime::Tensor, std::allocator<tt::runtime::Tensor> >&)
            # E        --- tt::run_program(tt::runtime::Binary&, int, std::vector<tt::Tensor, std::allocator<tt::Tensor> >&)
            # E        --- tt::ModelState::run_program(tt::ProgramType, std::vector<tt::Tensor, std::allocator<tt::Tensor> >)
            # forge/forge/compiled_graph_state.py:310: RuntimeError
            ExceptionCheck(
                class_name="RuntimeError",
                component=ComponentChecker.TTNN.value,
                message=[
                    M.starts_with("TT_FATAL"),
                    # E       RuntimeError: TT_FATAL @ ttforge/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/matmul_op.cpp:1538: a_shape[i] == b_shape[i]
                    M.contains("tt-metal/ttnn/cpp/ttnn/operations/matmul/device/matmul_op.cpp:"),
                    M.contains("a_shape[i] == b_shape[i]"),
                ],
                error_log=[
                    M.contains("bmm (non-bcast matmul) expects input tensors of shapes BCMK*BCKN=BCMN or equivalent"),
                    M.contains("ttnn::operations::matmul::Matmul::validate"),
                    M.last_line(M.contains("forge/compiled_graph_state.py:")),
                ],
            ),
        ],
    )

    MLIR_RUNTIME_OLD = FailingReason(
        description="MLIR runtime error old",
        checks=[
            # transpose	RuntimeError: TT_FATAL @ tt-metal/ttnn/cpp/ttnn/operations/data_movement/transpose/device/transpose_op.cpp:120: input_tensor.get_dtype() == DataType::BFLOAT16 || input_tensor.get_dtype() == DataType::
            # >       self.runtime_model_state.run_program(ProgramType.Forward, self.inputs)
            # E       RuntimeError: TT_FATAL @ /proj_sw/user_dev/vbrkic/src_bgd/ttforge/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/data_movement/transpose/device/transpose_op.cpp:120: input_tensor.get_dtype() == DataType::BFLOAT16 || input_tensor.get_dtype() == DataType::FLOAT32
            # E       info:
            # E       Error
            # E       backtrace:
            # E        --- ttnn::operations::data_movement::Transpose::validate(std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > const&) const
            # E        --- tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >::validate_on_program_cache_miss(tt::tt_metal::operation::DeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > > const&, tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >::tensor_args_t const&)
            # E        --- tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >::tensor_return_value_t ttnn::device_operation::detail::invoke<tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > > >(tt::stl::StrongType<unsigned char, ttnn::QueueIdTag>, tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >::operation_attributes_t const&, tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >::tensor_args_t const&)
            # E        --- std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > tt::tt_metal::operation::run<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >(tt::tt_metal::operation::DeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >&&, std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > const&, std::vector<std::optional<tt::tt_metal::Tensor const>, std::allocator<std::optional<tt::tt_metal::Tensor const> > > const&, std::vector<std::optional<tt::tt_metal::Tensor>, std::allocator<std::optional<tt::tt_metal::Tensor> > > const&, tt::stl::StrongType<unsigned char, ttnn::QueueIdTag>)
            # E        --- void tt::tt_metal::operation::launch_op_func<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >(std::function<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > (std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > const&, std::vector<std::optional<tt::tt_metal::Tensor const>, std::allocator<std::optional<tt::tt_metal::Tensor const> > > const&, std::vector<std::optional<tt::tt_metal::Tensor>, std::allocator<std::optional<tt::tt_metal::Tensor> > > const&)> const&, std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> >, std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> >&, std::vector<std::optional<tt::tt_metal::Tensor const>, std::allocator<std::optional<tt::tt_metal::Tensor const> > >, std::vector<std::optional<tt::tt_metal::Tensor>, std::allocator<std::optional<tt::tt_metal::Tensor> > >)
            # E        --- void tt::tt_metal::operation::launch_op<std::function<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > (std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > const&, std::vector<std::optional<tt::tt_metal::Tensor const>, std::allocator<std::optional<tt::tt_metal::Tensor const> > > const&, std::vector<std::optional<tt::tt_metal::Tensor>, std::allocator<std::optional<tt::tt_metal::Tensor> > > const&)>, std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >(std::function<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > (std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > const&, std::vector<std::optional<tt::tt_metal::Tensor const>, std::allocator<std::optional<tt::tt_metal::Tensor const> > > const&, std::vector<std::optional<tt::tt_metal::Tensor>, std::allocator<std::optional<tt::tt_metal::Tensor> > > const&)>&&, std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> >, std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> >&, std::vector<std::optional<tt::tt_metal::Tensor const>, std::allocator<std::optional<tt::tt_metal::Tensor const> > >, std::vector<std::optional<tt::tt_metal::Tensor>, std::allocator<std::optional<tt::tt_metal::Tensor> > >)
            # E        --- tt::tt_metal::operation::launch_with_autoformat(std::function<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > (std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > const&, std::vector<std::optional<tt::tt_metal::Tensor const>, std::allocator<std::optional<tt::tt_metal::Tensor const> > > const&, std::vector<std::optional<tt::tt_metal::Tensor>, std::allocator<std::optional<tt::tt_metal::Tensor> > > const&)>&&, std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > const&, std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> >&, std::vector<std::optional<tt::tt_metal::Tensor const>, std::allocator<std::optional<tt::tt_metal::Tensor const> > > const&, std::vector<std::optional<tt::tt_metal::Tensor>, std::allocator<std::optional<tt::tt_metal::Tensor> > > const&)
            # E        --- ttnn::operations::data_movement::ExecuteTranspose::invoke(tt::stl::StrongType<unsigned char, ttnn::QueueIdTag>, tt::tt_metal::Tensor const&, long const&, long const&, std::optional<tt::tt_metal::MemoryConfig> const&, std::optional<float> const&)
            # E        --- ttnn::operations::data_movement::ExecuteTranspose::invoke(tt::tt_metal::Tensor const&, long const&, long const&, std::optional<tt::tt_metal::MemoryConfig> const&, std::optional<float> const&)
            # E        --- tt::runtime::ttnn::operations::data_movement::run(tt::target::ttnn::TransposeOp const*, tt::runtime::ttnn::ProgramContext&)
            # E        --- tt::runtime::ttnn::ProgramExecutor::runOperation(tt::target::ttnn::Operation const*)
            # E        --- tt::runtime::ttnn::ProgramExecutor::execute()
            # E        --- tt::runtime::ttnn::runProgram(tt::tt_metal::distributed::MeshDevice&, tt::runtime::Binary, unsigned int, std::vector<tt::runtime::Tensor, std::allocator<tt::runtime::Tensor> >&)
            # E        --- tt::runtime::ttnn::submit(tt::runtime::Device, tt::runtime::Binary, unsigned int, std::vector<tt::runtime::Tensor, std::allocator<tt::runtime::Tensor> >&)
            # E        --- tt::runtime::submit(tt::runtime::Device, tt::runtime::Binary, unsigned int, std::vector<tt::runtime::Tensor, std::allocator<tt::runtime::Tensor> >&)
            # E        --- tt::run_program(tt::runtime::Binary&, int, std::vector<tt::Tensor, std::allocator<tt::Tensor> >&)
            # E        --- tt::ModelState::run_program(tt::ProgramType, std::vector<tt::Tensor, std::allocator<tt::Tensor> >)
            # forge/forge/compiled_graph_state.py:310: RuntimeError
            ExceptionCheck(
                class_name="RuntimeError",
                # component=ComponentChecker.TTNN.value,
                component=ComponentChecker.NONE.value,  # TODO remove
                message=[
                    M.starts_with("TT_FATAL"),
                    M.any(
                        M.regex(
                            "tt-metal/ttnn/cpp/ttnn/operations/data_movement/transpose/device/transpose_op.cpp:\\d+: input_tensor.get_dtype\\(\\) == DataType::BFLOAT16 || input_tensor.get_dtype\\(\\) == DataType::"
                        ),
                    ),
                ],
                error_log=[
                    M.contains("ttnn::operations::data_movement::Transpose::validate"),
                    M.last_line(M.contains("forge/compiled_graph_state.py:")),
                ],
            ),
        ],
    )

    INVALID_ARGUMENTS_TO_RESHAPE = FailingReason(
        description="Invalid arguments to reshape",
        checks=[
            # >       self.runtime_model_state.run_program(ProgramType.Forward, self.inputs)
            # E       RuntimeError: TT_FATAL @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/core/tensor/tensor_utils.cpp:54: new_volume == old_volume
            # E       info:
            # E       Invalid arguments to reshape
            # E       backtrace:
            # E        --- tt::tt_metal::infer_dims_for_reshape(tt::tt_metal::Tensor const&, tt::stl::Span<int const, 18446744073709551615ul>)
            # E        --- ttnn::operations::data_movement::ReshapeViewOperation::invoke(tt::stl::StrongType<unsigned char, ttnn::QueueIdTag>, tt::tt_metal::Tensor const&, tt::stl::Span<int const, 18446744073709551615ul>, std::optional<tt::tt_metal::MemoryConfig> const&, std::optional<std::variant<unsigned int, float> > const&)
            # E        --- tt::runtime::ttnn::operations::data_movement::run(tt::target::ttnn::ReshapeOp const*, tt::runtime::ttnn::ProgramContext&)
            # E        --- tt::runtime::ttnn::ProgramExecutor::execute()
            # E        --- tt::runtime::ttnn::runProgram(std::shared_ptr<tt::tt_metal::distributed::MeshDevice>, tt::runtime::Binary, unsigned int, std::vector<tt::runtime::Tensor, std::allocator<tt::runtime::Tensor> >&)
            # E        --- tt::runtime::ttnn::submit(tt::runtime::Device, tt::runtime::Binary, unsigned int, std::vector<tt::runtime::Tensor, std::allocator<tt::runtime::Tensor> >&)
            # E        --- tt::runtime::submit(tt::runtime::Device, tt::runtime::Binary, unsigned int, std::vector<tt::runtime::Tensor, std::allocator<tt::runtime::Tensor> >&)
            # E        --- tt::run_program(tt::runtime::Binary&, int, std::vector<tt::Tensor, std::allocator<tt::Tensor> >&)
            # E        --- tt::ModelState::run_program(tt::ProgramType, std::vector<tt::Tensor, std::allocator<tt::Tensor> >)
            # /opt/ttforge-toolchain/venv/lib/python3.10/site-packages/forge/compiled_graph_state.py:310: RuntimeError
            ExceptionCheck(
                class_name="RuntimeError",
                component=ComponentChecker.MLIR.value,
                message=[
                    M.starts_with("TT_FATAL"),
                    M.contains("new_volume == old_volume"),
                ],
                error_log=[
                    M.contains("Invalid arguments to reshape"),
                    M.last_line(M.contains("forge/compiled_graph_state.py:")),
                ],
            ),
        ],
    )

    FORGE_RUNTIME_OLD = FailingReason(
        description="Forge runtime error old",
        checks=[
            # max	RuntimeError: TT_ASSERT @ tt-forge-fe/forge/csrc/graph_lib/shape.cpp:230: v.front() == 1
            # >       inserted_node_id_mapping, context.fracture_chip_id_assignments = run_post_initial_graph_passes(
            #             graph, compiler_cfg, compiler_cfg.fracture_groups
            #         )
            # E       RuntimeError: TT_ASSERT @ /__w/tt-forge-fe/tt-forge-fe/forge/csrc/graph_lib/shape.cpp:217: v.front() == 1
            # E       info:
            # E       Cannot squeeze a non-zero dim
            # E       backtrace:
            # E        --- tt::graphlib::Shape::as_rank(unsigned int) const
            # E        --- tt::graphlib::handle_change_rank(tt::graphlib::Graph*, tt::graphlib::Edge)
            # E        --- std::vector<std::pair<long, long>, std::allocator<std::pair<long, long> > > tt::decompose_tt_forge_graph<(tt::DecomposeEpoch)0>(tt::graphlib::Graph*, std::shared_ptr<void>)
            # E        --- tt::run_post_initial_graph_passes(tt::graphlib::Graph*, pybind11::object, std::vector<std::tuple<std::vector<std::tuple<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::vector<int, std::allocator<int> >, std::vector<int, std::allocator<int> > >, std::allocator<std::tuple<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::vector<int, std::allocator<int> >, std::vector<int, std::allocator<int> > > > >, std::unordered_map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::vector<int, std::allocator<int> >, std::hash<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::equal_to<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::vector<int, std::allocator<int> > > > > >, std::allocator<std::tuple<std::vector<std::tuple<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::vector<int, std::allocator<int> >, std::vector<int, std::allocator<int> > >, std::allocator<std::tuple<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::vector<int, std::allocator<int> >, std::vector<int, std::allocator<int> > > > >, std::unordered_map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::vector<int, std::allocator<int> >, std::hash<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::equal_to<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::vector<int, std::allocator<int> > > > > > > > const&)
            # forge/compile.py:731: RuntimeError
            ExceptionCheck(
                class_name="RuntimeError",
                # component=ComponentChecker.FORGE.value,
                component=ComponentChecker.NONE.value,  # TODO remove
                message=[
                    M.starts_with("TT_ASSERT"),
                    M.regex("forge/csrc/graph_lib/shape.cpp:\\d+: v.front\\(\\) == 1"),
                ],
                error_log=[
                    M.contains("Cannot squeeze a non-zero dim"),
                    M.last_line(M.contains("forge/compile.py:")),
                ],
            ),
        ],
    )

    VALUE_CONVERSION_OVERFLOW = FailingReason(
        description="Value conversion overflow",
        checks=[
            # clamp	RuntimeError: value cannot be converted to type at::BFloat16 without overflow
            # >       ref_output = self.cpp_op_type.eval(values)
            # E       RuntimeError: value cannot be converted to type at::BFloat16 without overflow
            # forge/op/common.py:61: RuntimeError
            ExceptionCheck(
                class_name="RuntimeError",
                component=ComponentChecker.FORGE.value,
                message=[
                    M.starts_with("value cannot be converted to type at::BFloat16 without overflow"),
                ],
                error_log=[
                    M.contains(">       ref_output = self.cpp_op_type.eval(values)"),
                    M.last_line(M.contains("forge/op/common.py:")),
                ],
            ),
        ],
    )

    UNASSIGNED_PADDING_VARIABLE_OLD = FailingReason(
        description="Unassigned padding variable old",
        checks=[
            # >       op_attrs["padding"] = padding
            # E       UnboundLocalError: local variable 'padding' referenced before assignment
            # forge/forge/tvm_calls/relay/op/forge_passes.py:197: UnboundLocalError
            ExceptionCheck(
                class_name="UnboundLocalError",
                component=ComponentChecker.NONE.value,  # TODO remove
                message=[
                    M.starts_with("local variable 'padding' referenced before assignment"),
                ],
                error_log=[
                    M.contains('>       op_attrs["padding"] = padding'),
                    M.last_line(M.contains("forge/tvm_calls/relay/op/forge_passes.py:")),
                ],
            ),
        ],
    )

    TVM_RUNTIME_OLD = FailingReason(
        description="TVM runtime error old",
        checks=[
            # squeeze	tvm.error.InternalError: Traceback (most recent call last):
            # >       raise py_err
            # E       tvm.error.InternalError: Traceback (most recent call last):
            # E         8: tvm::runtime::PackedFuncObj::Extractor<tvm::runtime::PackedFuncSubObj<tvm::runtime::TypedPackedFunc<tvm::IRModule (tvm::transform::Pass, tvm::IRModule)>::AssignTypedLambda<tvm::transform::__mk_TVM9::{lambda(tvm::transform::Pass, tvm::IRModule)#1}>(tvm::transform::__mk_TVM9::{lambda(tvm::transform::Pass, tvm::IRModule)#1}, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}> >::Call(tvm::runtime::PackedFuncObj const*, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, tvm::runtime::TVMRetValue)
            # E         7: tvm::transform::Pass::operator()(tvm::IRModule) const
            # E         6: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
            # E         5: tvm::transform::ModulePassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
            # E         4: tvm::runtime::PackedFuncObj::Extractor<tvm::runtime::PackedFuncSubObj<tvm::runtime::TypedPackedFunc<tvm::IRModule (tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::relay::transform::InferType()::{lambda(tvm::IRModule, tvm::transform::PassContext const&)#1}>(tvm::relay::transform::InferType()::{lambda(tvm::IRModule, tvm::transform::PassContext const&)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}> >::Call(tvm::runtime::PackedFuncObj const*, tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)
            # E         3: tvm::relay::TypeInferencer::Infer(tvm::GlobalVar, tvm::relay::Function)
            # E         2: tvm::relay::TypeSolver::Solve()
            # E         1: tvm::runtime::PackedFuncObj::Extractor<tvm::runtime::PackedFuncSubObj<tvm::runtime::TypedPackedFunc<bool (tvm::runtime::Array<tvm::Type, void> const&, int, tvm::Attrs const&, tvm::TypeReporter const&)>::AssignTypedLambda<bool (*)(tvm::runtime::Array<tvm::Type, void> const&, int, tvm::Attrs const&, tvm::TypeReporter const&)>(bool (*)(tvm::runtime::Array<tvm::Type, void> const&, int, tvm::Attrs const&, tvm::TypeReporter const&))::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}> >::Call(tvm::runtime::PackedFuncObj const*, tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)
            # E         0: tvm::relay::SqueezeRel(tvm::runtime::Array<tvm::Type, void> const&, int, tvm::Attrs const&, tvm::TypeReporter const&)
            # E         File "/localdev/vbrkic/src/forge/tt-forge-fe/third_party/tvm/src/relay/op/tensor/transform.cc", line 2507
            # E       InternalError: Check failed: *axis_ptr == 1 (5 vs. 1) : cannot squeeze axis with dimension not equal to 1
            # third_party/tvm/python/tvm/_ffi/base.py:479: InternalError
            ExceptionCheck(
                class_name="tvm.error.InternalError",
                component=ComponentChecker.NONE.value,  # TODO remove  #  TODO Remove the duplicate
                message=[
                    M.starts_with("Traceback (most recent call last):"),
                ],
                error_log=[
                    M.regex("Check failed: .* : cannot squeeze axis with dimension not equal to 1"),
                    M.contains("third_party/tvm/src/relay/op/tensor/transform.cc"),
                    M.contains(">       raise py_err"),
                    M.last_line(M.contains("/tvm/_ffi/base.py:")),
                ],
            ),
        ],
    )

    ASSERT_STRIDE = FailingReason(
        description="Assert stride error",
        checks=[
            #     def populate_conv2d_transpose_args(graph, nid, compiler_cfg):
            #         ...
            # >       assert all([stride == strides[0] for stride in strides])
            # E       AssertionError
            # forge/forge/tvm_to_python.py:652: AssertionError
            ExceptionCheck(
                class_name="AssertionError",
                component=ComponentChecker.TVM.value,
                message=[
                    # M.starts_with(">       assert all([stride == strides[0] for stride in strides])"),
                ],
                error_log=[
                    M.contains("def populate_conv2d_transpose_args("),
                    M.contains(">       assert all([stride == strides[0] for stride in strides])"),
                    M.last_line(M.contains("forge/tvm_to_python.py:")),
                ],
            ),
        ],
    )

    ASSERT_GROUPS = FailingReason(
        description="Assert groups error",
        checks=[
            # >       assert groups == 1 or (in_channel is not None and groups == in_channel), "Only supports group of 1 or in_channel"
            # E       AssertionError: Only supports group of 1 or in_channel
            # forge/forge/tvm_to_python.py:697: AssertionError
            ExceptionCheck(
                class_name="AssertionError",
                component=ComponentChecker.TVM.value,
                message=[
                    # M.starts_with(">       assert groups == 1 or (in_channel is not None and groups == in_channel), "Only supports group of 1 or in_channel""),
                    M.starts_with("Only supports group of 1 or in_channel"),
                ],
                error_log=[
                    M.contains(
                        '>       assert groups == 1 or (in_channel is not None and groups == in_channel), "Only supports group of 1 or in_channel"'
                    ),
                    M.last_line(M.contains("forge/tvm_to_python.py:")),
                ],
            ),
        ],
    )

    ASSERT_DIM = FailingReason(
        description="Assert dim error",
        checks=[
            #     def populate_conv2d_transpose_args(graph, nid, compiler_cfg):
            #         ...
            #         dilation = [int(dilation) for dilation in node["attrs"]["dilation"][0]]
            # >       assert all([dim == dilation[0] for dim in dilation])
            # E       AssertionError
            # forge/forge/tvm_to_python.py:679: AssertionError
            ExceptionCheck(
                class_name="AssertionError",
                component=ComponentChecker.TVM.value,
                message=[
                    # M.starts_with(">       assert all([dim == dilation[0] for dim in dilation])"),
                ],
                error_log=[
                    M.contains("def populate_conv2d_transpose_args("),
                    M.contains(">       assert all([dim == dilation[0] for dim in dilation])"),
                    M.last_line(M.contains("forge/tvm_to_python.py:")),
                ],
            ),
        ],
    )

    INDEX_ERROR = FailingReason(
        description="Index error",
        checks=[
            # >       return torch.embedding(t_ops[1], t_ops[0].to(torch.int32))
            # E       IndexError: index out of range in self
            # forge/op/eval/forge/embedding.py:15: IndexError
            ExceptionCheck(
                class_name="IndexError",
                component=ComponentChecker.FORGE.value,
                message=[
                    M.equals("index out of range in self"),
                ],
                error_log=[
                    M.contains(">       return torch.embedding(t_ops[1], t_ops[0].to(torch.int32))"),
                    M.last_line(M.contains("forge/op/eval/forge/embedding.py:")),
                ],
            ),
        ],
    )

    INCORRECT_TENSOR_SHAPE = FailingReason(
        description="Incorrect tensor shape",
        checks=[
            # E       AssertionError: Setting a tensor value of incorrect shape: (3, 11, 9, 7) vs torch.Size([3, 11, 8, 9])
            # forge/forge_wheels/venv/lib/python3.10/site-packages/forge/tensor.py:414: AssertionError
            ExceptionCheck(
                class_name="AssertionError",
                component=ComponentChecker.FORGE.value,
                message=[
                    M.regex(
                        "Setting a tensor value of incorrect shape: \\(\\d+, \\d+, \\d+, \\d+\\) vs torch.Size\\(\\[\\d+, \\d+, \\d+, \\d+\\]\\)"
                    ),
                ],
                error_log=[
                    M.last_line(M.contains("forge/tensor.py:")),
                ],
            ),
        ],
    )

    PAD_BIGGER_THAN_HALF_OF_KERNEL = FailingReason(
        description="pad should be at most half of effective kernel size",
        checks=[
            # E           RuntimeError: pad should be at most half of effective kernel size, but got pad=1, kernel_size=1 and dilation=1
            # forge/forge_wheels/venv/lib/python3.10/site-packages/forge/op/eval/forge/pooling.py:319: RuntimeError
            ExceptionCheck(
                class_name="RuntimeError",
                component=ComponentChecker.FORGE.value,
                message=[
                    M.regex(
                        "pad should be at most half of effective kernel size, but got pad=\\d+, kernel_size=\\d+ and dilation=\\d+"
                    ),
                ],
                error_log=[
                    M.any(
                        M.last_line(M.contains("forge/op/eval/forge/pooling.py")),
                        M.last_line(M.contains("forge/op/common.py")),
                    ),
                ],
            ),
        ],
    )

    AXIS_OUT_OF_BOUNDS = FailingReason(
        description="Axis out of bounds error",
        checks=[
            # E       tvm.error.InternalError: Traceback (most recent call last):
            # E         10: tvm::runtime::PackedFuncObj::Extractor<tvm::runtime::PackedFuncSubObj<tvm::runtime::TypedPackedFunc<tvm::IRModule (tvm::transform::Pass, tvm::IRModule)>::AssignTypedLambda<tvm::transform::__mk_TVM9::{lambda(tvm::transform::Pass, tvm::IRModule)#1}>(tvm::transform::__mk_TVM9::{lambda(tvm::transform::Pass, tvm::IRModule)#1}, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}> >::Call(tvm::runtime::PackedFuncObj const*, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, tvm::runtime::TVMRetValue)
            # E         9: tvm::transform::Pass::operator()(tvm::IRModule) const
            # E         8: tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
            # E         7: tvm::transform::ModulePassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const
            # E         6: tvm::runtime::PackedFuncObj::Extractor<tvm::runtime::PackedFuncSubObj<tvm::runtime::TypedPackedFunc<tvm::IRModule (tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::relay::transform::InferType()::{lambda(tvm::IRModule, tvm::transform::PassContext const&)#1}>(tvm::relay::transform::InferType()::{lambda(tvm::IRModule, tvm::transform::PassContext const&)#1})::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}> >::Call(tvm::runtime::PackedFuncObj const*, tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)
            # E         5: tvm::relay::TypeInferencer::Infer(tvm::GlobalVar, tvm::relay::Function)
            # E         4: tvm::relay::TypeSolver::Solve()
            # E         3: tvm::runtime::PackedFuncObj::Extractor<tvm::runtime::PackedFuncSubObj<tvm::runtime::TypedPackedFunc<bool (tvm::runtime::Array<tvm::Type, void> const&, int, tvm::Attrs const&, tvm::TypeReporter const&)>::AssignTypedLambda<bool (*)(tvm::runtime::Array<tvm::Type, void> const&, int, tvm::Attrs const&, tvm::TypeReporter const&)>(bool (*)(tvm::runtime::Array<tvm::Type, void> const&, int, tvm::Attrs const&, tvm::TypeReporter const&))::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}> >::Call(tvm::runtime::PackedFuncObj const*, tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)
            # E         2: tvm::relay::ReduceRel(tvm::runtime::Array<tvm::Type, void> const&, int, tvm::Attrs const&, tvm::TypeReporter const&)
            # E         1: std::vector<tvm::PrimExpr, std::allocator<tvm::PrimExpr> > tvm::relay::ReduceShapeImpl<tvm::relay::ReduceAttrs>(std::vector<tvm::PrimExpr, std::allocator<tvm::PrimExpr> > const&, tvm::relay::ReduceAttrs const*, tvm::TypeReporter const&)
            # E         0: tvm::relay::GetReduceAxes(unsigned int, tvm::runtime::Array<tvm::Integer, void> const&, bool)
            # E         File "/__w/tt-forge-fe/tt-forge-fe/third_party/tvm/src/relay/op/tensor/reduce.cc", line 71
            # E       InternalError: Check failed: (axis < indim) is false: Axis out of bounds in reduce operator.
            # venv/lib/python3.10/site-packages/tvm/_ffi/base.py:479: InternalError
            ExceptionCheck(
                class_name="tvm.error.InternalError",
                component=ComponentChecker.TVM.value,
                error_log=[
                    M.contains(
                        "E       InternalError: Check failed: (axis < indim) is false: Axis out of bounds in reduce operator."
                    ),
                    M.last_line(M.contains("tvm/_ffi/base.py:")),
                ],
            ),
        ],
    )

    UNSUPPORTED_DILATION = FailingReason(
        description="Unsupported dilation, currently only support dilation = 1",
        checks=[
            # E       RuntimeError: TT_ASSERT @ /__w/tt-forge-fe/tt-forge-fe/forge/csrc/ops/op_max_pool_2d.cpp:123: dilation_height == 1 && dilation_width == 1
            # E       info:
            # E       info:
            # E       Currently only support dilation = 1
            ExceptionCheck(
                class_name="RuntimeError",
                component=ComponentChecker.FORGE.value,
                message=[
                    M.contains("forge/csrc/ops/op_max_pool_2d.cpp"),
                    M.contains("dilation_height == 1 && dilation_width == 1"),
                ],
                error_log=[
                    M.last_line(M.contains("forge/op/common.py")),
                ],
            ),
        ],
    )

    AUTOSHARDING_ERROR = FailingReason(
        description="autosharding could not determine valid shard scheme, please check tensor dimensions",
        checks=[
            # E       RuntimeError: TT_FATAL @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/pool/generic/generic_pools.cpp:110: sw_parallel_config.has_value()
            # E       info:
            # E       autosharding could not determine valid shard scheme, please check tensor dimensions
            ExceptionCheck(
                class_name="RuntimeError",
                component=ComponentChecker.TTNN.value,
                message=[
                    M.contains(
                        "third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/pool/generic/generic_pools.cpp"
                    ),
                    M.contains("sw_parallel_config.has_value()"),
                ],
                error_log=[
                    M.last_line(M.contains("forge/compiled_graph_state.py")),
                ],
            ),
        ],
    )

    CORE_CHANNELS_MODULUS_NOT_ZERO = FailingReason(
        description="Channels: 1, num core channels: 2; channels % num_cores_channels == 0",
        checks=[
            # E       RuntimeError: TT_FATAL @ /__w/tt-forge-fe/tt-forge-fe/third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp:314: channels % num_cores_channels == 0
            # E       info:
            # E       Channels: 1, num core channels: 2
            ExceptionCheck(
                class_name="RuntimeError",
                component=ComponentChecker.METAL.value,
                message=[
                    M.contains(
                        "third_party/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp"
                    ),
                    M.contains("channels % num_cores_channels == 0"),
                ],
                error_log=[
                    M.last_line(M.contains("forge/compiled_graph_state.py")),
                ],
            )
        ],
    )

    ASYMMETRIC_KERNEL_NOT_SUPPORTED = FailingReason(
        description="Asymmetric kernel is not supported",
        checks=[
            # def populate_maxpool2d_args(graph, nid, compiler_cfg):
            #         ...
            # >       assert all([dim == kernel_size[0] for dim in kernel_size])
            # E       AssertionError
            # /localdev/kmilanovic/src/forge/forge_wheels/venv/lib/python3.10/site-packages/forge/tvm_to_python.py
            ExceptionCheck(
                class_name="AssertionError",
                component=ComponentChecker.TVM.value,
                message=[],
                error_log=[
                    M.contains("def populate_maxpool2d_args("),
                    M.contains(">       assert all([dim == kernel_size[0] for dim in kernel_size])"),
                    M.last_line(M.contains("forge/tvm_to_python.py:")),
                ],
            )
        ],
    )

    ASYMMETRIC_STRIDE_NOT_SUPPORTED = FailingReason(
        description="Asymmetric stride is not supported",
        checks=[
            # def populate_maxpool2d_args(graph, nid, compiler_cfg):
            #         ...
            # >       assert all([stride == strides[0] for stride in strides])
            # E       AssertionError
            # /localdev/kmilanovic/src/forge/forge_wheels/venv/lib/python3.10/site-packages/forge/tvm_to_python.py:902: AssertionError
            ExceptionCheck(
                class_name="AssertionError",
                component=ComponentChecker.TVM.value,
                message=[],
                error_log=[
                    M.contains("def populate_maxpool2d_args("),
                    M.contains(">       assert all([stride == strides[0] for stride in strides])"),
                    M.last_line(M.contains("forge/tvm_to_python.py:")),
                ],
            )
        ],
    )

    ASYMMETRIC_DILATION_NOT_SUPPORTED = FailingReason(
        description="Asymmetric dilation is not supported",
        checks=[
            # def populate_maxpool2d_args(graph, nid, compiler_cfg):
            #         ...
            # >       assert all([dim == dilation[0] for dim in dilation])
            # E       AssertionError
            # /localdev/kmilanovic/src/forge/forge_wheels/venv/lib/python3.10/site-packages/forge/tvm_to_python.py:927: AssertionError
            ExceptionCheck(
                class_name="AssertionError",
                component=ComponentChecker.TVM.value,
                message=[],
                error_log=[
                    M.contains("def populate_maxpool2d_args("),
                    M.contains(">       assert all([dim == dilation[0] for dim in dilation])"),
                    M.last_line(M.contains("forge/tvm_to_python.py:")),
                ],
            )
        ],
    )
