# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Failing reasons definition for XLA

from enum import Enum
from typing import Optional

from .utils import ExceptionCheck, FailingReason, M


class ComponentChecker(Enum):
    """
    Enum representing different components which cause failures.
    """

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

    METAL = FailingReason(
        description="Metal",
        checks=[
            ExceptionCheck(
                error_log=[
                    M.contains("lib/libtt_metal.so"),
                    M.contains("lib/_ttnncpp.so"),
                    M.contains("lib/libTTMLIRRuntime.so"),
                    M.any(
                        M.last_line(M.contains("infra/runners/torch_device_runner.py")),
                    ),
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
                    M.any(
                        M.last_line(
                            M.contains("infra/runners/torch_device_runner.py:")
                        ),
                    ),
                ],
            ),
            ExceptionCheck(
                error_log=[
                    M.neg(M.contains("lib/libtt_metal.so")),
                    M.contains("lib/libTTMLIRCompiler.so"),
                    M.contains("lib/libTTMLIRRuntime.so"),
                    M.any(
                        M.last_line(
                            M.contains("infra/runners/torch_device_runner.py:")
                        ),
                    ),
                ],
            ),
        ],
    )

    TORCH = FailingReason(
        description="Torch",
        checks=[
            ExceptionCheck(
                error_log=[
                    M.neg(M.contains("lib/libtt_metal.so")),
                    M.neg(M.contains("lib/_ttnncpp.so")),
                    M.neg(M.contains("lib/libTTMLIRRuntime.so")),
                    M.any(
                        M.last_line(M.contains("torch/utils/_pytree.py:")),
                    ),
                ],
            ),
        ],
    )

    XLA = FailingReason(
        description="Xla",
        checks=[
            ExceptionCheck(
                message=[
                    M.neg(
                        M.any(
                            M.contains("lib/libTTMLIRCompiler.so"),
                            M.contains("lib/libtt_metal.so"),
                            M.contains("lib/_ttnncpp.so"),
                            M.contains("lib/libTTMLIRRuntime.so"),
                        )
                    ),
                ],
                error_log=[
                    M.any(
                        M.last_line(M.contains("tt_torch/backend/backend.py:")),
                        M.last_line(
                            M.contains("infra/runners/torch_device_runner.py:")
                        ),
                        M.last_line(M.contains("infra/comparators/comparator.py:")),
                        M.last_line(
                            M.contains(
                                "infra/testers/single_chip/model/torch_model_tester.py"
                            )
                        ),
                        M.last_line(
                            M.contains(
                                "forge/test/operators/pytorch/indexing/test_index_copy.py"
                            )
                        ),
                    ),
                ],
            ),
        ],
    )

    SWEEPS = FailingReason(
        description="Sweeps",
        checks=[
            ExceptionCheck(
                error_log=[
                    M.neg(M.contains("lib/libtt_metal.so")),
                    M.neg(M.contains("lib/_ttnncpp.so")),
                    M.neg(M.contains("lib/libTTMLIRRuntime.so")),
                    M.any(
                        M.last_line(
                            M.contains("forge/test/operators/utils/verify.py:")
                        ),
                    ),
                ],
            ),
        ],
    )


# Set the ComponentChecker.NONE value to avoid circular import issues
FailingReason.component_checker_none = ComponentChecker.NONE.value


class FailingReasons(Enum):
    """
    Enum representing different failing reasons with their checks.
    """

    def __repr__(self) -> str:
        return self.name

    @classmethod
    def find_by_description(cls, desc: str) -> Optional["FailingReasons"]:
        """Find failing reason by description."""
        failing_reasons = [
            xfail_reason
            for xfail_reason in FailingReasons
            if xfail_reason.value.description == desc
        ]
        if len(failing_reasons) == 0:
            return None
        elif len(failing_reasons) > 1:
            raise ValueError(
                f"Multiple xfail reasons {failing_reasons} found for description: {desc}"
            )
        return failing_reasons[0]

    UNCLASSIFIED = FailingReason(
        description="Unclassified error",
    )

    # Used when a model is marked as not supported and test is skipped.
    NOT_SUPPORTED_AND_SKIPPED = FailingReason(
        description="Model is not supported (skipped)",
    )

    # Used when a test is marked as incorrect result due to PCC,
    # but PCC assertion is disabled in the comparison configuration.
    INCORRECT_RESULT_PCC_DISABLED = FailingReason(
        description="Test marked w/ INCORRECT_RESULT. PCC check disabled.",
    )

    # Missing dependency: segmentation_models_pytorch
    SEGMENTATION_MODELS_PYTORCH_NOT_FOUND = FailingReason(
        description="ModuleNotFoundError: No module named 'segmentation_models_pytorch'",
        checks=[
            ExceptionCheck(
                class_name="ModuleNotFoundError",
                message=[
                    M.contains("No module named 'segmentation_models_pytorch'"),
                ],
            ),
        ],
    )

    # Missing tokenizer chat template configuration for chat template functions
    TOKENIZER_CHAT_TEMPLATE_NOT_SET = FailingReason(
        description="ValueError: Cannot use chat template functions because tokenizer.chat_template is not set and no template argument was passed",
        checks=[
            ExceptionCheck(
                class_name="ValueError",
                message=[
                    M.contains("Cannot use chat template functions"),
                    M.contains("tokenizer.chat_template is not set"),
                    M.contains("no template argument was passed"),
                ],
            ),
        ],
    )

    # Fatal error raised during xla_sync_multi call in backend
    XLA_SYNC_MULTI_FATAL_ERROR = FailingReason(
        description="Fatal error in xla_sync_multi call",
        checks=[
            ExceptionCheck(
                class_name="RuntimeError",
                component=ComponentChecker.XLA.value,
                message=[
                    M.contains("Fatal error"),
                ],
                error_log=[
                    M.contains(
                        ">           torch_xla._XLAC._xla_sync_multi(list(output), self.devices, wait=False)"
                    ),
                    M.any(
                        M.last_line(M.contains("tt_torch/backend/backend.py:")),
                        M.last_line(
                            M.contains("python_package/tt_torch/backend/backend.py:")
                        ),
                    ),
                ],
            ),
        ],
    )
    # # RuntimeError: Fatal Python error: Segmentation fault
    SEG_FAULT = FailingReason(
        description="Inference failed due to seg fault",
    )

    # # RuntimeError: Fatal Python error: Aborted
    FATAL_ERROR = FailingReason(
        description="Fatal error occured",
        checks=[
            # E           RuntimeError: Fatal error
            #
            # forge/infra/runners/torch_device_runner.py:40: RuntimeError
            ExceptionCheck(
                class_name="RuntimeError",
                component=ComponentChecker.XLA.value,
                message=[
                    M.contains("Fatal error"),
                ],
                error_log=[
                    M.last_line(M.contains("infra/runners/torch_device_runner.py:")),
                ],
            ),
        ],
    )

    HIGH_MEMORY = FailingReason(
        description="High memory usage",
    )

    INFERENCE_FROZEN = FailingReason(
        description="Inference frozen without error message",
    )

    DATA_TYPE_NOT_SUPPORTED = FailingReason(
        description="Data type is not supported, Can only work with bfloat16/float32 or int32/uint32 tensors",
        checks=[
            ExceptionCheck(
                class_name="RuntimeError",
                component=ComponentChecker.METAL.value,
                message=[
                    M.contains(
                        "Can only work with bfloat16/float32 or int32/uint32 tensors"
                    ),
                ],
                error_log=[
                    # M.contains("normalized_index >= 0 and normalized_index < rank"),
                    M.last_line(M.contains("infra/runners/torch_device_runner.py:")),
                ],
            ),
        ],
    )

    DATA_MISMATCH_PCC_LOW_RANGE = FailingReason(
        description="Data mismatch PCC in low range",
        checks=[
            # TODO: add log snippets
            ExceptionCheck(
                class_name="ValueError",
                component=ComponentChecker.SWEEPS.value,  # TODO: check component
                message=[
                    M.any(
                        M.starts_with(
                            "Comparison result 0 failed: PCC comparison failed: pcc is in invalid low range:"
                        ),
                    ),
                ],
                error_log=[
                    M.any(
                        M.last_line(M.contains("test/operators/utils/verify.py:")),
                    ),
                ],
            ),
        ],
    )

    DATA_MISMATCH_PCC_MEDIUM_RANGE = FailingReason(
        description="Data mismatch PCC in medium range",
        checks=[
            # TODO: add log snippets
            ExceptionCheck(
                class_name="ValueError",
                component=ComponentChecker.SWEEPS.value,  # TODO: check component
                message=[
                    M.any(
                        M.starts_with(
                            "Comparison result 0 failed: PCC comparison failed: pcc is in invalid medium range:"
                        ),
                    ),
                ],
                error_log=[
                    M.any(
                        M.last_line(M.contains("test/operators/utils/verify.py:")),
                    ),
                ],
            ),
        ],
    )

    DATA_MISMATCH_PCC_HIGH_RANGE = FailingReason(
        description="Data mismatch PCC in high range",
        checks=[
            # TODO: add log snippets
            ExceptionCheck(
                class_name="ValueError",
                component=ComponentChecker.SWEEPS.value,  # TODO: check component
                message=[
                    M.any(
                        M.starts_with(
                            "Comparison result 0 failed: PCC comparison failed: pcc is in invalid high range:"
                        ),
                    ),
                ],
                error_log=[
                    M.any(
                        M.last_line(M.contains("test/operators/utils/verify.py:")),
                    ),
                ],
            ),
        ],
    )

    DATA_MISMATCH_WRONG_PCC = FailingReason(
        description="Data mismatch PCC is wrong",
        checks=[
            # >           assert False, "\n".join(error_messages)
            # E           AssertionError: Comparison result 0 failed: PCC comparison failed. Calculated: pcc=0.959952175617218. Required: pcc=0.99.
            # forge/infra/comparators/comparator.py:169: AssertionError
            ExceptionCheck(
                class_name="AssertionError",
                component=ComponentChecker.XLA.value,
                message=[
                    M.starts_with("Comparison result 0 failed"),
                    M.contains("PCC comparison failed"),
                    M.contains("Calculated: pcc="),
                    M.neg(M.contains("Calculated: pcc=nan (invalid value)")),
                    M.neg(
                        M.contains("Allclose comparison failed")
                    ),  # Some test doesn't have pcc but compares allclose as well
                ],
                error_log=[
                    M.last_line(M.contains("infra/comparators/comparator.py:")),
                ],
            ),
        ],
    )

    DATA_MISMATCH_PCC_IS_NAN = FailingReason(
        description="Data mismatch PCC is nan",
        checks=[
            # Example error log (1):
            # >           assert False, "\n".join(error_messages)
            # E           AssertionError: Comparison result 0 failed: PCC comparison failed. Calculated: pcc=nan (invalid value). Required: pcc=0.99.
            # forge/infra/comparators/comparator.py:169: AssertionError
            #
            # Example error log (2): no pcc but compares allclose
            # >           assert False, "\n".join(error_messages)
            # E           AssertionError: Comparison result 0 failed: PCC comparison failed. Calculated: pcc=nan (invalid value). Required: pcc=0.99.
            # E           Allclose comparison failed. Required: atol=0.01, rtol=0.01.
            # forge/infra/comparators/comparator.py:169: AssertionError')
            ExceptionCheck(
                class_name="AssertionError",
                component=ComponentChecker.XLA.value,
                message=[
                    M.starts_with("Comparison result 0 failed"),
                    M.contains("PCC comparison failed"),
                    M.contains("Calculated: pcc=nan (invalid value)"),
                    M.neg(
                        M.contains("Allclose comparison failed")
                    ),  # Some test doesn't have pcc but compares allclose as well
                ],
                error_log=[
                    M.last_line(M.contains("infra/comparators/comparator.py:")),
                ],
            ),
        ],
    )

    DATA_MISMATCH_ALL_CLOSE = FailingReason(
        description="Data mismatch Allclose comparison failed",
        checks=[
            # >           assert False, "\n".join(error_messages)
            # E           AssertionError: Comparison result 0 failed: Allclose comparison failed. Required: atol=0.01, rtol=0.01.
            # forge/infra/comparators/comparator.py:169: AssertionError
            ExceptionCheck(
                class_name="AssertionError",
                component=ComponentChecker.XLA.value,
                message=[
                    M.starts_with("Comparison result 0 failed"),
                    M.contains("Allclose comparison failed"),
                ],
                error_log=[
                    M.last_line(M.contains("infra/comparators/comparator.py:")),
                ],
            ),
        ],
    )

    ERROR_CODE_13_TO_DEVICE = FailingReason(
        description="Error code 13 in to(device) call",
        checks=[
            # >           return x.to(device)
            # E           RuntimeError: Error code: 13
            # forge/infra/runners/torch_device_runner.py:52: RuntimeError
            ExceptionCheck(
                class_name="RuntimeError",
                component=ComponentChecker.XLA.value,
                message=[
                    M.equals("Error code: 13"),
                ],
                error_log=[
                    M.contains(">           return x.to(device)"),
                    M.last_line(M.contains("infra/runners/torch_device_runner.py:")),
                ],
            ),
        ],
    )

    ERROR_CODE_13_XLA_SYNC_MULTI = FailingReason(
        description="Error code 13 in xla_sync_multi call",
        checks=[
            # >           torch_xla._XLAC._xla_sync_multi(list(output), self.devices, wait=False)
            # E           ValueError: Error code: 13
            # /localdev/ctr-vbrkic/venv/sweeps/xla/lib/python3.11/site-packages/tt_torch/backend/backend.py:117: ValueError
            ExceptionCheck(
                class_name="ValueError",
                component=ComponentChecker.XLA.value,
                message=[
                    M.equals("Error code: 13"),
                ],
                error_log=[
                    M.contains(
                        ">           torch_xla._XLAC._xla_sync_multi(list(output), self.devices, wait=False)"
                    ),
                    M.last_line(M.contains("tt_torch/backend/backend.py:")),
                ],
            ),
        ],
    )

    OUT_OF_MEMORY = FailingReason(
        description="Out of memory error",
        checks=[
            # E           RuntimeError: TT_FATAL @ /__w/tt-xla/tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/impl/allocator/bank_manager.cpp:433: address.has_value()
            # E           info:
            # E           Out of Memory: Not enough space to allocate 85874464 B L1 buffer across 62 banks, where each bank needs to store 1385072 B, but bank size is only 1364928 B
            # E           backtrace:
            # E            --- /__w/tt-forge-sweeps/tt-forge-sweeps/env/venv/xla/lib/python3.11/site-packages/pjrt_plugin_tt/lib/libtt_metal.so(+0x567ede) [0x7f2ea69f9ede]
            # E            --- tt::tt_metal::BankManager::allocate_buffer(unsigned long, unsigned long, bool, CoreRangeSet const&amp;, std::optional&lt;unsigned int&gt;, ttsl::StrongType&lt;unsigned int, tt::tt_metal::AllocatorIDTag&gt;)
            # E            --- tt::tt_metal::Allocator::allocate_buffer(tt::tt_metal::Buffer*)
            # E            --- tt::tt_metal::Buffer::allocate_impl()
            # E            --- tt::tt_metal::Buffer::create(tt::tt_metal::IDevice*, unsigned long, unsigned long, tt::tt_metal::BufferType, tt::tt_metal::BufferShardingArgs const&amp;, std::optional&lt;bool&gt;, std::optional&lt;ttsl::StrongType&lt;unsigned char, tt::tt_metal::SubDeviceIdTag&gt; &gt;)
            # E            --- tt::tt_metal::distributed::MeshBuffer::create(std::variant&lt;tt::tt_metal::distributed::ReplicatedBufferConfig, tt::tt_metal::distributed::ShardedBufferConfig&gt; const&amp;, tt::tt_metal::distributed::DeviceLocalBufferConfig const&amp;, tt::tt_metal::distributed::MeshDevice*, std::optional&lt;unsigned long&gt;)
            # E            --- tt::tt_metal::tensor_impl::allocate_device_buffer(tt::tt_metal::distributed::MeshDevice*, tt::tt_metal::TensorSpec const&amp;)
            # E            --- tt::tt_metal::allocate_tensor_on_device(tt::tt_metal::TensorSpec const&amp;, tt::tt_metal::distributed::MeshDevice*)
            # E            --- tt::tt_metal::create_device_tensor(tt::tt_metal::TensorSpec const&amp;, tt::tt_metal::IDevice*)
            # E            --- /__w/tt-forge-sweeps/tt-forge-sweeps/env/venv/xla/lib/python3.11/site-packages/pjrt_plugin_tt/lib/_ttnncpp.so(_ZN2tt8tt_metal9operation29default_create_output_tensorsIN4ttnn10operations14sliding_window4halo19HaloDeviceOperationEEENS1_21program_output_helperIT_Xsr18has_create_programIS9_EE5valueEE4typeERKS9_RKSt6vectorINS0_6TensorESaISF_EERKSE_ISt8optionalISF_ESaISL_EE+0x180) [0x7f2ea629df40]
            # E            --- /__w/tt-forge-sweeps/tt-forge-sweeps/env/venv/xla/lib/python3.11/site-packages/pjrt_plugin_tt/lib/_ttnncpp.so(+0x101ddb9) [0x7f2ea629ddb9]
            # E            --- tt::tt_metal::operation::OldInfraDeviceOperation&lt;std::vector&lt;tt::tt_metal::Tensor, std::allocator&lt;tt::tt_metal::Tensor&gt; &gt; &gt;::create_output_tensors(tt::tt_metal::operation::DeviceOperation&lt;std::vector&lt;tt::tt_metal::Tensor, std::allocator&lt;tt::tt_metal::Tensor&gt; &gt; &gt; const&amp;, tt::tt_metal::operation::OldInfraDeviceOperation&lt;std::vector&lt;tt::tt_metal::Tensor, std::allocator&lt;tt::tt_metal::Tensor&gt; &gt; &gt;::tensor_args_t const&amp;)
            # E            --- tt::tt_metal::operation::OldInfraDeviceOperation&lt;std::vector&lt;tt::tt_metal::Tensor, std::allocator&lt;tt::tt_metal::Tensor&gt; &gt; &gt;::tensor_return_value_t ttnn::device_operation::detail::launch_on_device&lt;tt::tt_metal::operation::OldInfraDeviceOperation&lt;std::vector&lt;tt::tt_metal::Tensor, std::allocator&lt;tt::tt_metal::Tensor&gt; &gt; &gt; &gt;(tt::tt_metal::operation::OldInfraDeviceOperation&lt;std::vector&lt;tt::tt_metal::Tensor, std::allocator&lt;tt::tt_metal::Tensor&gt; &gt; &gt;::operation_attributes_t const&amp;, tt::tt_metal::operation::OldInfraDeviceOperation&lt;std::vector&lt;tt::tt_metal::Tensor, std::allocator&lt;tt::tt_metal::Tensor&gt; &gt; &gt;::tensor_args_t const&amp;)
            # E            --- tt::tt_metal::operation::OldInfraDeviceOperation&lt;std::vector&lt;tt::tt_metal::Tensor, std::allocator&lt;tt::tt_metal::Tensor&gt; &gt; &gt;::tensor_return_value_t ttnn::device_operation::detail::invoke&lt;tt::tt_metal::operation::OldInfraDeviceOperation&lt;std::vector&lt;tt::tt_metal::Tensor, std::allocator&lt;tt::tt_metal::Tensor&gt; &gt; &gt; &gt;(tt::tt_metal::operation::OldInfraDeviceOperation&lt;std::vector&lt;tt::tt_metal::Tensor, std::allocator&lt;tt::tt_metal::Tensor&gt; &gt; &gt;::operation_attributes_t const&amp;, tt::tt_metal::operation::OldInfraDeviceOperation&lt;std::vector&lt;tt::tt_metal::Tensor, std::allocator&lt;tt::tt_metal::Tensor&gt; &gt; &gt;::tensor_args_t const&amp;)
            # E            --- /__w/tt-forge-sweeps/tt-forge-sweeps/env/venv/xla/lib/python3.11/site-packages/pjrt_plugin_tt/lib/_ttnncpp.so(+0x6b9f6b) [0x7f2ea5939f6b]
            # E            --- std::vector&lt;tt::tt_metal::Tensor, std::allocator&lt;tt::tt_metal::Tensor&gt; &gt; tt::tt_metal::operation::run&lt;std::vector&lt;tt::tt_metal::Tensor, std::allocator&lt;tt::tt_metal::Tensor&gt; &gt; &gt;(tt::tt_metal::operation::DeviceOperation&lt;std::vector&lt;tt::tt_metal::Tensor, std::allocator&lt;tt::tt_metal::Tensor&gt; &gt; &gt;&amp;&amp;, std::vector&lt;tt::tt_metal::Tensor, std::allocator&lt;tt::tt_metal::Tensor&gt; &gt; const&amp;, std::vector&lt;std::optional&lt;tt::tt_metal::Tensor const&gt;, std::allocator&lt;std::optional&lt;tt::tt_metal::Tensor const&gt; &gt; &gt; const&amp;, std::vector&lt;std::optional&lt;tt::tt_metal::Tensor&gt;, std::allocator&lt;std::optional&lt;tt::tt_metal::Tensor&gt; &gt; &gt; const&amp;)
            # E            --- ttnn::operations::sliding_window::halo::halo_op(tt::tt_metal::Tensor const&amp;, ttnn::operations::sliding_window::SlidingWindowConfig const&amp;, unsigned int, bool, bool, tt::tt_metal::MemoryConfig const&amp;, bool, bool, bool)
            # E            --- ttnn::operations::sliding_window::halo::HaloOperation::invoke(tt::tt_metal::Tensor const&amp;, ttnn::operations::sliding_window::SlidingWindowConfig const&amp;, unsigned int, bool, bool, tt::tt_metal::MemoryConfig const&amp;, bool, bool, bool)
            # E            --- /__w/tt-forge-sweeps/tt-forge-sweeps/env/venv/xla/lib/python3.11/site-packages/pjrt_plugin_tt/lib/_ttnncpp.so(+0x10ac2e0) [0x7f2ea632c2e0]
            # E            --- /__w/tt-forge-sweeps/tt-forge-sweeps/env/venv/xla/lib/python3.11/site-packages/pjrt_plugin_tt/lib/_ttnncpp.so(+0x10a326f) [0x7f2ea632326f]
            # E            --- ttnn::operations::pool::AvgPool2DOp::invoke(tt::tt_metal::Tensor const&amp;, unsigned int, unsigned int, unsigned int, unsigned int, std::array&lt;unsigned int, 2ul&gt;, std::array&lt;unsigned int, 2ul&gt;, std::variant&lt;std::array&lt;unsigned int, 2ul&gt;, std::array&lt;unsigned int, 4ul&gt; &gt;, bool, bool, std::optional&lt;int&gt;, std::optional&lt;tt::tt_metal::MemoryConfig const&gt; const&amp;, std::optional&lt;tt::tt_metal::TensorMemoryLayout const&gt;, bool, bool, bool, tt::tt_metal::DataType, tt::tt_metal::Layout)
            # E            --- /__w/tt-forge-sweeps/tt-forge-sweeps/env/venv/xla/lib/python3.11/site-packages/pjrt_plugin_tt/lib/libTTMLIRRuntime.so(+0x297b2e) [0x7f2ec850ab2e]
            # E            --- /__w/tt-forge-sweeps/tt-forge-sweeps/env/venv/xla/lib/python3.11/site-packages/pjrt_plugin_tt/lib/libTTMLIRRuntime.so(+0x2977b9) [0x7f2ec850a7b9]
            # E            --- /__w/tt-forge-sweeps/tt-forge-sweeps/env/venv/xla/lib/python3.11/site-packages/pjrt_plugin_tt/lib/libTTMLIRRuntime.so(_ZN2tt7runtime4ttnn10operations4pool14runAvgPool2dOpEPKNS_6target4ttnn8Pool2dOpERNS1_17ProgramTensorPoolERKSt8functionIFNS_8tt_metal6TensorERKSD_jjjjSt5arrayIjLm2EESH_St7variantIJSH_SG_IjLm4EEEEbbSt8optionalIiERKSL_IKNSC_12MemoryConfigEESL_IKNSC_18TensorMemoryLayoutEEbEE+0x504) [0x7f2ec8509db4]
            # E            --- tt::runtime::ttnn::operations::pool::run(tt::target::ttnn::Pool2dOp const*, tt::runtime::ttnn::ProgramContext&amp;)
            # E            --- tt::runtime::ttnn::ProgramExecutor::execute()
            # E            --- tt::runtime::ttnn::submit(tt::runtime::Device, tt::runtime::Binary, unsigned int, std::vector&lt;tt::runtime::Tensor, std::allocator&lt;tt::runtime::Tensor&gt; &gt;&amp;)
            # E            --- tt::runtime::submit(tt::runtime::Device, tt::runtime::Binary, unsigned int, std::vector&lt;tt::runtime::Tensor, std::allocator&lt;tt::runtime::Tensor&gt; &gt;&amp;)
            # E            --- tt::pjrt::FlatbufferLoadedExecutableInstance::execute(PJRT_LoadedExecutable_Execute_Args*)
            # E            --- tt::pjrt::internal::onLoadedExecutableExecute(PJRT_LoadedExecutable_Execute_Args*)
            # E            --- xla::PjRtCApiLoadedExecutable::ExecuteWithSingleDevice(absl::lts_20230802::Span&lt;xla::PjRtBuffer* const&gt;, xla::PjRtDevice*, xla::ExecuteOptions const&amp;, std::optional&lt;xla::PjRtFuture&lt;void&gt; &gt;&amp;, bool)
            # E            --- xla::PjRtCApiLoadedExecutable::ExecutePortable(absl::lts_20230802::Span&lt;xla::PjRtBuffer* const&gt;, xla::PjRtDevice*, xla::ExecuteOptions const&amp;, std::optional&lt;xla::PjRtFuture&lt;void&gt; &gt;&amp;, bool)
            # E            --- torch_xla::runtime::PjRtComputationClient::ExecuteComputation(torch_xla::runtime::ComputationClient::Computation const&amp;, absl::lts_20230802::Span&lt;std::shared_ptr&lt;torch_xla::runtime::ComputationClient::Data&gt; const&gt;, std::__cxx11::basic_string&lt;char, std::char_traits&lt;char&gt;, std::allocator&lt;char&gt; &gt; const&amp;, torch_xla::runtime::ComputationClient::ExecuteComputationOptions const&amp;)
            # E            --- /__w/tt-forge-sweeps/tt-forge-sweeps/env/venv/xla/lib/python3.11/site-packages/_XLAC.cpython-311-x86_64-linux-gnu.so(+0x5f2b1ea) [0x7f2f0d11b1ea]
            # E            --- torch::lazy::MultiWait::Complete(std::function&lt;void ()&gt; const&amp;)
            # E            --- Eigen::ThreadPoolTempl&lt;tsl::thread::EigenEnvironment&gt;::WorkerLoop(int)
            # E            --- void absl::lts_20230802::internal_any_invocable::RemoteInvoker&lt;false, void, tsl::thread::EigenEnvironment::CreateThread(std::function&lt;void ()&gt;)::{lambda()#1}&amp;&gt;(absl::lts_20230802::internal_any_invocable::TypeErasedState*)
            # E            --- /__w/tt-forge-sweeps/tt-forge-sweeps/env/venv/xla/lib/python3.11/site-packages/_XLAC.cpython-311-x86_64-linux-gnu.so(+0xfc35751) [0x7f2f16e25751]
            # E            --- /lib/x86_64-linux-gnu/libc.so.6(+0x94ac3) [0x7f3013c2dac3]
            # E            --- /lib/x86_64-linux-gnu/libc.so.6(clone+0x44) [0x7f3013cbea74]
            # forge/infra/runners/torch_device_runner.py:40: RuntimeError
            ExceptionCheck(
                class_name="RuntimeError",
                # component=ComponentChecker.METAL.value,
                message=[
                    M.any(
                        M.regex(
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
                    M.any(
                        # Accept when comparator/runner is the last line
                        M.last_line(
                            M.contains("infra/runners/torch_device_runner.py:")
                        ),
                        # Also accept when the OOM message appears in the captured longrepr anywhere
                        M.regex(
                            "Out of Memory: Not enough space to allocate .* B (?:L1|L1_SMALL|DRAM) buffer across .* banks"
                        ),
                    ),
                ],
            ),
            # Circular buffers exceed L1 size
            ExceptionCheck(
                stderr=[
                    M.contains("circular buffers"),
                    M.contains("beyond max L1 size"),
                ],
            ),
            ExceptionCheck(
                stdout=[
                    M.contains("circular buffers"),
                    M.contains("beyond max L1 size"),
                ],
            ),
            # Out of Memory: Not enough space to allocate
            ExceptionCheck(
                stdout=[
                    M.contains("Out of Memory: Not enough space to allocate"),
                ],
            ),
            ExceptionCheck(
                stderr=[
                    M.contains("Out of Memory: Not enough space to allocate"),
                ],
            ),
            # Fallback: classify OOM when message is a StatusOr INTERNAL:13 error
            ExceptionCheck(
                message=[
                    M.contains("Bad StatusOr access"),
                    M.contains("Error code: 13"),
                ],
            ),
        ],
    )

    # MLIR/TTIR compilation errors - these show up in stdout/stderr with "error:" pattern
    # Example: loc("convolution.1254"): error: 'ttir.conv_transpose2d' op Number of input channels...
    MLIR_TTIR_COMPILATION_ERROR = FailingReason(
        description="MLIR/TTIR compilation error",
        checks=[
            # Matches MLIR errors in stderr (compilation output)
            ExceptionCheck(
                component=ComponentChecker.XLA.value,
                stderr=[
                    M.regex(r"loc\(.*\): error:.*'ttir\."),
                ],
            ),
            # Matches MLIR errors in stdout
            ExceptionCheck(
                component=ComponentChecker.XLA.value,
                stdout=[
                    M.regex(r"loc\(.*\): error:.*'ttir\."),
                ],
            ),
            # Fallback: check error_log too
            ExceptionCheck(
                component=ComponentChecker.XLA.value,
                error_log=[
                    M.regex(r"loc\(.*\): error:.*'ttir\."),
                ],
            ),
        ],
    )

    # MLIR/TTNN compilation errors
    MLIR_TTNN_COMPILATION_ERROR = FailingReason(
        description="MLIR/TTNN compilation error",
        checks=[
            ExceptionCheck(
                component=ComponentChecker.XLA.value,
                stderr=[
                    M.regex(r"loc\(.*\): error:.*'ttnn\."),
                ],
            ),
            ExceptionCheck(
                component=ComponentChecker.XLA.value,
                stdout=[
                    M.regex(r"loc\(.*\): error:.*'ttnn\."),
                ],
            ),
            ExceptionCheck(
                component=ComponentChecker.XLA.value,
                error_log=[
                    M.regex(r"loc\(.*\): error:.*'ttnn\."),
                ],
            ),
        ],
    )

    # StableHLO compilation errors
    MLIR_STABLEHLO_COMPILATION_ERROR = FailingReason(
        description="MLIR/StableHLO compilation error",
        checks=[
            ExceptionCheck(
                component=ComponentChecker.XLA.value,
                stderr=[
                    M.regex(r"loc\(.*\): error:.*'stablehlo\."),
                ],
            ),
            ExceptionCheck(
                component=ComponentChecker.XLA.value,
                stdout=[
                    M.regex(r"loc\(.*\): error:.*'stablehlo\."),
                ],
            ),
            ExceptionCheck(
                component=ComponentChecker.XLA.value,
                error_log=[
                    M.regex(r"loc\(.*\): error:.*'stablehlo\."),
                ],
            ),
        ],
    )

    MLIR_OP_VERIFICATION_ERROR = FailingReason(
        description="MLIR op verification error",
        checks=[
            ExceptionCheck(
                component=ComponentChecker.XLA.value,
                stderr=[
                    M.regex(r"loc\(.*\): error:"),
                    M.neg(M.contains("'ttir.")),
                    M.neg(M.contains("'ttnn.")),
                    M.neg(M.contains("stablehlo.")),
                ],
            ),
            ExceptionCheck(
                component=ComponentChecker.XLA.value,
                stdout=[
                    M.regex(r"loc\(.*\): error:"),
                    M.neg(M.contains("'ttir.")),
                    M.neg(M.contains("'ttnn.")),
                    M.neg(M.contains("stablehlo.")),
                ],
            ),
            ExceptionCheck(
                component=ComponentChecker.XLA.value,
                error_log=[
                    M.regex(r"error:.*'\w+\.\w+' op"),
                    M.neg(M.contains("'ttir.")),
                    M.neg(M.contains("'ttnn.")),
                    M.neg(M.contains("stablehlo.")),
                ],
            ),
            ExceptionCheck(
                component=ComponentChecker.XLA.value,
                stderr=[
                    M.regex(r"error:.*'\w+\.\w+' op"),
                    M.neg(M.contains("'ttir.")),
                    M.neg(M.contains("'ttnn.")),
                    M.neg(M.contains("stablehlo.")),
                ],
            ),
            ExceptionCheck(
                component=ComponentChecker.XLA.value,
                stdout=[
                    M.regex(r"error:.*'\w+\.\w+' op"),
                    M.neg(M.contains("'ttir.")),
                    M.neg(M.contains("'ttnn.")),
                    M.neg(M.contains("stablehlo.")),
                ],
            ),
        ],
    )

    TORCHVISION_DEFORM_IM2COL_BFLOAT16_NOT_IMPLEMENTED = FailingReason(
        description="deformable_im2col not implemented for bfloat16",
        checks=[
            # RuntimeError: "deformable_im2col" not implemented for 'BFloat16'
            ExceptionCheck(
                class_name="RuntimeError",
                error_log=[
                    M.contains("deformable_im2col"),
                    M.contains("not implemented"),
                    M.any(M.contains("BFloat16"), M.contains("bfloat16")),
                    M.any(
                        M.contains("torchvision/ops"),
                        M.contains("torchvision.ops"),
                        M.contains("torchvision.deform_conv2d"),
                    ),
                ],
            ),
        ],
    )

    TORCHVISION_NMS_BFLOAT16_NOT_IMPLEMENTED = FailingReason(
        description="nms_kernel not implemented for bfloat16",
        checks=[
            # RuntimeError: 'nms_kernel' not implemented for 'BFloat16'
            ExceptionCheck(
                class_name="RuntimeError",
                error_log=[
                    M.any(M.contains("nms_kernel"), M.contains("torchvision.ops.nms")),
                    M.contains("not implemented"),
                    M.any(M.contains("BFloat16"), M.contains("bfloat16")),
                    M.any(
                        M.contains("torchvision/ops"), M.contains("torchvision.ops.nms")
                    ),
                ],
            ),
        ],
    )

    AUTOSHAPE_FORWARD_ARGS_MISMATCH = FailingReason(
        description="AutoShape forward received wrong number of arguments",
        checks=[
            # TypeError: AutoShape.forward() takes from 2 to 5 positional arguments but 7 were given
            ExceptionCheck(
                class_name="TypeError",
                error_log=[
                    M.contains("AutoShape.forward"),
                    M.any(M.contains("positional arguments"), M.contains("were given")),
                ],
            ),
        ],
    )

    DYNAMO_FAKE_TENSORS_FX_NODE = FailingReason(
        description="Dynamo failed to run FX node with fake tensors",
        checks=[
            # TorchRuntimeError: Dynamo failed to run FX node with fake tensors
            ExceptionCheck(
                error_log=[
                    M.any(
                        M.contains("Dynamo failed to run FX node with fake tensors"),
                        M.contains("Dynamo failed to run FX node"),
                    ),
                    M.any(
                        M.contains("torch/_dynamo"),
                        M.contains("torch._dynamo"),
                        M.contains("dynamo"),
                    ),
                ],
            ),
        ],
    )

    TENSOR_DATA_NOT_ALLOCATED = FailingReason(
        description="Tensor has non-zero elements but data not allocated",
        checks=[
            # RuntimeError: The tensor has a non-zero number of elements, but its data is not allocated yet
            ExceptionCheck(
                class_name="RuntimeError",
                error_log=[
                    M.contains("The tensor has a non-zero number of elements"),
                    M.contains("data is not allocated yet"),
                ],
            ),
        ],
    )

    DYNAMO_UNSUPPORTED_MODULE_CONTAINER = FailingReason(
        description="Dynamo assertion: unsupported module container (e.g., SparseSequential)",
        checks=[
            # AssertionError from torch/_dynamo when encountering unsupported container like SparseSequential
            ExceptionCheck(
                class_name="AssertionError",
                error_log=[
                    M.any(
                        M.contains("torch/_dynamo/variables/nn_module.py"),
                        M.contains("torch._dynamo"),
                    ),
                    M.any(
                        M.contains("SparseSequential"),
                        M.regex(
                            "assert isinstance\\(base, \\(torch\\.nn\\.ModuleList, .*\\)\\)"
                        ),
                    ),
                ],
            ),
        ],
    )

    TENSOR_CANNOT_CONVERT_TO_SCALAR = FailingReason(
        description="Tensor with multiple elements cannot be converted to Scalar",
        checks=[
            # RuntimeError: a Tensor with 3145728 elements cannot be converted to Scalar
            ExceptionCheck(
                class_name="RuntimeError",
                error_log=[
                    M.regex(
                        "a Tensor with \\d+ elements cannot be converted to Scalar"
                    ),
                ],
            ),
        ],
    )

    PHI3V_UNEXPECTED_MAX_NEW_TOKENS = FailingReason(
        description="Phi3VForCausalLM.forward got unexpected keyword 'max_new_tokens'",
        checks=[
            # TypeError: Phi3VForCausalLM.forward() got an unexpected keyword argument 'max_new_tokens'
            ExceptionCheck(
                class_name="TypeError",
                error_log=[
                    M.contains("Phi3VForCausalLM.forward"),
                    M.contains("unexpected keyword argument"),
                    M.contains("max_new_tokens"),
                ],
            ),
        ],
    )

    MODEL_NOT_TORCH_MODULE = FailingReason(
        description="Model is not a torch.nn.Module",
        checks=[
            # AssertionError at torch_model_tester.py:70: assert isinstance(self._model, torch.nn.Module)
            ExceptionCheck(
                class_name="AssertionError",
                error_log=[
                    M.last_line(
                        M.contains(
                            "infra/testers/single_chip/model/torch_model_tester.py:"
                        )
                    ),
                ],
            ),
        ],
    )

    UNSUPPORTED_INPUT_DATA_TYPE = FailingReason(
        description="Unsupported input data type",
        checks=[
            # E           RuntimeError: TT_FATAL @ /__w/tt-xla/tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_device_operation.cpp:28: input_datatype == DataType::INT32
            # E           info:
            # E           Unsupported input data type '5' for UnaryOpType '73' (Bitwise operation).
            # E           backtrace:
            # E            --- /__w/tt-forge-sweeps/tt-forge-sweeps/env/venv/xla/lib/python3.11/site-packages/pjrt_plugin_tt/lib/libtt_metal.so(+0x654549) [0x7fc490ae2549]
            # E            --- ttnn::operations::unary::UnaryDeviceOperation::validate_on_program_cache_miss(ttnn::operations::unary::operation_attributes_t const&amp;, ttnn::operations::unary::tensor_args_t const&amp;)
            # E            --- /__w/tt-forge-sweeps/tt-forge-sweeps/env/venv/xla/lib/python3.11/site-packages/pjrt_plugin_tt/lib/_ttnncpp.so(_ZN4ttnn16device_operation6detail29launch_operation_with_adapterINS0_26MeshDeviceOperationAdapterINS_10operations5unary20UnaryDeviceOperationEEEEEvRKNT_22operation_attributes_tERKNS8_13tensor_args_tERNS8_21tensor_return_value_tEPN2tt8tt_metal11distributed10MeshDeviceE+0x170) [0x7fc46b9623d0]
            # E            --- ttnn::operations::unary::UnaryDeviceOperation::tensor_return_value_t ttnn::device_operation::detail::launch_on_device&lt;ttnn::operations::unary::UnaryDeviceOperation&gt;(ttnn::operations::unary::UnaryDeviceOperation::operation_attributes_t const&amp;, ttnn::operations::unary::UnaryDeviceOperation::tensor_args_t const&amp;)
            # E            --- ttnn::operations::unary::UnaryDeviceOperation::tensor_return_value_t ttnn::device_operation::detail::invoke&lt;ttnn::operations::unary::UnaryDeviceOperation&gt;(ttnn::operations::unary::UnaryDeviceOperation::operation_attributes_t const&amp;, ttnn::operations::unary::UnaryDeviceOperation::tensor_args_t const&amp;)
            # E            --- /__w/tt-forge-sweeps/tt-forge-sweeps/env/venv/xla/lib/python3.11/site-packages/pjrt_plugin_tt/lib/_ttnncpp.so(+0xb73c78) [0x7fc46b961c78]
            # E            --- /__w/tt-forge-sweeps/tt-forge-sweeps/env/venv/xla/lib/python3.11/site-packages/pjrt_plugin_tt/lib/_ttnncpp.so(+0xb739c9) [0x7fc46b9619c9]
            # E            --- /__w/tt-forge-sweeps/tt-forge-sweeps/env/venv/xla/lib/python3.11/site-packages/pjrt_plugin_tt/lib/_ttnncpp.so(+0xb6980b) [0x7fc46b95780b]
            # E            --- ttnn::operations::unary::ExecuteUnary&lt;(ttnn::operations::unary::UnaryOpType)73&gt;::invoke(tt::tt_metal::Tensor const&amp;, std::optional&lt;tt::tt_metal::MemoryConfig&gt; const&amp;, std::optional&lt;tt::tt_metal::Tensor&gt; const&amp;)
            # E            --- /__w/tt-forge-sweeps/tt-forge-sweeps/env/venv/xla/lib/python3.11/site-packages/pjrt_plugin_tt/lib/libTTMLIRRuntime.so(+0x27f0c6) [0x7fc4a44f20c6]
            # E            --- /__w/tt-forge-sweeps/tt-forge-sweeps/env/venv/xla/lib/python3.11/site-packages/pjrt_plugin_tt/lib/libTTMLIRRuntime.so(+0x27f039) [0x7fc4a44f2039]
            # E            --- /__w/tt-forge-sweeps/tt-forge-sweeps/env/venv/xla/lib/python3.11/site-packages/pjrt_plugin_tt/lib/libTTMLIRRuntime.so(+0x27a9ce) [0x7fc4a44ed9ce]
            # E            --- tt::runtime::ttnn::operations::eltwise::unary::run(tt::target::ttnn::EltwiseUnaryOp const*, tt::runtime::ttnn::ProgramContext&amp;)
            # E            --- tt::runtime::ttnn::ProgramExecutor::execute()
            # E            --- tt::runtime::ttnn::submit(tt::runtime::Device, tt::runtime::Binary, unsigned int, std::vector&lt;tt::runtime::Tensor, std::allocator&lt;tt::runtime::Tensor&gt; &gt;&amp;)
            # E            --- tt::runtime::submit(tt::runtime::Device, tt::runtime::Binary, unsigned int, std::vector&lt;tt::runtime::Tensor, std::allocator&lt;tt::runtime::Tensor&gt; &gt;&amp;)
            # E            --- tt::pjrt::FlatbufferLoadedExecutableInstance::execute(PJRT_LoadedExecutable_Execute_Args*)
            # E            --- tt::pjrt::internal::onLoadedExecutableExecute(PJRT_LoadedExecutable_Execute_Args*)
            # E            --- xla::PjRtCApiLoadedExecutable::ExecuteWithSingleDevice(absl::lts_20230802::Span&lt;xla::PjRtBuffer* const&gt;, xla::PjRtDevice*, xla::ExecuteOptions const&amp;, std::optional&lt;xla::PjRtFuture&lt;void&gt; &gt;&amp;, bool)
            # E            --- xla::PjRtCApiLoadedExecutable::ExecutePortable(absl::lts_20230802::Span&lt;xla::PjRtBuffer* const&gt;, xla::PjRtDevice*, xla::ExecuteOptions const&amp;, std::optional&lt;xla::PjRtFuture&lt;void&gt; &gt;&amp;, bool)
            # E            --- torch_xla::runtime::PjRtComputationClient::ExecuteComputation(torch_xla::runtime::ComputationClient::Computation const&amp;, absl::lts_20230802::Span&lt;std::shared_ptr&lt;torch_xla::runtime::ComputationClient::Data&gt; const&gt;, std::__cxx11::basic_string&lt;char, std::char_traits&lt;char&gt;, std::allocator&lt;char&gt; &gt; const&amp;, torch_xla::runtime::ComputationClient::ExecuteComputationOptions const&amp;)
            # E            --- /__w/tt-forge-sweeps/tt-forge-sweeps/env/venv/xla/lib/python3.11/site-packages/_XLAC.cpython-311-x86_64-linux-gnu.so(+0x5f2b1ea) [0x7fc4f291b1ea]
            # E            --- torch::lazy::MultiWait::Complete(std::function&lt;void ()&gt; const&amp;)
            # E            --- Eigen::ThreadPoolTempl&lt;tsl::thread::EigenEnvironment&gt;::WorkerLoop(int)
            # E            --- void absl::lts_20230802::internal_any_invocable::RemoteInvoker&lt;false, void, tsl::thread::EigenEnvironment::CreateThread(std::function&lt;void ()&gt;)::{lambda()#1}&amp;&gt;(absl::lts_20230802::internal_any_invocable::TypeErasedState*)
            # E            --- /__w/tt-forge-sweeps/tt-forge-sweeps/env/venv/xla/lib/python3.11/site-packages/_XLAC.cpython-311-x86_64-linux-gnu.so(+0xfc35751) [0x7fc4fc625751]
            # E            --- /lib/x86_64-linux-gnu/libc.so.6(+0x94ac3) [0x7fc5f9463ac3]
            # E            --- /lib/x86_64-linux-gnu/libc.so.6(clone+0x44) [0x7fc5f94f4a74]
            # forge/infra/runners/torch_device_runner.py:40: RuntimeError
            ExceptionCheck(
                class_name="RuntimeError",
                component=ComponentChecker.METAL.value,
                message=[
                    M.regex(
                        "Unsupported input data type .* for UnaryOpType .* \\(Bitwise operation\\)\\."
                    ),
                ],
                error_log=[
                    M.last_line(M.contains("infra/runners/torch_device_runner.py:")),
                ],
            ),
        ],
    )

    CIRCULAR_BUFFER_CLASH = FailingReason(
        description="Circular buffer clash",
        checks=[
            # x = &lt;[RuntimeError('torch_xla/csrc/tensor.cpp:217 : Check failed: handle-&gt;HasValue() \n*** Begin stack trace ***\n\ttsl::C...h ID 10179 while an async operation is in flight: UNKNOWN_SCALAR[]') raised in repr()] Tensor object at 0x7f2ed9ce5250&gt;
            #     def attempt_to_device(x):
            #         if hasattr(x, "to"):
            # &gt;           return x.to(device)
            #                    ^^^^^^^^^^^^
            # E           RuntimeError: TT_THROW @ /__w/tt-xla/tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/impl/program/program.cpp:916: tt::exception
            # E           info:
            # E           Statically allocated circular buffers in program 17780 clash with L1 buffers on core range [(x=0,y=0) - (x=6,y=7)]. L1 buffer allocated at 182016 and static circular buffer region ends at 216192
            # E           backtrace:
            # E            --- /__w/tt-forge-sweeps/tt-forge-sweeps/env/venv/xla/lib/python3.11/site-packages/pjrt_plugin_tt/lib/libtt_metal.so(+0x585c85) [0x7f2ea6a17c85]
            # E            --- tt::tt_metal::detail::ProgramImpl::validate_circular_buffer_region(tt::tt_metal::IDevice const*)
            # E            --- tt::tt_metal::distributed::MeshWorkloadImpl::compile(tt::tt_metal::distributed::MeshDevice*)
            # E            --- tt::tt_metal::distributed::EnqueueMeshWorkload(tt::tt_metal::distributed::MeshCommandQueue&amp;, tt::tt_metal::distributed::MeshWorkload&amp;, bool)
            # E            --- /__w/tt-forge-sweeps/tt-forge-sweeps/env/venv/xla/lib/python3.11/site-packages/pjrt_plugin_tt/lib/_ttnncpp.so(+0x6bd8e2) [0x7f2ea593d8e2]
            # E            --- /__w/tt-forge-sweeps/tt-forge-sweeps/env/venv/xla/lib/python3.11/site-packages/pjrt_plugin_tt/lib/_ttnncpp.so(_ZN4ttnn16device_operation6detail29launch_operation_with_adapterINS0_26MeshDeviceOperationAdapterIN2tt8tt_metal9operation23OldInfraDeviceOperationISt6vectorINS5_6TensorESaIS9_EEEEEEEEvRKNT_22operation_attributes_tERKNSE_13tensor_args_tERNSE_21tensor_return_value_tEPNS5_11distributed10MeshDeviceE+0x2c3) [0x7f2ea593b003]
            # E            --- tt::tt_metal::operation::OldInfraDeviceOperation&lt;std::vector&lt;tt::tt_metal::Tensor, std::allocator&lt;tt::tt_metal::Tensor&gt; &gt; &gt;::tensor_return_value_t ttnn::device_operation::detail::launch_on_device&lt;tt::tt_metal::operation::OldInfraDeviceOperation&lt;std::vector&lt;tt::tt_metal::Tensor, std::allocator&lt;tt::tt_metal::Tensor&gt; &gt; &gt; &gt;(tt::tt_metal::operation::OldInfraDeviceOperation&lt;std::vector&lt;tt::tt_metal::Tensor, std::allocator&lt;tt::tt_metal::Tensor&gt; &gt; &gt;::operation_attributes_t const&amp;, tt::tt_metal::operation::OldInfraDeviceOperation&lt;std::vector&lt;tt::tt_metal::Tensor, std::allocator&lt;tt::tt_metal::Tensor&gt; &gt; &gt;::tensor_args_t const&amp;)
            # E            --- tt::tt_metal::operation::OldInfraDeviceOperation&lt;std::vector&lt;tt::tt_metal::Tensor, std::allocator&lt;tt::tt_metal::Tensor&gt; &gt; &gt;::tensor_return_value_t ttnn::device_operation::detail::invoke&lt;tt::tt_metal::operation::OldInfraDeviceOperation&lt;std::vector&lt;tt::tt_metal::Tensor, std::allocator&lt;tt::tt_metal::Tensor&gt; &gt; &gt; &gt;(tt::tt_metal::operation::OldInfraDeviceOperation&lt;std::vector&lt;tt::tt_metal::Tensor, std::allocator&lt;tt::tt_metal::Tensor&gt; &gt; &gt;::operation_attributes_t const&amp;, tt::tt_metal::operation::OldInfraDeviceOperation&lt;std::vector&lt;tt::tt_metal::Tensor, std::allocator&lt;tt::tt_metal::Tensor&gt; &gt; &gt;::tensor_args_t const&amp;)
            # E            --- /__w/tt-forge-sweeps/tt-forge-sweeps/env/venv/xla/lib/python3.11/site-packages/pjrt_plugin_tt/lib/_ttnncpp.so(+0x6b9f6b) [0x7f2ea5939f6b]
            # E            --- std::vector&lt;tt::tt_metal::Tensor, std::allocator&lt;tt::tt_metal::Tensor&gt; &gt; tt::tt_metal::operation::run_without_autoformat&lt;std::vector&lt;tt::tt_metal::Tensor, std::allocator&lt;tt::tt_metal::Tensor&gt; &gt; &gt;(tt::tt_metal::operation::DeviceOperation&lt;std::vector&lt;tt::tt_metal::Tensor, std::allocator&lt;tt::tt_metal::Tensor&gt; &gt; &gt;&amp;&amp;, std::vector&lt;tt::tt_metal::Tensor, std::allocator&lt;tt::tt_metal::Tensor&gt; &gt; const&amp;, std::vector&lt;std::optional&lt;tt::tt_metal::Tensor const&gt;, std::allocator&lt;std::optional&lt;tt::tt_metal::Tensor const&gt; &gt; &gt; const&amp;, std::vector&lt;std::optional&lt;tt::tt_metal::Tensor&gt;, std::allocator&lt;std::optional&lt;tt::tt_metal::Tensor&gt; &gt; &gt; const&amp;)
            # E            --- /__w/tt-forge-sweeps/tt-forge-sweeps/env/venv/xla/lib/python3.11/site-packages/pjrt_plugin_tt/lib/_ttnncpp.so(_ZN4ttnn10operations4conv6conv2d6conv2dERKN2tt8tt_metal6TensorES7_St8optionalIS6_ERKNS0_14sliding_window19SlidingWindowConfigEjjbRKS8_INS0_5unary19BasicUnaryWithParamIJfEEEERKNS2_27Conv2dParallelizationConfigERKNS2_17Conv2dBlockConfigERKNS4_12MemoryConfigENS4_8DataTypeESt5arrayIjLm4EERKSt7variantIJNS_28GrayskullComputeKernelConfigENS_27WormholeComputeKernelConfigEEEbbbbbS8_IbE+0x731) [0x7f2ea5afe921]
            # E            --- /__w/tt-forge-sweeps/tt-forge-sweeps/env/venv/xla/lib/python3.11/site-packages/pjrt_plugin_tt/lib/_ttnncpp.so(_ZN4ttnn10operations4conv6conv2d9conv2d_L1ERKN2tt8tt_metal6TensorES7_PNS4_11distributed10MeshDeviceEjjjjjSt5arrayIjLm2EESC_St7variantIJSC_SB_IjLm4EEEESC_jRKSt8optionalIKNS4_8DataTypeEERKSG_IS6_ERKSG_IKNS2_12Conv2dConfigEERKSG_IKSD_IJNS_28GrayskullComputeKernelConfigENS_27WormholeComputeKernelConfigEEEERKSG_IKNS4_12MemoryConfigEE+0x1af5) [0x7f2ea5af6c55]
            # E            --- /__w/tt-forge-sweeps/tt-forge-sweeps/env/venv/xla/lib/python3.11/site-packages/pjrt_plugin_tt/lib/_ttnncpp.so(_ZN4ttnn10operations4conv6conv2d6conv2dERKN2tt8tt_metal6TensorES7_PNS4_11distributed10MeshDeviceEjjjjjSt5arrayIjLm2EESC_St7variantIJSC_SB_IjLm4EEEESC_jRKSt8optionalIKNS4_8DataTypeEERKSG_IS6_ERKSG_IKNS2_12Conv2dConfigEERKSG_IKSD_IJNS_28GrayskullComputeKernelConfigENS_27WormholeComputeKernelConfigEEEERKSG_IKNS4_12MemoryConfigEERKSG_IKNS2_17Conv2dSliceConfigEEbb+0x110) [0x7f2ea5af4e00]
            # E            --- /__w/tt-forge-sweeps/tt-forge-sweeps/env/venv/xla/lib/python3.11/site-packages/pjrt_plugin_tt/lib/_ttnncpp.so(_ZN4ttnn10operations4conv6conv2d15Conv2dOperation6invokeERKN2tt8tt_metal6TensorES8_PNS5_11distributed10MeshDeviceEjjjjjSt5arrayIjLm2EESD_St7variantIJSD_SC_IjLm4EEEESD_jRKSt8optionalIKNS5_8DataTypeEERKSH_IS7_ERKSH_IKNS2_12Conv2dConfigEERKSH_IKSE_IJNS_28GrayskullComputeKernelConfigENS_27WormholeComputeKernelConfigEEEERKSH_IKNS5_12MemoryConfigEERKSH_IKNS2_17Conv2dSliceConfigEEbb+0xca) [0x7f2ea5aff3ea]
            # E            --- /__w/tt-forge-sweeps/tt-forge-sweeps/env/venv/xla/lib/python3.11/site-packages/pjrt_plugin_tt/lib/libTTMLIRRuntime.so(+0x25783d) [0x7f2ec84ca83d]
            # E            --- /__w/tt-forge-sweeps/tt-forge-sweeps/env/venv/xla/lib/python3.11/site-packages/pjrt_plugin_tt/lib/libTTMLIRRuntime.so(+0x256d88) [0x7f2ec84c9d88]
            # E            --- tt::runtime::ttnn::operations::conv::run(tt::target::ttnn::Conv2dOp const*, tt::runtime::ttnn::ProgramContext&amp;)
            # E            --- tt::runtime::ttnn::ProgramExecutor::execute()
            # E            --- tt::runtime::ttnn::submit(tt::runtime::Device, tt::runtime::Binary, unsigned int, std::vector&lt;tt::runtime::Tensor, std::allocator&lt;tt::runtime::Tensor&gt; &gt;&amp;)
            # E            --- tt::runtime::submit(tt::runtime::Device, tt::runtime::Binary, unsigned int, std::vector&lt;tt::runtime::Tensor, std::allocator&lt;tt::runtime::Tensor&gt; &gt;&amp;)
            # E            --- tt::pjrt::FlatbufferLoadedExecutableInstance::execute(PJRT_LoadedExecutable_Execute_Args*)
            # E            --- tt::pjrt::internal::onLoadedExecutableExecute(PJRT_LoadedExecutable_Execute_Args*)
            # E            --- xla::PjRtCApiLoadedExecutable::ExecuteWithSingleDevice(absl::lts_20230802::Span&lt;xla::PjRtBuffer* const&gt;, xla::PjRtDevice*, xla::ExecuteOptions const&amp;, std::optional&lt;xla::PjRtFuture&lt;void&gt; &gt;&amp;, bool)
            # E            --- xla::PjRtCApiLoadedExecutable::ExecutePortable(absl::lts_20230802::Span&lt;xla::PjRtBuffer* const&gt;, xla::PjRtDevice*, xla::ExecuteOptions const&amp;, std::optional&lt;xla::PjRtFuture&lt;void&gt; &gt;&amp;, bool)
            # E            --- torch_xla::runtime::PjRtComputationClient::ExecuteComputation(torch_xla::runtime::ComputationClient::Computation const&amp;, absl::lts_20230802::Span&lt;std::shared_ptr&lt;torch_xla::runtime::ComputationClient::Data&gt; const&gt;, std::__cxx11::basic_string&lt;char, std::char_traits&lt;char&gt;, std::allocator&lt;char&gt; &gt; const&amp;, torch_xla::runtime::ComputationClient::ExecuteComputationOptions const&amp;)
            # E            --- /__w/tt-forge-sweeps/tt-forge-sweeps/env/venv/xla/lib/python3.11/site-packages/_XLAC.cpython-311-x86_64-linux-gnu.so(+0x5f2b1ea) [0x7f2f0d11b1ea]
            # E            --- torch::lazy::MultiWait::Complete(std::function&lt;void ()&gt; const&amp;)
            # E            --- Eigen::ThreadPoolTempl&lt;tsl::thread::EigenEnvironment&gt;::WorkerLoop(int)
            # E            --- void absl::lts_20230802::internal_any_invocable::RemoteInvoker&lt;false, void, tsl::thread::EigenEnvironment::CreateThread(std::function&lt;void ()&gt;)::{lambda()#1}&amp;&gt;(absl::lts_20230802::internal_any_invocable::TypeErasedState*)
            # E            --- /__w/tt-forge-sweeps/tt-forge-sweeps/env/venv/xla/lib/python3.11/site-packages/_XLAC.cpython-311-x86_64-linux-gnu.so(+0xfc35751) [0x7f2f16e25751]
            # E            --- /lib/x86_64-linux-gnu/libc.so.6(+0x94ac3) [0x7f3013c2dac3]
            # E            --- /lib/x86_64-linux-gnu/libc.so.6(clone+0x44) [0x7f3013cbea74]
            # forge/infra/runners/torch_device_runner.py:40: RuntimeError
            ExceptionCheck(
                class_name="RuntimeError",
                component=ComponentChecker.METAL.value,
                message=[
                    M.regex(
                        "Statically allocated circular buffers in program .* clash with L1 buffers on core range .*\\. L1 buffer allocated at .* and static circular buffer region ends at .*"
                    ),
                ],
                error_log=[
                    M.regex(
                        "RuntimeError: TT_THROW @ .*/tt_metal/impl/program/program.cpp:.*: tt::exception"
                    ),
                    M.last_line(M.contains("infra/runners/torch_device_runner.py:")),
                ],
            ),
        ],
    )

    INVALID_ARGUMENTS_TO_RESHAPE = FailingReason(
        description="Invalid arguments to reshape",
        checks=[
            #    x = &lt;[RuntimeError('TT_FATAL @ /__w/tt-xla/tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/c...--- /lib/x86_64-linux-gnu/libc.so.6(clone+0x44) [0x7f3013cbea74]\n') raised in repr()] Tensor object at 0x7f2e4052e030&gt;
            #     def attempt_to_device(x):
            #         if hasattr(x, "to"):
            # &gt;           return x.to(device)
            #                    ^^^^^^^^^^^^
            # E           RuntimeError: TT_FATAL @ /__w/tt-xla/tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/core/tensor/tensor_utils.cpp:55: new_volume == old_volume
            # E           info:
            # E           Invalid arguments to reshape
            # E           backtrace:
            # E            --- /__w/tt-forge-sweeps/tt-forge-sweeps/env/venv/xla/lib/python3.11/site-packages/pjrt_plugin_tt/lib/libTTMLIRCompiler.so(+0xa56a1f8) [0x7f2e722161f8]
            # E            --- tt::tt_metal::infer_dims_for_reshape(tt::tt_metal::Tensor const&amp;, std::span&lt;int const, 18446744073709551615ul&gt;)
            # E            --- ttnn::operations::data_movement::ReshapeViewOperation::invoke(tt::tt_metal::Tensor const&amp;, std::span&lt;int const, 18446744073709551615ul&gt;, std::optional&lt;tt::tt_metal::MemoryConfig&gt; const&amp;, std::optional&lt;std::variant&lt;unsigned int, float&gt; &gt; const&amp;)
            # E            --- /__w/tt-forge-sweeps/tt-forge-sweeps/env/venv/xla/lib/python3.11/site-packages/pjrt_plugin_tt/lib/libTTMLIRRuntime.so(+0x2695cc) [0x7f2ec84dc5cc]
            # E            --- tt::runtime::ttnn::operations::data_movement::run(tt::target::ttnn::ReshapeOp const*, tt::runtime::ttnn::ProgramContext&amp;)
            # E            --- tt::runtime::ttnn::ProgramExecutor::execute()
            # E            --- tt::runtime::ttnn::submit(tt::runtime::Device, tt::runtime::Binary, unsigned int, std::vector&lt;tt::runtime::Tensor, std::allocator&lt;tt::runtime::Tensor&gt; &gt;&amp;)
            # E            --- tt::runtime::submit(tt::runtime::Device, tt::runtime::Binary, unsigned int, std::vector&lt;tt::runtime::Tensor, std::allocator&lt;tt::runtime::Tensor&gt; &gt;&amp;)
            # E            --- tt::pjrt::FlatbufferLoadedExecutableInstance::execute(PJRT_LoadedExecutable_Execute_Args*)
            # E            --- tt::pjrt::internal::onLoadedExecutableExecute(PJRT_LoadedExecutable_Execute_Args*)
            # E            --- xla::PjRtCApiLoadedExecutable::ExecuteWithSingleDevice(absl::lts_20230802::Span&lt;xla::PjRtBuffer* const&gt;, xla::PjRtDevice*, xla::ExecuteOptions const&amp;, std::optional&lt;xla::PjRtFuture&lt;void&gt; &gt;&amp;, bool)
            # E            --- xla::PjRtCApiLoadedExecutable::ExecutePortable(absl::lts_20230802::Span&lt;xla::PjRtBuffer* const&gt;, xla::PjRtDevice*, xla::ExecuteOptions const&amp;, std::optional&lt;xla::PjRtFuture&lt;void&gt; &gt;&amp;, bool)
            # E            --- torch_xla::runtime::PjRtComputationClient::ExecuteComputation(torch_xla::runtime::ComputationClient::Computation const&amp;, absl::lts_20230802::Span&lt;std::shared_ptr&lt;torch_xla::runtime::ComputationClient::Data&gt; const&gt;, std::__cxx11::basic_string&lt;char, std::char_traits&lt;char&gt;, std::allocator&lt;char&gt; &gt; const&amp;, torch_xla::runtime::ComputationClient::ExecuteComputationOptions const&amp;)
            # E            --- /__w/tt-forge-sweeps/tt-forge-sweeps/env/venv/xla/lib/python3.11/site-packages/_XLAC.cpython-311-x86_64-linux-gnu.so(+0x5f2b1ea) [0x7f2f0d11b1ea]
            # E            --- torch::lazy::MultiWait::Complete(std::function&lt;void ()&gt; const&amp;)
            # E            --- Eigen::ThreadPoolTempl&lt;tsl::thread::EigenEnvironment&gt;::WorkerLoop(int)
            # E            --- void absl::lts_20230802::internal_any_invocable::RemoteInvoker&lt;false, void, tsl::thread::EigenEnvironment::CreateThread(std::function&lt;void ()&gt;)::{lambda()#1}&amp;&gt;(absl::lts_20230802::internal_any_invocable::TypeErasedState*)
            # E            --- /__w/tt-forge-sweeps/tt-forge-sweeps/env/venv/xla/lib/python3.11/site-packages/_XLAC.cpython-311-x86_64-linux-gnu.so(+0xfc35751) [0x7f2f16e25751]
            # E            --- /lib/x86_64-linux-gnu/libc.so.6(+0x94ac3) [0x7f3013c2dac3]
            # E            --- /lib/x86_64-linux-gnu/libc.so.6(clone+0x44) [0x7f3013cbea74]
            # forge/infra/runners/torch_device_runner.py:40: RuntimeError
            ExceptionCheck(
                class_name="RuntimeError",
                component=ComponentChecker.TTNN.value,
                message=[
                    M.regex(
                        ".*/src/tt-metal/ttnn/core/tensor/tensor_utils\\.cpp.* new_volume == old_volume.*"
                    ),
                    M.contains("Invalid arguments to reshape"),
                ],
                error_log=[
                    M.last_line(M.contains("infra/runners/torch_device_runner.py:")),
                ],
            ),
        ],
    )

    XLA_ASYNC_OPERATION_IN_FLIGHT = FailingReason(
        description="Trying to access XLA data while async operation is in flight",
        checks=[
            # RuntimeError('Check failed: handle->HasValue(): Trying to access XLA data for tensor with ID <id> while an async operation is in flight:')
            ExceptionCheck(
                class_name="RuntimeError",
                message=[
                    M.any(
                        M.contains("handle->HasValue"),
                        M.contains("torch_xla/csrc/tensor.cpp"),
                    ),
                    M.contains("async operation is in flight"),
                ],
            ),
        ],
    )

    MATMUL_UNSUPPORTED_DATA_FORMAT = FailingReason(
        description="Unsupported data format in matmul_op.cpp",
        checks=[
            #     x = &lt;[RuntimeError('TT_FATAL @ /__w/tt-xla/tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/c...--- /lib/x86_64-linux-gnu/libc.so.6(clone+0x44) [0x7f3013cbea74]\n') raised in repr()] Tensor object at 0x7f2eda82d970&gt;
            #     def attempt_to_device(x):
            #         if hasattr(x, "to"):
            # &gt;           return x.to(device)
            #                 ^^^^^^^^^^^^
            # E           RuntimeError: TT_FATAL @ /__w/tt-xla/tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/matmul_op.cpp:1700: is_floating_point(input_tensor_a.dtype())
            # E           info:
            # E           Unsupported data format
            # E           backtrace:
            # E            --- /__w/tt-forge-sweeps/tt-forge-sweeps/env/venv/xla/lib/python3.11/site-packages/pjrt_plugin_tt/lib/libTTMLIRCompiler.so(+0xa56a1f8) [0x7f2e722161f8]
            # E            --- ttnn::operations::matmul::Matmul::validate(std::vector&lt;tt::tt_metal::Tensor, std::allocator&lt;tt::tt_metal::Tensor&gt; &gt; const&amp;, std::vector&lt;std::optional&lt;tt::tt_metal::Tensor const&gt;, std::allocator&lt;std::optional&lt;tt::tt_metal::Tensor const&gt; &gt; &gt; const&amp;, std::vector&lt;std::optional&lt;tt::tt_metal::Tensor&gt;, std::allocator&lt;std::optional&lt;tt::tt_metal::Tensor&gt; &gt; &gt; const&amp;) const
            # E            --- /__w/tt-forge-sweeps/tt-forge-sweeps/env/venv/xla/lib/python3.11/site-packages/pjrt_plugin_tt/lib/_ttnncpp.so(+0xe4e21f) [0x7f2ea60ce21f]
            # E            --- /__w/tt-forge-sweeps/tt-forge-sweeps/env/venv/xla/lib/python3.11/site-packages/pjrt_plugin_tt/lib/_ttnncpp.so(+0xe4e197) [0x7f2ea60ce197]
            # E            --- /__w/tt-forge-sweeps/tt-forge-sweeps/env/venv/xla/lib/python3.11/site-packages/pjrt_plugin_tt/lib/_ttnncpp.so(_ZN4ttnn16device_operation6detail29launch_operation_with_adapterINS0_26MeshDeviceOperationAdapterIN2tt8tt_metal9operation23OldInfraDeviceOperationISt6vectorINS5_6TensorESaIS9_EEEEEEEEvRKNT_22operation_attributes_tERKNSE_13tensor_args_tERNSE_21tensor_return_value_tEPNS5_11distributed10MeshDeviceE+0x238) [0x7f2ea593af78]
            # E            --- tt::tt_metal::operation::OldInfraDeviceOperation&lt;std::vector&lt;tt::tt_metal::Tensor, std::allocator&lt;tt::tt_metal::Tensor&gt; &gt; &gt;::tensor_return_value_t ttnn::device_operation::detail::launch_on_device&lt;tt::tt_metal::operation::OldInfraDeviceOperation&lt;std::vector&lt;tt::tt_metal::Tensor, std::allocator&lt;tt::tt_metal::Tensor&gt; &gt; &gt; &gt;(tt::tt_metal::operation::OldInfraDeviceOperation&lt;std::vector&lt;tt::tt_metal::Tensor, std::allocator&lt;tt::tt_metal::Tensor&gt; &gt; &gt;::operation_attributes_t const&amp;, tt::tt_metal::operation::OldInfraDeviceOperation&lt;std::vector&lt;tt::tt_metal::Tensor, std::allocator&lt;tt::tt_metal::Tensor&gt; &gt; &gt;::tensor_args_t const&amp;)
            # E            --- tt::tt_metal::operation::OldInfraDeviceOperation&lt;std::vector&lt;tt::tt_metal::Tensor, std::allocator&lt;tt::tt_metal::Tensor&gt; &gt; &gt;::tensor_return_value_t ttnn::device_operation::detail::invoke&lt;tt::tt_metal::operation::OldInfraDeviceOperation&lt;std::vector&lt;tt::tt_metal::Tensor, std::allocator&lt;tt::tt_metal::Tensor&gt; &gt; &gt; &gt;(tt::tt_metal::operation::OldInfraDeviceOperation&lt;std::vector&lt;tt::tt_metal::Tensor, std::allocator&lt;tt::tt_metal::Tensor&gt; &gt; &gt;::operation_attributes_t const&amp;, tt::tt_metal::operation::OldInfraDeviceOperation&lt;std::vector&lt;tt::tt_metal::Tensor, std::allocator&lt;tt::tt_metal::Tensor&gt; &gt; &gt;::tensor_args_t const&amp;)
            # E            --- /__w/tt-forge-sweeps/tt-forge-sweeps/env/venv/xla/lib/python3.11/site-packages/pjrt_plugin_tt/lib/_ttnncpp.so(+0x6b9f6b) [0x7f2ea5939f6b]
            # E            --- std::vector&lt;tt::tt_metal::Tensor, std::allocator&lt;tt::tt_metal::Tensor&gt; &gt; tt::tt_metal::operation::run&lt;std::vector&lt;tt::tt_metal::Tensor, std::allocator&lt;tt::tt_metal::Tensor&gt; &gt; &gt;(tt::tt_metal::operation::DeviceOperation&lt;std::vector&lt;tt::tt_metal::Tensor, std::allocator&lt;tt::tt_metal::Tensor&gt; &gt; &gt;&amp;&amp;, std::vector&lt;tt::tt_metal::Tensor, std::allocator&lt;tt::tt_metal::Tensor&gt; &gt; const&amp;, std::vector&lt;std::optional&lt;tt::tt_metal::Tensor const&gt;, std::allocator&lt;std::optional&lt;tt::tt_metal::Tensor const&gt; &gt; &gt; const&amp;, std::vector&lt;std::optional&lt;tt::tt_metal::Tensor&gt;, std::allocator&lt;std::optional&lt;tt::tt_metal::Tensor&gt; &gt; &gt; const&amp;)
            # E            --- ttnn::operations::matmul::matmul(tt::tt_metal::Tensor const&amp;, tt::tt_metal::Tensor const&amp;, std::optional&lt;tt::tt_metal::Tensor const&gt; const&amp;, ttnn::operations::matmul::Matmul const&amp;, std::optional&lt;tt::tt_metal::Tensor&gt; const&amp;)
            # E            --- ttnn::operations::matmul::bound_matmul(tt::tt_metal::Tensor const&amp;, tt::tt_metal::Tensor const&amp;, std::optional&lt;tt::tt_metal::Tensor const&gt; const&amp;, ttnn::operations::matmul::Matmul const&amp;, std::optional&lt;tt::tt_metal::Tensor&gt;&amp;)
            # E            --- /__w/tt-forge-sweeps/tt-forge-sweeps/env/venv/xla/lib/python3.11/site-packages/pjrt_plugin_tt/lib/_ttnncpp.so(_ZN4ttnn10operations6matmul15MatmulOperation6invokeERKN2tt8tt_metal6TensorES7_bbRKSt8optionalIKNS4_12MemoryConfigEES8_IKNS4_8DataTypeEERKS8_IKSt7variantIJNS1_28MatmulMultiCoreProgramConfigENS1_33MatmulMultiCoreReuseProgramConfigENS1_42MatmulMultiCoreReuseMultiCastProgramConfigENS1_44MatmulMultiCoreReuseMultiCast1DProgramConfigENS1_53MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigEEEERKS8_IKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEES8_IKSH_IJNS_28GrayskullComputeKernelConfigENS_27WormholeComputeKernelConfigEEEES8_IKNS_5types8CoreGridEERKS8_IKNS4_4TileEES8_IS5_ERKS8_IKNS4_12experimental20GlobalCircularBufferEERKS8_IN4ttsl10StrongTypeIhNS4_14SubDeviceIdTagEEEE+0x2fd) [0x7f2ea60b4b1d]
            # E            --- /__w/tt-forge-sweeps/tt-forge-sweeps/env/venv/xla/lib/python3.11/site-packages/pjrt_plugin_tt/lib/libTTMLIRRuntime.so(+0x2928e0) [0x7f2ec85058e0]
            # E            --- /__w/tt-forge-sweeps/tt-forge-sweeps/env/venv/xla/lib/python3.11/site-packages/pjrt_plugin_tt/lib/libTTMLIRRuntime.so(+0x292303) [0x7f2ec8505303]
            # E            --- tt::runtime::ttnn::operations::matmul::run(tt::target::ttnn::MatmulOp const*, tt::runtime::ttnn::ProgramContext&amp;)
            # E            --- tt::runtime::ttnn::ProgramExecutor::execute()
            # E            --- tt::runtime::ttnn::submit(tt::runtime::Device, tt::runtime::Binary, unsigned int, std::vector&lt;tt::runtime::Tensor, std::allocator&lt;tt::runtime::Tensor&gt; &gt;&amp;)
            # E            --- tt::runtime::submit(tt::runtime::Device, tt::runtime::Binary, unsigned int, std::vector&lt;tt::runtime::Tensor, std::allocator&lt;tt::runtime::Tensor&gt; &gt;&amp;)
            # E            --- tt::pjrt::FlatbufferLoadedExecutableInstance::execute(PJRT_LoadedExecutable_Execute_Args*)
            # E            --- tt::pjrt::internal::onLoadedExecutableExecute(PJRT_LoadedExecutable_Execute_Args*)
            # E            --- xla::PjRtCApiLoadedExecutable::ExecuteWithSingleDevice(absl::lts_20230802::Span&lt;xla::PjRtBuffer* const&gt;, xla::PjRtDevice*, xla::ExecuteOptions const&amp;, std::optional&lt;xla::PjRtFuture&lt;void&gt; &gt;&amp;, bool)
            # E            --- xla::PjRtCApiLoadedExecutable::ExecutePortable(absl::lts_20230802::Span&lt;xla::PjRtBuffer* const&gt;, xla::PjRtDevice*, xla::ExecuteOptions const&amp;, std::optional&lt;xla::PjRtFuture&lt;void&gt; &gt;&amp;, bool)
            # E            --- torch_xla::runtime::PjRtComputationClient::ExecuteComputation(torch_xla::runtime::ComputationClient::Computation const&amp;, absl::lts_20230802::Span&lt;std::shared_ptr&lt;torch_xla::runtime::ComputationClient::Data&gt; const&gt;, std::__cxx11::basic_string&lt;char, std::char_traits&lt;char&gt;, std::allocator&lt;char&gt; &gt; const&amp;, torch_xla::runtime::ComputationClient::ExecuteComputationOptions const&amp;)
            # E            --- /__w/tt-forge-sweeps/tt-forge-sweeps/env/venv/xla/lib/python3.11/site-packages/_XLAC.cpython-311-x86_64-linux-gnu.so(+0x5f2b1ea) [0x7f2f0d11b1ea]
            # E            --- torch::lazy::MultiWait::Complete(std::function&lt;void ()&gt; const&amp;)
            # E            --- Eigen::ThreadPoolTempl&lt;tsl::thread::EigenEnvironment&gt;::WorkerLoop(int)
            # E            --- void absl::lts_20230802::internal_any_invocable::RemoteInvoker&lt;false, void, tsl::thread::EigenEnvironment::CreateThread(std::function&lt;void ()&gt;)::{lambda()#1}&amp;&gt;(absl::lts_20230802::internal_any_invocable::TypeErasedState*)
            # E            --- /__w/tt-forge-sweeps/tt-forge-sweeps/env/venv/xla/lib/python3.11/site-packages/_XLAC.cpython-311-x86_64-linux-gnu.so(+0xfc35751) [0x7f2f16e25751]
            # E            --- /lib/x86_64-linux-gnu/libc.so.6(+0x94ac3) [0x7f3013c2dac3]
            # E            --- /lib/x86_64-linux-gnu/libc.so.6(clone+0x44) [0x7f3013cbea74]
            # forge/infra/runners/torch_device_runner.py:40: RuntimeError
            ExceptionCheck(
                class_name="RuntimeError",
                component=ComponentChecker.TTNN.value,
                message=[
                    M.regex(
                        ".*/src/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/matmul_op\\.cpp.* is_floating_point\\(input_tensor_a\\.dtype\\(\\)\\).*"
                    ),
                    M.contains("Unsupported data format"),
                ],
                error_log=[
                    M.last_line(M.contains("infra/runners/torch_device_runner.py:")),
                ],
            ),
        ],
    )

    SOFTMAX_NUMERIC_STABLE = FailingReason(
        description="For softmax, cannot enable both large_kernel and numeric_stable",
        checks=[
            #     x = &lt;[RuntimeError('TT_FATAL @ /__w/tt-xla/tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/c...--- /lib/x86_64-linux-gnu/libc.so.6(clone+0x44) [0x7f3013cbea74]\n') raised in repr()] Tensor object at 0x7f2d2d57aed0&gt;
            #     def attempt_to_device(x):
            #         if hasattr(x, "to"):
            # &gt;           return x.to(device)
            #                 ^^^^^^^^^^^^
            # E           RuntimeError: TT_FATAL @ /__w/tt-xla/tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/normalization/softmax/device/softmax_program_factory.cpp:818: !attributes.numeric_stable
            # E           info:
            # E           For softmax, cannot enable both large_kernel and numeric_stable
            # E           backtrace:
            # E            --- /__w/tt-forge-sweeps/tt-forge-sweeps/env/venv/xla/lib/python3.11/site-packages/pjrt_plugin_tt/lib/libTTMLIRCompiler.so(+0xa56a1f8) [0x7f2e722161f8]
            # E            --- ttnn::operations::normalization::softmax::program::SoftmaxProgramFactoryAttentionOptimized::create(ttnn::operations::normalization::softmax::operation_attributes_t const&amp;, ttnn::operations::normalization::softmax::tensor_args_t const&amp;, tt::tt_metal::Tensor&amp;)
            # E            --- /__w/tt-forge-sweeps/tt-forge-sweeps/env/venv/xla/lib/python3.11/site-packages/pjrt_plugin_tt/lib/_ttnncpp.so(+0x10def83) [0x7f2ea635ef83]
            # E            --- /__w/tt-forge-sweeps/tt-forge-sweeps/env/venv/xla/lib/python3.11/site-packages/pjrt_plugin_tt/lib/_ttnncpp.so(+0x10de5c9) [0x7f2ea635e5c9]
            # E            --- /__w/tt-forge-sweeps/tt-forge-sweeps/env/venv/xla/lib/python3.11/site-packages/pjrt_plugin_tt/lib/_ttnncpp.so(_ZN4ttnn16device_operation6detail29launch_operation_with_adapterINS0_26MeshDeviceOperationAdapterINS_10operations13normalization7softmax22SoftmaxDeviceOperationEEEEEvRKNT_22operation_attributes_tERKNS9_13tensor_args_tERNS9_21tensor_return_value_tEPN2tt8tt_metal11distributed10MeshDeviceE+0x274) [0x7f2ea6352cd4]
            # E            --- ttnn::operations::normalization::softmax::SoftmaxDeviceOperation::tensor_return_value_t ttnn::device_operation::detail::launch_on_device&lt;ttnn::operations::normalization::softmax::SoftmaxDeviceOperation&gt;(ttnn::operations::normalization::softmax::SoftmaxDeviceOperation::operation_attributes_t const&amp;, ttnn::operations::normalization::softmax::SoftmaxDeviceOperation::tensor_args_t const&amp;)
            # E            --- ttnn::operations::normalization::softmax::SoftmaxDeviceOperation::tensor_return_value_t ttnn::device_operation::detail::invoke&lt;ttnn::operations::normalization::softmax::SoftmaxDeviceOperation&gt;(ttnn::operations::normalization::softmax::SoftmaxDeviceOperation::operation_attributes_t const&amp;, ttnn::operations::normalization::softmax::SoftmaxDeviceOperation::tensor_args_t const&amp;)
            # E            --- /__w/tt-forge-sweeps/tt-forge-sweeps/env/venv/xla/lib/python3.11/site-packages/pjrt_plugin_tt/lib/_ttnncpp.so(+0x10e017b) [0x7f2ea636017b]
            # E            --- /__w/tt-forge-sweeps/tt-forge-sweeps/env/venv/xla/lib/python3.11/site-packages/pjrt_plugin_tt/lib/_ttnncpp.so(+0x10dfda3) [0x7f2ea635fda3]
            # E            --- ttnn::operations::normalization::softmax::softmax(tt::tt_metal::Tensor const&amp;, signed char, tt::tt_metal::MemoryConfig, std::optional&lt;std::variant&lt;ttnn::GrayskullComputeKernelConfig, ttnn::WormholeComputeKernelConfig&gt; const&gt;, bool)
            # E            --- ttnn::operations::normalization::ExecuteSoftmax::invoke(tt::tt_metal::Tensor const&amp;, int, std::optional&lt;tt::tt_metal::MemoryConfig&gt; const&amp;, std::optional&lt;std::variant&lt;ttnn::GrayskullComputeKernelConfig, ttnn::WormholeComputeKernelConfig&gt; const&gt;, bool)
            # E            --- /__w/tt-forge-sweeps/tt-forge-sweeps/env/venv/xla/lib/python3.11/site-packages/pjrt_plugin_tt/lib/libTTMLIRRuntime.so(+0x296347) [0x7f2ec8509347]
            # E            --- tt::runtime::ttnn::operations::normalization::run(tt::target::ttnn::SoftmaxOp const*, tt::runtime::ttnn::ProgramContext&amp;)
            # E            --- tt::runtime::ttnn::ProgramExecutor::execute()
            # E            --- tt::runtime::ttnn::submit(tt::runtime::Device, tt::runtime::Binary, unsigned int, std::vector&lt;tt::runtime::Tensor, std::allocator&lt;tt::runtime::Tensor&gt; &gt;&amp;)
            # E            --- tt::runtime::submit(tt::runtime::Device, tt::runtime::Binary, unsigned int, std::vector&lt;tt::runtime::Tensor, std::allocator&lt;tt::runtime::Tensor&gt; &gt;&amp;)
            # E            --- tt::pjrt::FlatbufferLoadedExecutableInstance::execute(PJRT_LoadedExecutable_Execute_Args*)
            # E            --- tt::pjrt::internal::onLoadedExecutableExecute(PJRT_LoadedExecutable_Execute_Args*)
            # E            --- xla::PjRtCApiLoadedExecutable::ExecuteWithSingleDevice(absl::lts_20230802::Span&lt;xla::PjRtBuffer* const&gt;, xla::PjRtDevice*, xla::ExecuteOptions const&amp;, std::optional&lt;xla::PjRtFuture&lt;void&gt; &gt;&amp;, bool)
            # E            --- xla::PjRtCApiLoadedExecutable::ExecutePortable(absl::lts_20230802::Span&lt;xla::PjRtBuffer* const&gt;, xla::PjRtDevice*, xla::ExecuteOptions const&amp;, std::optional&lt;xla::PjRtFuture&lt;void&gt; &gt;&amp;, bool)
            # E            --- torch_xla::runtime::PjRtComputationClient::ExecuteComputation(torch_xla::runtime::ComputationClient::Computation const&amp;, absl::lts_20230802::Span&lt;std::shared_ptr&lt;torch_xla::runtime::ComputationClient::Data&gt; const&gt;, std::__cxx11::basic_string&lt;char, std::char_traits&lt;char&gt;, std::allocator&lt;char&gt; &gt; const&amp;, torch_xla::runtime::ComputationClient::ExecuteComputationOptions const&amp;)
            # E            --- /__w/tt-forge-sweeps/tt-forge-sweeps/env/venv/xla/lib/python3.11/site-packages/_XLAC.cpython-311-x86_64-linux-gnu.so(+0x5f2b1ea) [0x7f2f0d11b1ea]
            # E            --- torch::lazy::MultiWait::Complete(std::function&lt;void ()&gt; const&amp;)
            # E            --- Eigen::ThreadPoolTempl&lt;tsl::thread::EigenEnvironment&gt;::WorkerLoop(int)
            # E            --- void absl::lts_20230802::internal_any_invocable::RemoteInvoker&lt;false, void, tsl::thread::EigenEnvironment::CreateThread(std::function&lt;void ()&gt;)::{lambda()#1}&amp;&gt;(absl::lts_20230802::internal_any_invocable::TypeErasedState*)
            # E            --- /__w/tt-forge-sweeps/tt-forge-sweeps/env/venv/xla/lib/python3.11/site-packages/_XLAC.cpython-311-x86_64-linux-gnu.so(+0xfc35751) [0x7f2f16e25751]
            # E            --- /lib/x86_64-linux-gnu/libc.so.6(+0x94ac3) [0x7f3013c2dac3]
            # E            --- /lib/x86_64-linux-gnu/libc.so.6(clone+0x44) [0x7f3013cbea74]
            # forge/infra/runners/torch_device_runner.py:40: RuntimeError:
            ExceptionCheck(
                class_name="RuntimeError",
                component=ComponentChecker.TTNN.value,
                message=[
                    M.regex(
                        ".*/src/tt-metal/ttnn/cpp/ttnn/operations/normalization/softmax/device/softmax_program_factory\\.cpp.* !attributes\\.numeric_stable.*"
                    ),
                    M.contains(
                        "For softmax, cannot enable both large_kernel and numeric_stable"
                    ),
                ],
                error_log=[
                    M.last_line(M.contains("infra/runners/torch_device_runner.py:")),
                ],
            ),
        ],
    )

    TRANSPOSE_UNSUPPORTED_DATA_TYPE = FailingReason(
        description="Unsupported data type for input tensor",
        checks=[
            # x = &lt;[RuntimeError('TT_FATAL @ /__w/tt-xla/tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/c...--- /lib/x86_64-linux-gnu/libc.so.6(clone+0x44) [0x7fc5f94f4a74]\n') raised in repr()] Tensor object at 0x7fc314777890&gt;
            #     def attempt_to_device(x):
            #         if hasattr(x, "to"):
            # &gt;           return x.to(device)
            #                 ^^^^^^^^^^^^
            # E           RuntimeError: TT_FATAL @ /__w/tt-xla/tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/data_movement/transpose/device/transpose_op.cpp:119: input_tensor.dtype() == DataType::BFLOAT16 || input_tensor.dtype() == DataType::FLOAT32 || input_tensor.dtype() == DataType::INT32
            # E           info:
            # E           Unsupported data type for input tensor
            # E           backtrace:
            # E            --- /__w/tt-forge-sweeps/tt-forge-sweeps/env/venv/xla/lib/python3.11/site-packages/pjrt_plugin_tt/lib/libTTMLIRCompiler.so(+0xa56a1f8) [0x7fc45a2161f8]
            # E            --- ttnn::operations::data_movement::Transpose::validate(std::vector&lt;tt::tt_metal::Tensor, std::allocator&lt;tt::tt_metal::Tensor&gt; &gt; const&amp;) const
            # E            --- /__w/tt-forge-sweeps/tt-forge-sweeps/env/venv/xla/lib/python3.11/site-packages/pjrt_plugin_tt/lib/_ttnncpp.so(_ZN4ttnn16device_operation6detail29launch_operation_with_adapterINS0_26MeshDeviceOperationAdapterIN2tt8tt_metal9operation23OldInfraDeviceOperationISt6vectorINS5_6TensorESaIS9_EEEEEEEEvRKNT_22operation_attributes_tERKNSE_13tensor_args_tERNSE_21tensor_return_value_tEPNS5_11distributed10MeshDeviceE+0x238) [0x7fc46b4a8f78]
            # E            --- tt::tt_metal::operation::OldInfraDeviceOperation&lt;std::vector&lt;tt::tt_metal::Tensor, std::allocator&lt;tt::tt_metal::Tensor&gt; &gt; &gt;::tensor_return_value_t ttnn::device_operation::detail::launch_on_device&lt;tt::tt_metal::operation::OldInfraDeviceOperation&lt;std::vector&lt;tt::tt_metal::Tensor, std::allocator&lt;tt::tt_metal::Tensor&gt; &gt; &gt; &gt;(tt::tt_metal::operation::OldInfraDeviceOperation&lt;std::vector&lt;tt::tt_metal::Tensor, std::allocator&lt;tt::tt_metal::Tensor&gt; &gt; &gt;::operation_attributes_t const&amp;, tt::tt_metal::operation::OldInfraDeviceOperation&lt;std::vector&lt;tt::tt_metal::Tensor, std::allocator&lt;tt::tt_metal::Tensor&gt; &gt; &gt;::tensor_args_t const&amp;)
            # E            --- tt::tt_metal::operation::OldInfraDeviceOperation&lt;std::vector&lt;tt::tt_metal::Tensor, std::allocator&lt;tt::tt_metal::Tensor&gt; &gt; &gt;::tensor_return_value_t ttnn::device_operation::detail::invoke&lt;tt::tt_metal::operation::OldInfraDeviceOperation&lt;std::vector&lt;tt::tt_metal::Tensor, std::allocator&lt;tt::tt_metal::Tensor&gt; &gt; &gt; &gt;(tt::tt_metal::operation::OldInfraDeviceOperation&lt;std::vector&lt;tt::tt_metal::Tensor, std::allocator&lt;tt::tt_metal::Tensor&gt; &gt; &gt;::operation_attributes_t const&amp;, tt::tt_metal::operation::OldInfraDeviceOperation&lt;std::vector&lt;tt::tt_metal::Tensor, std::allocator&lt;tt::tt_metal::Tensor&gt; &gt; &gt;::tensor_args_t const&amp;)
            # E            --- /__w/tt-forge-sweeps/tt-forge-sweeps/env/venv/xla/lib/python3.11/site-packages/pjrt_plugin_tt/lib/_ttnncpp.so(+0x6b9f6b) [0x7fc46b4a7f6b]
            # E            --- std::vector&lt;tt::tt_metal::Tensor, std::allocator&lt;tt::tt_metal::Tensor&gt; &gt; tt::tt_metal::operation::run&lt;std::vector&lt;tt::tt_metal::Tensor, std::allocator&lt;tt::tt_metal::Tensor&gt; &gt; &gt;(tt::tt_metal::operation::DeviceOperation&lt;std::vector&lt;tt::tt_metal::Tensor, std::allocator&lt;tt::tt_metal::Tensor&gt; &gt; &gt;&amp;&amp;, std::vector&lt;tt::tt_metal::Tensor, std::allocator&lt;tt::tt_metal::Tensor&gt; &gt; const&amp;, std::vector&lt;std::optional&lt;tt::tt_metal::Tensor const&gt;, std::allocator&lt;std::optional&lt;tt::tt_metal::Tensor const&gt; &gt; &gt; const&amp;, std::vector&lt;std::optional&lt;tt::tt_metal::Tensor&gt;, std::allocator&lt;std::optional&lt;tt::tt_metal::Tensor&gt; &gt; &gt; const&amp;)
            # E            --- /__w/tt-forge-sweeps/tt-forge-sweeps/env/venv/xla/lib/python3.11/site-packages/pjrt_plugin_tt/lib/_ttnncpp.so(+0x92863c) [0x7fc46b71663c]
            # E            --- ttnn::operations::data_movement::ExecuteTranspose::invoke(tt::tt_metal::Tensor const&amp;, long const&amp;, long const&amp;, std::optional&lt;tt::tt_metal::MemoryConfig&gt; const&amp;, std::optional&lt;float&gt; const&amp;)
            # E            --- /__w/tt-forge-sweeps/tt-forge-sweeps/env/venv/xla/lib/python3.11/site-packages/pjrt_plugin_tt/lib/_ttnncpp.so(+0x9cc8ac) [0x7fc46b7ba8ac]
            # E            --- /__w/tt-forge-sweeps/tt-forge-sweeps/env/venv/xla/lib/python3.11/site-packages/pjrt_plugin_tt/lib/_ttnncpp.so(+0x9cc4eb) [0x7fc46b7ba4eb]
            # E            --- ttnn::operations::data_movement::detail::permute_impl(tt::tt_metal::Tensor const&amp;, ttsl::SmallVector&lt;unsigned int, 8ul&gt; const&amp;, tt::tt_metal::MemoryConfig const&amp;, std::optional&lt;float&gt; const&amp;)
            # E            --- ttnn::operations::data_movement::ExecutePermute::invoke(tt::tt_metal::Tensor const&amp;, ttsl::SmallVector&lt;long, 8ul&gt; const&amp;, std::optional&lt;tt::tt_metal::MemoryConfig&gt; const&amp;, std::optional&lt;float&gt; const&amp;)
            # E            --- /__w/tt-forge-sweeps/tt-forge-sweeps/env/venv/xla/lib/python3.11/site-packages/pjrt_plugin_tt/lib/libTTMLIRRuntime.so(+0x267e62) [0x7fc4a44dae62]
            # E            --- tt::runtime::ttnn::operations::data_movement::run(tt::target::ttnn::PermuteOp const*, tt::runtime::ttnn::ProgramContext&amp;)
            # E            --- tt::runtime::ttnn::ProgramExecutor::execute()
            # E            --- tt::runtime::ttnn::submit(tt::runtime::Device, tt::runtime::Binary, unsigned int, std::vector&lt;tt::runtime::Tensor, std::allocator&lt;tt::runtime::Tensor&gt; &gt;&amp;)
            # E            --- tt::runtime::submit(tt::runtime::Device, tt::runtime::Binary, unsigned int, std::vector&lt;tt::runtime::Tensor, std::allocator&lt;tt::runtime::Tensor&gt; &gt;&amp;)
            # E            --- tt::pjrt::FlatbufferLoadedExecutableInstance::execute(PJRT_LoadedExecutable_Execute_Args*)
            # E            --- tt::pjrt::internal::onLoadedExecutableExecute(PJRT_LoadedExecutable_Execute_Args*)
            # E            --- xla::PjRtCApiLoadedExecutable::ExecuteWithSingleDevice(absl::lts_20230802::Span&lt;xla::PjRtBuffer* const&gt;, xla::PjRtDevice*, xla::ExecuteOptions const&amp;, std::optional&lt;xla::PjRtFuture&lt;void&gt; &gt;&amp;, bool)
            # E            --- xla::PjRtCApiLoadedExecutable::ExecutePortable(absl::lts_20230802::Span&lt;xla::PjRtBuffer* const&gt;, xla::PjRtDevice*, xla::ExecuteOptions const&amp;, std::optional&lt;xla::PjRtFuture&lt;void&gt; &gt;&amp;, bool)
            # E            --- torch_xla::runtime::PjRtComputationClient::ExecuteComputation(torch_xla::runtime::ComputationClient::Computation const&amp;, absl::lts_20230802::Span&lt;std::shared_ptr&lt;torch_xla::runtime::ComputationClient::Data&gt; const&gt;, std::__cxx11::basic_string&lt;char, std::char_traits&lt;char&gt;, std::allocator&lt;char&gt; &gt; const&amp;, torch_xla::runtime::ComputationClient::ExecuteComputationOptions const&amp;)
            # E            --- /__w/tt-forge-sweeps/tt-forge-sweeps/env/venv/xla/lib/python3.11/site-packages/_XLAC.cpython-311-x86_64-linux-gnu.so(+0x5f2b1ea) [0x7fc4f291b1ea]
            # E            --- torch::lazy::MultiWait::Complete(std::function&lt;void ()&gt; const&amp;)
            # E            --- Eigen::ThreadPoolTempl&lt;tsl::thread::EigenEnvironment&gt;::WorkerLoop(int)
            # E            --- void absl::lts_20230802::internal_any_invocable::RemoteInvoker&lt;false, void, tsl::thread::EigenEnvironment::CreateThread(std::function&lt;void ()&gt;)::{lambda()#1}&amp;&gt;(absl::lts_20230802::internal_any_invocable::TypeErasedState*)
            # E            --- /__w/tt-forge-sweeps/tt-forge-sweeps/env/venv/xla/lib/python3.11/site-packages/_XLAC.cpython-311-x86_64-linux-gnu.so(+0xfc35751) [0x7fc4fc625751]
            # E            --- /lib/x86_64-linux-gnu/libc.so.6(+0x94ac3) [0x7fc5f9463ac3]
            # E            --- /lib/x86_64-linux-gnu/libc.so.6(clone+0x44) [0x7fc5f94f4a74]
            # forge/infra/runners/torch_device_runner.py:40: RuntimeError
            ExceptionCheck(
                class_name="RuntimeError",
                component=ComponentChecker.TTNN.value,
                message=[
                    M.regex(
                        ".*/src/tt-metal/ttnn//cpp/ttnn/operations/data_movement/transpose/device/transpose_op\\.cpp:\\d+: input_tensor\\.get_dtype\\(\\) == DataType::BFLOAT16 || input_tensor\\.get_dtype\\(\\) == DataType::FLOAT32 || input_tensor\\.get_dtype\\(\\) == DataType::INT32"
                    ),
                    M.contains("Unsupported data type for input tensor"),
                ],
                error_log=[
                    M.last_line(M.contains("infra/runners/torch_device_runner.py:")),
                ],
            ),
        ],
    )

    INPUT_TENSOR_PAD_SHAPE = FailingReason(
        description="Input tensor batch size must be 1",
        checks=[
            # >           return x.to(device)
            # E           RuntimeError: TT_FATAL @ /__w/tt-xla/tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/kv_cache/device/update_cache_op.cpp:38: input_tensor.padded_shape()[0] == 1
            # E           info:
            # E           Error
            # E           backtrace:
            # E            --- /root/.pyenv/versions/3.11.13/lib/python3.11/site-packages/pjrt_plugin_tt/lib/libTTMLIRCompiler.so(+0xa5e3fe8) [0x7f738e210fe8]
            # E            --- ttnn::operations::kv_cache::UpdateCache::validate(std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > const&) const
            # E            --- /root/.pyenv/versions/3.11.13/lib/python3.11/site-packages/pjrt_plugin_tt/lib/_ttnncpp.so(_ZN4ttnn16device_operation6detail29launch_operation_with_adapterINS0_26MeshDeviceOperationAdapterIN2tt8tt_metal9operation23OldInfraDeviceOperationISt6vectorINS5_6TensorESaIS9_EEEEEEEEvRKNT_22operation_attributes_tERKNSE_13tensor_args_tERNSE_21tensor_return_value_tEPNS5_11distributed10MeshDeviceE+0x238) [0x7f739749e038]
            # E            --- tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >::tensor_return_value_t ttnn::device_operation::detail::launch_on_device<tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > > >(tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >::operation_attributes_t const&, tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >::tensor_args_t const&)
            # E            --- tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >::tensor_return_value_t ttnn::device_operation::detail::invoke<tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > > >(tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >::operation_attributes_t const&, tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >::tensor_args_t const&)
            # E            --- /root/.pyenv/versions/3.11.13/lib/python3.11/site-packages/pjrt_plugin_tt/lib/_ttnncpp.so(+0x6c202b) [0x7f739749d02b]
            # E            --- std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > tt::tt_metal::operation::run<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >(tt::tt_metal::operation::DeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >&&, std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > const&, std::vector<std::optional<tt::tt_metal::Tensor const>, std::allocator<std::optional<tt::tt_metal::Tensor const> > > const&, std::vector<std::optional<tt::tt_metal::Tensor>, std::allocator<std::optional<tt::tt_metal::Tensor> > > const&)
            # E            --- /root/.pyenv/versions/3.11.13/lib/python3.11/site-packages/pjrt_plugin_tt/lib/_ttnncpp.so(+0xe03201) [0x7f7397bde201]
            # E            --- ttnn::operations::kv_cache::FillCacheOperation::invoke(tt::tt_metal::Tensor const&, tt::tt_metal::Tensor const&, unsigned int)
            # E            --- /root/.pyenv/versions/3.11.13/lib/python3.11/site-packages/pjrt_plugin_tt/lib/libTTMLIRRuntime.so(+0x289877) [0x7f73d4782877]
            # E            --- tt::runtime::ttnn::operations::kv_cache::run(tt::target::ttnn::FillCacheOp const*, tt::runtime::ttnn::ProgramContext&)
            # E            --- tt::runtime::ttnn::ProgramExecutor::execute()
            # E            --- tt::runtime::ttnn::submit(tt::runtime::Device, tt::runtime::Binary, unsigned int, std::vector<tt::runtime::Tensor, std::allocator<tt::runtime::Tensor> >&)
            # E            --- tt::runtime::submit(tt::runtime::Device, tt::runtime::Binary, unsigned int, std::vector<tt::runtime::Tensor, std::allocator<tt::runtime::Tensor> >&)
            # E            --- tt::pjrt::FlatbufferLoadedExecutableInstance::execute(PJRT_LoadedExecutable_Execute_Args*)
            # E            --- tt::pjrt::internal::onLoadedExecutableExecute(PJRT_LoadedExecutable_Execute_Args*)
            # E            --- xla::PjRtCApiLoadedExecutable::ExecuteWithSingleDevice(absl::lts_20230802::Span<xla::PjRtBuffer* const>, xla::PjRtDevice*, xla::ExecuteOptions const&, std::optional<xla::PjRtFuture<void> >&, bool)
            # E            --- xla::PjRtCApiLoadedExecutable::ExecutePortable(absl::lts_20230802::Span<xla::PjRtBuffer* const>, xla::PjRtDevice*, xla::ExecuteOptions const&, std::optional<xla::PjRtFuture<void> >&, bool)
            # E            --- torch_xla::runtime::PjRtComputationClient::ExecuteComputation(torch_xla::runtime::ComputationClient::Computation const&, absl::lts_20230802::Span<std::shared_ptr<torch_xla::runtime::ComputationClient::Data> const>, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, torch_xla::runtime::ComputationClient::ExecuteComputationOptions const&)
            # E            --- /root/.pyenv/versions/3.11.13/lib/python3.11/site-packages/_XLAC.cpython-311-x86_64-linux-gnu.so(+0x6ddf416) [0x7f74143aa416]
            # E            --- torch::lazy::MultiWait::Complete(std::function<void ()> const&)
            # E            --- Eigen::ThreadPoolTempl<tsl::thread::EigenEnvironment>::WorkerLoop(int)
            # E            --- void absl::lts_20230802::internal_any_invocable::RemoteInvoker<false, void, tsl::thread::EigenEnvironment::CreateThread(std::function<void ()>)::{lambda()#1}&>(absl::lts_20230802::internal_any_invocable::TypeErasedState*)
            # E            --- /root/.pyenv/versions/3.11.13/lib/python3.11/site-packages/_XLAC.cpython-311-x86_64-linux-gnu.so(+0x11678542) [0x7f741ec43542]
            # E            --- /lib/x86_64-linux-gnu/libc.so.6(+0x94ac3) [0x7f752b943ac3]
            # E            --- /lib/x86_64-linux-gnu/libc.so.6(clone+0x44) [0x7f752b9d4a04]
            # forge/infra/runners/torch_device_runner.py:52: RuntimeError
            ExceptionCheck(
                class_name="RuntimeError",
                component=ComponentChecker.TTNN.value,
                message=[
                    M.regex(
                        "TT_FATAL @ .*/src/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/kv_cache/device/update_cache_op.cpp"
                    ),
                    M.contains("input_tensor.padded_shape()[0] == 1"),
                ],
                error_log=[
                    M.contains(">           return x.to(device)"),
                    M.last_line(M.contains("infra/runners/torch_device_runner.py:")),
                ],
            ),
        ],
    )

    COMPUTE_WITH_STORAGE_GRID_SIZE = FailingReason(
        description="Compute with storage grid size mismatch",
        checks=[
            # >           return x.to(device)
            # E           RuntimeError: TT_FATAL @ /__w/tt-xla/tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/kv_cache/device/update_cache_op.cpp:56: (num_blocks_of_work <= compute_with_storage_grid_size.x * compute_with_storage_grid_size.y)
            # E           info:
            # E           Error
            # E           backtrace:
            # E            --- /root/.pyenv/versions/3.11.13/lib/python3.11/site-packages/pjrt_plugin_tt/lib/libTTMLIRCompiler.so(+0xa5e3fe8) [0x7f410a210fe8]
            # E            --- ttnn::operations::kv_cache::UpdateCache::validate(std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > const&) const
            # E            --- /root/.pyenv/versions/3.11.13/lib/python3.11/site-packages/pjrt_plugin_tt/lib/_ttnncpp.so(_ZN4ttnn16device_operation6detail29launch_operation_with_adapterINS0_26MeshDeviceOperationAdapterIN2tt8tt_metal9operation23OldInfraDeviceOperationISt6vectorINS5_6TensorESaIS9_EEEEEEEEvRKNT_22operation_attributes_tERKNSE_13tensor_args_tERNSE_21tensor_return_value_tEPNS5_11distributed10MeshDeviceE+0x238) [0x7f412619d038]
            # E            --- tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >::tensor_return_value_t ttnn::device_operation::detail::launch_on_device<tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > > >(tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >::operation_attributes_t const&, tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >::tensor_args_t const&)
            # E            --- tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >::tensor_return_value_t ttnn::device_operation::detail::invoke<tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > > >(tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >::operation_attributes_t const&, tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >::tensor_args_t const&)
            # E            --- /root/.pyenv/versions/3.11.13/lib/python3.11/site-packages/pjrt_plugin_tt/lib/_ttnncpp.so(+0x6c202b) [0x7f412619c02b]
            # E            --- std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > tt::tt_metal::operation::run<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >(tt::tt_metal::operation::DeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >&&, std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > const&, std::vector<std::optional<tt::tt_metal::Tensor const>, std::allocator<std::optional<tt::tt_metal::Tensor const> > > const&, std::vector<std::optional<tt::tt_metal::Tensor>, std::allocator<std::optional<tt::tt_metal::Tensor> > > const&)
            # E            --- /root/.pyenv/versions/3.11.13/lib/python3.11/site-packages/pjrt_plugin_tt/lib/_ttnncpp.so(+0xe03201) [0x7f41268dd201]
            # E            --- ttnn::operations::kv_cache::FillCacheOperation::invoke(tt::tt_metal::Tensor const&, tt::tt_metal::Tensor const&, unsigned int)
            # E            --- /root/.pyenv/versions/3.11.13/lib/python3.11/site-packages/pjrt_plugin_tt/lib/libTTMLIRRuntime.so(+0x289877) [0x7f416c3d4877]
            # E            --- tt::runtime::ttnn::operations::kv_cache::run(tt::target::ttnn::FillCacheOp const*, tt::runtime::ttnn::ProgramContext&)
            # E            --- tt::runtime::ttnn::ProgramExecutor::execute()
            # E            --- tt::runtime::ttnn::submit(tt::runtime::Device, tt::runtime::Binary, unsigned int, std::vector<tt::runtime::Tensor, std::allocator<tt::runtime::Tensor> >&)
            # E            --- tt::runtime::submit(tt::runtime::Device, tt::runtime::Binary, unsigned int, std::vector<tt::runtime::Tensor, std::allocator<tt::runtime::Tensor> >&)
            # E            --- tt::pjrt::FlatbufferLoadedExecutableInstance::execute(PJRT_LoadedExecutable_Execute_Args*)
            # E            --- tt::pjrt::internal::onLoadedExecutableExecute(PJRT_LoadedExecutable_Execute_Args*)
            # E            --- xla::PjRtCApiLoadedExecutable::ExecuteWithSingleDevice(absl::lts_20230802::Span<xla::PjRtBuffer* const>, xla::PjRtDevice*, xla::ExecuteOptions const&, std::optional<xla::PjRtFuture<void> >&, bool)
            # E            --- xla::PjRtCApiLoadedExecutable::ExecutePortable(absl::lts_20230802::Span<xla::PjRtBuffer* const>, xla::PjRtDevice*, xla::ExecuteOptions const&, std::optional<xla::PjRtFuture<void> >&, bool)
            # E            --- torch_xla::runtime::PjRtComputationClient::ExecuteComputation(torch_xla::runtime::ComputationClient::Computation const&, absl::lts_20230802::Span<std::shared_ptr<torch_xla::runtime::ComputationClient::Data> const>, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, torch_xla::runtime::ComputationClient::ExecuteComputationOptions const&)
            # E            --- /root/.pyenv/versions/3.11.13/lib/python3.11/site-packages/_XLAC.cpython-311-x86_64-linux-gnu.so(+0x6ddf416) [0x7f41936aa416]
            # E            --- torch::lazy::MultiWait::Complete(std::function<void ()> const&)
            # E            --- Eigen::ThreadPoolTempl<tsl::thread::EigenEnvironment>::WorkerLoop(int)
            # E            --- void absl::lts_20230802::internal_any_invocable::RemoteInvoker<false, void, tsl::thread::EigenEnvironment::CreateThread(std::function<void ()>)::{lambda()#1}&>(absl::lts_20230802::internal_any_invocable::TypeErasedState*)
            # E            --- /root/.pyenv/versions/3.11.13/lib/python3.11/site-packages/_XLAC.cpython-311-x86_64-linux-gnu.so(+0x11678542) [0x7f419df43542]
            # E            --- /lib/x86_64-linux-gnu/libc.so.6(+0x94ac3) [0x7f42aac31ac3]
            # E            --- /lib/x86_64-linux-gnu/libc.so.6(clone+0x44) [0x7f42aacc2a04]
            # forge/infra/runners/torch_device_runner.py:52: RuntimeError
            ExceptionCheck(
                class_name="RuntimeError",
                component=ComponentChecker.TTNN.value,
                message=[
                    M.regex(
                        "TT_FATAL @ .*/src/tt-metal/ttnn/cpp/ttnn/operations/kv_cache/device/update_cache_op.cpp"
                    ),
                    M.contains(
                        "(num_blocks_of_work <= compute_with_storage_grid_size.x * compute_with_storage_grid_size.y)"
                    ),
                ],
                error_log=[
                    M.contains(">           return x.to(device)"),
                    M.last_line(M.contains("infra/runners/torch_device_runner.py:")),
                ],
            ),
        ],
    )

    CIRCULAR_BUFFER_ON_CORE_RANGE = FailingReason(
        description="Circular buffer on core range exceeds L1 size",
        checks=[
            # >           return x.to(device)
            # E           RuntimeError: TT_THROW @ /__w/tt-xla/tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/impl/program/program.cpp:907: tt::exception
            # E           info:
            # E           Statically allocated circular buffers on core range [(x=0,y=0) - (x=0,y=0)] grow to 5081888 B which is beyond max L1 size of 1499136 B
            # E           backtrace:
            # E            --- /root/.pyenv/versions/3.11.13/lib/python3.11/site-packages/pjrt_plugin_tt/lib/libtt_metal.so(+0x515a3b) [0x7fa6eea13a3b]
            # E            --- tt::tt_metal::detail::ProgramImpl::validate_circular_buffer_region(tt::tt_metal::IDevice const*)
            # E            --- tt::tt_metal::distributed::MeshWorkloadImpl::compile(tt::tt_metal::distributed::MeshDevice*)
            # E            --- tt::tt_metal::distributed::EnqueueMeshWorkload(tt::tt_metal::distributed::MeshCommandQueue&, tt::tt_metal::distributed::MeshWorkload&, bool)
            # E            --- /root/.pyenv/versions/3.11.13/lib/python3.11/site-packages/pjrt_plugin_tt/lib/_ttnncpp.so(+0x6c59a2) [0x7fa6ed99e9a2]
            # E            --- /root/.pyenv/versions/3.11.13/lib/python3.11/site-packages/pjrt_plugin_tt/lib/_ttnncpp.so(_ZN4ttnn16device_operation6detail29launch_operation_with_adapterINS0_26MeshDeviceOperationAdapterIN2tt8tt_metal9operation23OldInfraDeviceOperationISt6vectorINS5_6TensorESaIS9_EEEEEEEEvRKNT_22operation_attributes_tERKNSE_13tensor_args_tERNSE_21tensor_return_value_tEPNS5_11distributed10MeshDeviceE+0x2c3) [0x7fa6ed99c0c3]
            # E            --- tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >::tensor_return_value_t ttnn::device_operation::detail::launch_on_device<tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > > >(tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >::operation_attributes_t const&, tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >::tensor_args_t const&)
            # E            --- tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >::tensor_return_value_t ttnn::device_operation::detail::invoke<tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > > >(tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >::operation_attributes_t const&, tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >::tensor_args_t const&)
            # E            --- /root/.pyenv/versions/3.11.13/lib/python3.11/site-packages/pjrt_plugin_tt/lib/_ttnncpp.so(+0x6c202b) [0x7fa6ed99b02b]
            # E            --- std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > tt::tt_metal::operation::run<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >(tt::tt_metal::operation::DeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >&&, std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > const&, std::vector<std::optional<tt::tt_metal::Tensor const>, std::allocator<std::optional<tt::tt_metal::Tensor const> > > const&, std::vector<std::optional<tt::tt_metal::Tensor>, std::allocator<std::optional<tt::tt_metal::Tensor> > > const&)
            # E            --- /root/.pyenv/versions/3.11.13/lib/python3.11/site-packages/pjrt_plugin_tt/lib/_ttnncpp.so(+0xe02d6d) [0x7fa6ee0dbd6d]
            # E            --- ttnn::operations::kv_cache::UpdateCacheOperation::invoke(tt::tt_metal::Tensor const&, tt::tt_metal::Tensor const&, unsigned int, unsigned int, std::optional<std::variant<ttnn::GrayskullComputeKernelConfig, ttnn::WormholeComputeKernelConfig> const>)
            # E            --- /root/.pyenv/versions/3.11.13/lib/python3.11/site-packages/pjrt_plugin_tt/lib/libTTMLIRRuntime.so(+0x28a6b5) [0x7fa73c3d66b5]
            # E            --- tt::runtime::ttnn::operations::kv_cache::run(tt::target::ttnn::UpdateCacheOp const*, tt::runtime::ttnn::ProgramContext&)
            # E            --- tt::runtime::ttnn::ProgramExecutor::execute()
            # E            --- tt::runtime::ttnn::submit(tt::runtime::Device, tt::runtime::Binary, unsigned int, std::vector<tt::runtime::Tensor, std::allocator<tt::runtime::Tensor> >&)
            # E            --- tt::runtime::submit(tt::runtime::Device, tt::runtime::Binary, unsigned int, std::vector<tt::runtime::Tensor, std::allocator<tt::runtime::Tensor> >&)
            # E            --- tt::pjrt::FlatbufferLoadedExecutableInstance::execute(PJRT_LoadedExecutable_Execute_Args*)
            # E            --- tt::pjrt::internal::onLoadedExecutableExecute(PJRT_LoadedExecutable_Execute_Args*)
            # E            --- xla::PjRtCApiLoadedExecutable::ExecuteWithSingleDevice(absl::lts_20230802::Span<xla::PjRtBuffer* const>, xla::PjRtDevice*, xla::ExecuteOptions const&, std::optional<xla::PjRtFuture<void> >&, bool)
            # E            --- xla::PjRtCApiLoadedExecutable::ExecutePortable(absl::lts_20230802::Span<xla::PjRtBuffer* const>, xla::PjRtDevice*, xla::ExecuteOptions const&, std::optional<xla::PjRtFuture<void> >&, bool)
            # E            --- torch_xla::runtime::PjRtComputationClient::ExecuteComputation(torch_xla::runtime::ComputationClient::Computation const&, absl::lts_20230802::Span<std::shared_ptr<torch_xla::runtime::ComputationClient::Data> const>, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, torch_xla::runtime::ComputationClient::ExecuteComputationOptions const&)
            # E            --- /root/.pyenv/versions/3.11.13/lib/python3.11/site-packages/_XLAC.cpython-311-x86_64-linux-gnu.so(+0x6ddf416) [0x7fa762eaa416]
            # E            --- torch::lazy::MultiWait::Complete(std::function<void ()> const&)
            # E            --- Eigen::ThreadPoolTempl<tsl::thread::EigenEnvironment>::WorkerLoop(int)
            # E            --- void absl::lts_20230802::internal_any_invocable::RemoteInvoker<false, void, tsl::thread::EigenEnvironment::CreateThread(std::function<void ()>)::{lambda()#1}&>(absl::lts_20230802::internal_any_invocable::TypeErasedState*)
            # E            --- /root/.pyenv/versions/3.11.13/lib/python3.11/site-packages/_XLAC.cpython-311-x86_64-linux-gnu.so(+0x11678542) [0x7fa76d743542]
            # E            --- /lib/x86_64-linux-gnu/libc.so.6(+0x94ac3) [0x7fa87a451ac3]
            # E            --- /lib/x86_64-linux-gnu/libc.so.6(clone+0x44) [0x7fa87a4e2a04]
            # forge/infra/runners/torch_device_runner.py:52: RuntimeError
            ExceptionCheck(
                class_name="RuntimeError",
                component=ComponentChecker.METAL.value,
                message=[
                    M.regex(
                        "TT_THROW @ .*/src/tt-metal/tt_metal/impl/program/program\\.cpp"
                    ),
                    M.regex(
                        "Statically allocated circular buffers on core range .* grow to .* B which is beyond max L1 size of .* B"
                    ),
                ],
                error_log=[
                    M.contains(">           return x.to(device)"),
                    M.last_line(M.contains("infra/runners/torch_device_runner.py:")),
                ],
            ),
        ],
    )

    NOT_XLA_TENSOR = FailingReason(
        description="Check failed: xtensor: Input tensor is not an XLA tensor: torch.LongTensor",
        checks=[
            # def forward(self, x, y):
            #     return self.operator(x, self.dim, self.index, y)
            #
            # E       RuntimeError: Check failed: xtensor: Input tensor is not an XLA tensor: torch.LongTensor
            # forge/test/operators/pytorch/indexing/test_index_copy.py:68: RuntimeError
            ExceptionCheck(
                class_name="RuntimeError",
                component=ComponentChecker.XLA.value,
                message=[
                    M.contains(
                        "Check failed: xtensor: Input tensor is not an XLA tensor: torch.LongTensor"
                    ),
                ],
                error_log=[
                    M.last_line(
                        M.contains(
                            "forge/test/operators/pytorch/indexing/test_index_copy.py"
                        )
                    ),
                ],
            ),
        ],
    )

    UNSUPPORTED_DATA_FORMAT_REPEAT_INTERLEAVE = FailingReason(
        description="Can only work with UINT16, BFLOAT16, UINT32, INT32, FLOAT32 data types",
        checks=[
            # E           RuntimeError: TT_FATAL @ /__w/tt-xla/tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/data_movement/repeat/device/repeat_device_operation.cpp:27: input_tensor_a.dtype() == tt::tt_metal::DataType::UINT16 or input_tensor_a.dtype() == tt::tt_metal::DataType::BFLOAT16 or input_tensor_a.dtype() == tt::tt_metal::DataType::UINT32 or input_tensor_a.dtype() == tt::tt_metal::DataType::INT32 or input_tensor_a.dtype() == tt::tt_metal::DataType::FLOAT32
            # E           info:
            # E           Can only work with UINT16, BFLOAT16, UINT32, INT32, FLOAT32 data types
            # E           backtrace:
            # E            --- /root/.pyenv/versions/3.11.13/lib/python3.11/site-packages/pjrt_plugin_tt/lib/libTTMLIRCompiler.so(+0xa5e3fe8) [0x7f738e210fe8]
            # E            --- ttnn::RepeatDeviceOperation::validate(std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > const&) const
            # E            --- /root/.pyenv/versions/3.11.13/lib/python3.11/site-packages/pjrt_plugin_tt/lib/_ttnncpp.so(_ZN4ttnn16device_operation6detail29launch_operation_with_adapterINS0_26MeshDeviceOperationAdapterIN2tt8tt_metal9operation23OldInfraDeviceOperationISt6vectorINS5_6TensorESaIS9_EEEEEEEEvRKNT_22operation_attributes_tERKNSE_13tensor_args_tERNSE_21tensor_return_value_tEPNS5_11distributed10MeshDeviceE+0x238) [0x7f739749e038]
            # E            --- tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >::tensor_return_value_t ttnn::device_operation::detail::launch_on_device<tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > > >(tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >::operation_attributes_t const&, tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >::tensor_args_t const&)
            # E            --- tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >::tensor_return_value_t ttnn::device_operation::detail::invoke<tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > > >(tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >::operation_attributes_t const&, tt::tt_metal::operation::OldInfraDeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >::tensor_args_t const&)
            # E            --- /root/.pyenv/versions/3.11.13/lib/python3.11/site-packages/pjrt_plugin_tt/lib/_ttnncpp.so(+0x6c202b) [0x7f739749d02b]
            # E            --- std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > tt::tt_metal::operation::run<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >(tt::tt_metal::operation::DeviceOperation<std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > >&&, std::vector<tt::tt_metal::Tensor, std::allocator<tt::tt_metal::Tensor> > const&, std::vector<std::optional<tt::tt_metal::Tensor const>, std::allocator<std::optional<tt::tt_metal::Tensor const> > > const&, std::vector<std::optional<tt::tt_metal::Tensor>, std::allocator<std::optional<tt::tt_metal::Tensor> > > const&)
            # E            --- ttnn::operations::data_movement::detail::repeat_last_dim_rm(tt::tt_metal::Tensor const&, unsigned int, tt::tt_metal::MemoryConfig const&)
            # E            --- ttnn::operations::data_movement::RepeatOperation::invoke(tt::tt_metal::Tensor const&, ttsl::SmallVector<unsigned int, 8ul> const&, std::optional<tt::tt_metal::MemoryConfig> const&)
            # E            --- ttnn::operations::data_movement::RepeatOperation::invoke(tt::tt_metal::Tensor const&, tt::tt_metal::Shape const&)
            # E            --- /root/.pyenv/versions/3.11.13/lib/python3.11/site-packages/pjrt_plugin_tt/lib/libTTMLIRRuntime.so(+0x26a616) [0x7f73d4763616]
            # E            --- tt::runtime::ttnn::operations::data_movement::run(tt::target::ttnn::RepeatOp const*, tt::runtime::ttnn::ProgramContext&)
            # E            --- tt::runtime::ttnn::ProgramExecutor::execute()
            # E            --- tt::runtime::ttnn::submit(tt::runtime::Device, tt::runtime::Binary, unsigned int, std::vector<tt::runtime::Tensor, std::allocator<tt::runtime::Tensor> >&)
            # E            --- tt::runtime::submit(tt::runtime::Device, tt::runtime::Binary, unsigned int, std::vector<tt::runtime::Tensor, std::allocator<tt::runtime::Tensor> >&)
            # E            --- tt::pjrt::FlatbufferLoadedExecutableInstance::execute(PJRT_LoadedExecutable_Execute_Args*)
            # E            --- tt::pjrt::internal::onLoadedExecutableExecute(PJRT_LoadedExecutable_Execute_Args*)
            # E            --- xla::PjRtCApiLoadedExecutable::ExecuteWithSingleDevice(absl::lts_20230802::Span<xla::PjRtBuffer* const>, xla::PjRtDevice*, xla::ExecuteOptions const&, std::optional<xla::PjRtFuture<void> >&, bool)
            # E            --- xla::PjRtCApiLoadedExecutable::ExecutePortable(absl::lts_20230802::Span<xla::PjRtBuffer* const>, xla::PjRtDevice*, xla::ExecuteOptions const&, std::optional<xla::PjRtFuture<void> >&, bool)
            # E            --- torch_xla::runtime::PjRtComputationClient::ExecuteComputation(torch_xla::runtime::ComputationClient::Computation const&, absl::lts_20230802::Span<std::shared_ptr<torch_xla::runtime::ComputationClient::Data> const>, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, torch_xla::runtime::ComputationClient::ExecuteComputationOptions const&)
            # E            --- /root/.pyenv/versions/3.11.13/lib/python3.11/site-packages/_XLAC.cpython-311-x86_64-linux-gnu.so(+0x6ddf416) [0x7f74143aa416]
            # E            --- torch::lazy::MultiWait::Complete(std::function<void ()> const&)
            # E            --- Eigen::ThreadPoolTempl<tsl::thread::EigenEnvironment>::WorkerLoop(int)
            # E            --- void absl::lts_20230802::internal_any_invocable::RemoteInvoker<false, void, tsl::thread::EigenEnvironment::CreateThread(std::function<void ()>)::{lambda()#1}&>(absl::lts_20230802::internal_any_invocable::TypeErasedState*)
            # E            --- /root/.pyenv/versions/3.11.13/lib/python3.11/site-packages/_XLAC.cpython-311-x86_64-linux-gnu.so(+0x11678542) [0x7f741ec43542]
            # E            --- /lib/x86_64-linux-gnu/libc.so.6(+0x94ac3) [0x7f752b943ac3]
            # E            --- /lib/x86_64-linux-gnu/libc.so.6(clone+0x44) [0x7f752b9d4a04]
            #
            # forge/infra/runners/torch_device_runner.py:52: RuntimeError
            ExceptionCheck(
                class_name="RuntimeError",
                component=ComponentChecker.TTNN.value,
                message=[
                    M.regex(
                        "TT_FATAL @ .*/src/tt-metal/ttnn/cpp/ttnn/operations/data_movement/repeat/device/repeat_device_operation.cpp:\\d+: input_tensor_a.dtype\\(\\) == tt::tt_metal::DataType::UINT16 or input_tensor_a.dtype\\(\\) == tt::tt_metal::DataType::BFLOAT16 or input_tensor_a.dtype\\(\\) == tt::tt_metal::DataType::UINT32 or input_tensor_a.dtype\\(\\) == tt::tt_metal::DataType::INT32 or input_tensor_a.dtype\\(\\) == tt::tt_metal::DataType::FLOAT32"
                    ),
                ],
                error_log=[
                    M.last_line(M.contains("infra/runners/torch_device_runner.py")),
                ],
            ),
        ],
    )

    CAUSAL_LM_OUTPUT_WITH_PAST_NO_SHAPE = FailingReason(
        description="CausalLMOutputWithPast object has no attribute shape",
        checks=[
            # >       random_grad = torch.randn(cpu_res.shape, dtype=cpu_res.dtype)
            # E       AttributeError: 'CausalLMOutputWithPast' object has no attribute 'shape'
            # tests/infra/testers/single_chip/model/torch_model_tester.py:153: AttributeError
            ExceptionCheck(
                class_name="AttributeError",
                component=ComponentChecker.XLA.value,
                message=[
                    M.equals(
                        "'CausalLMOutputWithPast' object has no attribute 'shape'"
                    ),
                ],
                error_log=[
                    M.contains(
                        ">       random_grad = torch.randn(cpu_res.shape, dtype=cpu_res.dtype)"
                    ),
                    M.last_line(
                        M.contains(
                            "infra/testers/single_chip/model/torch_model_tester.py:"
                        )
                    ),
                ],
            ),
        ],
    )

    NODE_ARITY_MISMATCH = FailingReason(
        description="Node arity mismatch",
        checks=[
            # E               ValueError: Node arity mismatch; expected 291, but got 290.
            # /localdev/ctr-vbrkic/venv/sweeps/xla/lib/python3.11/site-packages/torch/utils/_pytree.py:935: ValueError
            ExceptionCheck(
                class_name="ValueError",
                component=ComponentChecker.TORCH.value,
                message=[
                    M.regex("Node arity mismatch; expected .*, but got .*"),
                ],
                error_log=[
                    M.last_line(M.contains("torch/utils/_pytree.py:")),
                ],
            ),
        ],
    )

    BAD_STATUS0R_ACCESS = FailingReason(
        description="Bad StatusOr access",
        checks=[
            # E           RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
            # env/venv/xla/lib/python3.11/site-packages/tt_torch/backend/backend.py:99: RuntimeError
            ExceptionCheck(
                class_name="RuntimeError",
                component=ComponentChecker.XLA.value,
                message=[
                    M.contains("Bad StatusOr access: INTERNAL: Error code: 13"),
                ],
                error_log=[
                    M.any(
                        M.last_line(M.contains("tt_torch/backend/backend.py:")),
                        M.last_line(M.contains("infra/runners/torch_device_runner.py")),
                    ),
                ],
            ),
        ],
    )
