# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Callable, Sequence

import jax
from infra.connectors import JaxDeviceConnector
from infra.evaluators import ComparisonConfig
from infra.runners import JaxDeviceRunner
from infra.utilities import (
    Framework,
    ShardingMode,
    Tensor,
    compile_jax_multichip_workload,
    enable_shardy,
    random_tensor,
    sanitize_test_name,
)
from infra.workloads import JaxMultichipWorkload, Workload

from tests.infra.testers.compiler_config import CompilerConfig

from ...base_tester import BaseTester


class JaxMultichipOpTester(BaseTester):
    """
    A tester for evaluating operations in a multichip JAX execution environment.

    This class extends `BaseTester` and provides functionality for testing
    operations using a specified device mesh, input sharding specifications,
    and output sharding specifications.

    Attributes
    ----------
        _device_mesh: jax.Mesh
            The device mesh over which the computation is distributed.

        _in_spec: tuple[jax.sharding.PartitionSpec]
            The sharding specifications for the input tensors.

        _out_spec: jax.sharding.PartitionSpec
            The sharding specification for the output tensor.
    """

    def __init__(
        self,
        in_specs: tuple[jax.sharding.PartitionSpec],
        out_spec: jax.sharding.PartitionSpec,
        mesh_shape: tuple,
        axis_names: tuple,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        compiler_config: CompilerConfig = None,
    ) -> None:
        if compiler_config is None:
            compiler_config = CompilerConfig()
        self._compiler_config = compiler_config
        self._in_specs = in_specs
        self._out_spec = out_spec
        self._mesh_shape = mesh_shape
        self._axis_names = axis_names
        # Placeholders for objects that will be set in `_initialize_meshes`.
        # Easier to spot if located in constructor instead of dynamically creating them
        # somewhere in methods.
        self._device_mesh: jax.sharding.Mesh = None
        self._cpu_mesh: jax.sharding.Mesh = None

        super().__init__(comparison_config, Framework.JAX)
        self._initialize_meshes()

    def _initialize_meshes(self) -> None:
        self._device_mesh = self._get_tt_device_mesh()
        self._cpu_mesh = self._get_cpu_device_mesh()

    def test_with_random_inputs(
        self,
        executable: Callable,
        input_shapes: Sequence[tuple],
        sharding_mode: ShardingMode,
        minval: float = 0.0,
        maxval: float = 1.0,
    ) -> None:
        """
        Tests an input executable with random inputs in range [`minval`, `maxval`) by
        running it on a mesh of TT devices and comparing it to output of the cpu
        executable ran with the same input.
        """
        inputs = [
            random_tensor(shape=shape, minval=minval, maxval=maxval)
            for shape in input_shapes
        ]
        device_workload = JaxMultichipWorkload(
            executable=executable,
            args=inputs,
            device_mesh=self._device_mesh,
            in_specs=self._in_specs,
            out_spec=self._out_spec,
            sharding_mode=sharding_mode,
        )
        cpu_workload = JaxMultichipWorkload(
            executable=executable,
            args=inputs,
            device_mesh=self._cpu_mesh,
            in_specs=self._in_specs,
            out_spec=self._out_spec,
            sharding_mode=sharding_mode,
        )
        self.test(device_workload, cpu_workload)

    def test(
        self, device_workload: JaxMultichipWorkload, cpu_workload: JaxMultichipWorkload
    ) -> None:
        """
        Runs test by running `workload` on TT device and 'cpu_workload' on the CPU and
        comparing the results.
        """
        with self._device_mesh:
            self._compile_for_tt_device(device_workload)
            device_res = self._run_on_multichip_device(device_workload)

        with self._cpu_mesh:
            self._compile_for_cpu(cpu_workload)
            cpu_res = self._run_on_multichip_device(cpu_workload)

        self._comparison_evaluator.compare(device_res, cpu_res)

    def _compile_for_cpu(self, workload: Workload) -> None:
        """Compile JAX multichip workload for CPU."""
        assert isinstance(workload, JaxMultichipWorkload)
        compile_jax_multichip_workload(workload, compiler_options={})

    def _compile_for_tt_device(self, workload: Workload) -> None:
        """Compile JAX multichip workload for TT device."""
        assert isinstance(workload, JaxMultichipWorkload)
        compile_jax_multichip_workload(
            workload, self._compiler_config.to_jax_compiler_options()
        )

    # --- Convenience wrappers ---

    def _run_on_multichip_device(self, compiled_workload: Workload) -> Tensor:
        """Runs multichip workload on a multichip device."""
        assert isinstance(self._device_runner, JaxDeviceRunner)
        return self._device_runner.run_on_multichip_device(compiled_workload)

    def _get_tt_device_mesh(self) -> jax.sharding.Mesh:
        """Returns TT device mesh with specified `shape` and `axis_names`."""
        assert isinstance(self._device_runner, JaxDeviceRunner) and isinstance(
            self._device_runner.connector, JaxDeviceConnector
        )
        return self._device_runner.connector.get_tt_device_mesh(
            self._mesh_shape, self._axis_names
        )

    def _get_cpu_device_mesh(self) -> jax.sharding.Mesh:
        """Returns CPU mesh with specified `shape` and `axis_names`."""
        assert isinstance(self._device_runner, JaxDeviceRunner) and isinstance(
            self._device_runner.connector, JaxDeviceConnector
        )
        return self._device_runner.connector.get_cpu_device_mesh(
            self._mesh_shape, self._axis_names
        )

    def serialize_on_device(
        self, workload: JaxMultichipWorkload, output_prefix: str
    ) -> None:
        """
        Serializes a workload on TT device with proper compiler configuration.

        Args:
            workload: The workload to serialize
            output_prefix: Base path and filename prefix for output files
        """
        compiler_options = self._compiler_config.to_jax_compiler_options()

        # For multichip, we need to compile the workload first within the device mesh
        with self._device_mesh:
            self._compile(workload, compiler_options)

        # Then serialize using the device runner
        self._device_runner.serialize_on_device(
            workload, output_prefix, compiler_options=compiler_options
        )


def run_jax_multichip_op_test_with_random_inputs(
    executable: Callable,
    input_shapes: Sequence[tuple],
    mesh_shape: tuple,
    axis_names: tuple,
    in_specs: Sequence[jax.sharding.PartitionSpec],
    out_specs: jax.sharding.PartitionSpec,
    use_shardy: bool,
    sharding_mode: ShardingMode,
    minval: float = 0.0,
    maxval: float = 1.0,
    comparison_config: ComparisonConfig = ComparisonConfig(),
    compiler_config: CompilerConfig = None,
) -> None:
    """
    Tests an input executable with random inputs in range [`minval`, `maxval`) by
    running it on a mesh of TT devices and comparing it to output of the cpu executable
    ran with the same input. The xla backend used the shardy dialect if `use_shardy` is
    True, otherwise it uses GSPMD.
    """
    with enable_shardy(use_shardy):
        tester = JaxMultichipOpTester(
            in_specs,
            out_specs,
            mesh_shape,
            axis_names,
            comparison_config,
            compiler_config,
        )
        tester.test_with_random_inputs(
            executable, input_shapes, sharding_mode, minval, maxval
        )


def serialize_jax_multichip_op(
    executable: Callable,
    inputs: Sequence[Tensor],
    output_prefix: str,
    mesh_shape: tuple,
    axis_names: tuple,
    in_specs: Sequence[jax.sharding.PartitionSpec],
    out_specs: jax.sharding.PartitionSpec,
    sharding_mode: ShardingMode,
    compiler_config: CompilerConfig = None,
) -> None:
    """
    Serializes a JAX multichip op with given inputs to disk.

    Args:
        executable: The operation/function to serialize
        inputs: Input tensors for the operation
        output_prefix: Base path and filename prefix for output files
        mesh_shape: Shape of the device mesh
        axis_names: Names of the mesh axes
        in_specs: Input sharding specifications
        out_specs: Output sharding specification
        sharding_mode: The sharding mode to use
        compiler_config: Compiler configuration options
    """
    if compiler_config is None:
        compiler_config = CompilerConfig()

    tester = JaxMultichipOpTester(
        in_specs,
        out_specs,
        mesh_shape,
        axis_names,
        compiler_config=compiler_config,
    )

    workload = JaxMultichipWorkload(
        executable=executable,
        args=inputs,
        device_mesh=tester._device_mesh,
        in_specs=in_specs,
        out_spec=out_specs,
        sharding_mode=sharding_mode,
    )

    # Serialize workload on TT device
    tester.serialize_on_device(workload, output_prefix)


def serialize_jax_multichip_op_with_random_inputs(
    executable: Callable,
    input_shapes: Sequence[tuple],
    test_name: str,
    mesh_shape: tuple,
    axis_names: tuple,
    in_specs: Sequence[jax.sharding.PartitionSpec],
    out_specs: jax.sharding.PartitionSpec,
    sharding_mode: ShardingMode,
    minval: float = 0.0,
    maxval: float = 1.0,
    compiler_config: CompilerConfig = None,
) -> None:
    """
    Serializes a JAX multichip op with random inputs to disk.

    Args:
        executable: The operation/function to serialize
        input_shapes: Shapes for random input generation
        test_name: Test name to generate output prefix from
        mesh_shape: Shape of the device mesh
        axis_names: Names of the mesh axes
        in_specs: Input sharding specifications
        out_specs: Output sharding specification
        sharding_mode: The sharding mode to use
        minval: Minimum value for random inputs (default: 0.0)
        maxval: Maximum value for random inputs (default: 1.0)
        compiler_config: Compiler configuration options
    """

    clean_name = sanitize_test_name(test_name)
    output_prefix = f"output_artifact/{clean_name}"

    inputs = [
        random_tensor(shape=shape, minval=minval, maxval=maxval)
        for shape in input_shapes
    ]

    serialize_jax_multichip_op(
        executable,
        inputs,
        output_prefix,
        mesh_shape,
        axis_names,
        in_specs,
        out_specs,
        sharding_mode,
        compiler_config,
    )
