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
from tests.infra.testers.tester import Tester


class JaxMultichipOpTester:
    """Tester for evaluating operations in a multichip JAX execution environment.

    Uses a Tester internally for device runner and evaluator setup, but manages
    mesh creation and multichip-specific compilation/execution directly.
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

        # Use Tester for device runner and evaluator
        self._tester = Tester(
            Framework.JAX, comparison_config, compiler_config=compiler_config
        )
        self._device_runner = self._tester.device_runner

        self._device_mesh: jax.sharding.Mesh = None
        self._cpu_mesh: jax.sharding.Mesh = None
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
        request=None,
    ) -> None:
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
        self.test(device_workload, cpu_workload, request=request)

    def test(
        self,
        device_workload: JaxMultichipWorkload,
        cpu_workload: JaxMultichipWorkload,
        request=None,
    ) -> None:
        if request:
            if request.config.getoption(
                "--serialize", False
            ) or request.node.get_closest_marker("filecheck"):
                assert (
                    False
                ), "Serialization/filecheck not supported through JAX multichip op/graph testers yet."

        with self._device_mesh:
            self._compile_for_tt_device(device_workload)
            device_res = self._run_on_multichip_device(device_workload)

        with self._cpu_mesh:
            self._compile_for_cpu(cpu_workload)
            cpu_res = self._run_on_multichip_device(cpu_workload)

        self._tester.evaluator.evaluate(device_res, cpu_res)

    def _compile_for_cpu(self, workload: Workload) -> None:
        assert isinstance(workload, JaxMultichipWorkload)
        compile_jax_multichip_workload(workload, compiler_options={})

    def _compile_for_tt_device(self, workload: Workload) -> None:
        assert isinstance(workload, JaxMultichipWorkload)
        compile_jax_multichip_workload(
            workload, self._compiler_config.to_jax_compiler_options()
        )

    def _run_on_multichip_device(self, compiled_workload: Workload) -> Tensor:
        assert isinstance(self._device_runner, JaxDeviceRunner)
        return self._device_runner.run_on_multichip_device(compiled_workload)

    def _get_tt_device_mesh(self) -> jax.sharding.Mesh:
        assert isinstance(self._device_runner, JaxDeviceRunner) and isinstance(
            self._device_runner.connector, JaxDeviceConnector
        )
        return self._device_runner.connector.get_tt_device_mesh(
            self._mesh_shape, self._axis_names
        )

    def _get_cpu_device_mesh(self) -> jax.sharding.Mesh:
        assert isinstance(self._device_runner, JaxDeviceRunner) and isinstance(
            self._device_runner.connector, JaxDeviceConnector
        )
        return self._device_runner.connector.get_cpu_device_mesh(
            self._mesh_shape, self._axis_names
        )

    def serialize_on_device(
        self, workload: JaxMultichipWorkload, output_prefix: str
    ) -> None:
        compiler_options = self._compiler_config.to_jax_compiler_options()

        with self._device_mesh:
            self._compile_for_tt_device(workload)

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
    request=None,
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
            executable, input_shapes, sharding_mode, minval, maxval, request=request
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
    """Serializes a JAX multichip op with given inputs to disk."""
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
    """Serializes a JAX multichip op with random inputs to disk."""
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
