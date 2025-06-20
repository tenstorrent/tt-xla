# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Callable, Sequence

import jax
from comparators import ComparisonConfig
from connectors import DeviceConnectorFactory
from connectors.jax_device_connector import JaxDeviceConnector
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding
from runners.jax_device_runner import JaxDeviceRunner
from utilities.types import Framework
from utilities.workloads.jax_workload import (
    JaxMultichipWorkload,
    ShardingMode,
    enable_shardy,
)

from .base_tester import BaseTester


class JaxMultichipOpTester(BaseTester):
    """
    A tester for evaluating operations in a multichip JAX execution environment.

    This class extends `BaseTester` and provides functionality for testing
    operations using a specified device mesh, input sharding specifications,
    and output sharding specifications.

    TODO this class is hardcoded to jax. No multichip support for torch in infra yet.

    Attributes:
        device_mesh (jax.Mesh): The device mesh over which the computation is distributed.
        in_specs (tuple): The sharding specifications for the input tensors.
        out_specs (jax.sharding.PartitionSpec): The sharding specification for the output tensor.
    """

    # -------------------- Public methods --------------------

    def __init__(
        self,
        in_specs: tuple[jax.sharding.PartitionSpec],
        out_specs: jax.sharding.PartitionSpec,
        mesh_shape: tuple,
        axis_names: tuple,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        framework: Framework = Framework.JAX,
    ) -> None:
        self._in_specs = in_specs
        self._out_specs = out_specs
        self._mesh_shape = mesh_shape
        self._axis_names = axis_names
        # Placeholders for objects that will be set in `_initialize_all_components`.
        # Easier to spot if located in constructor instead of dynamically creating them
        # somewhere in methods.
        self._device_mesh: jax.sharding.Mesh = None
        self._cpu_mesh: jax.sharding.Mesh = None

        super().__init__(comparison_config, framework)

    def test(
        self,
        multichip_workload: JaxMultichipWorkload,
        cpu_workload: JaxMultichipWorkload,
        sharding_mode: ShardingMode,
    ) -> None:
        """
        Runs test by running `workload` on TT device and 'cpu_workload' on the CPU and comparing the results.
        """
        assert isinstance(self._device_runner, JaxDeviceRunner)

        with self._device_mesh:
            compiled_device_workload = JaxMultichipWorkload(
                self._compile_for_device(multichip_workload.executable, sharding_mode),
                multichip_workload.args,
                multichip_workload.kwargs,
                device_mesh=self._device_mesh,
                in_specs=self._in_specs,
            )
            device_res = self._device_runner.run_on_multichip_device(
                compiled_device_workload, sharding_mode
            )

        with self._cpu_mesh:
            compiled_cpu_workload = JaxMultichipWorkload(
                self._compile_for_cpu(cpu_workload.executable, sharding_mode),
                cpu_workload.args,
                cpu_workload.kwargs,
                device_mesh=self._cpu_mesh,
                in_specs=self._in_specs,
            )

            cpu_res = self._device_runner.run_on_multichip_device(
                compiled_cpu_workload, sharding_mode
            )

        self._compare(device_res, cpu_res)

    def test_with_random_inputs(
        self,
        executable: Callable,
        input_shapes: Sequence[tuple],
        sharding_mode: ShardingMode,
        minval: float = 0.0,
        maxval: float = 1.0,
    ) -> None:
        """
        Tests an input executable with random inputs in range [`minval`, `maxval`) by running it on
        a mesh of TT devices and comparing it to output of the cpu executable ran with the same
        input.
        """
        inputs = [
            jax.random.uniform(
                key=jax.random.key(0), shape=shape, minval=minval, maxval=maxval
            )
            for shape in input_shapes
        ]
        device_workload = JaxMultichipWorkload(
            executable,
            inputs,
            device_mesh=self._device_mesh,
            in_specs=self._in_specs,
        )
        cpu_workload = JaxMultichipWorkload(
            executable,
            inputs,
            device_mesh=self._cpu_mesh,
            in_specs=self._in_specs,
        )

        self.test(device_workload, cpu_workload, sharding_mode)

    # -------------------- Private methods --------------------

    # @override
    def _initialize_all_components(self) -> None:
        self._initialize_framework_specific_helpers()
        self._initialize_meshes()

    def _initialize_meshes(self) -> None:
        # TODO hardcoded to Jax classes.
        assert isinstance(self._device_runner, JaxDeviceRunner) and isinstance(
            self._device_runner.connector, JaxDeviceConnector
        )
        self._device_mesh = self._device_runner.connector.get_tt_device_mesh(
            self._mesh_shape, self._axis_names
        )
        self._cpu_mesh = self._device_runner.connector.get_cpu_device_mesh(
            self._mesh_shape, self._axis_names
        )

    def _compile_for_cpu(
        self,
        executable: Callable,
        sharding_mode: ShardingMode,
        static_argnames: Sequence[str] = None,
    ) -> Callable:
        """Sets up `executable` for just-in-time compile and execution on CPU"""
        module_sharded = (
            shard_map(
                executable,
                mesh=self._cpu_mesh,
                in_specs=self._in_specs,
                out_specs=self._out_specs,
            )
            if sharding_mode.requires_shard_map
            else executable
        )
        output_sharding = NamedSharding(self._cpu_mesh, self._out_specs)
        return jax.jit(
            module_sharded,
            out_shardings=output_sharding,
            static_argnames=static_argnames,
        )

    def _compile_for_device(
        self,
        executable: Callable,
        sharding_mode: ShardingMode,
        static_argnames: Sequence[str] = None,
    ) -> Callable:
        """Sets up executable for just-in-time compile and execution on multichip device."""
        module_sharded = (
            shard_map(
                executable,
                mesh=self._device_mesh,
                in_specs=self._in_specs,
                out_specs=self._out_specs,
            )
            if sharding_mode.requires_shard_map
            else executable
        )
        output_sharding = NamedSharding(self._device_mesh, self._out_specs)
        return jax.jit(
            module_sharded,
            out_shardings=output_sharding,
            static_argnames=static_argnames,
        )


def run_multichip_test_with_random_inputs(
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
    framework: Framework = Framework.JAX,
) -> None:
    """
    Tests an input executable with random inputs in range [`minval`, `maxval`) by running it on a
    mesh of TT devices and comparing it to output of the cpu executable ran with the same input.
    The xla backend used the shardy dialect if `use_shardy` is True, otherwise it uses GSPMD.
    """
    device_connector = DeviceConnectorFactory(framework).create_connector()
    assert isinstance(device_connector, JaxDeviceConnector)

    with enable_shardy(use_shardy), device_connector.simulate_cpu_mesh(mesh_shape):
        tester = JaxMultichipOpTester(
            in_specs, out_specs, mesh_shape, axis_names, comparison_config
        )
        tester.test_with_random_inputs(
            executable, input_shapes, sharding_mode, minval, maxval
        )
