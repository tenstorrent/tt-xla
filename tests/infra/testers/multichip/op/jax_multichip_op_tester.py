# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Callable, Sequence

import jax
from infra.comparators import ComparisonConfig
from infra.connectors import JaxDeviceConnector
from infra.runners import JaxDeviceRunner
from infra.utilities import (
    Framework,
    ShardingMode,
    Tensor,
    enable_shardy,
    random_tensor,
)
from infra.workloads import JaxMultichipWorkload, Workload
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding

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

    # -------------------- Public methods --------------------

    def __init__(
        self,
        in_specs: tuple[jax.sharding.PartitionSpec],
        out_spec: jax.sharding.PartitionSpec,
        mesh_shape: tuple,
        axis_names: tuple,
        comparison_config: ComparisonConfig = ComparisonConfig(),
    ) -> None:
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

    def test(
        self, device_workload: JaxMultichipWorkload, cpu_workload: JaxMultichipWorkload
    ) -> None:
        """
        Runs test by running `workload` on TT device and 'cpu_workload' on the CPU and
        comparing the results.
        """
        with self._device_mesh:
            compiled_device_workload = self._compile(device_workload)
            device_res = self._run_on_multichip_device(compiled_device_workload)

        with self._cpu_mesh:
            compiled_cpu_workload = self._compile(cpu_workload)
            cpu_res = self._run_on_multichip_device(compiled_cpu_workload)

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
        Tests an input executable with random inputs in range [`minval`, `maxval`) by
        running it on a mesh of TT devices and comparing it to output of the cpu
        executable ran with the same input.
        """
        inputs = [
            random_tensor(shape=shape, minval=minval, maxval=maxval)
            for shape in input_shapes
        ]
        device_workload = JaxMultichipWorkload(
            executable,
            inputs,
            device_mesh=self._device_mesh,
            in_specs=self._in_specs,
            out_spec=self._out_spec,
            sharding_mode=sharding_mode,
        )
        cpu_workload = JaxMultichipWorkload(
            executable,
            inputs,
            device_mesh=self._cpu_mesh,
            in_specs=self._in_specs,
            out_spec=self._out_spec,
            sharding_mode=sharding_mode,
        )
        self.test(device_workload, cpu_workload)

    # -------------------- Private methods --------------------

    # @override
    def _initialize_all_components(self) -> None:
        self._initialize_meshes()

    def _initialize_meshes(self) -> None:
        self._device_mesh = self._get_tt_device_mesh()
        self._cpu_mesh = self._get_cpu_device_mesh()

    # @override
    def _compile(self, workload: Workload) -> Workload:
        """
        Sets up `workload.executable` for just-in-time compile and execution.

        `workload.device_mesh` defines for which device (TT or CPU) it will be compiled.
        """
        assert isinstance(workload, JaxMultichipWorkload)

        module_sharded_executable = (
            shard_map(
                workload.executable,
                mesh=workload.device_mesh,
                in_specs=workload.in_specs,
                out_specs=workload.out_spec,
            )
            if workload.sharding_mode.requires_shard_map
            else workload.executable
        )
        output_sharding = NamedSharding(workload.device_mesh, workload.out_spec)

        workload.executable = jax.jit(
            module_sharded_executable,
            out_shardings=output_sharding,
            static_argnames=workload.static_argnames,
        )
        return workload

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
) -> None:
    """
    Tests an input executable with random inputs in range [`minval`, `maxval`) by
    running it on a mesh of TT devices and comparing it to output of the cpu executable
    ran with the same input. The xla backend used the shardy dialect if `use_shardy` is
    True, otherwise it uses GSPMD.
    """
    with enable_shardy(use_shardy):
        tester = JaxMultichipOpTester(
            in_specs, out_specs, mesh_shape, axis_names, comparison_config
        )
        tester.test_with_random_inputs(
            executable, input_shapes, sharding_mode, minval, maxval
        )
