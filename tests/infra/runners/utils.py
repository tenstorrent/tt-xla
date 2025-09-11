# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from functools import wraps
from typing import Callable

from infra.utilities import Framework
from infra.workloads.workload import Workload

from .device_runner_factory import DeviceRunnerFactory


def run_on_tt_device(framework: Framework):
    """Runs any decorated function `f` on TT device."""

    def decorator(f: Callable):
        @wraps(f)
        def wrapper(*args, **kwargs):
            workload = Workload(
                framework=framework, executable=f, args=args, kwargs=kwargs
            )
            runner = DeviceRunnerFactory.create_runner(framework)
            return runner.run_on_tt_device(workload)

        return wrapper

    return decorator


def run_on_cpu(framework: Framework):
    """Runs any decorated function `f` on CPU."""

    def decorator(f: Callable):
        @wraps(f)
        def wrapper(*args, **kwargs):
            workload = Workload(
                framework=framework, executable=f, args=args, kwargs=kwargs
            )
            runner = DeviceRunnerFactory.create_runner(framework)
            return runner.run_on_cpu(workload)

        return wrapper

    return decorator
