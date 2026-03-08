# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import torch_xla.runtime as xr
from infra.connectors.torch_device_connector import TorchDeviceConnector
from infra.utilities import Framework, Mesh, Model
from tt_jax import serialize_compiled_artifacts_to_disk
from tt_torch import parse_compiled_artifacts_from_cache_to_disk


class Workload:
    """Class encapsulating workload (executable/model with its inputs).

    Workload needs both model and executable fields depending on the tests,
    for example model tests use both and op tests use only executable.
    Please also pay attention that run_on_cpu decorator used in _match_data_types()
    creates a workload with executable.
    """

    def __init__(
        self,
        framework: Framework,
        executable: Optional[Callable] = None,
        compiled_executable: Optional[Callable] = None,
        model: Optional[Model] = None,
        args: Optional[Sequence[Any]] = None,
        kwargs: Optional[Mapping[str, Any]] = None,
        static_argnames: Optional[Sequence[str]] = None,
    ) -> None:

        print(f"\n[DEBUG][Workload.__init__] CALLED", flush=True)
        print(f"  framework = {framework}", flush=True)
        print(f"  executable = {type(executable).__name__ if executable else None}", flush=True)
        print(f"  compiled_executable = {type(compiled_executable).__name__ if compiled_executable else None}", flush=True)
        print(f"  model = {type(model).__name__ if model else None}", flush=True)
        print(f"  args count = {len(args) if args else 0}", flush=True)
        print(f"  kwargs keys = {list(kwargs.keys()) if kwargs else []}", flush=True)

        self.framework = framework

        assert (
            executable is not None or model is not None
        ), f"Workload must either have executable or model provided"

        self.executable = executable
        self.compiled_executable = compiled_executable
        self.model = model

        assert (
            args is not None or kwargs is not None
        ), f"Workload must either have args or kwargs provided"

        self.args = args or []
        self.kwargs = kwargs or {}
        # TODO: Move static_argnames out of Workload.
        # This field is JAX-specific and only used in compile functions.
        # Currently needed because _safely_put_workload_on_device relies on it to avoid putting those args on device.
        # Consider reworking _safely_put_workload_on_device to eliminate the need for static_argnames in Workload.
        self.static_argnames = static_argnames or []
        print(f"[DEBUG][Workload.__init__] DONE", flush=True)

    @property
    def is_jax(self) -> bool:
        return self.framework == Framework.JAX

    @property
    def is_torch(self) -> bool:
        return self.framework == Framework.TORCH

    def execute(self) -> Any:
        """Calls callable passing stored args and kwargs directly."""
        print(f"[DEBUG][Workload.execute] CALLED — has compiled_executable={self.compiled_executable is not None}, has model={self.model is not None}, has executable={self.executable is not None}", flush=True)
        if self.compiled_executable is not None:
            print(f"[DEBUG][Workload.execute] Using compiled_executable ({type(self.compiled_executable).__name__}) with {len(self.args)} args, {list(self.kwargs.keys())} kwargs", flush=True)
            result = self.compiled_executable(*self.args, **self.kwargs)
            print(f"[DEBUG][Workload.execute] DONE", flush=True)
            return result
        elif self.model is not None:
            print(f"[DEBUG][Workload.execute] Using model ({type(self.model).__name__})", flush=True)
            return self.model(*self.args, **self.kwargs)
        elif self.executable is not None:
            print(f"[DEBUG][Workload.execute] Using executable ({type(self.executable).__name__})", flush=True)
            return self.executable(*self.args, **self.kwargs)
        else:
            raise ValueError(
                "No model, compiled_executable, or executable provided in Workload."
            )

    def serialize(self, output_prefix: str, compiler_options=None) -> None:
        """Serialize the workload compilation artifacts to disk.

        Args:
            output_prefix: Base path and filename prefix for output files.
            compiler_options: Optional JAX compiler options dict (ignored for Torch)
        """
        if self.is_jax:
            # Get the executable to serialize
            executable = self.model if self.model else self.executable
            if executable is None:
                raise ValueError("No executable or model to serialize")

            # Serialize with the workload's args and kwargs
            serialize_compiled_artifacts_to_disk(
                executable,
                *self.args,
                output_prefix=output_prefix,
                compiler_options=compiler_options,
                **self.kwargs,
            )
        elif self.is_torch:
            cache_dir = TorchDeviceConnector.get_cache_dir()

            # Clear existing cache files to ensure we get exactly one file
            cache_dir_path = Path(cache_dir)
            if cache_dir_path.exists():
                for f in cache_dir_path.iterdir():
                    if f.is_file():
                        f.unlink()

            xr.clear_computation_cache()

            self.execute()
            parse_compiled_artifacts_from_cache_to_disk(cache_dir, output_prefix)

            # Recreate the cache directory after parsing
            Path(cache_dir).mkdir(parents=True, exist_ok=True)

        else:
            raise ValueError(
                f"Unsupported framework for serialization: {self.framework}"
            )
