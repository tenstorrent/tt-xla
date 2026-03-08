# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from infra.evaluators import (
    ComparisonConfig,
    Evaluator,
    EvaluatorFactory,
    QualityConfig,
)
from infra.runners import DeviceRunner, DeviceRunnerFactory
from infra.utilities import Framework, sanitize_test_name

from tests.infra.utilities.filecheck_utils import (
    run_filecheck,
    validate_filecheck_results,
)

FILECHECK_DIR = Path(__file__).parent.parent.parent / "filecheck"


def _debug_summarize(obj, max_depth=2):
    """Summarize an object for debug printing without dumping huge tensors."""
    import torch
    if isinstance(obj, torch.Tensor):
        return f"Tensor(shape={list(obj.shape)}, dtype={obj.dtype}, device={obj.device})"
    elif isinstance(obj, torch.nn.Module):
        return f"{type(obj).__name__}(params={sum(p.numel() for p in obj.parameters())})"
    elif isinstance(obj, (list, tuple)):
        if max_depth <= 0:
            return f"{type(obj).__name__}(len={len(obj)})"
        items = [_debug_summarize(x, max_depth - 1) for x in obj[:5]]
        suffix = f"...+{len(obj)-5} more" if len(obj) > 5 else ""
        return f"{type(obj).__name__}([{', '.join(items)}{suffix}])"
    elif isinstance(obj, dict):
        if max_depth <= 0:
            return f"dict(keys={list(obj.keys())[:5]})"
        items = {k: _debug_summarize(v, max_depth - 1) for k, v in list(obj.items())[:5]}
        return f"dict({items})"
    elif isinstance(obj, str):
        return repr(obj[:100])
    else:
        return repr(obj)


class BaseTester(ABC):
    """Abstract base class all testers must inherit."""

    def __init__(
        self,
        evaluator_type: str,
        comparison_config: Optional[ComparisonConfig] = None,
        framework: Optional[Framework] = None,
        quality_config: Optional[QualityConfig] = None,
        metric_names: Optional[List[str]] = None,
        metric_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        """Protected constructor for subclasses to use."""
        print(f"\n[DEBUG][BaseTester.__init__] CALLED", flush=True)
        print(f"  evaluator_type = {evaluator_type}", flush=True)
        print(f"  comparison_config = {comparison_config}", flush=True)
        print(f"  framework = {framework}", flush=True)
        print(f"  quality_config = {quality_config}", flush=True)
        print(f"  metric_names = {metric_names}", flush=True)
        print(f"  metric_kwargs = {metric_kwargs}", flush=True)
        self._evaluator_type = evaluator_type
        self._comparison_config = (
            comparison_config if comparison_config is not None else ComparisonConfig()
        )
        self._framework = framework
        self._quality_config = (
            quality_config if quality_config is not None else QualityConfig()
        )
        self._metric_names = metric_names
        self._metric_kwargs = metric_kwargs

        # Placeholders for objects that will be set during
        # `_initialize_framework_specific_helpers`. Easier to spot if located in
        # constructor instead of dynamically creating them somewhere in methods.
        self._device_runner: DeviceRunner = None
        self._evaluator: Optional[Evaluator] = None

        # Automatically initialize framework-specific helpers
        self._initialize_framework_specific_helpers()
        print(f"[DEBUG][BaseTester.__init__] DONE — device_runner={type(self._device_runner).__name__}, evaluator={type(self._evaluator).__name__ if self._evaluator else None}", flush=True)

    def _initialize_framework_specific_helpers(self) -> None:
        """
        Initializes `self._device_runner` and `self._evaluator`.

        Based on the framework instantiates a DeviceRunner (which internally
        instantiates a DeviceConnector singleton, ensuring plugin registration and
        connection to the device) and a Comparator, instantiates and stores the concrete
        model instance and finally packs model or its forward method and its arguments
        in a Workload.

        This function triggers connection to device.
        """
        print(f"\n[DEBUG][BaseTester._initialize_framework_specific_helpers] CALLED — framework={self._framework}", flush=True)
        if self._framework is not None:
            # Creating runner will register plugin and connect the device properly.
            self._device_runner = DeviceRunnerFactory.create_runner(self._framework)
            print(f"[DEBUG][BaseTester._initialize_framework_specific_helpers] Created device_runner: {type(self._device_runner).__name__}", flush=True)
        self._initialize_evaluator()

    def _initialize_evaluator(self) -> None:
        """Initialize evaluator using factory with stored params."""
        print(f"\n[DEBUG][BaseTester._initialize_evaluator] CALLED — evaluator_type={self._evaluator_type}", flush=True)
        # Skip if quality evaluator needs lazy init (no metric_names yet)
        if self._evaluator_type == "quality" and not self._metric_names:
            print(f"[DEBUG][BaseTester._initialize_evaluator] SKIPPED (quality evaluator, no metric_names)", flush=True)
            return

        self._evaluator = EvaluatorFactory.create_evaluator(
            evaluation_type=self._evaluator_type,
            framework=self._framework,
            comparison_config=self._comparison_config,
            quality_config=self._quality_config,
            metric_names=self._metric_names,
            metric_kwargs=self._metric_kwargs,
        )
        print(f"[DEBUG][BaseTester._initialize_evaluator] Created evaluator: {type(self._evaluator).__name__}", flush=True)

    def serialize_compilation_artifacts(
        self, test_name: str, workload: Workload
    ) -> None:
        """Serialize the model with the appropriate output prefix.

        Args:
            test_name: Test name to generate output prefix from.
        """
        clean_name = sanitize_test_name(test_name)
        output_prefix = f"output_artifact/{clean_name}"
        self.serialize_on_device(workload, output_prefix)

    @abstractmethod
    def serialize_on_device(self, workload: Workload, output_prefix: str) -> None:
        """
        Serializes the model workload on TT device with proper compiler configuration.

        Args:
            workload: Workload to serialize
            output_prefix: Base path and filename prefix for output files
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def handle_filecheck_and_serialization(self, request, workload: Workload) -> None:
        """
        Serializes workload if --serialize flag is set or filecheck patterns are specified,
        then runs filecheck validation if patterns are provided.

        Args:
            request: pytest request fixture
            workload: Workload to serialize
        """
        if not request:
            return

        test_id = request.node.name

        # Check if serialization is requested
        serialize = request.config.getoption("--serialize", False)

        # Check for filecheck pattern files from pytest marker
        filecheck_marker = request.node.get_closest_marker("filecheck")
        pattern_files = (
            filecheck_marker.args[0]
            if filecheck_marker and filecheck_marker.args
            else None
        )

        # Serialize workload if requested OR if pattern files are specified
        if serialize or pattern_files:
            self.serialize_compilation_artifacts(test_name=test_id, workload=workload)

        # Run filecheck if pattern files are specified
        if pattern_files:
            self._run_filecheck(pattern_files, test_id=test_id)

    def _run_filecheck(self, pattern_files: list, test_id: str) -> None:
        """Run filecheck with validation."""
        self._validate_filecheck_mark(
            pattern_files, test_id=test_id, where="pytest mark"
        )

        filecheck_results = run_filecheck(
            test_node_name=test_id,
            irs_filepath="output_artifact",
            pattern_files=pattern_files,
        )
        validate_filecheck_results(filecheck_results)

    def _validate_filecheck_mark(
        self, pattern_files, *, test_id: str, where: str
    ) -> None:
        """Validate filecheck marker arguments."""
        if not pattern_files:
            return
        if not isinstance(pattern_files, list):
            print(
                f"WARNING: 'filecheck' mark should pass a list in {where}. Found: {type(pattern_files).__name__}"
            )
            return
        for pattern_file in pattern_files:
            if not isinstance(pattern_file, str):
                print(
                    f"WARNING: filecheck entry should be a string in {where}. Found: {type(pattern_file).__name__}"
                )
                continue
            pattern_path = FILECHECK_DIR / pattern_file
            if not pattern_path.exists():
                print(
                    f"WARNING: filecheck pattern file not found: {pattern_path}\n         Referenced in test '{test_id}'"
                )
