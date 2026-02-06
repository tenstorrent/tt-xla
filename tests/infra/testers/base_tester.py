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
        if self._framework is not None:
            # Creating runner will register plugin and connect the device properly.
            self._device_runner = DeviceRunnerFactory.create_runner(self._framework)
        self._initialize_evaluator()

    def _initialize_evaluator(self) -> None:
        """Initialize evaluator using factory with stored params."""
        # Skip if quality evaluator needs lazy init (no metric_names yet)
        if self._evaluator_type == "quality" and not self._metric_names:
            return

        self._evaluator = EvaluatorFactory.create_evaluator(
            evaluation_type=self._evaluator_type,
            framework=self._framework,
            comparison_config=self._comparison_config,
            quality_config=self._quality_config,
            metric_names=self._metric_names,
            metric_kwargs=self._metric_kwargs,
        )

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
